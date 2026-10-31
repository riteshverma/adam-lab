"""Shared code: char-level corpus, a width-configurable GPT, and a hand-written Adam.

The Adam here is written from the paper's equations (not torch.optim.Adam) so that bias correction can be switched
off and the update-to-weight ratio can be logged. `adam_by_hand.py` proves it matches torch.optim.Adam.
"""
import copy, glob, json, math, os, sysconfig, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

dev = "cuda" if torch.cuda.is_available() else "cpu"
HERE = os.path.dirname(os.path.abspath(__file__))
FIGS = os.path.join(HERE, "figs"); os.makedirs(FIGS, exist_ok=True)


def load_data():
    files = sorted(glob.glob(os.path.join(sysconfig.get_paths()["stdlib"], "*.py")))
    text = ""
    for f in files:
        try: text += open(f, encoding="utf-8").read()
        except Exception: pass
        if len(text) > 3_000_000: break
    text = "".join(c for c in text if ord(c) < 128)
    chars = sorted(set(text)); stoi = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long, device=dev)
    n = int(0.9 * len(data))
    return data[:n], data[n:], len(chars)

TRAIN, VAL, V = load_data()


def get_batch(split, B, T, gen):
    d = TRAIN if split == "train" else VAL
    ix = torch.randint(len(d) - T - 1, (B,), generator=gen).to(d.device)
    idx = ix[:, None] + torch.arange(T, device=d.device)[None]
    return d[idx], d[idx + 1]


class Block(nn.Module):
    def __init__(s, C, H):
        super().__init__()
        s.H = H; s.ln1 = nn.LayerNorm(C); s.qkv = nn.Linear(C, 3*C, bias=False); s.proj = nn.Linear(C, C, bias=False)
        s.ln2 = nn.LayerNorm(C); s.fc = nn.Linear(C, 4*C, bias=False); s.out = nn.Linear(4*C, C, bias=False)
    def forward(s, x):
        B, T, C = x.shape; H = s.H; D = C // H
        q, k, v = s.qkv(s.ln1(x)).split(C, dim=2)
        q, k, v = [z.view(B, T, H, D).transpose(1, 2) for z in (q, k, v)]
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, C)
        x = x + s.proj(a)
        return x + s.out(F.gelu(s.fc(s.ln2(x))))


class GPT(nn.Module):
    """Standard parameterisation (GPT-2 init, std 0.02), pre-LN, untied head. Width C, heads C/64 (min 4)."""
    def __init__(s, C=256, L=4, T=128):
        super().__init__()
        H = max(4, C // 64)
        s.wte = nn.Embedding(V, C); s.wpe = nn.Embedding(T, C)
        s.blocks = nn.ModuleList([Block(C, H) for _ in range(L)])
        s.lnf = nn.LayerNorm(C); s.lm_head = nn.Linear(C, V, bias=False)
        for m in s.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)): nn.init.normal_(m.weight, std=0.02)
        for b in s.blocks:
            nn.init.normal_(b.proj.weight, std=0.02/math.sqrt(2*L)); nn.init.normal_(b.out.weight, std=0.02/math.sqrt(2*L))
    def forward(s, idx, targets=None):
        x = s.wte(idx) + s.wpe(torch.arange(idx.shape[1], device=idx.device))
        for b in s.blocks: x = b(x)
        logits = s.lm_head(s.lnf(x))
        loss = None if targets is None else F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def make_model(C, L=4, seed=0):
    torch.manual_seed(seed); return GPT(C, L).to(dev)


class Adam:
    """Adam from the paper's equations. State kept in fp32 next to fp32 master weights."""
    def __init__(s, params, betas=(0.9, 0.999), eps=1e-8, bias_correction=True):
        s.params = list(params); s.b1, s.b2 = betas; s.eps = eps; s.bc = bias_correction; s.t = 0
        s.m = [torch.zeros_like(p) for p in s.params]; s.v = [torch.zeros_like(p) for p in s.params]
    @torch.no_grad()
    def step(s, lr, log_ratio=False):
        s.t += 1; t = s.t; ratios = []
        c1 = 1 - s.b1 ** t if s.bc else 1.0
        c2 = 1 - s.b2 ** t if s.bc else 1.0
        for p, m, v in zip(s.params, s.m, s.v):
            g = p.grad
            m.mul_(s.b1).add_(g, alpha=1 - s.b1)
            v.mul_(s.b2).addcmul_(g, g, value=1 - s.b2)
            upd = (m / c1) / ((v / c2).sqrt() + s.eps)
            if log_ratio: ratios.append(lr * upd.norm() / p.norm())
            p.add_(upd, alpha=-lr)
        return torch.stack(ratios) if log_ratio else None
    def state(s): return copy.deepcopy((s.m, s.v, s.t))
    def load(s, st): s.m, s.v, s.t = copy.deepcopy(st[0]), copy.deepcopy(st[1]), st[2]


VAL_BATCHES = None
@torch.no_grad()
def val_loss(model, n=40, B=16, T=128):
    """Mean CE on a fixed set of held-out batches (same for every model, so differences are paired)."""
    global VAL_BATCHES
    if VAL_BATCHES is None:
        g = torch.Generator().manual_seed(123)
        VAL_BATCHES = [get_batch("val", B, T, g) for _ in range(n)]
    model.eval(); tot = 0.0
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"):
        for x, y in VAL_BATCHES[:n]: tot += model(x, y)[1].item()
    model.train(); return tot / n


class Run:
    """A resumable training run: step(lr) does one forward/backward/Adam step on the next batch of a seeded data stream."""
    def __init__(s, C=256, L=4, seed=0, B=16, T=128, clip=1.0, **adam_kw):
        s.model = make_model(C, L, seed); s.opt = Adam(s.model.parameters(), **adam_kw)
        s.gen = torch.Generator().manual_seed(1000 + seed); s.B, s.T, s.clip = B, T, clip
        s.names = [n for n, _ in s.model.named_parameters()]; s.losses = []; s.gnorms = []
    def step(s, lr, log_ratio=False):
        x, y = get_batch("train", s.B, s.T, s.gen)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev == "cuda"): _, loss = s.model(x, y)
        s.model.zero_grad(set_to_none=True); loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(s.model.parameters(), s.clip)
        r = s.opt.step(lr, log_ratio)
        s.losses.append(loss.detach()); s.gnorms.append(gn.detach()); return r
    def snapshot(s): return (copy.deepcopy(s.model.state_dict()), s.opt.state(), s.gen.get_state(), len(s.losses))
    def restore(s, snap):
        s.model.load_state_dict(snap[0]); s.opt.load(snap[1]); s.gen.set_state(snap[2])
        s.losses = s.losses[:snap[3]]; s.gnorms = s.gnorms[:snap[3]]
    def train_losses(s): return torch.stack(s.losses).float().cpu().numpy()


def savefig(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIGS, name), dpi=110); return name
def dump(obj, name):
    json.dump(obj, open(os.path.join(HERE, name), "w"), indent=1, default=float)
