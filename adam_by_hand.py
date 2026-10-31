"""Part 1: reproduce Adam by hand for one weight and five gradients, then check every quantity against PyTorch."""
import math
import torch
from common import *

lr, b1, b2, eps = 0.01, 0.9, 0.999, 1e-8
w0 = 1.0
grads = [0.5, -0.3, 0.8, 0.1, -0.6]

# ---- by hand: plain Python floats (IEEE double), straight from the paper --------------------------------------
m = v = 0.0; w = w0; hand = []
for t, g in enumerate(grads, 1):
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * g * g
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    step = lr * mhat / (math.sqrt(vhat) + eps)
    w = w - step
    hand.append(dict(t=t, g=g, m=m, v=v, mhat=mhat, vhat=vhat, step=step, w=w))

# ---- PyTorch: float64 parameter, torch.optim.Adam; m, v come straight from its state ---------------------------
p = torch.nn.Parameter(torch.tensor([w0], dtype=torch.float64))
opt = torch.optim.Adam([p], lr=lr, betas=(b1, b2), eps=eps)
ref = []
for t, g in enumerate(grads, 1):
    before = p.item()
    p.grad = torch.tensor([g], dtype=torch.float64); opt.step()
    st = opt.state[p]; m_t, v_t = st["exp_avg"].item(), st["exp_avg_sq"].item()
    ref.append(dict(m=m_t, v=v_t, mhat=m_t / (1 - b1 ** t), vhat=v_t / (1 - b2 ** t), step=before - p.item(), w=p.item()))

# ---- and my own `Adam` class from common.py (used everywhere else in this repo) --------------------------------
q = torch.nn.Parameter(torch.tensor([w0], dtype=torch.float64)); mine = Adam([q], (b1, b2), eps); mine_w = []
for g in grads:
    q.grad = torch.tensor([g], dtype=torch.float64); mine.step(lr); mine_w.append(q.item())

print(f"lr={lr} betas=({b1},{b2}) eps={eps} w0={w0}\n")
hdr = f"{'t':>2} {'g':>6} | {'m':>14} {'v':>14} {'m_hat':>14} {'v_hat':>14} {'step':>14} {'w':>14}"
print("BY HAND"); print(hdr)
for h in hand: print(f"{h['t']:>2} {h['g']:>6.2f} | {h['m']:>14.10f} {h['v']:>14.10e} {h['mhat']:>14.10f} {h['vhat']:>14.10e} {h['step']:>14.10f} {h['w']:>14.10f}")
print("\nPYTORCH torch.optim.Adam (float64)"); print(hdr)
for h, r in zip(hand, ref): print(f"{h['t']:>2} {h['g']:>6.2f} | {r['m']:>14.10f} {r['v']:>14.10e} {r['mhat']:>14.10f} {r['vhat']:>14.10e} {r['step']:>14.10f} {r['w']:>14.10f}")

print("\nmax |hand - torch| per quantity:")
diffs = {}
for k in ["m", "v", "mhat", "vhat", "step", "w"]:
    diffs[k] = max(abs(h[k] - r[k]) for h, r in zip(hand, ref)); print(f"  {k:<5} {diffs[k]:.3e}")
d_mine = max(abs(h["w"] - x) for h, x in zip(hand, mine_w)); print(f"  w (common.Adam vs hand) {d_mine:.3e}")

# ---- float32 reality check: same thing in the dtype people actually train in ------------------------------------
p32 = torch.nn.Parameter(torch.tensor([w0], dtype=torch.float32)); o32 = torch.optim.Adam([p32], lr=lr, betas=(b1, b2), eps=eps)
for g in grads: p32.grad = torch.tensor([g], dtype=torch.float32); o32.step()
d32 = abs(p32.item() - hand[-1]["w"]); print(f"  w (torch fp32 vs hand fp64) {d32:.3e}   (fp32 rounding, ~1e-8)")

# ---- whole-model check: common.Adam == torch.optim.Adam over 10 real steps of a real transformer ---------------
def traj(use_mine):
    torch.manual_seed(0); mdl = GPT(128, 4).to(dev); gen = torch.Generator().manual_seed(1)
    o = Adam(mdl.parameters()) if use_mine else torch.optim.Adam(mdl.parameters(), lr=1e-3)
    for _ in range(10):
        x, y = get_batch("train", 16, 128, gen); _, l = mdl(x, y); mdl.zero_grad(); l.backward()
        o.step(1e-3) if use_mine else o.step()
    return torch.cat([q.detach().flatten() for q in mdl.parameters()])
a, b = traj(True), traj(False)
print(f"\nwhole transformer, 10 steps, fp32: max |param diff| common.Adam vs torch.optim.Adam = {(a-b).abs().max():.2e} (params ~1e-2 in size)")

dump(dict(hyper=dict(lr=lr, b1=b1, b2=b2, eps=eps, w0=w0, grads=grads), hand=hand, torch=ref, max_abs_diff=diffs,
          common_adam_vs_hand=d_mine, fp32_vs_hand=d32, model_param_maxdiff=(a - b).abs().max().item()), "results_adam_by_hand.json")
