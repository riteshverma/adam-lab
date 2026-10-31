"""Part 3: log the update-to-weight ratio ||lr*adam_update|| / ||w|| for every parameter tensor, every step, with
different linear-warmup lengths (then constant lr), and find the step at which warmup stops changing it.

Method: the ratio is lr_t * ||m_hat/(sqrt(v_hat)+eps)|| / ||w||. During warmup lr_t rises linearly, so the ratio is pushed up;
once lr is constant the ratio can only fall (Adam's normalised update shrinks as gradient noise starts to dominate m_hat).
So the ratio peaks where the lr ramp stops pushing it. I measure that peak step for warmup lengths W = 25, 50, 100, 200
and check that it follows W, and I also measure how long the warmup run's ratio stays different from the no-warmup run's.
(LayerNorm biases are excluded: they start at exactly 0, so their ratio is undefined.)"""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

C, STEPS, PEAK = 256, 300, 3e-3
SEEDS = [0, 1, 2]
WS = [1, 25, 50, 100, 200]               # W=1 means "no warmup"
lr_fn = lambda W: (lambda t: PEAK * min(1.0, (t + 1) / W))

def run(W, seed):
    r = Run(C, 4, seed); out = [r.step(lr_fn(W)(t), log_ratio=True) for t in range(STEPS)]
    return r.names, torch.stack(out).cpu().numpy(), r.train_losses()

R, LOSS = {}, {}
for W in WS:
    runs = [run(W, s) for s in SEEDS]; names = runs[0][0]
    R[W] = np.mean([r[1] for r in runs], 0); LOSS[W] = np.mean([r[2] for r in runs], 0)
keep = [i for i, n in enumerate(names) if not n.endswith(".bias")]
def group(n):
    for k in ["wte", "wpe", "qkv", "proj", "fc", "out", "lm_head"]:
        if n.endswith(k + ".weight"): return k
    return "layernorm"
G = {}
for i in keep: G.setdefault(group(names[i]), []).append(i)
print("tensors per group:", {k: len(v) for k, v in G.items()}, "\n")
sm = lambda a, k=9: np.convolve(a, np.ones(k) / k, mode="same")

# ---- 1. ratio at selected steps, every layer group ---------------------------------------------------------------
print(f"update-to-weight ratio, mean of {len(SEEDS)} seeds, peak lr {PEAK}  (W=1: no warmup; W=100: linear warmup over 100 steps)")
print(f"{'group':<10}" + "".join(f"{'W=%d t=%d' % (W, t):>13}" for W, t in [(1, 1), (1, 20), (1, 100), (100, 1), (100, 50), (100, 100), (100, 150), (100, 300)]))
for k, idx in G.items():
    v = lambda W, t: R[W][t - 1, idx].mean()
    print(f"{k:<10}" + "".join(f"{v(W, t):>13.4f}" for W, t in [(1, 1), (1, 20), (1, 100), (100, 1), (100, 50), (100, 100), (100, 150), (100, 300)]))

# ---- 2. where does the ratio stop being pushed by the lr ramp? ---------------------------------------------------
print("\nstep at which the (smoothed) ratio peaks, by warmup length W:")
agg = {W: np.exp(np.mean(np.log(R[W][:, [i for i in keep if group(names[i]) != "layernorm"]]), 1)) for W in WS}   # geometric mean over the weight matrices
peaks = {}
print(f"{'W':>5} | {'all-matrix geo-mean':>20} | " + " ".join(f"{k:>7}" for k in G))
for W in WS:
    pk = int(np.argmax(sm(agg[W]))) + 1
    per = {k: int(np.argmax(sm(R[W][:, idx].mean(1)))) + 1 for k, idx in G.items()}
    peaks[W] = dict(overall=pk, per_group=per)
    print(f"{W:>5} | {pk:>20} | " + " ".join(f"{per[k]:>7}" for k in G))

# ---- 3. how long does a warmup run differ from the no-warmup run? ---------------------------------------------
print("\nmedian over tensors of |ratio_W / ratio_none - 1|, at selected steps (smoothed):")
Rn = np.stack([sm(R[1][:, j]) for j in keep], 1)
gapres = {}
print(f"{'W':>5}" + "".join(f"{'t=%d' % t:>9}" for t in [10, 50, 100, 150, 200, 250, 290]))
for W in WS[1:]:
    Rw_ = np.stack([sm(R[W][:, j]) for j in keep], 1); gap = np.median(np.abs(Rw_ / Rn - 1), 1); gapres[W] = gap
    print(f"{W:>5}" + "".join(f"{gap[t-1]:>9.2f}" for t in [10, 50, 100, 150, 200, 250, 290]))
print("\nfinal-ish train loss (mean of last 20 steps, 3 seeds): " + ", ".join(f"W={W}: {LOSS[W][-20:].mean():.3f}" for W in WS))
print("train loss at step 20:                                   " + ", ".join(f"W={W}: {LOSS[W][19]:.3f}" for W in WS))

dump(dict(C=C, peak=PEAK, peaks=peaks, loss_last20={W: float(LOSS[W][-20:].mean()) for W in WS}, loss_t20={W: float(LOSS[W][19]) for W in WS},
          ratio_by_group={W: {k: R[W][:, idx].mean(1).tolist() for k, idx in G.items()} for W in WS}, names=names), "results_ratios.json")

fig, ax = plt.subplots(1, 4, figsize=(22, 4.4)); t = np.arange(1, STEPS + 1); cols = dict(zip(G, plt.cm.tab10.colors))
for a, W in zip(ax[:2], [1, 100]):
    for k, idx in G.items(): a.plot(t, R[W][:, idx].mean(1), c=cols[k], label=k)
    a.set(title="No warmup: ||update||/||w|| per layer group" if W == 1 else "Linear warmup, 100 steps", yscale="log", xlabel="step", ylim=(1e-3, 0.6))
ax[0].legend(fontsize=8, ncol=2); ax[1].axvline(100, c="k", ls=":")
for W, c in zip(WS, plt.cm.viridis(np.linspace(0, .9, len(WS)))):
    ax[2].plot(t, agg[W], c=c, label=f"W={W}" if W > 1 else "no warmup"); ax[2].axvline(W, c=c, ls=":", lw=.8)
ax[2].set(title="Geo-mean ratio over weight matrices; dotted = end of warmup", yscale="log", xlabel="step"); ax[2].legend(fontsize=8)
ax[3].plot([W for W in WS[1:]], [peaks[W]["overall"] for W in WS[1:]], "o-", label="step where ratio peaks"); ax[3].plot([0, 220], [0, 220], "k--", label="y = W")
ax[3].set(title="Ratio peak step vs warmup length", xlabel="warmup length W", ylabel="peak step"); ax[3].legend()
savefig(fig, "ratios.png")
