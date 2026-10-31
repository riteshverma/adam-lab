"""Part 5: learning-rate sweep at widths 256, 512, 1024 (4 layers, standard parameterisation, 300 steps, warmup 20 + cosine to 10%),
loss vs LR with the three minima marked, then an extrapolation to width 4096.

Each width gets the same grid (sqrt(2) spacing) and the same seeds; the grid is extended downward/upward until the minimum is
strictly inside it, so no width is judged with its optimum sitting on the edge (the 'badly tuned baseline' failure)."""
import math, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

STEPS, WARM, MINR = 300, 20, 0.1
SEEDS = [0, 1]
WIDTHS = [256, 512, 1024]
B, T = 16, 128

def lr_at(t, peak):
    if t < WARM: return peak * (t + 1) / WARM
    p = (t - WARM) / (STEPS - WARM); return peak * (MINR + (1 - MINR) * 0.5 * (1 + math.cos(math.pi * p)))

def final_loss(C, peak, seed):
    r = Run(C, 4, seed, B=B, T=T)
    for t in range(STEPS): r.step(lr_at(t, peak))
    l = val_loss(r.model); tl = r.train_losses()[-1]
    del r; torch.cuda.empty_cache()
    return l if np.isfinite(l) and l < 10 else 10.0            # divergence is recorded as 10 (> ln V = 4.6, i.e. worse than uniform)

grid_exp = {256: range(-21, -14), 512: range(-22, -15), 1024: range(-23, -16)}   # initial window of k (lr = 2^(k/2)), extended below if the min is on the edge
res = {C: {} for C in WIDTHS}; t0 = time.time()
def ensure(C, k):
    if k in res[C]: return
    res[C][k] = [final_loss(C, 2.0 ** (k / 2), s) for s in SEEDS]                # peak lr = 2^(k/2): sqrt(2) spacing
for C in WIDTHS:
    lo, hi = min(grid_exp[C]), max(grid_exp[C])
    for k in range(lo, hi + 1): ensure(C, k)
    while True:                                                                  # extend until the minimum is strictly interior
        ks = sorted(res[C]); mean = {k: np.mean(res[C][k]) for k in ks}; kb = min(mean, key=mean.get)
        if kb == ks[0]: ensure(C, ks[0] - 1)
        elif kb == ks[-1]: ensure(C, ks[-1] + 1)
        else: break
    print(f"width {C}: {len(res[C])} lrs x {len(SEEDS)} seeds done ({time.time()-t0:.0f}s)", flush=True)

def analyse(C):
    ks = sorted(res[C]); lrs = np.array([2.0 ** (k / 2) for k in ks]); m = np.array([np.mean(res[C][k]) for k in ks]); sd = np.array([np.std(res[C][k], ddof=1) for k in ks])
    i = int(np.argmin(m))
    # quadratic fit in log2(lr) through the best point and its neighbours -> sub-grid minimum
    x = np.log2(lrs[i-1:i+2]); y = m[i-1:i+2]; a, b, _ = np.polyfit(x, y, 2)
    xs = -b / (2 * a) if a > 0 else x[1]; xs = float(np.clip(xs, x[0], x[-1]))
    # basin: LRs within 0.02 nats of the best grid point (a robustness measure)
    ok = lrs[m <= m[i] + 0.02]
    return dict(lrs=lrs, mean=m, sd=sd, i_best=i, lr_best_grid=float(lrs[i]), lr_best_fit=float(2 ** xs), loss_best=float(m[i]), basin=(float(ok.min()), float(ok.max())))
A = {C: analyse(C) for C in WIDTHS}

print(f"\nval loss after {STEPS} steps (mean of {len(SEEDS)} seeds; 10.0 = diverged). rows = peak lr")
all_lrs = sorted({k for C in WIDTHS for k in res[C]})
print(f"{'lr':>9} | " + " | ".join(f"{'C=%d' % C:>14}" for C in WIDTHS))
for k in all_lrs:
    cells = []
    for C in WIDTHS:
        if k in res[C]:
            m = np.mean(res[C][k]); cells.append(f"{m:>8.4f}{'*' if k == 2 * math.log2(A[C]['lr_best_grid']) else ' '}({np.std(res[C][k], ddof=1):.3f})")
        else: cells.append(" " * 14)
    print(f"{2 ** (k / 2):>9.3g} | " + " | ".join(cells))
print("\nminima:")
for C in WIDTHS:
    a = A[C]; print(f"  C={C:>5}: best grid lr {a['lr_best_grid']:.3g} (loss {a['loss_best']:.4f}); quadratic-fit lr* {a['lr_best_fit']:.3g}; within 0.02 nats of best: {a['basin'][0]:.3g} .. {a['basin'][1]:.3g}")

# ---- extrapolation: lr*(C) = a * C^-alpha, fit on the three minima (log-log) -------------------------------------
Cs = np.array(WIDTHS, float); y = np.log2([A[C]["lr_best_fit"] for C in WIDTHS])
slope, icpt = np.polyfit(np.log2(Cs), y, 1)
print(f"\nlog-log fit over 3 widths: lr* ~ C^{slope:.2f}   (standard parameterisation theory for Adam: -1 for hidden matrices)")
pred_free = 2 ** (icpt + slope * np.log2(4096))
pred_m1 = {C: A[C]["lr_best_fit"] * C / 4096 for C in WIDTHS}                     # exact 1/width scaling from each measured point
# bootstrap over seeds: resample seeds per (width, lr), refit the minima, refit the slope, predict 4096
rng = np.random.default_rng(0); boots_free, boots_m1 = [], []
for _ in range(400):
    ys = []
    for C in WIDTHS:
        ks = sorted(res[C]); mm = np.array([np.mean(rng.choice(res[C][k], len(SEEDS))) for k in ks]); i = int(np.argmin(mm))
        if 0 < i < len(ks) - 1:
            x = np.array(ks[i-1:i+2]) / 2.0; a, b, _ = np.polyfit(x, mm[i-1:i+2], 2); xs = np.clip(-b / (2 * a) if a > 0 else x[1], x[0], x[-1])
        else: xs = ks[i] / 2.0
        ys.append(xs)
    s_, i_ = np.polyfit(np.log2(Cs), ys, 1); boots_free.append(2 ** (i_ + s_ * 12)); boots_m1.append(2 ** (ys[-1]) * 1024 / 4096)
lo_f, hi_f = np.percentile(boots_free, [5, 95]); lo_m, hi_m = np.percentile(boots_m1, [5, 95])
print(f"width 4096 prediction, free exponent ({slope:.2f}): {pred_free:.2e}   bootstrap 90% interval {lo_f:.2e} .. {hi_f:.2e}")
print(f"width 4096 prediction, exponent fixed at -1 from the C=1024 minimum: {pred_m1[1024]:.2e}   (from C=256: {pred_m1[256]:.2e}, C=512: {pred_m1[512]:.2e}); bootstrap 90%: {lo_m:.2e} .. {hi_m:.2e}")

dump(dict(steps=STEPS, seeds=SEEDS, widths=WIDTHS, slope=slope, pred_4096_free=pred_free, pred_4096_m1_from_1024=pred_m1[1024], boot_free=[lo_f, hi_f], boot_m1=[lo_m, hi_m],
          per_width={C: dict(lr_best_grid=A[C]["lr_best_grid"], lr_best_fit=A[C]["lr_best_fit"], loss_best=A[C]["loss_best"], basin=A[C]["basin"],
                              curve={str(2 ** (k / 2)): res[C][k] for k in sorted(res[C])}) for C in WIDTHS}), "results_lr_sweep.json")

fig, ax = plt.subplots(1, 2, figsize=(14, 4.8)); cols = {256: "C0", 512: "C1", 1024: "C2"}
for C in WIDTHS:
    a = A[C]; ax[0].errorbar(a["lrs"], a["mean"], a["sd"], c=cols[C], marker="o", ms=4, capsize=2, label=f"width {C}")
    ax[0].scatter([a["lr_best_fit"]], [a["loss_best"]], c=cols[C], marker="*", s=380, edgecolors="k", zorder=5)
ax[0].set(xscale="log", xlabel="peak learning rate", ylabel="val loss after 300 steps", title="Loss vs LR (stars = minima)", ylim=(None, 3.6)); ax[0].legend()
ax[1].plot(WIDTHS, [A[C]["lr_best_fit"] for C in WIDTHS], "o", c="k", label="measured minima")
cc = np.array([256, 4096]); ax[1].plot(cc, 2 ** (icpt + slope * np.log2(cc)), "--", c="C3", label=f"free fit: C^{slope:.2f}")
ax[1].plot(cc, A[1024]["lr_best_fit"] * 1024 / cc, ":", c="C4", label="C^-1 from width 1024")
ax[1].errorbar([4096], [pred_free], [[pred_free - lo_f], [hi_f - pred_free]], c="C3", marker="s", capsize=4)
ax[1].errorbar([4096], [pred_m1[1024]], [[pred_m1[1024] - lo_m], [hi_m - pred_m1[1024]]], c="C4", marker="s", capsize=4)
ax[1].set(xscale="log", yscale="log", xlabel="width", ylabel="optimal peak lr", title="Extrapolation to width 4096 (bars: bootstrap 90%)"); ax[1].legend()
savefig(fig, "lr_sweep.png")
