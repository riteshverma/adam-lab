"""Part 4: 300-step cosine vs 300-step WSD, both stopped at step 200. Each schedule gets its own tuned peak LR
(the comparison is only accepted once both sides are tuned), 3 seeds, paired by seed (same init, same data stream)."""
import math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

C, TOTAL, WARM, STOP, COOL = 256, 300, 20, 200, 60
MINR = 0.1                                           # both schedules end at 10% of peak
DECAY_START = TOTAL - COOL                            # WSD: stable until 240, then decay for the last 20%
SEEDS = [0, 1, 2]
LRS = [5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3, 2.8e-3, 4e-3]      # fine grid (sqrt(2) steps) around the optimum; first coarse pass put WSD at the edge of the grid

def cosine(t, peak):
    if t < WARM: return peak * (t + 1) / WARM
    p = (t - WARM) / (TOTAL - WARM); return peak * (MINR + (1 - MINR) * 0.5 * (1 + math.cos(math.pi * p)))
def wsd(t, peak):
    if t < WARM: return peak * (t + 1) / WARM
    if t < DECAY_START: return peak
    p = (t - DECAY_START) / (TOTAL - DECAY_START); return peak * (1 - (1 - MINR) * p)
SCHED = dict(cosine=cosine, wsd=wsd)

def one(name, peak, seed):
    """train to 300 on `name`, recording val loss every 20 steps; at step 200 also run a 60-step linear cooldown branch."""
    f = SCHED[name]; r = Run(C, 4, seed); vals = {}; branch = None
    for t in range(TOTAL):
        if t == STOP:
            vals[STOP] = val_loss(r.model); snap = r.snapshot(); lr0 = f(t - 1, peak)       # lr in force at the last step before stopping
            r2 = Run(C, 4, seed); r2.restore(snap)
            for k in range(COOL): r2.step(lr0 + (MINR * peak - lr0) * (k + 1) / COOL)
            branch = val_loss(r2.model); del r2
        r.step(f(t, peak))
        if (t + 1) % 20 == 0 and t + 1 != STOP: vals[t + 1] = val_loss(r.model)
    vals[TOTAL] = val_loss(r.model)
    return dict(vals=vals, at200=vals[STOP], cooled200=branch, final=vals[TOTAL], train=r.train_losses())

t0 = time.time(); res = {}
for name in SCHED:
    for lr in LRS:
        res[(name, lr)] = [one(name, lr, s) for s in SEEDS]
    print(f"{name} done ({time.time()-t0:.0f}s)", flush=True)

def agg(name, lr, key): a = np.array([x[key] for x in res[(name, lr)]]); return a.mean(), a.std(ddof=1)
print(f"\nval loss, mean over {len(SEEDS)} seeds (sd), by peak LR   [C={C}, {TOTAL} steps, warmup {WARM}, end at {MINR:.0%} of peak]")
for key, label in [("at200", "stopped at step 200 as-is"), ("cooled200", f"step-200 checkpoint + {COOL}-step linear cooldown"), ("final", "full 300-step schedule")]:
    print(f"\n{label}\n{'peak lr':>9} | {'cosine':>16} | {'WSD':>16}")
    for lr in LRS:
        c, w = agg("cosine", lr, key), agg("wsd", lr, key); mk = lambda n, lr_: " *" if lr_ == min(LRS, key=lambda z: agg(n, z, key)[0]) else "  "
        print(f"{lr:>9.4g} | {c[0]:>8.4f} ({c[1]:.3f}){mk('cosine', lr)}| {w[0]:>8.4f} ({w[1]:.3f}){mk('wsd', lr)}")

best = {}
for name in SCHED:
    b = min(LRS, key=lambda lr: agg(name, lr, "at200")[0]); best[name] = b
print(f"\ntuned peak LR (chosen on the step-200 loss): cosine {best['cosine']:g}, WSD {best['wsd']:g}")
rows = {}
for name in SCHED:
    b = best[name]; rows[name] = {k: dict(mean=agg(name, b, k)[0], sd=agg(name, b, k)[1], per_seed=[x[k] for x in res[(name, b)]]) for k in ("at200", "cooled200", "final")}
print("\nHEAD TO HEAD, each at its own tuned LR")
print(f"{'':<34}{'cosine':>20}{'WSD':>20}{'WSD - cosine (paired)':>26}")
for k, label in [("at200", "stopped at step 200"), ("cooled200", f"200 + {COOL}-step cooldown"), ("final", "run to step 300")]:
    c, w = rows["cosine"][k], rows["wsd"][k]; d = np.array(w["per_seed"]) - np.array(c["per_seed"])
    print(f"{label:<34}{c['mean']:>12.4f} ({c['sd']:.3f}){w['mean']:>12.4f} ({w['sd']:.3f}){d.mean():>+14.4f} (sd {d.std(ddof=1):.3f}, per seed {', '.join(f'{x:+.3f}' for x in d)})")
c_lr_at_200 = cosine(STOP - 1, best["cosine"]) / best["cosine"]; w_lr_at_200 = wsd(STOP - 1, best["wsd"]) / best["wsd"]
print(f"\nlr in force when stopped, as a fraction of peak: cosine {c_lr_at_200:.2f}, WSD {w_lr_at_200:.2f}")
# what an untuned comparison would have said: both at the SAME lr
print("\nsame-LR comparisons (what you would conclude if you tuned only one side), loss stopped at 200, WSD - cosine:")
for lr in LRS: print(f"   both at {lr:>8.4g}: {agg('wsd', lr, 'at200')[0] - agg('cosine', lr, 'at200')[0]:+.4f}")
dump(dict(C=C, total=TOTAL, warm=WARM, stop=STOP, cool=COOL, lrs=LRS, best=best, head_to_head=rows,
          grid={f"{n}@{lr}": {k: agg(n, lr, k) for k in ("at200", "cooled200", "final")} for n in SCHED for lr in LRS}), "results_schedules.json")

fig, ax = plt.subplots(1, 3, figsize=(17, 4.4)); tt = np.arange(TOTAL)
for n, c in [("cosine", "C0"), ("wsd", "C1")]:
    ax[0].plot(tt, [SCHED[n](t, 1.0) for t in tt], c, label=n.upper() if n == "wsd" else "cosine")
ax[0].axvline(STOP, c="k", ls=":"); ax[0].set(title="LR schedules (fraction of peak); dotted = stop at 200", xlabel="step"); ax[0].legend()
for n, c in [("cosine", "C0"), ("wsd", "C1")]:
    v = np.array([[x["vals"][s] for s in sorted(x["vals"])] for x in res[(n, best[n])]]); steps = sorted(res[(n, best[n])][0]["vals"])
    ax[1].errorbar(steps, v.mean(0), v.std(0, ddof=1), c=c, marker="o", ms=3, label=f"{n} (lr {best[n]:g})")
    ax[1].scatter([STOP + COOL], [np.mean([x["cooled200"] for x in res[(n, best[n])]])], c=c, marker="*", s=140, zorder=5)
ax[1].axvline(STOP, c="k", ls=":"); ax[1].set(title="Val loss (mean±sd, 3 seeds); stars = step-200 checkpoint + 60-step cooldown", xlabel="step", ylim=(2.0, 3.4)); ax[1].legend()
for n, c in [("cosine", "C0"), ("wsd", "C1")]:
    ax[2].plot(LRS, [agg(n, lr, "at200")[0] for lr in LRS], "o-", c=c, label=f"{n}: stopped at 200"); ax[2].plot(LRS, [agg(n, lr, "final")[0] for lr in LRS], "s--", c=c, label=f"{n}: step 300")
ax[2].set(title="LR tuning for each side", xscale="log", xlabel="peak lr", ylabel="val loss"); ax[2].legend(fontsize=8)
savefig(fig, "schedules.png")
