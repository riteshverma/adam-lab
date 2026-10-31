"""Part 2: switch bias correction off, plot the first twenty steps both ways, and find when the difference stops mattering."""
import math
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

b1, b2 = 0.9, 0.999
R = {}

# ---- (a) the identity: update_without / update_with = (1-b1^t)/sqrt(1-b2^t), independent of the gradients (eps->0) --
def ratio(t, b1=b1, b2=b2): return (1 - b1 ** t) / math.sqrt(1 - b2 ** t)
def steps_until(thr, b1=b1, b2=b2):
    """last t at which |ratio-1| > thr (so from t+1 on, the two updates differ by less than thr)."""
    last = 0
    for t in range(1, 200000):
        if abs(ratio(t, b1, b2) - 1) > thr: last = t
        elif t > 50 and last and t > 3 * last: break
    return last + 1
print("ratio  uncorrected-update / corrected-update  at t = 1,2,5,10,20,50,100,1000:")
print("  " + "  ".join(f"t={t}:{ratio(t):.2f}" for t in [1, 2, 5, 10, 20, 50, 100, 1000]))
tmax = max(range(1, 200), key=ratio); print(f"  worst case t={tmax}: uncorrected update is {ratio(tmax):.2f}x too big")
R["identity"] = dict(peak_t=tmax, peak_ratio=ratio(tmax), thresholds={})
print("\nsteps after which |uncorrected/corrected - 1| < threshold for good (beta1=0.9):")
print(f"  {'beta2':>6} | {'10%':>6} {'5%':>6} {'1%':>6} {'0.1%':>7}")
for B2 in [0.95, 0.98, 0.99, 0.999]:
    row = {thr: steps_until(thr, b1, B2) for thr in [0.10, 0.05, 0.01, 0.001]}
    R["identity"]["thresholds"][str(B2)] = row
    print(f"  {B2:>6} | " + " ".join(f"{row[k]:>6}" for k in row))

# ---- (b) real gradients for one weight: 20 steps both ways ------------------------------------------------------
mdl = make_model(256, 4, seed=0); gen = torch.Generator().manual_seed(5)
w = mdl.blocks[0].fc.weight; idx = (7, 11); gs = []
for _ in range(20):
    x, y = get_batch("train", 16, 128, gen); mdl.zero_grad(); mdl(x, y)[1].backward(); gs.append(w.grad[idx].item())
def trace(bc, lr=1.0):
    m = v = 0.0; out = []
    for t, g in enumerate(gs, 1):
        m = b1 * m + (1 - b1) * g; v = b2 * v + (1 - b2) * g * g
        mh = m / (1 - b1 ** t) if bc else m; vh = v / (1 - b2 ** t) if bc else v
        out.append(lr * mh / (math.sqrt(vh) + 1e-8))
    return out
u_bc, u_no = trace(True), trace(False)
print("\nreal gradient sequence for fc.weight[7,11] (std %.2e); update/lr, t=1..20:" % np.std(gs))
print("  with BC   :", " ".join(f"{u:+.2f}" for u in u_bc)); print("  without BC:", " ".join(f"{u:+.2f}" for u in u_no))
print("  |no|/|with|:", " ".join(f"{abs(a/b):.2f}" for a, b in zip(u_no, u_bc)))

# ---- (c) effect on training: no warmup, lr 3e-3, 3 seeds, with vs without BC -------------------------------------
curves = {True: [], False: []}; final = {True: [], False: []}; ratios20 = {True: [], False: []}
for bc in (True, False):
    for seed in range(3):
        r = Run(256, 4, seed, bias_correction=bc); rr = []
        for t in range(300):
            rt = r.step(3e-3, log_ratio=(t < 20))
            if rt is not None: rr.append(rt.cpu().numpy())
        curves[bc].append(r.train_losses()); final[bc].append(val_loss(r.model)); ratios20[bc].append(np.mean(rr, axis=1))
cb, cn = np.mean(curves[True], 0), np.mean(curves[False], 0)
print(f"\ntraining 300 steps, lr 3e-3, no warmup, mean of 3 seeds: val loss with BC {np.mean(final[True]):.4f}  without BC {np.mean(final[False]):.4f}")
print("train loss t=1..20 with BC   :", " ".join(f"{x:.2f}" for x in cb[:20]))
print("train loss t=1..20 without BC:", " ".join(f"{x:.2f}" for x in cn[:20]))
gap = np.abs(cb - cn)
sm = lambda a, k=9: np.convolve(a, np.ones(k) / k, mode="same")
# the loss curves agree "within noise" once the smoothed gap is under 0.05 nats for good
ok = next((t for t in range(len(gap)) if np.all(sm(gap)[t:-5] < 0.05)), None)
print(f"smoothed |loss gap| < 0.05 nats for good from step {ok}")
R["training"] = dict(val_with=float(np.mean(final[True])), val_without=float(np.mean(final[False])), loss_gap_below_0p05_from=ok,
                     first20_with=cb[:20].tolist(), first20_without=cn[:20].tolist())

fig, ax = plt.subplots(1, 4, figsize=(20, 4.2)); t20 = np.arange(1, 21)
ax[0].plot(t20, u_bc, "o-", label="with bias correction"); ax[0].plot(t20, u_no, "s-", label="no bias correction")
ax[0].axhline(0, c="k", lw=.5); ax[0].set(title="One weight, real gradients: update / lr", xlabel="step"); ax[0].legend()
ax[1].plot(t20, [abs(a / b) for a, b in zip(u_no, u_bc)], "o-", label="measured |no|/|with|"); ax[1].plot(t20, [ratio(t) for t in t20], "k--", label="(1-b1^t)/sqrt(1-b2^t)")
ax[1].set(title="Update-size ratio, first 20 steps", xlabel="step", ylabel="uncorrected / corrected"); ax[1].legend()
tt = np.arange(1, 8001); ax[2].plot(tt, [ratio(t) for t in tt], label="beta2=0.999")
for B2, c in [(0.99, "C1"), (0.95, "C2")]: ax[2].plot(tt, [ratio(t, b1, B2) for t in tt], c, label=f"beta2={B2}")
for thr in [0.05, 0.01]: ax[2].axhline(1 + thr, c="gray", ls=":", lw=.8)
ax[2].set(title="Same ratio to step 8000", xlabel="step", xscale="log"); ax[2].legend()
ax[3].plot(np.arange(1, 41), cb[:40], label="with BC"); ax[3].plot(np.arange(1, 41), cn[:40], label="without BC")
ax[3].set(title="Train loss, first 40 steps (lr 3e-3, no warmup, 3 seeds)", xlabel="step"); ax[3].legend()
savefig(fig, "bias_correction.png"); dump(R, "results_bias_correction.json")
