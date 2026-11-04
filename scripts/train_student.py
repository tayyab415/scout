"""Train the student's trust network on recorded episodes and save its weights. CPU, a few seconds.

    python scripts/train_student.py --records runs/env/final runs/env/final2 runs/env/cmp1 ... \
        --exclude-houses configs/frozen_final3.json --out assets/brains/student_trust.npz

Houses listed in --exclude-houses never enter training or validation. Validation is a held-out fifth of the training
houses (by house), used only to pick the stopping epoch.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scout.student.data import dataset  # noqa: E402
from scout.student.trust import FEATURE_NAMES, TrustMLP  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--exclude-houses", default="", help="json with a 'houses' list (the evaluation plan)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "assets", "brains", "student_trust.npz"))
    ap.add_argument("--hidden", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    excl = set(json.load(open(a.exclude_houses))["houses"]) if a.exclude_houses else set()
    d = dataset(a.records, excl)
    X, y, house = d["X"], d["y"], d["house"]
    houses = sorted(set(house))
    rng = np.random.default_rng(a.seed)
    val_h = set(rng.choice(houses, size=max(1, len(houses) // 5), replace=False).tolist())
    va = np.array([h in val_h for h in house])
    print(f"{len(d['episodes'])} episodes, {len(houses)} houses ({len(val_h)} held out for early stopping), "
          f"{len(y)} remembered objects, {int(y.sum())} real ({y.mean():.1%})")
    net = TrustMLP(hidden=a.hidden, seed=a.seed)
    fit = net.fit(X[~va], y[~va], X[va], y[va], epochs=a.epochs, seed=a.seed)
    p_va = net.predict(X[va])
    acc = float(((p_va >= 0.5) == (y[va] > 0.5)).mean())
    base = float(max(y[va].mean(), 1 - y[va].mean()))
    print(f"{net.n_params()} parameters; best epoch {fit['best_epoch']}, val loss {fit['best_val_loss']:.3f}; "
          f"val accuracy {acc:.3f} (majority {base:.3f})")
    # feature effect: mean change in p when each standardised feature moves +1 sd (a rough look, not a claim)
    base_p = net.predict(X[va]).mean()
    effects = {}
    for i, name in enumerate(FEATURE_NAMES):
        Xp = X[va].copy()
        Xp[:, i] += net.sd[i]
        effects[name] = round(float(net.predict(Xp).mean() - base_p), 3)
    top = sorted(effects.items(), key=lambda kv: -abs(kv[1]))[:8]
    print("largest feature effects (+1 sd):", ", ".join(f"{k} {v:+.3f}" for k, v in top))
    net.save(a.out, n_train=int((~va).sum()), n_val=int(va.sum()), houses_train=sorted(set(house[~va])),
             houses_val=sorted(val_h), episodes=d["episodes"], best_epoch=fit["best_epoch"] or 0,
             val_accuracy=acc, seed=a.seed, hidden=a.hidden, excluded_houses=sorted(excl))
    json.dump(dict(curve=fit["curve"], effects=effects, n_train=int((~va).sum()), n_val=int(va.sum()),
                   houses_train=sorted(set(house[~va])), houses_val=sorted(val_h), episodes=d["episodes"],
                   val_accuracy=acc, majority=base, n_params=net.n_params()),
              open(os.path.splitext(a.out)[0] + "_training.json", "w"), indent=1)
    print("saved", os.path.abspath(a.out))


if __name__ == "__main__":
    main()
