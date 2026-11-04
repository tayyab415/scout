"""Leave-one-house-out cross-validation of the student on the training runs only. Used once, before the final
evaluation records were scored with the chosen setting, to pick the hidden width and the yes/no threshold.

    python scripts/student_cv.py --records runs/env/final runs/env/final2 ... --exclude-houses configs/frozen_final3.json

For each held-out house the network is trained on the other houses, then every remembered category in the held-out
episodes is answered yes/no the way the student answers the quiz. The score counts a yes as right when the most
trusted object is real, and a no as right when no remembered object of that category is real. Categories with no
remembered object are left out, since every memory-only brain answers those the same way.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scout.student.data import episode_arm, label_rows  # noqa: E402
from scout.student.trust import TrustMLP, features  # noqa: E402


def quiz_style(rows, p, tau):
    byc = {}
    for r, pi in zip(rows, p):
        byc.setdefault(r["label"], []).append((float(pi), r["y"]))
    ok = n = 0
    for lst in byc.values():
        pi, y = max(lst)
        n += 1
        ok += (y > 0.5) if pi >= tau else (not any(yy > 0.5 for _, yy in lst))
    return ok, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--exclude-houses", default="")
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    excl = set(json.load(open(a.exclude_houses))["houses"]) if a.exclude_houses else set()
    eps = []
    for root in a.records:
        for man in sorted(glob.glob(f"{root}/*/manifest.json")):
            m = json.load(open(man))
            arm = episode_arm(os.path.dirname(man))
            if m["house"] in excl or arm is None:
                continue
            rows = label_rows(arm)
            if rows:
                eps.append((m["house"], rows))
    houses = sorted({h for h, _ in eps})
    print(f"{len(eps)} episodes, {len(houses)} houses")
    print("| hidden | threshold | held-out categories right |")
    print("|---:|---:|---:|")
    for hidden in (8, 16):
        for tau in (0.4, 0.5, 0.6):
            ok = n = 0
            for h in houses:
                tr = [r for hh, r in eps if hh != h]
                X = np.concatenate([features(r) for r in tr])
                y = np.array([x["y"] for r in tr for x in r])
                net = TrustMLP(hidden=hidden, seed=0)
                net.fit(X, y, epochs=a.epochs, seed=0)
                for rows in (r for hh, r in eps if hh == h):
                    o, k = quiz_style(rows, net.predict(features(rows)), tau)
                    ok += o
                    n += k
            print(f"| {hidden} | {tau} | {ok}/{n} ({ok / n:.3f}) |")
    ok = n = 0
    for _, rows in eps:
        byc = {}
        for r in rows:
            byc.setdefault(r["label"], []).append(r)
        for lst in byc.values():
            lst.sort(key=lambda r: (not r["confident"], -r["last"]))
            n += 1
            ok += lst[0]["y"] > 0.5
    print(f"\nrule (confident first, then most recent; always yes) on the same categories: {ok}/{n} ({ok / n:.3f})")


if __name__ == "__main__":
    main()
