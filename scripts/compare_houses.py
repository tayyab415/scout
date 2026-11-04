"""Per-house comparison of brains on the same frozen houses and seed: verification outcome per arm and quiz score.

    python scripts/compare_houses.py records_final3 records_student
Only houses present in the second directory are shown, so a partial run compares like with like.
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scout.env.scoring import score_episode  # noqa: E402

ABBR = {"correct": "ok", "abstain": "abst", "wrong": "WRONG", "wrong target": "wtgt", "ineligible": "inel",
        "no answer": "none", None: "-"}


def load(root):
    out = {}
    for m in glob.glob(f"{root}/*/manifest.json"):
        d = os.path.dirname(m)
        man = json.load(open(m))
        r = score_episode(d)
        brain = os.path.basename(d).split("_procthor")[0]
        out[(brain, man["house"])] = r
    return out


def main():
    base, new = load(sys.argv[1]), load(sys.argv[2])
    houses = sorted({h for _, h in new})
    brains = sorted({b for b, _ in base}) + sorted({b for b, _ in new})
    allr = {**base, **new}
    print("| house | " + " | ".join(brains) + " |")
    print("|---|" + "---|" * len(brains))
    tot = defaultdict(lambda: [0, 0, 0])
    for h in houses:
        cells = []
        for b in brains:
            r = allr.get((b, h))
            if r is None:
                cells.append("")
                continue
            u = (r.get("unchanged") or {}).get("verify") or {}
            mv = (r.get("moved") or {}).get("verify") or {}
            q = (r.get("unchanged") or {}).get("quiz")
            qs = f"{round(q['accuracy'] * q['n'])}/{q['n']}" if q else "no quiz"
            pair = u.get("outcome") == "correct" and mv.get("outcome") == "correct"
            if q:
                tot[b][0] += round(q["accuracy"] * q["n"]); tot[b][1] += q["n"]
            tot[b][2] += int(pair)
            cells.append(f"{ABBR.get(u.get('outcome'), u.get('outcome'))} / {ABBR.get(mv.get('outcome'), mv.get('outcome'))}"
                         f"{' (pair)' if pair else ''}; quiz {qs}")
        print(f"| {h.replace('procthor-10k-val_', '')} | " + " | ".join(cells) + " |")
    print("| **total** | " + " | ".join(f"{tot[b][2]} pairs; quiz {tot[b][0]}/{tot[b][1]}" for b in brains) + " |")
    print("\ncells: verify unchanged / moved (ok, abst = abstain, wtgt = wrong target, none = no answer, - = not run); quiz correct/questions")


if __name__ == "__main__":
    main()
