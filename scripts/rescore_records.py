"""Rescore the published final3 records on a laptop (no GPU, no house data) and print every verification outcome
and the memory-quiz denominators.

    curl -LO https://github.com/tayyab415/scout/releases/download/final3-evidence/final3-records.tar.gz
    mkdir records && tar xzf final3-records.tar.gz -C records
    python scripts/rescore_records.py records

Floor coverage needs the house layouts, so it is not recomputed here; everything else in the leaderboard is.
"""
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scout.env.scoring import score_episode  # noqa: E402

KINDS = ["correct", "abstain", "wrong", "wrong target", "ineligible", "no answer", "not run"]


def main():
    root = sys.argv[1]
    plan = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", "frozen_final3.json")))
    out = defaultdict(lambda: defaultdict(int))
    quiz = defaultdict(lambda: [0, 0, 0])
    total = defaultdict(int)
    for b in plan["brains"]:
        for h in plan["houses"]:
            d = f"{root}/{b}_{h}_s{plan.get('seed', 0)}"
            r = score_episode(d, None) if os.path.exists(f"{d}/manifest.json") else {}
            for arm in ("unchanged", "moved"):
                v = (r.get(arm) or {}).get("verify")
                k = v["outcome"] if v else "not run"
                out[(b, arm)][k] += 1
                total[k] += 1
            q = (r.get("unchanged") or {}).get("quiz")
            if q:
                quiz[b][0] += q["n"]
                quiz[b][1] += round(q["accuracy"] * q["n"])
                quiz[b][2] += q["errors"]
    print("| brain | arm | " + " | ".join(KINDS) + " |")
    for (b, arm), c in sorted(out.items()):
        print(f"| {b} | {arm} | " + " | ".join(str(c[k]) for k in KINDS) + " |")
    print("all arms: " + ", ".join(f"{total[k]} {k}" for k in KINDS))
    print("\nmemory quiz (questions / correct / errors):")
    for b, (n, c, e) in sorted(quiz.items()):
        print(f"  {b}: {n} / {c} / {e}")


if __name__ == "__main__":
    main()
