"""Rescore the published final3 records on a laptop (no GPU, no house data) and print every verification outcome
and the memory-quiz denominators.

    curl -LO https://github.com/tayyab415/scout/releases/download/final3-evidence/final3-records.tar.gz
    mkdir records && tar xzf final3-records.tar.gz -C records
    python scripts/rescore_records.py records
    python scripts/rescore_records.py records_student frozen_final3_student.json   # the live student run

Floor coverage needs the house layouts, so it is not recomputed here; everything else in the leaderboard is.
"""
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scout.env.scoring import score_episode  # noqa: E402

KINDS = ["correct", "abstain", "wrong", "wrong target", "ineligible", "no answer", "not run"]


def episode_dirs(root: str, brains: list[str], houses: list[str], seed: int) -> dict:
    """Map (brain, house) to its record directory. A run published under a brain's older label keeps that label, so
    after every exact match is taken, a house with exactly one unclaimed directory hands it to the one brain that
    still has none for that house."""
    out, claimed = {}, set()
    for b in brains:
        for h in houses:
            d = os.path.join(root, f"{b}_{h}_s{seed}")
            if os.path.exists(os.path.join(d, "manifest.json")):
                out[(b, h)] = d
                claimed.add(os.path.basename(d))
    for h in houses:
        spare = [p for p in sorted(glob.glob(os.path.join(root, f"*_{h}_s{seed}")))
                 if os.path.basename(p) not in claimed and os.path.exists(os.path.join(p, "manifest.json"))]
        missing = [b for b in brains if (b, h) not in out]
        if len(spare) == 1 and len(missing) == 1:
            out[(missing[0], h)] = spare[0]
    return out


def main():
    root = sys.argv[1]
    plan_file = sys.argv[2] if len(sys.argv) > 2 else "frozen_final3.json"  # e.g. frozen_final3_student.json
    plan = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "configs", plan_file)))
    out = defaultdict(lambda: defaultdict(int))
    quiz = defaultdict(lambda: [0, 0, 0])
    total = defaultdict(int)
    dirs = episode_dirs(root, plan["brains"], plan["houses"], plan.get("seed", 0))
    for b in plan["brains"]:
        for h in plan["houses"]:
            d = dirs.get((b, h), os.path.join(root, f"{b}_{h}_s{plan.get('seed', 0)}"))
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
