"""Score one recorded episode on a laptop, no GPU.

    python scripts/score_episode.py examples/sample_episode
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scout.env.scoring import score_episode  # noqa: E402
from scout.world.layout import Layout  # noqa: E402

d = sys.argv[1]
lay = glob.glob(f"{d}/layout_*.npz")
r = score_episode(d, Layout.load(lay[0]) if lay else None)
for arm in ("unchanged", "moved"):
    if arm in r:
        v, rc = r[arm].get("verify") or {}, r[arm].get("verify_recall") or {}
        print(f"{arm:9s}  walking back to look: {v.get('outcome')!s:12s}  memory-persistence rule: {rc.get('outcome')}")
q = (r.get("unchanged") or {}).get("quiz")
if q:
    print(f"memory quiz: {q['accuracy']:.2f} over {q['n']} questions ({q['errors']} errors); "
          f"citations: {r['unchanged']['cite']['valid']}/{r['unchanged']['cite']['cited']} valid")
json.dump(r, open(os.path.join(d, "score.json"), "w"), indent=1, default=float)
