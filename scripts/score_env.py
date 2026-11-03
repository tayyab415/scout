"""Score every episode of a tag and print the leaderboard. Downloads only the small files (no videos or frames).

    python scripts/score_env.py --tag cmp
Writes runs/env/<tag>/scores.json and runs/env/<tag>/leaderboard.md.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from scout.env.scoring import score_episode  # noqa: E402
from scout.world.layout import Layout  # noqa: E402

VOL = os.environ.get("SCOUT_VOLUME", "tribe-lab-claude")


def fetch(tag, root):
    """Copy the small files of every episode of a tag from the Modal volume (machine-readable listing)."""
    r = subprocess.run(["modal", "volume", "ls", "--json", VOL, f"/scout/env/{tag}"], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"modal volume ls failed: {r.stderr[:300]}")
    eps = [os.path.basename(x["filename"].rstrip("/")) for x in json.loads(r.stdout) if x.get("type", "dir") == "dir"]
    for e in eps:
        d = f"{root}/{e}"
        os.makedirs(d, exist_ok=True)
        subprocess.run(["modal", "volume", "get", "--force", VOL, f"/scout/env/{tag}/{e}/manifest.json", f"{d}/manifest.json"],
                       capture_output=True)
        for arm in ("unchanged", "moved", "explore_only"):
            os.makedirs(f"{d}/{arm}", exist_ok=True)
            for f in ("log.json", "memory.json", "evaluator.json", "maps.npz", "map_meta.json"):
                subprocess.run(["modal", "volume", "get", "--force", VOL, f"/scout/env/{tag}/{e}/{arm}/{f}", f"{d}/{arm}/{f}"],
                               capture_output=True)
            if not os.listdir(f"{d}/{arm}"):
                os.rmdir(f"{d}/{arm}")
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layouts", default="data/layouts")
    ap.add_argument("--no-fetch", action="store_true")
    a = ap.parse_args()
    root = f"runs/env/{a.tag}"
    if not a.no_fetch:
        fetch(a.tag, root)
    # the frozen plan is the authoritative episode list: every planned episode stays in its brain's denominator
    plan_file = f"configs/frozen_{a.tag}.json"
    if os.path.exists(plan_file):
        plan = json.load(open(plan_file))
        planned = [(b, h, f"{b}_{h}_s{plan.get('seed', 0)}") for b in plan["brains"] for h in plan["houses"]]
    else:
        planned = []
        for e in sorted(x for x in os.listdir(root) if os.path.isdir(f"{root}/{x}")):
            planned.append((e.split("_")[0], e.split("_", 1)[1].rsplit("_s", 1)[0], e))
    scores = []
    for b, h, e in planned:
        d = f"{root}/{e}"
        if not os.path.exists(f"{d}/manifest.json"):
            scores.append(dict(episode=e, plan_brain=b, house=h, error="missing manifest (crashed, still running or never started)"))
            continue
        man = json.load(open(f"{d}/manifest.json"))
        lay = Layout.load(f"{a.layouts}/{man['house']}.npz")
        sc = score_episode(d, lay)
        sc["plan_brain"] = b
        scores.append(sc)
    json.dump(scores, open(f"{root}/scores.json", "w"), indent=1, default=float)

    by = defaultdict(list)
    for s in scores:
        by[s["plan_brain"]].append(s)
    lines = ["| brain | episodes planned | completed | target frozen | verify unchanged (correct / abstain / wrong / other) | "
             "verify moved (correct / abstain / wrong / other) | memory-persistence rule, unchanged / moved (correct) | quiz accuracy (errors) | affirmative answers with valid evidence | "
             "own checks (correct / abstain / wrong) | floor seen (median) | falls / got up |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for b, ss in sorted(by.items()):
        done = [s for s in ss if "error" not in s]

        def tally(arm, key="verify"):
            c = defaultdict(int)
            for s in done:
                v = s.get(arm, {}).get(key)
                c[v["outcome"] if v else "not run"] += 1
            other = sum(v for k, v in c.items() if k not in ("correct", "abstain", "wrong"))
            return f"{c['correct']} / {c['abstain']} / {c['wrong']} / {other}"
        quiz = [s["unchanged"]["quiz"] for s in done if s.get("unchanged", {}).get("quiz")]
        cite = [s["unchanged"]["cite"] for s in done if s.get("unchanged", {}).get("cite")]
        chk = defaultdict(int)
        for s in done:  # the exploration prefix is shared by both arms: count own checks once, from the unchanged arm
            arm = "unchanged" if "unchanged" in s else "explore_only"
            for k, v in s.get(arm, {}).get("checks", {}).items():
                chk[k] += v
        fl = [s[arm]["explore"]["floor_seen"] for s in done for arm in ("unchanged", "explore_only") if s.get(arm, {}).get("explore")]
        falls = sum(s.get("unchanged", s.get("explore_only", {})).get("falls", 0) for s in done)
        ups = sum(s.get("unchanged", s.get("explore_only", {})).get("got_up", 0) for s in done)
        qa = f"{np.mean([q['accuracy'] for q in quiz]):.2f} ({sum(q['errors'] for q in quiz)})" if quiz else "-"
        ev = f"{sum(c['affirmative_supported'] for c in cite)}/{sum(c['affirmative'] for c in cite)}" if cite else "-"
        lines.append(f"| {b} | {len(ss)} | {len(done)} | {sum(1 for s in done if s.get('target'))} | {tally('unchanged')} | "
                     f"{tally('moved')} | {tally('unchanged', 'verify_recall').split(' / ')[0]} / "
                     f"{tally('moved', 'verify_recall').split(' / ')[0]} | {qa} | {ev} | {chk['correct']} / {chk['abstain']} / {chk['wrong']} | "
                     f"{(str(round(100 * float(np.median(fl)))) + '%') if fl else '-'} | {falls} / {ups} |")
    # every verification outcome, separately, per brain and arm (no "other" bucket)
    kinds = ["correct", "abstain", "wrong", "wrong target", "ineligible", "no answer", "not run"]
    lines += ["", "Verification outcomes per arm (physical check; 'not run' = no target frozen, crashed, or moved arm skipped):", "",
              "| brain | arm | " + " | ".join(kinds) + " |", "|---|---|" + "---|" * len(kinds)]
    for b, ss in sorted(by.items()):
        for arm in ("unchanged", "moved"):
            c = defaultdict(int)
            for s in ss:
                v = s.get(arm, {}).get("verify") if "error" not in s else None
                c[v["outcome"] if v else "not run"] += 1
            lines.append(f"| {b} | {arm} | " + " | ".join(str(c[k]) for k in kinds) + " |")
    lines += ["", "Falls, recoveries and own checks are counted on the unchanged arm (it shares the exploration with the moved arm)."]
    errs = [s for s in scores if "error" in s]
    md = "\n".join(lines) + ("\n\nNot completed: " + ", ".join(f"{s['episode']} ({s['error']})" for s in errs) if errs else "")
    open(f"{root}/leaderboard.md", "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
