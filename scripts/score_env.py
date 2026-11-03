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

VOL = "tribe-lab-claude"


def fetch(tag, root):
    ls = subprocess.run(["modal", "volume", "ls", VOL, f"/scout/env/{tag}"], capture_output=True, text=True).stdout
    eps = [ln.strip().split("/")[-1] for ln in ls.splitlines() if ln.strip()]
    for e in eps:
        d = f"{root}/{e}"
        os.makedirs(d, exist_ok=True)
        subprocess.run(["modal", "volume", "get", "--force", VOL, f"/scout/env/{tag}/{e}/manifest.json", f"{d}/manifest.json"],
                       capture_output=True)
        for arm in ("unchanged", "moved", "explore_only"):
            for f in ("log.json", "memory.json", "evaluator.json", "maps.npz", "map_meta.json"):
                os.makedirs(f"{d}/{arm}", exist_ok=True)
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
    eps = sorted(os.listdir(root)) if a.no_fetch else fetch(a.tag, root)
    scores = []
    for e in eps:
        d = f"{root}/{e}"
        if not os.path.exists(f"{d}/manifest.json"):
            scores.append(dict(episode=e, error="no manifest (crashed or still running)"))
            continue
        man = json.load(open(f"{d}/manifest.json"))
        lay = Layout.load(f"{a.layouts}/{man['house']}.npz")
        scores.append(score_episode(d, lay))
    json.dump(scores, open(f"{root}/scores.json", "w"), indent=1, default=float)

    by = defaultdict(list)
    for s in scores:
        if "brain" in s:
            by[s["brain"]].append(s)
    lines = ["| brain | episodes | target frozen | verify unchanged (correct / abstain / wrong) | verify moved (correct / abstain / wrong) | "
             "quiz accuracy | citations valid | own checks (correct / abstain / wrong) | floor seen (median) | falls / got up |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for b, ss in sorted(by.items()):
        def tally(arm):
            c = defaultdict(int)
            for s in ss:
                v = s.get(arm, {}).get("verify")
                if v:
                    c[v["outcome"]] += 1
            return f"{c['correct']} / {c['abstain'] + c['no answer']} / {c['wrong']}"
        quiz = [s["unchanged"]["quiz"]["accuracy"] for s in ss if s.get("unchanged", {}).get("quiz")]
        cite = [s["unchanged"]["cite"] for s in ss if s.get("unchanged", {}).get("cite")]
        cv, cn = sum(c["valid"] for c in cite), sum(c["cited"] for c in cite)
        chk = defaultdict(int)
        for s in ss:
            for arm in ("unchanged", "moved", "explore_only"):
                for k, v in s.get(arm, {}).get("checks", {}).items():
                    chk[k] += v
        fl = [s[arm]["explore"]["floor_seen"] for s in ss for arm in ("unchanged", "explore_only") if s.get(arm, {}).get("explore")]
        falls = sum(s.get("unchanged", s.get("explore_only", {})).get("falls", 0) for s in ss)
        ups = sum(s.get("unchanged", s.get("explore_only", {})).get("got_up", 0) for s in ss)
        lines.append(f"| {b} | {len(ss)} | {sum(1 for s in ss if s.get('target'))} | {tally('unchanged')} | {tally('moved')} | "
                     f"{np.mean(quiz):.2f} (n={len(quiz)}) | {cv}/{cn} | {chk['correct']} / {chk['abstain']} / {chk['wrong']} | "
                     f"{np.median(fl) if fl else float('nan'):.0%} | {falls} / {ups} |")
    errs = [s for s in scores if "error" in s]
    md = "\n".join(lines) + ("\n\nNot scored: " + ", ".join(f"{s['episode']} ({s['error']})" for s in errs) if errs else "")
    open(f"{root}/leaderboard.md", "w").write(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
