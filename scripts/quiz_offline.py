"""Give a brain the memory quiz on recorded episodes, on a laptop, and score it with the environment's own scorer.

    python scripts/quiz_offline.py records --brains student rules

For every episode with a recorded quiz, each brain answers the 16 questions from the saved memory table (the same
text the original brain saw at quiz time) and `scout.env.scoring.score_quiz` / `score_citations` grade the answers
against the evaluator's visibility record, exactly as in the leaderboard. The recorded answers of the brain that ran
the episode are rescored alongside as "recorded".

This measures answering only. The memories were built during the recorded runs, by whichever brain ran them, so no
brain here gets credit for exploring or re-checking.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from scout.brains import make_brain  # noqa: E402
from scout.env.core import PROMPTS  # noqa: E402
from scout.env.scoring import score_citations, score_quiz  # noqa: E402


def quiz_from_table(brain, mem: dict, t_q: float) -> list[dict]:
    """The same record `Agent.quiz` writes, built from a saved memory."""
    out = []
    for cat in PROMPTS:
        q = f"Did you see a {cat}? If so, where was it?"
        a = brain.answer(mem["table"], q) or {}
        valid = isinstance(a.get("observed"), bool) and isinstance(a.get("answer"), str) and bool(a.get("answer", "").strip()) \
            and "_error" not in a
        oid = a.get("object_id")
        xy = None
        if isinstance(oid, int) and 0 <= oid < len(mem["tracks"]):
            tr = mem["tracks"][oid]
            xy = mem["sightings"][tr["sightings"][-1]]["xyz"][:2]
        cited = a.get("cited_sightings", [])
        cited = cited if isinstance(cited, list) else []
        out.append(dict(t=t_q, category=cat, question=q, answer=a.get("answer", ""), observed=bool(a.get("observed")) if valid else None,
                        valid=valid, error=None if valid else a.get("_error", "invalid reply"), object_id=oid, xy=xy,
                        cited=[c for c in cited if isinstance(c, int)], malformed_citations=[c for c in cited if not isinstance(c, int)]))
    return out


class ExactRuleBrain:
    """Control: the rule brain's quiz answer with the student's whole-word category matching, no network."""
    name = "rules, whole-word match"

    def __init__(self):
        from scout.brains import RuleBrain
        from scout.student.trust import CATEGORIES, parse_table
        self.rule, self.cats, self.parse = RuleBrain(), CATEGORIES, parse_table

    def answer(self, table, question):
        import re
        q = question.lower()
        named = [c for c in self.cats if re.search(rf"\b{re.escape(c)}\b", q)]
        if not named:
            return self.rule.answer(table, question)
        cat = max(named, key=len)
        rows = [r for r in self.parse(table) if r["label"] == cat]
        rows.sort(key=lambda r: (not r["confident"], -r["last"]))
        if not rows:
            return {"answer": "I did not see one.", "cited_sightings": [], "observed": False, "object_id": None}
        r = rows[0]
        return {"answer": f"I saw a {r['label']}, last at {r['last']:.0f} s.", "cited_sightings": r["ids"][-1:],
                "observed": True, "object_id": r["id"]}


def _brain(name: str):
    if name == "rules-exact":
        return ExactRuleBrain()
    if name == "student-trust-all":  # ablation: the network only ranks; every remembered category is answered yes
        from scout.student.trust import StudentBrain
        return StudentBrain(threshold=0.0)
    return make_brain(name)


def run(root: str, brains: list[str]) -> dict:
    totals = defaultdict(lambda: dict(n=0, correct=0, errors=0, cited=0, valid=0, affirmative=0, supported=0, episodes=0))
    per_episode = []
    for man_path in sorted(glob.glob(f"{root}/*/manifest.json")):
        ep = os.path.dirname(man_path)
        arm = f"{ep}/unchanged"
        if not os.path.exists(f"{arm}/log.json"):
            continue
        log = json.load(open(f"{arm}/log.json"))
        if not log.get("quiz"):
            continue
        mem, ev = json.load(open(f"{arm}/memory.json")), json.load(open(f"{arm}/evaluator.json"))
        man = json.load(open(man_path))
        row = dict(episode=os.path.basename(ep), house=man["house"], ran_by=man["brain"])
        answers = {"recorded": log["quiz"]}
        for b in brains:
            t_q = log["quiz"][0]["t"] if log["quiz"][0].get("t") is not None else log["t_end"]
            answers[b] = quiz_from_table(_brain(b), mem, t_q)
        for name, qz in answers.items():
            q = score_quiz(dict(quiz=qz, looks=log["looks"], t_end=log["t_end"]), mem, ev)
            c = score_citations(dict(quiz=qz, looks=log["looks"], t_end=log["t_end"]), mem, ev)
            correct = int(round(q["accuracy"] * q["n"]))
            row[name] = dict(correct=correct, n=q["n"], errors=q["errors"], citations_valid=c["valid"], citations=c["cited"],
                             affirmative=c["affirmative"], affirmative_supported=c["affirmative_supported"])
            for k in ((name, man["brain"]), (name, "all")):
                T = totals[k]
                T["n"] += q["n"]; T["correct"] += correct; T["errors"] += q["errors"]; T["cited"] += c["cited"]
                T["valid"] += c["valid"]; T["affirmative"] += c["affirmative"]; T["supported"] += c["affirmative_supported"]
                T["episodes"] += 1
        per_episode.append(row)
    return dict(totals={f"{k[0]} | {k[1]}": v for k, v in totals.items()}, episodes=per_episode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records")
    ap.add_argument("--brains", nargs="+", default=["student", "rules", "rules-exact", "student-trust-all"])
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    r = run(a.records, a.brains)
    print(f"memory quiz on recorded memories under {a.records}\n")
    print("| answers by | memories from | episodes | questions | correct | errors | affirmative answers with a valid citation |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for k, T in sorted(r["totals"].items()):
        who, frm = k.split(" | ")
        print(f"| {who} | {frm} | {T['episodes']} | {T['n']} | {T['correct']} | {T['errors']} | {T['supported']}/{T['affirmative']} |")
    if a.out:
        json.dump(r, open(a.out, "w"), indent=1)
        print("\nsaved", a.out)


if __name__ == "__main__":
    main()
