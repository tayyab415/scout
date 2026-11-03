"""Additional CPU probes for the advertised reply/evidence contracts.

The baseline test records existing behavior as a characterization, not a policy
requirement. The finding is that the demo must not call it an independent model answer.
"""
from pathlib import Path
import sys

import numpy as np
import pytest

ROOT = Path(__file__).parent / ".."
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tests"))
from scout.env.agent import Agent
from scout.env.scoring import score_citations, score_quiz, score_verify
from test_review2 import evidence


class ReplyBrain:
    def __init__(self, reply):
        self.reply = reply

    def answer(self, table, question):
        return self.reply


@pytest.mark.parametrize("reply", [
    {"observed": False},
    {"observed": False, "answer": [], "object_id": None, "cited_sightings": []},
], ids=["missing-answer", "answer-is-list"])
def test_malformed_answer_is_not_a_valid_correct_negative(reply):
    a = Agent(ReplyBrain(reply), mind=False)
    log, mem, ev = evidence()
    ev["visibility"] = []
    log["quiz"] = a.quiz(["sofa"], t=100.)
    result = score_quiz(log, mem, ev)
    print("raw:", reply, "normalized:", log["quiz"], "score:", result)
    assert result["accuracy"] == 0. and result["errors"] == 1


def test_valid_negative_control():
    reply = dict(observed=False, answer="I did not see a sofa.", object_id=None, cited_sightings=[])
    a = Agent(ReplyBrain(reply), mind=False)
    log, mem, ev = evidence()
    ev["visibility"] = []
    log["quiz"] = a.quiz(["sofa"], t=100.)
    assert score_quiz(log, mem, ev)["accuracy"] == 1.


def test_malformed_citations_remain_in_citation_denominator():
    a = Agent(ReplyBrain(dict(observed=True, answer="I saw a sofa.", object_id=0,
                             cited_sightings=[0, "bogus-id"])), mind=False)
    a.mem.add("sofa", .9, [1, 1, .5], 10., 0, [0, 0])
    log, mem, ev = evidence()
    log["quiz"] = a.quiz(["sofa"], t=100.)
    result = score_citations(log, mem, ev)
    print("two supplied citations:", log["quiz"], "scored:", result)
    assert result["cited"] == 2 and result["invalid_id"] == 1


def test_citation_actual_capture_must_precede_question():
    log, mem, ev = evidence()
    # A corrupt/stale sighting time points at a later, otherwise matching capture.
    # The scorer must verify the frame timestamp as well as the sighting timestamp.
    log["looks"][0]["t"] = ev["visibility"][0]["t"] = 50.
    log["quiz"] = [dict(t=20., category="sofa", observed=True, xy=[1, 1], cited=[0])]
    result = score_citations(log, mem, ev)
    print("sighting t=10, question t=20, actual cited capture t=50:", result)
    assert result["valid"] == 0 and result["future"] == 1


def test_memory_persistence_baseline_is_labelled_as_a_rule():
    class Brain:
        recall = True

        def resolve(self, table, question):
            return {"object_id": 0}

        def phrase(self, *a):
            raise AssertionError("the baseline verdict must not be phrased as the model's own judgement")
    a = Agent(Brain(), mind=False)
    a.mem.add("sofa", .9, [1, 1, .5], 10., 0, [0, 0])
    a.ask("Is the sofa you saw at 10 s still where it was?")
    a._decide(20., np.zeros(2))
    ans = a.qa[-1]
    assert ans["from_memory"] and "persistence" in ans["rule"] and ans["say"].startswith("From memory, without looking")


def test_spoken_verification_answer_is_the_scored_verdict():
    from scout.env.agent import canonical_answer
    assert "could not find it there now" in canonical_answer("not found", "sofa", 10.0)
    assert "just saw it there again" in canonical_answer("still there", "sofa", 10.0)
    assert "could not recognise" in canonical_answer("unclear", "sofa", 10.0, occupied=True)
