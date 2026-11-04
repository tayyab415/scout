import numpy as np

from scout.eval.episodes import explore_then_find, objectnav
from scout.policies import NearestFrontier, RandomFrontier
from scout.teacher.prompt import compass, describe, parse
from scout.agent import Body, frontier_table, look_around
from scout.memory.explored import Memory
from test_world import corridor_house


def test_objectnav_finds_fridge_in_other_room():
    lay = corridor_house()
    r = objectnav(lay, NearestFrontier(), (0.8, 0.8), "Fridge", budget_m=30)
    assert r["success"], r
    assert r["walked_m"] >= r["optimal_m"] - 0.3
    assert 0 < r["spl"] <= 1


def test_objectnav_respects_budget():
    lay = corridor_house()
    r = objectnav(lay, RandomFrontier(1), (0.8, 0.8), "Fridge", budget_m=1.0)
    assert not r["success"] and r["walked_m"] <= 1.0 + 1e-6


def test_explore_then_find_uses_memory():
    lay = corridor_house()
    r = explore_then_find(lay, NearestFrontier(), (0.8, 0.8), "Fridge", explore_m=20, find_m=20)
    assert r["in_memory"] and r["success"]
    assert r["explore"]["coverage"] > 0.8


def test_prompt_only_mentions_seen_things():
    lay = corridor_house()
    mem = Memory(lay)
    body = Body(np.array([2.5, 1.5]))
    mem.look(body.xy, np.pi)  # looking at the bed, away from the kitchen
    fr, _ = frontier_table(mem, body)
    text = describe(mem, body, fr, "Fridge", 30)
    assert "bed at (1.0, 1.5)" in text
    assert "fridge at" not in text.lower().replace("find the fridge", "")
    assert "kitchen" not in text.lower()


def test_parse_handles_messy_replies():
    assert parse('{"choice": 2, "why": "corridor"}', 3) == (2, "corridor")
    assert parse('Sure! ```json\n{"choice": "1", "why": "x"}\n```', 3)[0] == 1
    assert parse('choice: [0] because', 3)[0] == 0
    assert parse('{"choice": 7}', 3)[0] is None
    assert parse("no idea", 3)[0] is None


def test_compass():
    assert compass(1, 0) == "east" and compass(0, 1) == "north" and compass(-1, -1) == "south-west"
