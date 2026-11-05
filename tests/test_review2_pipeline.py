"""Failure-denominator probe. No cloud calls or model requests."""
import argparse
import ast
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).parent / ".."


def test_crashed_planned_episode_remains_in_its_brain_denominator(tmp_path, monkeypatch):
    root = tmp_path / "runs/env/probe"
    complete = root / "spark_house_a_s0"
    crashed = root / "spark_house_b_s0"
    complete.mkdir(parents=True)
    crashed.mkdir()  # episode created its output dir, then failed before final manifest
    (complete / "manifest.json").write_text(json.dumps(dict(house="house_a", brain="spark")))
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/frozen_probe.json").write_text(json.dumps(dict(
        houses=["house_a", "house_b"], brains=["spark"], explore_s=240, seed=0, tag="probe")))
    tree = ast.parse((ROOT / "scripts/score_env.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    namespace = dict(argparse=argparse, os=os, json=json, np=np, defaultdict=defaultdict,
                     Layout=NS(load=lambda p: None),
                     score_episode=lambda d, layout: dict(episode=Path(d).name, brain="spark", target=None))
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(ROOT / "scripts/score_env.py"), "exec"), namespace)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["score_env.py", "--tag", "probe", "--no-fetch"])
    namespace["main"]()
    board = (root / "leaderboard.md").read_text()
    row = next(line for line in board.splitlines() if line.startswith("| spark |"))
    assert row.split("|")[2].strip() == "2", "A crashed planned episode disappeared from the brain denominator"
