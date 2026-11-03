# Scout environment: comparison runs

Every run below is kept, with its frozen plan in `configs/frozen_<tag>.json`. Scores come from
`modal run scripts/modal_env.py::score --tag <tag>`.

## Pilot 1: `cmp1` (24 episodes, superseded)

- **Plan:** 6 houses × 4 brains, code from before review round 2.
- **Finding:** every verification answer was "no check". After a close inspection the body could stand inside an
  obstacle's clearance margin, and the planner then had 1 reachable cell of 7,328.
- **Fix:** the body may step out of a margin it already stands in (`SelfMap.planning_mask`). Kept as a pilot only.

## Pilot 2: `final` (32 episodes, superseded as an evaluation)

8 fresh houses × 4 brains (spark = Muse Spark 1.3 on Meta Model API, llama4 = Llama 4 Maverick, rules,
random). All 32 planned episodes completed.

| brain | target frozen | verify unchanged (correct / abstain / wrong / other) | verify moved (correct / abstain / wrong / other) | memory-persistence rule, unchanged / moved correct | quiz accuracy (errors) | affirmative answers with valid evidence | falls / got up |
|---|---|---|---|---|---|---|---|
| llama4 | 6 | 2 / 4 / 0 / 2 | 1 / 4 / 0 / 3 | 5 / 0 | 0.61 (1) | 58/75 | 3 / 3 |
| random | 7 | 1 / 0 / 0 / 7 | 0 / 1 / 1 / 6 | 1 / 0 | 0.16 (0) | 5/76 | 5 / 5 |
| rules | 4 | 3 / 1 / 0 / 4 | 1 / 3 / 0 / 4 | 4 / 0 | 0.77 (0) | 37/56 | 5 / 5 |
| spark | 5 | 1 / 4 / 0 / 3 | 0 / 5 / 0 / 3 | 5 / 0 | 0.81 (0) | 45/60 | 6 / 5 |

What it showed:

- **The memory-persistence rule was never right when the object had moved** (0 in every brain's row). In the
  unchanged arm it was right whenever the brain picked the right object. This is the stale-memory failure the
  environment is built to expose.
- **The memory quiz separates the brains.** Ranked against what each agent's camera actually showed: spark 0.81,
  rules 0.77, llama4 0.61, random 0.16.
- **The physical check almost never gave a wrong verdict, but it abstained most of the time.** The per-check
  evidence (`modal run scripts/modal_env.py::evidence --tag final`) showed three causes:
  1. no reachable viewpoint;
  2. it arrived without ever facing the spot (0 looks in the 40 s budget);
  3. the object stood against a wall, so after the move the depth camera saw the wall within 0.35 m of the spot.
- **Falls:** 19 falls in 32 episodes; the body got up on its own 18 times.

Fixes, after which these 8 houses count as development houses:
- the viewpoint is the nearest reachable cell 1.2-3.5 m from the spot with a clear line of sight on the agent's own
  map;
- the check budget is 75 s, including turning to face the spot;
- "seeing past" the spot needs 0.2 m (simulated depth is ideal);
- cells the body has stood on stay passable.

Development check after the fixes (`dev3`, the same failure houses): val_749 and val_631 now answer correctly in
both arms. val_227 then exposed the trail-trapping case, which was fixed and is being rechecked (`dev4`).

## `final2` (stopped after 9 of 32 episodes)

Two flaws found during the run and in the development checks. The run was stopped so they did not reach a final
evaluation. Its houses now count as development houses.

- **Identity.** The target was one of several dining chairs. In the moved arm a neighbouring chair within 1.5 m
  counted as "still there" (dev4, val_227). Fixes:
  - a hit needs a same-category detection within 0.6 m of the remembered spot, or a same-category box over the
    remembered pixel while depth shows something at the remembered distance;
  - the evaluator only freezes targets with no other instance of their category within 2.5 m.
- **Reach.** Muse Spark's questions often named an object seen through a doorway in an unmapped area: unreachable in
  4 of its 7 finished episodes. Fix: walk to the reachable cell nearest the object, then search for a viewpoint
  again.
- **After the fixes (dev5, the same failure houses):**
  - val_227: unchanged correct, moved abstains (it was wrong before);
  - val_105 and val_726: both arms abstain. In val_726 something still stands at the TV's spot after the move,
    probably its stand, and the answer says so.

  The check now prefers abstaining to guessing.

## Final evaluation: `final3` (8 brand-new houses × 4 brains, 32 episodes)

Results are added here when the run finishes. Houses: `configs/final3_houses.json`; 52 houses were excluded as
used before.

## Branch replay

120 identical steps from one snapshot, run twice: maximum qpos difference 3.9e-12. 3 of 8 compared head frames are
identical; the others differ by at most 1 intensity level in at most 0.002% of pixels (GPU rounding).
