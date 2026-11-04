# Scout environment: comparison runs

Every run below is kept, with its frozen plan in `configs/frozen_<tag>.json`. Scores come from
`modal run scripts/modal_env.py::score --tag <tag>`.

## Pilot 1: `cmp1` (24 episodes, superseded)

- **Plan:** 6 houses × 4 brains, code from before review round 2.
- **Finding:** every verification answer was "no check". After a close inspection the body could stand inside an
  obstacle's clearance margin, and the planner then had 1 reachable cell of 7,328.
- **Fix:** the body may step out of a margin it already stands in (`SelfMap.planning_mask`). Kept as a pilot only.

## Pilot 2: `final` (32 episodes, superseded as an evaluation)

8 fresh houses × 4 brains (terra = GPT-6.1 Terra as a stand-in for Muse Spark, llama4 = Llama 4 Maverick, rules,
random). All 32 planned episodes completed.

| brain | target frozen | verify unchanged (correct / abstain / wrong / other) | verify moved (correct / abstain / wrong / other) | memory-persistence rule, unchanged / moved correct | quiz accuracy (errors) | affirmative answers with valid evidence | falls / got up |
|---|---|---|---|---|---|---|---|
| llama4 | 6 | 2 / 4 / 0 / 2 | 1 / 4 / 0 / 3 | 5 / 0 | 0.61 (1) | 58/75 | 3 / 3 |
| random | 7 | 1 / 0 / 0 / 7 | 0 / 1 / 1 / 6 | 1 / 0 | 0.16 (0) | 5/76 | 5 / 5 |
| rules | 4 | 3 / 1 / 0 / 4 | 1 / 3 / 0 / 4 | 4 / 0 | 0.77 (0) | 37/56 | 5 / 5 |
| terra | 5 | 1 / 4 / 0 / 3 | 0 / 5 / 0 / 3 | 5 / 0 | 0.81 (0) | 45/60 | 6 / 5 |

What it showed:

- **The memory-persistence rule was never right when the object had moved** (0 in every brain's row). In the
  unchanged arm it was right whenever the brain picked the right object. This is the stale-memory failure the
  environment is built to expose.
- **The memory quiz separates the brains.** Ranked against what each agent's camera actually showed: terra 0.81,
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
- **Reach.** Terra's questions often named an object seen through a doorway in an unmapped area: unreachable in
  4 of its 7 finished episodes. Fix: walk to the reachable cell nearest the object, then search for a viewpoint
  again.
- **After the fixes (dev5, the same failure houses):**
  - val_227: unchanged correct, moved abstains (it was wrong before);
  - val_105 and val_726: both arms abstain. In val_726 something still stands at the TV's spot after the move,
    probably its stand, and the answer says so.

  The check now prefers abstaining to guessing.

## Final evaluation: `final3` (8 never-seen houses × 4 brains, 32 of 32 planned episodes)

Plan: `configs/frozen_final3.json`. The houses (`configs/final3_houses.json`) were picked after 52 houses had been
used for development. The brain is GPT-6.1 Terra on AWS Bedrock, standing in for Muse Spark.

| brain | episodes planned | completed | target frozen | verify unchanged (correct / abstain / wrong / other) | verify moved (correct / abstain / wrong / other) | memory-persistence rule, unchanged / moved (correct) | quiz accuracy (errors) | affirmative answers with valid evidence | own checks (correct / abstain / wrong) | floor seen (median) | falls / got up |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama4 | 8 | 8 | 6 | 2 / 2 / 0 / 4 | 1 / 2 / 0 / 5 | 3 / 0 | 0.50 (16) | 54/64 | 19 / 18 / 1 | 96% | 15 / 15 |
| random | 8 | 8 | 6 | 0 / 0 / 0 / 8 | 0 / 0 / 0 / 8 | 0 / 0 | 0.17 (0) | 2/64 | 8 / 12 / 1 | 83% | 0 / 0 |
| rules | 8 | 8 | 7 | 4 / 1 / 0 / 3 | 2 / 2 / 0 / 4 | 5 / 0 | 0.83 (0) | 61/87 | 3 / 8 / 0 | 98% | 17 / 17 |
| terra | 8 | 8 | 7 | 3 / 2 / 0 / 3 | 4 / 1 / 0 / 3 | 7 / 0 | 0.82 (0) | 61/82 | 29 / 28 / 6 | 93% | 3 / 3 |

Verification outcomes per arm (physical check; 'not run' = no target frozen, crashed, or moved arm skipped):

| brain | arm | correct | abstain | wrong | wrong target | ineligible | no answer | not run |
|---|---|---|---|---|---|---|---|---|
| llama4 | unchanged | 2 | 2 | 0 | 2 | 0 | 0 | 2 |
| llama4 | moved | 1 | 2 | 0 | 0 | 0 | 2 | 3 |
| random | unchanged | 0 | 0 | 0 | 6 | 0 | 0 | 2 |
| random | moved | 0 | 0 | 0 | 4 | 0 | 1 | 3 |
| rules | unchanged | 4 | 1 | 0 | 2 | 0 | 0 | 1 |
| rules | moved | 2 | 2 | 0 | 2 | 0 | 1 | 1 |
| terra | unchanged | 3 | 2 | 0 | 0 | 0 | 2 | 1 |
| terra | moved | 4 | 1 | 0 | 0 | 0 | 2 | 1 |

Falls, recoveries and own checks are counted on the unchanged arm (it shares the exploration with the moved arm).


**Memory-quiz denominators** (16 questions per episode where the memory quiz ran):

| brain | questions | correct | errors |
| --- | --- | --- | --- |
| terra | 112 | 92 | 0 |
| rules | 112 | 93 | 0 |
| llama4 | 96 | 48 | 16 (invalid replies, counted as wrong) |
| random | 96 | 16 | 0 |

**What final3 shows:**

- **Verification outcomes over all 64 arms:** 16 correct, 10 abstained, 16 wrong target, 8 no answer in time,
  14 not run. The separate `wrong` category (right object, wrong verdict) is 0. These are prompted
  verification outcomes; the agent's own re-checks are scored separately in the leaderboard.
- **The memory-persistence rule was never right in a moved arm** (0 for every brain). In the unchanged arms it was
  right whenever the brain picked the right object: terra 7, rules 5, llama4 3.
- **Clean verification pairs** (the check correct in both arms, the rule correct when unchanged and wrong when
  moved): 6. They are terra on val_893 (dresser), val_752 (sofa) and val_330 (dresser); llama4 on val_300 (sofa);
  rules on val_893 (laptop) and val_300 (dresser). The public sample and the demo use terra / val_752.
- **Failures kept:**
  - no target frozen (for example, val_167 stalls with no unexplored openings left and nothing eligible to move);
  - wrong target (the random brain picks a random object);
  - no answer in time: a question waits behind a re-check already in progress, a known flaw that affects all
    brains alike.
- **Falls:** 35 falls; the body got itself back up 35 times.

**Provenance** (`docs/results/final3/provenance.json`, `docs/results/final3/aws_runs.jsonl`):

- 24 episodes ran on Modal (L40S) from the frozen-plan commit `1cfe674`. They were launched with
  `modal_env.py::compare`; the client was lost when the build host restarted at 22:21 UTC.
- 8 episodes ran on AWS (g5.2xlarge, A10G) with `scripts/run_local.py`, at commits `872684a` (2 episodes) and
  `ba31a83` (6 episodes).
- `git diff` of agent, core, perception, memory, body, world, mind and brains between `1cfe674` and both AWS
  commits is empty. The runner moved into `src/scout/env/episode.py` and gained provenance fields, nothing else.
- **Retry history:**
  - first AWS attempt: rules/val_330 and random/val_893 completed;
  - the other 6 random episodes crashed with CUDA out-of-memory, because each worker process kept the previous
    episode's GPU memory;
  - after the fix (one fresh process per episode), all 6 completed;
  - all attempts are in `aws_runs.jsonl`.
- **SAM 3:** the commit is recorded per episode in its manifest. **Models:** `us.openai.gpt-6.1-terra` and
  `us.meta.llama4-maverick-17b-instruct-v1:0` on Bedrock (us-east-1).

**Recordings and records:** public, no credentials needed, in the GitHub release
[`final3-evidence`](https://github.com/tayyab415/scout/releases/tag/final3-evidence): the demo cut, branch shot,
replay recording, two failure recordings and the records of all 32 episodes. `docs/evidence.md` lists each file.
`scripts/rescore_records.py` rebuilds the outcome table and quiz denominators from those records on a CPU; its output
is in `final3/rescore_from_release.txt` and matches the table above. The full set of arm videos (about 7 GB) stays in
the team's S3 bucket.

## Branch replay

120 identical steps from one snapshot, run twice:

- **Modal L40S:** maximum qpos difference 3.9e-12; 3 of 8 compared head frames identical.
- **AWS A10G:** 5.8e-12; 7 of 8 identical (`docs/results/evidence/replay_check.txt`, `replay_result.json`; recording
  in the `final3-evidence` release).
- **Remaining differences:** at most 1 intensity level, in at most 0.002% of pixels. We have not isolated the cause.
