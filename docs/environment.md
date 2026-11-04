# Scout environment: API, tasks and scoring

## API (`src/scout/env/`)

```python
from scout.env.core import ScoutEnv, PROMPTS
from scout.env.agent import Agent
from scout.brains import make_brain            # "terra", "llama4", "rules", "random", or "llm:<bedrock model id>"

env = ScoutEnv("procthor-10k-val_val_935", seed=0, eyes=Sam31())   # GPU: Motivo body, MuJoCo house, SAM 3.1
env.reset()                                     # random free start from the seed
agent = Agent(make_brain("terra"))
while env.t < 240:
    agent.tick(env)                             # one 1/30 s step: perceive (every 0.5 s), decide, steer, env.step(z)
snap, saved = env.snapshot(), copy.deepcopy(agent)   # branch point
env.move_object(instance_id, to_xy)             # evaluator intervention (hidden from the agent)
agent.ask("Is the sofa still where you last saw it?")
```

- `ScoutEnv.observe(detect)` is everything the agent may use. `truth_objects()`, `visible_instances()` and
  `move_object()` are evaluator-only.
- Path clearance: every planned path cell keeps 0.25 m from the nearest obstacle cell, except cells the body has
  already stood on. It fits there, so it can always retrace its own trail. When the body already stands inside a
  margin, it may step out within 0.6 m.
- One clock: `env.t = env.k / 30`. Every recorded stream (observer frames, head frames, events, map snapshots)
  carries `t`. Fall recovery runs on the same clock.

## Brain interface (`src/scout/brains.py`)

| Call | Input | Output |
| --- | --- | --- |
| `decide(table, state)` | memory table; time, unexplored openings, stale objects, recent events | `{"do": "explore"}` or `{"do": "check", "object_id": N}` |
| `resolve(table, question)` | memory table, question | `{"object_id": N or null, "say": ...}` |
| `phrase(question, remembered, finding)` | the verdict from the body's check | the spoken answer |
| `answer(table, question)` | memory table, quiz question | `{"answer", "cited_sightings", "observed", "object_id"}` |

Every brain gets the same table and state strings.

## How a check reaches a verdict (the body does this, not the brain)

If no viewpoint is reachable yet, the body first walks to the reachable cell nearest the object, since its map
grows on the way, and searches again. The body walks to the nearest reachable cell 1.2-3.5 m from the remembered spot that has a clear line of sight on
its own map, and turns to face it. It then looks for 4.5 s, or 1.5 s if it sees the object. The whole check has
a 75 s budget.

| Verdict | Condition |
| --- | --- |
| still there | SAM 3.1 detects the same category within 0.6 m of the remembered spot (instance level: a neighbouring chair is a different chair) |
| not found | no such detection, **and** on at least 2 looks the depth camera sees more than 0.2 m past the remembered spot, so the spot is now empty |
| unclear | anything else: out of view, occluded, or something is still there that SAM did not recognise |

"Not found" never claims the object was moved. Spoken answers say "I couldn't find it where I last saw it".

## Episode protocol (`scripts/modal_env.py::run_episode`)

1. **Reset and explore.** Reset with the seed and explore for `explore_s` (240 s). The brain decides re-checks
   every 20 s and on events.
2. **Freeze the target (evaluator).** It is the movable object instance the agent has seen most. It must match a
   memory track seen at least 3 times, within 1.5 m, have no other instance of its category within 2.5 m (so the question names an identifiable object), be at
   least 4 m from the body, and not be in view. If no instance
   qualifies, the episode records `no_target_reason` and stops.
3. **Freeze the question.** It is fixed before branching: "Is the <category> still where you last saw it?".
4. **Pick a destination.** It is in a different room, at least 4 m from the source and 6 m from the body.
5. **Snapshot** the environment and the agent.
6. **Run two arms from the snapshot:**
   - **moved:** the evaluator moves the target;
   - **unchanged:** nothing changes.

   Both arms then run 20 s on their own, get the same question, and run until the answer (90 s timeout), plus 3 s.
7. **Memory quiz** in the unchanged arm: "Did you see a <category>? If so, where was it?" for all 16 categories.

## Scoring (`src/scout/env/scoring.py`)

| Task | Score |
| --- | --- |
| verify | unchanged → "still there" is correct; moved → "not found" is correct; "unclear" or no check is an abstention. Both arms are reported, so a brain that always says "not found" scores 1/2. |
| remember | A category counts as seen only if one of its instances covered at least 0.5% of a head frame the agent captured (evaluator segmentation at every look). "Yes" is correct only if the category was seen **and** the answer places it within 1.5 m of a seen instance. "No" is correct only if it was never seen. |
| cite | Each cited sighting must exist, come from a frame captured before the question, and that frame's segmentation must contain an instance of the category. |
| checks | The brain's own re-checks, scored against the truth at that time (moves applied). Abstentions are reported. |
| explore | Share of the true walkable floor that the agent's map saw by the freeze time; rooms entered; metres walked; stuck events; share of 0.5 s intervals spent below 0.1 m/s. |
| stay up | Falls, recoveries (standing again within 8 s), and early ends. |

## Outputs per episode (`/data/scout/env/<tag>/<episode>/` on the Modal volume)

- `manifest.json`: the spec, start pose, frozen target, destination, question, arms and SAM 3 commit.
- `observer_prefix.mp4`, `observer_<arm>.mp4`: the stable viewer camera. Walls and ceilings are hidden from this
  camera only.
- `<arm>/` contains:
  - `log.json`: events, looks, decisions, QA, quiz, observer frame times;
  - `memory.json`;
  - `evaluator.json`: truth positions, moves and per-look visibility;
  - `maps.npz` and `map_meta.json`;
  - `frames/*.jpg`: every head frame.
- `<arm>.mp4`: the synchronised three-panel video with evidence cards (`scripts/env_video.py`).

## Branch replay check (`modal run scripts/modal_env.py::replay`)

The check takes one snapshot in house val_935, then runs the same 120-step action sequence twice from it, with no
SAM and no language model. Result on 3 Nov:
- body state: maximum qpos difference 3.9e-12;
- head images: 3 of 8 compared frames are identical; the others differ by at most 1 intensity level, in at most
  0.002% of pixels;
- the policy and renderer run on the GPU, so traces are reproducible to rounding, not bit-for-bit.

## Speed (measured 3 Nov)

- **Episode speed.** One episode runs about 2.2× slower than real time on a Modal L40S. On an AWS g5.2xlarge
  (A10G) it runs about 7× slower, with 2 episodes sharing the GPU.
- **Where the time goes.** A py-spy profile of a running episode on the A10G (479 samples) puts 95.6% of the time
  in SAM 3.1 detection, about 1% each in physics and policy and in the observer render, and under 0.5% in mapping
  and planning.
- **Why.** The eyes drive SAM 3.1's video predictor once per category: 16 one-frame sessions per look, each of
  which re-encodes the image. FlashAttention 3 is unavailable on Ampere GPUs.
- **Not yet applied:**
  - encode each frame once and run all 16 text prompts on it;
  - SAM 3.1's `torch.compile` mode, which its documentation says gives about 2×.

  Both were left out so the frozen comparison kept one detection pipeline.

### Measured: SAM 3.1 weights in image mode (`scripts/sam_speed_test.py`, 4 Nov, A10G)

The SAM 3.1 checkpoint's detector weights were loaded into SAM 3's image model, which encodes each frame once and
then runs the 16 text prompts. Test set: 20 real head-camera frames from final3 (Terra, val_752).

| | Current pipeline (one video session per category) | Image mode |
| --- | --- | --- |
| Seconds per look | 5.23 | 1.67 (3.1x faster) |
| Detections (score >= 0.5) | 28 | 33 |
| Same detect / no-detect decision per category | | 98.4% |
| Mask IoU when both detect | | 0.98 |

**Caveat:** 4 weights of one neck level (`backbone.vision_backbone.convs.3.*`) have no counterpart in the
checkpoint, so they stay randomly initialised. The 5 extra detections have not been checked against the truth. Not
adopted for the comparison; it is the first thing to validate for faster runs.

### Physics options when the house is attached

MuJoCo warns about an attach conflict and keeps the humanoid's physics options (solver, gravity, contact limits)
instead of the house file's. That is intended: Motivo was trained with HumEnv's settings.
