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

The body walks to a viewpoint 2-3 m from the remembered spot and turns to face it. It then looks for 4.5 s, or
1.5 s if it sees the object.

| Verdict | Condition |
| --- | --- |
| still there | SAM 3.1 detects the same category on the same memory track, or within 1.5 m of the remembered spot |
| not found | no such detection, **and** on at least 2 looks the depth camera sees more than 0.35 m past the remembered spot, so the spot is now empty |
| unclear | anything else: out of view, occluded, or something is still there that SAM did not recognise |

"Not found" never claims the object was moved. Spoken answers say "I couldn't find it where I last saw it".

## Episode protocol (`scripts/modal_env.py::run_episode`)

1. **Reset and explore.** Reset with the seed and explore for `explore_s` (240 s). The brain decides re-checks
   every 20 s and on events.
2. **Freeze the target (evaluator).** It is the movable object instance the agent has seen most. It must match a
   memory track seen at least 3 times, within 1.5 m, be at least 4 m from the body, and not be in view. If no instance
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
