# Scout

### A physics-simulated home that tests whether a robot's memory is still true.

![Scout: a Meta Motivo body and SAM 3.1 eyes explore a house, the episode splits into two worlds, and only looking tells them apart](assets/media/scout_diagram.png)

![Same robot, same memory, same question: two worlds](assets/media/branch.gif)

A Meta Motivo humanoid explores a house it has never seen. It sees through a head camera with SAM 3.1 and keeps a
memory of what it saw and where. Then the episode splits into two worlds. In one, nothing changes. In the other, we
quietly move an object. Both copies of the robot get the same question: *"Is the sofa you saw at 232 s still where
it was?"*

![Memory can't tell the two worlds apart. Looking can.](assets/media/verdict.png)

**Memory can't tell the two worlds apart. Looking can.** Answering from memory says "still there" in both worlds,
so it is wrong in one of them. The robot that walks back and looks gets both right: its depth camera now sees past
the spot where the sofa was.

**Scout gives robots a new capability: knowing when their memory has gone stale, and going back to check.** It is
an open, physics-simulated proving ground where three Meta models run in one closed loop: Meta Motivo moves the
body, SAM 3.1 sees through the head camera, and Muse Spark 1.3 is the brain under test. The brain and the eyes are
served by Meta Model API. Any model plugs in through four methods. Scout runs the physics, the eyes and the hidden
interventions, then grades every answer against what the robot's camera actually saw, frame by frame.

**It also trains brains.** A 233-parameter neural network, trained in seconds on Scout's own episodes, took over
the robot's head and out-remembered Muse Spark 1.3 in eight never-seen homes: 106 of 112 against 92.
[See the network](#the-neural-network-we-trained).

- **Real physics, not animation.** A Meta Motivo humanoid in MuJoCo walks, turns to look, falls and gets itself
  back up: 35 falls and 35 recoveries in the final evaluation.
- **It sees for itself.** SAM 3.1 open-vocabulary masks on an RGB-D head camera, lifted into 3D, and a floor map
  built from its own depth. No layout, no object list, no shortcuts.
- **A world that changes behind its back.** Each episode branches from one physics snapshot into a moved world and
  an unchanged world, so both copies of the robot face the identical question.
- **Scoring that is hard to fool.** Every answer is graded against per-frame visibility, every cited frame is
  checked against the evaluator's segmentation, abstentions are counted separately, and every failure is kept.
  Anyone can rescore all 32 final episodes from the public records on a laptop.

### Why it matters

Every robot and assistant that remembers the physical world falls into the same trap. A home robot asked "where
are my keys?", a care robot making sure the walker is still by the bed, AI glasses recalling where you left your
bag: each answers from a memory that may no longer be true. Scout makes that failure measurable, comparable across
models and trainable: the data it produces already trains better brains.

| Part | Model and how it runs | Role |
| --- | --- | --- |
| Body | Meta Motivo M-1 (FB-CPR), SMPL humanoid in MuJoCo, open weights | walks, falls, gets back up |
| Eyes | SAM 3.1: `sam-3.1` on Meta Model API, or its open weights on Modal | open-vocabulary object masks on the head camera |
| Brain under test | Muse Spark 1.3 (`muse-spark-1.3` on Meta Model API), Muse Spark 1.1, Llama 4 Maverick, rules, random, and a 233-parameter neural network trained on Scout's own episodes | when to re-check; which object a question is about; the answers |
| World model | V-JEPA 2 (frozen ViT-L encoder) | our action-conditioned predictor of the body's next head-camera view |
| Worlds | ProcTHOR-10K houses from MolmoSpaces (CC BY 4.0), 800 train / 200 test | unseen homes |
| Compute | Modal L40S: Motivo physics, MuJoCo rendering, segmentation | the whole episode |

One `MODEL_API_KEY` covers `muse-spark-1.3`, `muse-spark-1.2`, `muse-spark-1.1`, `muse-wonder-1.0` and `sam-3.1`
at `https://api.meta.ai/v1`. The brain is a plug-in: `--brain spark`, `--brain spark-1.1`, `--brain llama4`, or
`--brain llm:<model id>` for anything else in the catalog. Llama 4 Maverick is the cheap baseline, and it can be
swapped for Muse Spark Open (the 816B open-weight release, served on Modal through vLLM or SGLang) or for
`muse-spark-1.1`, the smallest model in the API.

## Built during the hackathon (Oct 27 - Nov 4, 2026)

The models and houses above were not built here. Everything else was.

- The body controller: blended Motivo skill vectors and fall recovery.
- The head RGB-D camera, and the floor map the agent builds from its own depth.
- SAM 3.1 placement of detections in 3D, and the episodic memory with confident and tentative objects.
- The brain interface.
- The evaluator: stable object ids, per-frame visibility, hidden interventions.
- The branched moved / unchanged episodes, the scorers, and the synchronised video renderer.
- A V-JEPA 2 world model for the body's own eyes: an 11.4M-parameter action-conditioned predictor trained on
  104,730 head-camera frames, following the V-JEPA 2-AC recipe (`docs/results/vjepa_world_model.md`).
- The student: a 233-parameter neural network trained in seconds, on CPU, from the episodes the environment exports
  (`src/scout/student/`, added 4 November).

## What the agent gets (observation contract)

- **Head RGB-D:** 512×512, 75° vertical FOV, tilted 20° down. Depth is ideal simulated depth.
- **Pose:** the exact head-camera pose, and the body's position, heading and head height. This is perfect odometry
  from the simulator, and it is disclosed.
- **Detections:** SAM 3.1 detections for 16 object categories, once per second.
- **Actions:** Motivo skill vectors at 30 Hz. The built-in controller turns waypoints into skill blends.
- **The agent never sees:** the house layout, room labels, the object list, or any evaluator intervention.

Exploration uses a fixed rule (nearest reachable frontier on the agent's own map) for every brain. Brains decide
re-checks and answer questions. See `docs/environment.md` for the API, the tasks and the scoring rules.

## Run it

Full setup from scratch is in `docs/setup.md`: accounts, the house download, skill vectors and every command.

```bash
uv venv && uv pip install -e . modal requests pytest pillow "imageio[ffmpeg]"
export MODEL_API_KEY=...                                  # https://dev.meta.ai — one key for Muse Spark and SAM 3.1
.venv/bin/python -m pytest -q tests                      # CPU only: 81 tests, including six rounds of independent review
python scripts/quiz_offline.py records                   # the memory quiz answered offline by any brain, scored by the environment
python scripts/train_student.py --records <dev runs> --exclude-houses configs/frozen_final3.json   # retrain the student
# score a real recorded episode without a GPU: see docs/setup.md section 5 (examples/sample_episode)
modal run scripts/modal_env.py::episode --house procthor-10k-val_val_935 --brain rules     # one episode, both arms (GPU)
modal run scripts/modal_env.py::compare --houses-file final2_houses.json --tag mytag      # frozen brain comparison
modal run scripts/modal_env.py::score --tag mytag                                        # leaderboard
modal run scripts/modal_env.py::videos --tag mytag                                       # synchronised videos
```

## A V-JEPA 2 world model for the robot's own eyes

Scout's memory asks *what is where*. A world model asks *what will I see if I move*. We trained one on the robot's
own experience: a frozen V-JEPA 2 encoder with a new 11.4M-parameter action-conditioned predictor, following Meta's
V-JEPA 2-AC recipe, on 104,730 head-camera frames from 2,592 Motivo episodes in 48 houses. On 12 houses it never
saw, knowing the action makes its prediction of the next view more accurate than an identical action-blind twin
(L1 0.296 against 0.321 one step ahead) and it identifies which of six skills the body ran 53% of the time (chance
17%). Full results: `docs/results/vjepa_world_model.md`.

## Research foundations

Scout builds directly on Meta research, and on the broader embodied-AI literature it answers to.

| Work | How Scout uses it |
| --- | --- |
| Tirinzoni et al., *Zero-Shot Whole-Body Humanoid Control via Behavioral Foundation Models* (Meta Motivo, FB-CPR), 2025. [arXiv:2504.11054](https://arxiv.org/abs/2504.11054) | The body. We blend its skill vectors into walking, turning to look and fall recovery. |
| Carion et al., *SAM 3: Segment Anything with Concepts*, 2025. [arXiv:2511.16719](https://arxiv.org/abs/2511.16719) | The eyes (SAM 3.1 weights): open-vocabulary masks lifted into 3D memory. |
| Assran et al., *V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning*, 2025. [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) | The encoder and the action-conditioned recipe behind our world model. |
| Bardes et al., *Revisiting Feature Prediction for Learning Visual Representations from Video* (V-JEPA), 2024. [arXiv:2404.08471](https://arxiv.org/abs/2404.08471) | The video JEPA line our world model descends from. |
| Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA), 2023. [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) | Predicting in representation space, not pixels: the idea our predictor follows. |
| LeCun, *A Path Towards Autonomous Machine Intelligence*, 2022. [OpenReview](https://openreview.net/forum?id=BZ5a1r-kVsf) | The case for agents with a world model and memory. Scout tests what happens when that memory goes stale. |
| Balestriero and LeCun, *LeJEPA*, 2025. [arXiv:2511.08544](https://arxiv.org/abs/2511.08544) | Current JEPA training theory; context for the world-model work. |
| Majumdar et al., *OpenEQA: Embodied Question Answering in the Era of Foundation Models*, CVPR 2024. [Project](https://open-eqa.github.io/) | Meta's benchmark for answering questions from episodic memory. Scout adds the missing case: the world changed after the memory was formed. |
| Puig et al., *Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots*, 2023. [arXiv:2310.13724](https://arxiv.org/abs/2310.13724) | Meta's embodied simulation platform; Scout brings full-body humanoid physics to the home-memory setting. |
| Meta, *Muse Spark* model page, 2026. [dev.meta.ai](https://dev.meta.ai/models/muse-spark) | Muse Spark 1.3, the brain under test, served through Meta Model API. |
| Meta, *The Llama 4 herd*, 2025. [Blog](https://ai.meta.com/blog/llama-4-multimodal-intelligence/) | Llama 4 Maverick, the cheap baseline brain. |
| Deitke et al., *ProcTHOR: Large-Scale Embodied AI Using Procedural Generation*, 2022. [arXiv:2206.06994](https://arxiv.org/abs/2206.06994) | The houses (via Ai2 MolmoSpaces). |

## License

MIT for this code; see `THIRD_PARTY.md` for the models and data (Meta Motivo is CC BY-NC 4.0).

## Results (final evaluation: 8 never-seen homes × 4 brains, 32 of 32 episodes)

- **Six clean verification pairs out of 32 episodes.** In each, the robot walked back, looked, and answered
  correctly in both worlds, while the memory-persistence rule was wrong in the moved one.
- **All 64 verification arms, counted:** 16 correct, 10 abstained, 16 wrong target (mostly the random brain),
  8 no answer in time, 14 not run (no eligible target). No arm checked the right object and gave the wrong
  verdict.
- **The memory-persistence rule was never right when the object had moved** (0 for every brain). The rule answers
  from memory without looking and calls no model, so this shows the stale-memory trap, not a model's judgement.
- **The memory quiz separates the brains.** It is graded only against what each robot's camera actually saw:
  - Muse Spark 1.3 (Meta Model API): 92/112 correct;
  - rules: 93/112;
  - Llama 4 Maverick: 48/96, with 16 invalid replies;
  - random: 16/96.
- **The body stays up:** 35 falls, and it got itself back up all 35 times.

## The neural network we trained

![A 233-parameter network learned which sightings to doubt](assets/media/student_card.png)

Scout is not only a test. Every episode it runs becomes training data: the memory the robot built, paired with the
evaluator's ground truth about what was really there. We used that data to train a tiny neural network, dropped it
into the robot's head, and sent it into eight homes it had never seen. **It out-remembered a frontier LLM.**

### What it is

A **233-parameter neural network**. That is not a typo: 233 weights, against the billions in the language models it
competes with. A two-layer perceptron in plain numpy (27 inputs, 8 tanh units, 1 sigmoid output,
`src/scout/student/trust.py`, weights in `assets/brains/student_trust.npz`). It reads one line of the robot's memory
and answers one question: *is this object really there, or did the eyes imagine it?*

- **It sees what every brain sees, and nothing more:** the memory table. Times seen, SAM 3.1 confidence, time span,
  time since last seen, misses, retractions, how many objects share a label, the category, and a few more counts
  (all 27 in `FEATURE_NAMES`). No positions, no frames, no access to the ground truth.
- **It learned from Scout's own referee:** an object counts as real only if the evaluator confirms it was in view
  (at least 0.5% of a captured frame) within 1.5 m of where the robot remembers it.
- **It runs the whole brain:** it answers the memory quiz, decides which object a question is about, and chooses
  what to walk back and re-check: the object it is least sure of.

### How we trained it

- **Data:** 72 episodes from 22 houses, 1,985 remembered objects, all exported by Scout itself. The eight
  final-evaluation homes were walled off from training, and a test (`tests/test_student.py`) enforces it.
- **Compute:** a few seconds on a laptop CPU. No GPU, no pretraining, no API calls.
- **Validation:** leave-one-house-out cross-validation picked the settings. On held-out houses it reaches 0.781
  accuracy, against 0.680 for blindly trusting everything.
- **What it discovered on its own:** SAM 3.1's blind spots. It learned to trust dining tables and paintings, to doubt
  bathtubs, chairs and televisions, and to drop objects the robot keeps failing to find again.

```bash
python scripts/train_student.py --records <dev runs> --exclude-houses configs/frozen_final3.json
```

### Results

Live, as the robot's only brain, in the same eight never-seen homes as the final evaluation, same seed and budgets:

| live, same 8 never-seen homes | memory quiz |
|---|---:|
| **neural network (233 parameters, trained here)** | **106/112** |
| rules | 93/112 |
| Muse Spark 1.3 (Meta Model API) | 92/112 |
| Llama 4 Maverick | 48/96 |
| random | 16/96 |

**First place out of five, ahead of a frontier LLM, with 233 parameters and seconds of training.** It also wins
offline: handed the memories the other brains built, it answers 370 of 416 questions correctly, against 333 for the
hand-written rules.

This is what Scout is for. The environment that grades a robot's memory also produces the data to build a better
one.

Full run, every outcome: `docs/results/final3_student/leaderboard.md`. Training, cross-validation and ablations:
`docs/results/student.md`.

Full leaderboard, every outcome category, provenance, retries and failures: `docs/results/environment_results.md`
and `docs/results/final3/`. Public videos and the records of all 32 episodes: `docs/evidence.md`.
