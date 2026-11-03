# Scout: an embodied memory environment built on the Meta stack

**Can a robot tell what it knows from what it needs to check? Plug in a brain and find out.**

Scout puts a physics-simulated humanoid in a home it has never seen. It explores with its own eyes, builds its own map
and memory, and is later asked: *"Is the sofa still where you last saw it?"* Sometimes the evaluator has secretly
moved the sofa, and sometimes it has not. The body has to walk back, look, and answer from fresh evidence. Brains are
plug-ins, so different models can be compared on the same houses with the same body, eyes and budgets.

| Part | Model / asset | Role |
| --- | --- | --- |
| Body | Meta Motivo M-1 (FB-CPR), SMPL humanoid in MuJoCo | walks, falls, gets back up |
| Eyes | SAM 3.1 (open weights, `facebook/sam3.1`) | open-vocabulary object masks on the head camera |
| Brain under test | GPT-6.1 Terra (stand-in for Muse Spark), Llama 4 Maverick, rules, random | when to re-check; which object a question is about; the answers |
| Worlds | ProcTHOR-10K houses from MolmoSpaces (CC BY 4.0), 800 train / 200 test | unseen homes |

## Built during the hackathon (Oct 27 - Nov 4, 2026)

The models and houses above were not built here. Everything else was.

- The body controller: blended Motivo skill vectors and fall recovery.
- The head RGB-D camera, and the floor map the agent builds from its own depth.
- SAM 3.1 placement of detections in 3D, and the episodic memory with confident and tentative objects.
- The brain interface.
- The evaluator: stable object ids, per-frame visibility, hidden interventions.
- The branched moved / unchanged episodes, the scorers, and the synchronised video renderer.

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
uv venv && uv pip install -e . modal boto3 pytest pillow "imageio[ffmpeg]"
.venv/bin/python -m pytest -q tests                      # CPU only, includes three rounds of independent review tests
# score a real recorded episode without a GPU: see docs/setup.md section 5 (examples/sample_episode)
modal run scripts/modal_env.py::episode --house procthor-10k-val_val_935 --brain rules     # one episode, both arms (GPU)
modal run scripts/modal_env.py::compare --houses-file final2_houses.json --tag mytag      # frozen brain comparison
modal run scripts/modal_env.py::score --tag mytag                                        # leaderboard
modal run scripts/modal_env.py::videos --tag mytag                                       # synchronised videos
```

## License

MIT for this code; see `THIRD_PARTY.md` for the models and data (Meta Motivo is CC BY-NC 4.0).

## Results

All results, including failures and negative findings, are in `docs/results/`.
