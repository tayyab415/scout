# Setup from scratch

Episodes need a GPU: either [Modal](https://modal.com) (sections 3-4) or any Linux GPU machine (section 4b). Scoring
and the CPU tests run anywhere.

## 1. Local tools

```bash
git clone https://github.com/tayyab415/scout && cd scout
uv venv && uv pip install -e . modal boto3 pytest pillow "imageio[ffmpeg]"
source .venv/bin/activate                  # or prefix commands with .venv/bin/ (python, modal)
python -m pytest -q tests        # CPU only: map, memory, scorers, recording and review contracts
modal token new                            # Modal account
export SCOUT_VOLUME=scout-data             # any Modal volume name; created on first use
```

## 2. Access you need

| What | Why | How |
| --- | --- | --- |
| Hugging Face access to `facebook/sam3.1` | SAM 3.1 open weights (gated; request access on the model page) | `huggingface-cli login`, so a token sits at `~/.cache/huggingface/token`; it is passed to Modal as an ephemeral secret |
| AWS Bedrock in us-east-1 with `us.openai.gpt-6.1-terra` and `us.meta.llama4-maverick-17b-instruct-v1:0` | the LLM brains (the rules and random brains need nothing) | default profile in `~/.aws/credentials`, passed to Modal as an ephemeral secret |
| Modal GPU (L40S) | Motivo body, MuJoCo rendering, SAM 3.1 | about 2.2 x slower than real time per episode |

The Motivo M-1 weights (`facebook/metamotivo-M-1`) download automatically and need no login.

## 3. Data on the volume

```bash
modal run scripts/modal_fetch_houses.py                        # 1,000 ProcTHOR-10K val houses + THOR assets (CC BY 4.0)
modal run scripts/modal_layouts.py                             # evaluator layouts (walkable grid, rooms, objects)
modal volume put $SCOUT_VOLUME assets/skills/z_motion.pt /scout/z_motion.pt   # Motivo skill vectors (50 KB, in the repo)
```

The skill vectors were made here, by Motivo's reward inference on its own motion buffer (`scripts/modal_skills.py`,
`scripts/modal_motion_tests.py`). The repo ships the result, so you do not need the buffer.

## 4. Run

```bash
modal run scripts/modal_env.py::episode --house procthor-10k-val_val_935 --brain rules   # one episode, both arms
modal run scripts/modal_env.py::compare --houses-file final_houses.json --brains terra,llama4,rules,random --tag mytag
modal run scripts/modal_env.py::score --tag mytag                # leaderboard, computed next to the data
modal run scripts/modal_env.py::videos --tag mytag               # synchronised videos with evidence cards
modal run scripts/modal_env.py::replay                           # branch replay determinism check
```

## 4b. Run on your own GPU machine, without Modal

The final evaluation's last 8 episodes ran this way on an AWS g5.2xlarge (A10G, 24 GB, Deep Learning AMI).

```bash
sudo apt-get install -y ffmpeg libegl1 libgl1
uv venv /opt/venv --python 3.12 && source /opt/venv/bin/activate
uv pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
git clone https://github.com/facebookresearch/sam3.git /opt/sam3 && uv pip install -e /opt/sam3
uv pip install "imageio[ffmpeg]" pillow einops decord pycocotools psutil scipy pandas opencv-python-headless \
    mujoco==3.17.0 h5py boto3 "metamotivo[huggingface,humenv] @ git+https://github.com/facebookresearch/metamotivo.git"
uv pip install -e .
export MUJOCO_GL=egl SAM3_DIR=/opt/sam3 HF_TOKEN_PATH=~/.cache/huggingface/token
```

Put the data under `/data` (or pass `--data-root`): `house/{scenes,objects,layouts}` from section 3 (copy them off
the Modal volume with `modal volume get`) and `scout/z_motion.pt` from `assets/skills/`.

```bash
python scripts/run_local.py --house procthor-10k-val_val_935 --brain rules --tag dev            # one episode
python scripts/run_local.py --plan configs/frozen_final3.json --tag final3 --parallel 2         # every missing episode of a plan
python scripts/run_local.py --brain-module mybrain.py --brain mybrain --house procthor-10k-val_val_752 --tag mine
```

Each episode runs in a fresh process; outputs land in `/data/scout/env/<tag>/<episode>/` and every attempt, crashes
included, is appended to `run_local.jsonl`. Two episodes in parallel fit in 24 GB.

## 5. Score a recorded episode without a GPU

`examples/sample_episode/` is one real episode from the final evaluation (`final3`, Terra, house val_752): logs,
memory, evaluator records, maps, the head frames the evidence cards cite, and that house's layout. The physical check
is correct in both worlds; the memory-persistence rule is wrong in the moved one.

```bash
python scripts/score_episode.py examples/sample_episode
```

To rescore all 32 final3 episodes from the public records, see `docs/evidence.md` (`scripts/rescore_records.py`).
