# Setup from scratch

Everything heavy runs on [Modal](https://modal.com) GPUs. Scoring and the CPU tests run anywhere.

## 1. Local tools

```bash
git clone <this repo> scout && cd scout
uv venv && uv pip install -e . modal boto3 pytest pillow "imageio[ffmpeg]"
.venv/bin/python -m pytest -q tests        # CPU only: map, memory, scorers, recording and review contracts
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

## 5. Score a recorded episode without a GPU

`examples/sample_episode/` is one real episode from the final evaluation (`final3`, Terra, house val_752): logs,
memory, evaluator records, maps, the head frames the evidence cards cite, and that house's layout. The physical check
is correct in both worlds; the memory-persistence rule is wrong in the moved one.

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'src')
from scout.env.scoring import score_episode
from scout.world.layout import Layout
import json, glob
d = 'examples/sample_episode'
lay = Layout.load(glob.glob(d + '/layout_*.npz')[0])
print(json.dumps(score_episode(d, lay), indent=1, default=float)[:3000])"
```
