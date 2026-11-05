# Submission draft (Cerebral Valley form)

**Project name:** Scout

**Tagline:** A physics-simulated home that catches robots trusting stale memories, and trains them to check.

**Track / problem statement:** Push the Frontier (robotics: give a machine a new capability, and a way to measure it).

## Description

Home robots answer questions from memory: "Is the sofa still in the living room?" But the world changes when
they are not looking, and a robot that trusts a stale memory is confidently wrong. Scout gives robots a new
capability, knowing when to doubt what they remember and walking back to check, and a controlled way to
measure it. Three Meta models run in one closed loop: Meta Motivo for the body, SAM 3.1 for the eyes, and Llama 4
Maverick among the brains under test.

- **The setup.** A Meta Motivo humanoid explores a house it has never seen, in full physics, through a head camera.
  SAM 3.1 recognises objects. Depth builds the robot's own map. Every sighting is remembered with the frame it came
  from.
- **The two worlds.** Then the episode branches from one physics snapshot into two worlds. In one, nothing changes.
  In the other, we quietly move an object. Both copies of the robot get the identical question.
- **What it shows.** The baseline is a fixed memory-persistence rule: it answers from memory without looking and
  calls no model. It says "still there" in both worlds, so it is wrong in one of them. The robot that walks back and
  looks can get both right: its depth camera sees past the spot where the object was.

Brains are plug-ins. Any model decides when to re-check, which memory a question is about, and how to answer.
Scout scores it:
- against what the robot's camera actually saw, frame by frame;
- with citations checked against the evaluator's segmentation;
- with abstentions counted separately;
- with every failure kept.

## Why it matters

Any system that remembers the physical world faces this: home and care robots, warehouse robots, AI glasses that
recall where you left things. Each answers from memory that may have gone stale. Scout turns that failure into
something you can measure, compare across models and train against. The environment that grades a robot's memory
also produces the data to build a better one: a 233-parameter neural network trained on Scout's own episodes
out-remembered a frontier LLM in homes it had never seen.

## How we use Meta's models

We go past a single API call: three Meta models run together inside a physics simulation, each doing a job the
others depend on.

| Model | Role |
| --- | --- |
| **Meta Motivo M-1** | The body: zero-shot whole-body humanoid control in cluttered houses. We blend its skill vectors into walking, turning to look and fall recovery: 35 falls, 35 recoveries. |
| **SAM 3.1** (open weights) | The eyes: open-vocabulary masks on a moving head camera, lifted into 3D with depth and stored as citable memories, each tied to the frame it came from. |
| **Llama 4 Maverick** | A brain under test: decides what to re-check, which memory a question refers to, and how to answer. |
| **Muse Spark** (brain slot) | The interface is model-agnostic. In these runs the slot is filled by Muse Spark 1.3 on Meta Model API, as a stand-in. |

## What we built during the hackathon

- **The environment:** the head RGB-D camera, the self-built depth map, SAM 3.1 detections placed in 3D, and
  episodic memory.
- **The body controller:** blended Motivo skills, turning to face a spot, and fall recovery.
- **The evaluator:** object identities, per-frame visibility, hidden interventions, and snapshot/restore branching.
- **The rest:** scorers, the brain interface, synchronised videos, and the frozen comparison on unseen homes.
- **A neural network trained on Scout's own data:** 233 parameters, trained in seconds on a CPU, that learned on
  its own which of SAM 3.1's detections to trust. Dropped into the robot's head in eight never-seen homes, it took
  first place of five brains on the memory quiz: 106/112, ahead of a frontier LLM (92) and the rule brain (93).
- **Testing:** 81 tests, written over six rounds of independent adversarial review, with recorded runs.

## Results

Final evaluation: 8 ProcTHOR homes the system had never seen × 4 brains (Muse Spark 1.3, Llama 4 Maverick, a rule
brain, a random brain). All 32 planned episodes ran; every one is counted, failures included.

- **Six clean verification pairs out of 32 episodes.** In each, the robot checked the right object and answered
  correctly in both worlds, while the memory-persistence rule was wrong in the moved world. Four are Muse Spark or
  Llama 4; two are the rule brain.
- **All 64 verification arms:** 16 correct, 10 abstained, 16 wrong target (mostly the random brain), 8 no answer in
  time, 14 not run (no eligible target). None checked the right object and gave the wrong verdict.
- **The memory-persistence rule was right 0 times when the object had moved,** for every brain.
- **The memory quiz** is graded against what each robot's camera actually saw: Muse Spark 92/112, rules 93/112,
  Llama 4 48/96 (16 invalid replies), random 16/96.
- **The body fell 35 times and got itself back up 35 times.**
- **The environment's data trains brains.** A 233-parameter neural network trained on 72 earlier episodes from 22
  other houses ran live in the same eight homes and took first place on the memory quiz: 106/112, against Muse Spark 92
  and rules 93. Full run: `docs/results/final3_student/leaderboard.md`.

Leaderboard: `docs/results/final3/leaderboard.md`. Write-up with provenance and retries:
`docs/results/environment_results.md`. Anyone can rescore every number from the public records on a laptop
(`docs/evidence.md`).

## Links

- Repo: https://github.com/tayyab415/scout
- Demo video: (to be added after recording)
- Evidence (silent demo cut, branch shot, replay, failures, test recordings, records of all 32 episodes):
  https://github.com/tayyab415/scout/releases/tag/final3-evidence

## Short post for X (draft)

> Same robot. Same memory. Same question. We moved the sofa in one world and not the other.
> Memory can't tell the two worlds apart. Looking can.
> Scout: a physics-simulated home that tests whether a robot can verify what it remembers. Meta Motivo body, SAM 3.1 eyes,
> plug in any brain. Our 233-parameter neural network, trained on Scout's own episodes, out-remembers a frontier LLM.
> github.com/tayyab415/scout
