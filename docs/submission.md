# Submission draft (Cerebral Valley form)

**Project name:** Scout

**Tagline:** A physics-simulated home that tests whether a robot's memory is still true.

**Track / problem statement:** Push the Frontier (robotics: give a machine a new capability, and a way to measure it).

## Description

Home robots answer questions from memory: "Is the sofa still in the living room?" But the world changes when
they are not looking. Scout is an open simulation environment that tests whether a robot can tell what it knows from
what it needs to check.

- **The setup.** A Meta Motivo humanoid explores a house it has never seen, in full physics, through a head camera.
  SAM 3.1 recognises objects. Depth builds the robot's own map. Every sighting is remembered with the frame it came
  from.
- **The two worlds.** Then the episode branches from one physics snapshot into two worlds. In one, nothing changes.
  In the other, we quietly move an object. Both copies of the robot get the identical question.
- **What it shows.** Answering from memory says "still there" in both worlds, so it is wrong in one of them. The
  robot that walks back and looks gets both right: its depth camera sees past the spot where the object was.

Brains are plug-ins. Any model decides when to re-check, which memory a question is about, and how to answer.
Scout scores it:
- against what the robot's camera actually saw, frame by frame;
- with citations checked against the evaluator's segmentation;
- with abstentions counted separately;
- with every failure kept.

## How we use Meta's models

| Model | Role |
| --- | --- |
| **Meta Motivo M-1** | The body: whole-body physics control. It walks, turns to look, falls and gets itself back up. |
| **SAM 3.1** (open weights) | The eyes: open-vocabulary masks on the head camera, placed in 3D with depth. |
| **Muse Spark** (brain slot) | In these runs the brain is GPT-6.1 Terra on AWS Bedrock, as a stand-in. Llama 4 Maverick is also compared. |

## What we built during the hackathon

- **The environment:** the head RGB-D camera, the self-built depth map, SAM 3.1 detections placed in 3D, and
  episodic memory.
- **The body controller:** blended Motivo skills, turning to face a spot, and fall recovery.
- **The evaluator:** object identities, per-frame visibility, hidden interventions, and snapshot/restore branching.
- **The rest:** scorers, the brain interface, synchronised videos, and the frozen comparison on unseen homes.
- **Testing:** 70+ tests, including five rounds of independent adversarial review.

## Results

See `runs/env/final3/leaderboard.md` and `docs/results/environment_results.md` in the repository: 8 never-seen
houses × 4 brains, every planned episode listed, failures included. The final numbers go here after the final run.

## Links

- Repo: https://github.com/tayyab415/scout
- Demo video: (to be added after recording)

## Short post for X (draft)

> Same robot. Same memory. Same question. We moved the sofa in one world and not the other.
> Memory can't tell the two worlds apart. Looking can.
> Scout: a physics-simulated home that tests whether a robot's memory is still true. Meta Motivo body, SAM 3.1 eyes,
> plug in any brain. github.com/tayyab415/scout
