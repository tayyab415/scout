# Bridge test: V-JEPA 2 video features → Motivo skill vector

Run 3 Nov 2026 (`scripts/modal_bridge.py`, 26 min on one L40S). 3,000 training clips, 300 held-out clips. Each clip
is 4.3 s of Motivo running one skill vector z from the same standing pose, filmed from a random camera.

## Behaviours used

- 1,274 random z.
- 1,009 single-pose goal z from the motion buffer. The buffer rows are not in time order, so 8-step motion windows
  were not possible.
- 1,017 mixes of two named skills (walks, turns).
- 8% of clips end in a fall.

## Results on held-out clips

The motion error is the mean distance between matching body parts of the original and the copy, in metres, over
4.3 s. Identification asks whether a copy's motion is closer to its own original than to the other 299 (chance 0.3%).

| What drives the copy | Motion error, mean | Median | Identified (top 1) | Mean rank (of 300) |
| --- | --- | --- | --- | --- |
| True z plus action noise (ceiling) | 0.11 | 0.07 | 92.7% | 0.5 |
| V-JEPA 2 on silhouettes → linear map | 1.07 | 0.52 | 7.0% | 77 |
| V-JEPA 2 on normal renders → linear map | 1.09 | 0.54 | 6.7% | 79 |
| Nearest training clip in V-JEPA space | 1.18 | 0.57 | 9.7% | 86 |
| Raw silhouette pixels → linear map | 1.50 | 0.63 | 2.7% | 130 |
| Average z (one behaviour for everything) | 1.06 | 0.49 | 0.3% | 150 |
| Random training z | 1.82 | 0.99 | 0.3% | 152 |

Motion error by behaviour type (mean, metres):

| Type | n | V-JEPA silhouettes | Average z |
| --- | --- | --- | --- |
| Skill mixes (walking, turning) | 97 | 1.31 | 1.78 |
| Pose goals | 84 | 0.90 | 1.04 |
| Random z | 119 | 1.00 | 0.50 |

## What this shows

- **The video carries real information about the behaviour.** Copies are identified 23 times more often than
  chance, and the mean rank halves.
- **V-JEPA 2 beats raw pixels clearly:** rank 77 vs 130, error 1.07 vs 1.50.
- **Silhouettes lose nothing compared with normal renders.** So a SAM 3.1 silhouette of a real person is a
  workable input.
- **The copies are weak.**
  - Overall error equals the "do the average thing" baseline.
  - For moving behaviours the copy is 26% better than average.
  - For random z the copy is worse. Most random z just stand, and the copy invents motion.
  - It is far from the ceiling: 7% identified vs 93%.
- **Looking at the clips** (`runs/bridge/sheet.jpg`, `runs/bridge/bridge.mp4`):
  - Six of eight showcase originals barely move, so they cannot tell the methods apart.
  - Clip 3006: the original lunges in a crouch, and the copy also crouches and lunges. Its position error is high
    (1.49 m) because it ends in a different place.
  - Clip 3007: the original falls to the floor and the copy goes to the floor. The average baseline stands in both.

## Problems with this test

- The behaviour set is dull. Most random and pose-goal z stand or fidget.
- The error includes where the body ends up. A copy that does the right movement in the wrong place scores worse
  than standing still.
- A linear map to z regresses toward the mean, and many z give the same behaviour.
