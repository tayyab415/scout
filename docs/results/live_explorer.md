# Live explorer: physics body, head RGB-D, SAM 3.1 eyes, episodic memory, cited answers

Objective (agreed with the judge thread, 3 Nov 2026): a physics humanoid explores an unfamiliar home, remembers
what it saw, goes back when unsure, and answers questions with footage from its own experience.

## What the body knows, and what it does not

| Agent side | Evaluator only |
| --- | --- |
| Head RGB-D camera (75° FOV, tilted down 20°) | House layout, room labels, object list |
| Its own position (perfect odometry from the simulator, disclosed) | Object positions (used only for scoring) |
| SAM 3.1 detections on head RGB, placed in 3D with head depth | The evaluator's object moves (disclosed in the video) |
| A floor map it builds from depth | |

## Eyes: SAM 3.1 on house furniture (`scripts/modal_eyes.py`)

- **Setup:** 24 eye-height views in 4 held-out houses, 25 categories. Ground truth is the simulator segmentation,
  and an object counts as visible if it covers at least 0.5% of the image.
- **SAM 3.1** (multiplex checkpoint, one-frame clips):
  - recall 83% (34/41);
  - precision 60%, with 12 of the 23 false alarms being "desk", likely a gap in the answer-key naming;
  - mask IoU 0.89 when found;
  - 2.5 s per image.
- **SAM 3 image model:** recall 85%, precision 56%, 0.76 s per image.
- **Small objects** (cup, bowl, apple, book) never covered 0.5% of a standing-height view. The demo uses larger
  objects.

## Gate: one continuous minute, house val_935 (`runs/live/gate60`)

- **Run:** 60 s, no falls, 16 waypoints, 1 stuck event recovered by replanning.
- **Memory:** 23 objects; 20 of them within 1.5 m of a real object of that category. The three that fail: house
  plant and laptop have no answer-key entry (the plant is visible in its frames), and one "toilet" is 7 m off.
- **Questions** (Muse Spark 1.3 as stand-in for Muse Spark): each answer cites the right frames, and for an unseen
  bathtub it answered "I did not see a bathtub."

## Robustness batch 1 (before the bed and window fixes)

5 preselected held-out houses × 10 min. Every run is kept.

| House | Survived | Rooms visited | Floor seen | Walked |
| --- | --- | --- | --- | --- |
| val_632 | fell at 29 s (walked into a bed) | 1/10 | 16% | 9 m |
| val_650 | 600 s | 2/8 | 44% | 65 m |
| val_997 | 600 s | 1/10 | 21% | 12 m |
| val_706 | 600 s | 1/4 | 26% | 2 m |
| val_329 | 600 s | 4/4 | 92% | 31 m |

Causes found from the maps and video:

- **Ray carving marked floor under low furniture as walkable**, so it walked into a bed. Fix: the head is tilted
  down 20°, and only the first 60% of each ray is carved.
- **The physics engine's endless floor plane showed through windows.** The body mapped a "yard" it could never
  reach and chased frontiers there. Fix: the floor plane is hidden from cameras; it still collides.
- **It switched between two frontiers every 2 s.** Fix: it commits to a frontier until it gets there, or makes no
  progress for 6 s, in which case that spot is banned.

## The mind (Muse Spark 1.3): checks from memory

In run `mind180b` (val_935, 180 s), 8 checks were started by the mind. Scored against truth:

- **Correct:**
  - a "bed" seen once, 4.6 m from any real bed, was retracted after a second look;
  - a mislocated "chair" track was marked gone.
- **Wrong:** two "painting gone" verdicts, where the painting was there. SAM missed it at close range and at an
  angle, and matching by estimated 3D position failed.
- **Fixes:**
  - check from a viewpoint 2.5 m away, facing the object;
  - a hit means a same-label box covering the remembered spot;
  - "missing" needs 3 misses in a row, 1-3.5 m away, near the centre of view.
- **The moved TV:** the evaluator moved it at 50 s, but the body never returned to that room. The evaluator now
  moves only objects that are out of view and more than 6 m away. After exploring, the mind re-checks the objects
  it has not seen for longest.
