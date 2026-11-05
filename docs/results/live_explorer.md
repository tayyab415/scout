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
- **Questions** (Muse Spark 1.3 on Meta Model API): each answer cites the right frames, and for an unseen
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

## Robustness batch 2 (fixes in, mind on, the same 5 houses)

- 5/5 ran the full 600 s. There were 2 falls; both times it got up on its own (holding Motivo's stand skill).
- Floor seen: 44 / 99 / 72 / 49 / 100%, up from 16 / 44 / 21 / 26 / 92%. Rooms visited: 3/10, 6/8, 4/10, 2/4, 4/4.
- These houses informed the fixes, so this is regression evidence, not generalisation.

## 20 untouched held-out houses × 10 min (mind off) — generalisation

| Measure | Result |
| --- | --- |
| Ran the full 600 s | 20/20 |
| Falls, got up on its own | 6, 6 |
| Floor seen | median 95%, mean 79%, range 12-100% |
| Rooms visited | 61/96; all rooms in 5/20 houses |
| Confident objects (3+ sightings) within 1.5 m of a real one of that type | 361/446 (81%) |
| Tentative objects (1-2 sightings) | 89/194 (46%) |

## Self-triggered checks by the mind (batch 2, all 5 houses)

- 123 checks were started and 121 finished. 39 reached a verdict; 25 of those were correct (64% of decided, 21% of
  finished). 14 real objects were wrongly declared gone.
- The moved object was noticed spontaneously in 0/3 runs. The body did not go back to the old place.

## Question-triggered verification (house val_935, one trial each)

Asked at 300 s: "Is the television still where you last saw it?" The body chooses the remembered object, the
route and the viewpoint itself.

- **Unchanged:** "I remembered seeing the television there before, and I just saw it there again." Correct.
- **Moved** (by the evaluator to the bathroom at 150 s): "I couldn't find it where I last saw it." Correct.
  Then, asked "Then find it.": "I walked through the places I know and did not see one." The search failed.
- **An earlier unchanged trial** said "couldn't get a clear look". The check ignored detections more than 5 m
  away; fixed.

## Frontier experiment: does V-JEPA 2 see failure coming? (`scripts/modal_failure.py`)

- **Data:** the 20 untouched runs, one head-camera look every 0.5 s.
- **Label:** a stuck or fall event in the next 4 s (base rate 3.7%).
- **Split:** 15 houses train, 5 test. Logistic regression; the regularisation strength was chosen on training
  houses only.

| Features | Test AUC | Test average precision |
| --- | --- | --- |
| Recent motion (speed over 1 s and 2 s, turn rate, command, head height) | 0.956 | 0.39 |
| Frozen V-JEPA 2 ViT-L on the last two head frames | 0.507 | 0.07 |
| Both | 0.559 | 0.11 |

- **Result: negative.** Frozen V-JEPA 2 features did not predict this body's failures, and adding them to motion
  made predictions worse.
- **Caveat:** "stuck" is defined by lack of progress, so the motion baseline partly sees the label coming. Even
  so, V-JEPA alone is at chance.
