# The student: a 233-parameter brain trained on Scout's own episodes

Added on 4 November, after the final evaluation was frozen. The frozen leaderboard
(`final3/leaderboard.md`) is unchanged; this is a separate, offline result.

## What it is

Scout exports every episode: the memory table the brain saw, the head frames, and the evaluator's record of what
was really in view. That is training data. The student is the smallest brain we could train on it: a two-layer
perceptron in plain numpy (27 inputs, 8 tanh units, 1 output, 233 parameters, `src/scout/student/trust.py`) that
reads one row of the memory table and outputs the probability that the remembered object is really there.

- **Input:** only the memory table every brain receives. Per remembered object: times seen, best SAM 3.1
  confidence, time span, time since last seen, misses, retracted/gone verdicts, the confident flag, how many other
  objects share its label, how many objects are remembered, when in the run it was first seen, and the category.
  No positions, no frames, nothing from the evaluator.
- **Label:** the memory quiz's own criterion. An object is real if its category was in view of the head camera
  (at least 0.5% of a captured frame) and its remembered position is within 1.5 m of a visible instance.
- **Training data:** 72 recorded episodes from the development runs (tags cmp1, dev3, dev4, dev5, final, final2,
  smoke, smoke2), one arm per episode, 22 houses, 1985 remembered objects of which 1372 (69%) were real. None of
  the eight final-evaluation houses is in the training set; the test in `tests/test_student.py` checks that
  against the weights file.
- **Training:** Adam on the weighted cross-entropy, 400 epochs, a few seconds on a laptop CPU. Four of the 22 houses
  are held out to pick the stopping epoch (epoch 43). Held-out accuracy 0.781 against a majority rate of 0.680
  (`training_log.txt`, `training.json`).
- **Hidden width and threshold** (8 units, say yes at probability 0.4 or more) were picked by leave-one-house-out
  cross-validation on the training runs only, before the final-evaluation records were scored with that setting
  (`cv.txt`, `scripts/student_cv.py`).

What the network learned, roughly (change in its output when one standardised feature rises by one standard
deviation, on the held-out houses): SAM confidence +0.15; dining tables +0.11 and paintings +0.06 are trusted
more; bathtubs −0.05, chairs −0.05 and televisions −0.03 less; each miss −0.04. It has learned SAM 3.1's
false-positive profile in these houses.

## How it answers

- **Quiz** ("Did you see a sofa? If so, where was it?"): take the remembered sofas, trust the most probable one;
  say yes with its location if the probability is at least 0.4, otherwise no. Categories are matched as whole
  words, so an armchair question never matches a chair.
- **Verification question:** the question is about the most trusted object of that category.
- **Re-checks after exploration:** look again at the object it is least sure about (probability nearest 0.5),
  skipping the object it checked last and any it has checked twice, since the agent refuses those.

All three were run live in the simulator on the eight final-evaluation houses (below).

## Result: the memory quiz on the final-evaluation records, answered offline

`scripts/quiz_offline.py` gives a brain the saved memory table of each final-evaluation episode (the same text the
original brain saw at quiz time) and scores its 16 answers with the environment's scorer, against the evaluator's
visibility record. 26 of the 32 episodes reached the quiz (416 questions). The memories were built during the
original runs, by whichever brain ran them, so this measures answering only, not exploring.

| answers by | questions | correct | affirmative answers with a valid citation |
|---|---:|---:|---:|
| the brains that ran the episodes (Muse Spark, Llama 4, rules, random; recorded answers) | 416 | 249 | 178/297 |
| rule brain (as in the leaderboard) | 416 | 333 | 224/329 |
| rule brain with whole-word category matching | 416 | 352 | 224/310 |
| student, network ranks only, every remembered category answered yes | 416 | 355 | 228/310 |
| **student** | 416 | **370** | 218/265 |
| ceiling of any memory-only answer (a real object of the category is in memory, or none was in view) | 416 | 405 | |

By whose memories the student answered from:

| memories built by | questions | the brain that ran it | rule brain | student |
|---|---:|---:|---:|---:|
| Muse Spark 1.3 (Meta Model API) | 112 | 92 | 84 | 96 |
| rules | 112 | 93 | 93 | 103 |
| Llama 4 Maverick | 96 | 48 (16 invalid replies) | 81 | 88 |
| random | 96 | 16 | 75 | 83 |

Per-episode numbers: `quiz_final3.json`. The student's remaining errors on the 416 questions: 27 real objects it did
not trust, 13 spurious objects it trusted, 7 right category but wrong object, 6 categories that were in view but
never recognised (no brain can answer those from memory).

## Result: live, in the simulator, on the final-evaluation houses

The student then ran as the brain in all 8 final-evaluation houses, same seed and budgets
(`final3_student/leaderboard.md`). Each brain answers from the memory its own exploration built.

| live, same 8 houses | memory quiz | affirmative answers with valid evidence | verification arms correct | clean pairs |
|---|---:|---:|---:|---:|
| **student** | **106/112** | **68/80** | 5/16 | 1 |
| Muse Spark 1.3 (Meta Model API) | 92/112 | 61/82 | 7/16 | 3 |
| rules | 93/112 | 61/87 | 6/16 | 2 |
| Llama 4 Maverick | 48/96 | 54/64 | 3/16 | 1 |
| random | 16/96 | 2/64 | 0/16 | 0 |

It is the best of the five at the memory quiz and the worst of the three serious brains at physical verification.
One of its failures is its own: in val_489 it ignored the time named in the question and checked the wrong dresser.

## Read this carefully

- The whole-word matching is worth 19 of the 37 questions gained over the rule brain. The network is worth 18
  over the rule brain with the same matching, and 15 over itself with the threshold removed.
- The final-evaluation records were scored twice. The first setting (16 hidden units, threshold 0.5, the rule
  brain's substring matching) scored 343. The setting above was then chosen by cross-validation on the training
  runs, with whole-word matching added, and scored 370. Both numbers are reported.
- The offline score is on memories other brains built. The live score is on memories the student built, so its
  denominators match the other brains but the memories do not.
- The live run was launched after `final3` was frozen and published; it is reported separately and the frozen
  leaderboard is unchanged. The gain is on 112 live questions from 7 houses (one house had no quiz): a
  demonstration that the environment's exported episodes train a brain that transfers to unseen houses.

## Card

`assets/media/student_card.png` (`scripts/student_card.py`) shows the network's trust on the sample episode's memory
(final3, Muse Spark, val_752): its most and least trusted objects, against the evaluator's record. One of the shown rows
is a mistake: a real television seen once, trusted at 0.11.

## Reproduce

```bash
# records: the public final3 archive (docs/evidence.md) plus the development runs
python scripts/train_student.py --records <dev run dirs...> --exclude-houses configs/frozen_final3.json
python scripts/student_cv.py    --records <dev run dirs...> --exclude-houses configs/frozen_final3.json
python scripts/quiz_offline.py  records            # the table above, from the public final3 records
```

The shipped weights are `assets/brains/student_trust.npz` (21 KB, with the training houses and episodes recorded
inside). The development-run records are on the team's S3 volume (`scripts/modal_to_s3.py`); the final3 records
are public.
