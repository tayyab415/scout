# Demo narration script (2:52)

Opening line: **"Same robot. Same memory. Same question."**

`scripts/demo_cut.py` assembles the video from one recorded episode (final3, Muse Spark, house val_752) and its branch
shot (`scripts/branch_video.py`). The cut is **2:52.067** long, and exploration plays at **12x**. It is the released 2:43.067 silent cut with the
student card (`scripts/student_card.py`, 9 s) inserted at 2:17.067, before the plug-in card. Every clip is a
real recording. Every number comes from the final leaderboard (`final3`), except the student card's, which come from
the live student run (`docs/results/final3_student/leaderboard.md`). The silent cut is
`demo_final_silent_student.mp4` in the [`final3-evidence`](https://github.com/tayyab415/scout/releases/tag/final3-evidence)
release (the 2:43 cut without the card is `demo_final_silent.mp4`). The narrator records the voice over it.

The times below are the cut's real boundaries. Leave the real-time branch quiet in places; it does not need talk
throughout.

| Video time | Screen | Suggested narration |
| --- | --- | --- |
| 0:00–0:08 | Both worlds, decisive look and answer | "Same robot. Same memory. Same question. We moved the sofa in one world. Watch it check." |
| 0:08–0:11 | Branch verdict | "Remembering says yes. Looking gives two different answers." |
| 0:11–0:16 | Title | "This is Scout, an open environment for testing embodied memory." |
| 0:16–0:36 | Exploration, 12x playback | "The body is Meta Motivo, moving in physics. SAM 3.1 recognises objects through its head camera. Depth builds its floor map. Each sighting keeps its location, time and image. This exploration is shown at twelve times speed." |
| 0:36–1:43.067 | Full branch, real time | "We split one saved state into two worlds. In one, the evaluator moves the sofa without telling the robot. Both copies get the same question. The brain picks the remembered object; the controller walks back and looks." Pause. "The first copy sees the sofa and finishes. Its panel freezes while the other keeps checking." Near 1:39: "Now the second camera sees past the remembered spot. The sofa is absent there." |
| 1:43.067–1:49.067 | Branch verdict | "The memory-persistence rule assumes the sofa stayed put. The physical check distinguishes these two worlds." |
| 1:49.067–1:56.067 | Moved evidence | "Here is its earlier sighting, and the later image of that same spot." |
| 1:56.067–2:03.067 | Unchanged evidence | "In the unchanged world, the sofa is still there. Both answers match the evaluator." |
| 2:03.067–2:17.067 | Leaderboard | "Across thirty-two episodes, six pairs succeeded like this one. We report abstentions, wrong objects, timeouts and missing trials. The memory quiz is scored against what each camera saw." |
| 2:17.067–2:26.067 | Student card | "Every episode is exported, so the environment trains brains too. We trained a 233-parameter network on earlier episodes from other houses. It learned which sightings to doubt. As the robot's brain in the eight unseen homes, it remembered best of all five brains. It is not better at the physical check." |
| 2:26.067–2:35.067 | Plug-in interface | "Register your own brain, compare it on the same tasks, and score recorded episodes on a laptop." |
| 2:35.067–2:43.067 | Hackathon contributions | "We built the controller, mapping, memory, interventions, scorers, interface and the student during the hackathon, using the existing models and houses." |
| 2:43.067–2:52.067 | Disclosure | "Exploration follows a fixed rule. Depth and pose are ideal. Muse Spark 1.3 runs on Meta Model API. Targets are isolated objects." |

## Who decides what (say it this way)

- **The body's check decides "still there", "not found" or "unclear".** Fixed rules on SAM 3.1 detections and depth:
  "not found" needs the depth camera to see past the remembered spot.
- **The brain** (Muse Spark 1.3, standing in for Muse Spark) **decides** when to re-check during exploration and
  which remembered object a question is about. It also answers the memory quiz, citing frames.
- **The memory-only answer is a rule, not the brain:** whatever memory says, without looking.
- **Targets** are objects with no other object of the same kind within 2.5 m, so the question names one
  identifiable object.

## What must stay true on screen

- Never relabel Muse Spark as Muse Spark.
- The student's quiz result is offline, on memories other brains built. Say "answers the memory quiz", not "explores better".
- "Not found" never means "it detected the move".
- Use the final leaderboard's numbers, including abstentions, wrong targets and episodes that did not run.
- Never say "zero wrong verdicts in 64 arms". The outcomes are 16 correct, 10 abstained, 16 wrong target, 8 no answer
  and 14 not run. The separate `wrong` category is 0. These are prompted verification outcomes; the agent's own
  re-checks are scored separately.
