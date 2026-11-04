# 3-minute demo script

Opening line: **"Can a robot verify what it remembers?"**

`scripts/demo_cut.py` assembles the video, under 3:00, from one recorded episode and its branch shot
(`scripts/branch_video.py`). Every clip is a real recording;
every number comes from the final leaderboard (`final3`). The narrator records the voice over the cut.

| Time (approx.) | On screen | Voice-over (read as written, or adapt) |
| --- | --- | --- |
| 0:00-0:14 | Hook: the last 10 s of the split (both worlds answering), then the graded verdict card | "Same robot. Same memory. Same question. In one world we moved the sofa; in the other we didn't. Memory can't tell the two worlds apart. Looking can." |
| 0:14-0:20 | Title card: "A physics-simulated home that tests whether a robot's memory is still true." | "This is Scout." |
| 0:20-0:45 | Exploration, three panels, labelled "10x playback" | "A Meta Motivo humanoid, in full physics, explores a home it has never seen. SAM 3.1 recognises objects in its head camera; depth builds its own map; every object is pinned where it was seen, with the frame it was seen in." |
| 0:45-1:50 | The split, in real time: World A and World B side by side. The world that finishes first freezes with a label. World B reaches the empty spot around 303 s. | "Here we branch the episode from one physics snapshot. In World B we quietly move the sofa. Both robots get the same question. The brain picks which memory it is about. The body walks back, turns to face the spot, and looks. In World A the sofa is there. In World B its depth camera sees straight past where the sofa was." |
| 1:50-1:58 | Verdict card | "From memory alone, a fixed persistence rule, the answer is 'still there' in both worlds, so it's wrong in one. Looking gets both right." |
| 1:58-2:16 | Evidence cards (moved, unchanged), coloured by the scorer | "Every answer cites the frame it is based on." |
| 2:16-2:32 | Leaderboard | "Any brain plugs in. On eight homes it had never seen: correct, abstained and wrong, with denominators, plus the memory quiz graded against what each camera actually saw." |
| 2:32-2:42 | Plug-in card | "Register a brain in three lines, run it, and score recorded episodes on a laptop." |
| 2:42-2:51 | Built during the hackathon | "Everything on this card was built this week; the models and houses were used as is." |
| 2:51-3:00 | Disclosure card | "Exploration is a fixed rule; depth and pose come from the simulator; the brain here is GPT-6.1 Terra standing in for Muse Spark. Every failure is in the repo." |

## Who decides what (say it this way)

- **The body's check decides "still there", "not found" or "unclear".** Fixed rules on SAM 3.1 detections and depth:
  "not found" needs the depth camera to see past the remembered spot.
- **The brain** (GPT-6.1 Terra, standing in for Muse Spark) **decides** when to re-check during exploration and
  which remembered object a question is about. It also answers the memory quiz, citing frames.
- **The memory-only answer is a rule, not the brain:** whatever memory says, without looking.
- **Targets** are objects with no other object of the same kind within 2.5 m, so the question names one
  identifiable object.

## What must stay true on screen

- Never relabel Terra as Muse Spark.
- "Not found" never means "it detected the move".
- Use the final leaderboard's numbers, including abstentions, wrong targets and episodes that did not run.
