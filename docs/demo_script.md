# 3-minute demo script

Opening line: **"Can a robot verify what it remembers?"**

`scripts/demo_cut.py` assembles the video, under 3:00, from one recorded episode. Every clip is a real recording;
every number comes from the final leaderboard (`final3`). The narrator records the voice over the cut.

| Time (approx.) | On screen | Voice-over (read as written, or adapt) |
| --- | --- | --- |
| 0:00-0:10 | Evidence card, moved arm: what it remembered, and what it saw when it went back | "This robot saw a sofa here. Then, while it wasn't looking, we moved it. Watch what happens when we ask about it." |
| 0:10-0:16 | Title card | "This is Scout: an environment where a physics-simulated humanoid explores homes it has never seen." |
| 0:16-0:41 | Exploration, three panels, labelled "N× playback" | "The body is Meta Motivo, in full physics. SAM 3.1 recognises objects in its head camera. It builds its own map from depth and pins each object it sees, with the frame it saw it in." |
| 0:41-1:40 | The hidden move (on-screen caption), the question, the walk back, the look, in real time | "We branch the episode: in one copy we move the object, in the other we leave it. Then we ask: is it still where you saw it? The brain picks which memory the question is about. The body walks back, turns to face the spot, and looks." |
| 1:40-1:58 | Both evidence cards, coloured by the scorer | "Answering from memory alone, a fixed persistence rule, it says yes, it's still there. That's stale. After looking, it says it couldn't find it, because its depth camera now sees past that spot. In the copy where nothing moved, it says it just saw it there again." |
| 1:58-2:14 | Leaderboard: correct, abstain and wrong counts with denominators | "Any brain plugs in. On eight homes it had never seen: memory-quiz accuracy, answers backed by the right frame, and how often checking was correct, abstained or wrong." |
| 2:14-2:23 | Plug-in card: the interface, the compare command, the GPU-free scoring command, the URL | "Four methods make a brain. One command compares brains. Recorded episodes score on a laptop." |
| 2:23-2:34 | Disclosure card | "Exploration is a fixed rule. Depth and pose come from the simulator. Muse Spark 1.3 stands in for Muse Spark. Every failure is in the repo." |

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
- "Not found" never means "it detected the move".
- Use the final leaderboard's numbers, including abstentions, wrong targets and episodes that did not run.
