# 3-minute demo script

Opening line: **"Can a robot tell what it knows from what it needs to check? Plug in a brain and find out."**

All clips are rendered from real episodes. Every number on screen comes from `runs/env/final/leaderboard.md`.

| Time | On screen | Voice-over (read as written, or adapt) | Source file |
| --- | --- | --- | --- |
| 0:00-0:15 | Observer view of the humanoid walking into an unseen house; title card | "This is Scout. A physics-simulated human, Meta Motivo, is dropped into a home it has never seen. No map, no list of objects. It only has a head camera." | `final/<episode>/unchanged.mp4`, first 15 s |
| 0:15-0:45 | Three panels: body, head camera with SAM 3.1 boxes, the map it builds itself | "SAM 3.1 tells it what it is looking at. Depth from the same camera builds its own map. Everything it remembers is pinned where it saw it, with the frame it saw it in." | same, 30-60 s |
| 0:45-1:05 | A fall and recovery, if the final runs include one (otherwise skip) | "It can fall. When it does, it gets itself back up." | episode with a `got_up` event |
| 1:05-1:20 | Caption "[evaluator, hidden from the agent] moved the television" | "Then we change the world while it isn't looking. In one copy of the episode we move an object. In the other copy, branched from the exact same moment, we leave it." | `moved.mp4` at the move time |
| 1:20-1:50 | The question appears. The body walks back and looks | "Then we ask: is the television you saw earlier still where it was?" | `moved.mp4` around the question |
| 1:50-2:15 | Evidence cards: memory-only answer vs checked answer, moved and unchanged | "Answering from memory, the brain says yes, it's there. That's a stale answer. Going back to check, it says it couldn't find it where it last saw it, because its depth camera now sees past that spot. In the unchanged copy it says it just saw it there again. A brain that always says 'gone' can't score well." | evidence cards at the end of `moved.mp4` and `unchanged.mp4` |
| 2:15-2:45 | Leaderboard table | "Any brain plugs in. On eight homes it had never seen, here is how GPT-6.1 Terra (our stand-in for Muse Spark), Llama 4, a rule baseline and random compare: on checking, on memory questions with cited frames, and on staying up." | `runs/env/final/leaderboard.md` |
| 2:45-3:00 | Disclosure card | "Exploration uses a fixed rule. Depth and pose come from the simulator. Every failure is in the repo. Body: Meta Motivo. Eyes: SAM 3.1. Brain: plug in yours." | README |

## What must stay true on screen

- Say "GPT-6.1 Terra, standing in for Muse Spark". Do not relabel Terra results as Muse Spark.
- Say exploration is a fixed rule. The brain decides re-checks and answers.
- "Not found" means the spot is now empty. Never say it "detected that the object was moved".
- Use the numbers from the final leaderboard, including the abstentions and errors.
