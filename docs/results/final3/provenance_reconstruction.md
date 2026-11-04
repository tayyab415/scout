# Provenance fields filled after the run

The 24 episodes that ran on Modal predate the provenance fields in the episode manifest, so their manifests have no
`scout_commit` and no `brain_model`. The AWS episodes record both, but the 8 rules and random episodes call no model, so
their `brain_model` stays null. In `provenance.json` we filled the missing values after the run. Each filled row
lists the filled keys in `reconstructed` and carries a `reconstruction_note`. The original manifests are unchanged; the
records in the `final3-evidence` release are the originals.

## `scout_commit` (24 Modal rows): `1cfe674af54fe9d6f853014948b1a8ff968bb3d4`

- The compare job was launched at about 21:42 UTC on 3 November from `~/scout`. The launch uploaded `src/` and
  `configs/` from that working tree (`modal_launch_excerpt.txt`, header).
- `1cfe674` ("Frozen final3 plan", 21:42:56 UTC) was committed 25 seconds after the launch and adds only
  `configs/frozen_final3.json`. The code it records is the code of its parent.
- The launch log's 22 `EP` lines are the episodes reported before the log stopped at 22:17 UTC. The other 2 Modal
  episodes (rules/val_489, random/val_167) were already running and finished on Modal after the client was lost. The
  `resume` entrypoint at 22:31 UTC found 24 complete and started nothing: Modal refused the new calls because the app
  had been stopped (`modal_resume.log`). So all 24 Modal episodes come from the one launch at 21:42.
- Commits made after the launch (`a068199` onwards) cannot reach episodes that were already running: Modal ran the
  code uploaded at launch.
- Behaviour code (agent, core, perception, memory, body, world, mind, brains) is identical between `1cfe674` and the
  AWS commits `872684a` and `ba31a83`. The only later source changes are the runner moving into
  `src/scout/env/episode.py` and the scorer changes listed below.

Limit: the launch does not record whether the working tree was clean. We know of no uncommitted edits at that time,
but we cannot prove it from the log.

## `brain_model` (16 LLM rows)

The manifests record the brain's display name, which `LLMBrain` derives from the model id
(`src/scout/brains.py` at `1cfe674`, lines 23-24 and 46):

| display name in manifest | Bedrock model id |
| --- | --- |
| `terra (Muse Spark stand-in)` | `us.openai.gpt-6.1-terra` |
| `llama4-maverick` | `us.meta.llama4-maverick-17b-instruct-v1:0` |

A model passed as `llm:<id>` would show the raw id as its name, so the display name identifies the model.

## Scorer revision

The leaderboard, `rescore_from_release.txt` and the README numbers were computed with the scorer at the current
`main` (`src/scout/env/scoring.py`, last changed in `34d485d`, round-4 review: verify bound to the frozen instance).
Scoring reads only the recorded files, so rescoring with a later scorer does not need a rerun.
