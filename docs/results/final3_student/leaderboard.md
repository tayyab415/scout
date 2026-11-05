# Live student run on the final-evaluation houses (`final3_student`)

Run on 4 November 2026, 16:40 to 20:17 UTC, after the final evaluation (`final3`) was frozen and published. Same 8
houses, same seed (0), same 240 s exploration budget, same body, eyes, check rules and scorer as `final3`. The only
change is the brain: `student`, the 233-parameter trust network (`docs/results/student.md`). The `final3` rows below
are the frozen leaderboard, unchanged.

Host: one AWS g5.xlarge (A10G, 24 GB), two episodes at a time, `scripts/run_local.py --plan` with
`configs/frozen_final3_student.json`. All 8 planned episodes completed; none crashed (`run_local.jsonl`).

| brain | episodes | target frozen | memory quiz (errors) | affirmative answers with valid evidence | verification arms correct (of 16) | clean pairs | wrong-target arms | own checks (correct / abstain / wrong) | falls / got up |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| **student (233 parameters, live)** | 8 | 7 | **106/112** (0) | **68/80** | 5 | 1 | 2 | 26 / 25 / 3 | 1 / 1 |
| muse-spark-1.3 | 8 | 7 | 92/112 (0) | 61/82 | 7 | 3 | 0 | 29 / 28 / 6 | 3 / 3 |
| rules | 8 | 7 | 93/112 (0) | 61/87 | 6 | 2 | 4 | 3 / 8 / 0 | 17 / 17 |
| llama4 | 8 | 6 | 48/96 (16) | 54/64 | 3 | 1 | 2 | 19 / 18 / 1 | 15 / 15 |
| random | 8 | 6 | 16/96 (0) | 2/64 | 0 | 0 | 16 | 8 / 12 / 1 | 0 / 0 |

Floor coverage is not recomputed here (the house layouts are not in the records). The memory-persistence rule was
wrong in every moved arm for the student too (0 correct), as for every brain.

Every verification arm of the student, from the records (`rescore.txt`):

| arm | correct | abstain | wrong | wrong target | ineligible | no answer | not run |
|---|---:|---:|---:|---:|---:|---:|---:|
| unchanged | 4 | 2 | 0 | 1 | 0 | 0 | 1 |
| moved | 1 | 3 | 0 | 1 | 0 | 1 | 2 |

Per house, all five brains: `per_house.md`.

## What it shows

- **Memory quiz: best of the five brains.** 106 of 112, against 92 for Muse Spark and 93 for rules, on the same houses.
  Each brain answers from the memory its own exploration built, so the memories differ between brains.
- **Unchanged world: 4 of 6 checked arms correct, none wrong.** Moved world: 1 correct. The moved-world weakness is
  mostly the body's check, shared by every brain: across all recorded runs, the failed moved-world checks on the
  right target had no usable view of the remembered spot (0 looks past it, 0 looks at something there).
- **Verification pairs: 1** (val_752, the demo house; Muse Spark also succeeds there). Muse Spark has 3, rules 2. The student is
  not better at the physical check.

## Failures

- **val_489, wrong target in both arms.** The house has four dressers. The question named "the dresser you saw at
  158 s"; the student's `resolve` ignores the time and picks its most trusted dresser, a different one. Its look at
  that dresser was right. The fix (prefer the remembered object whose sightings are closest to the time in the
  question) is **not** applied, so the code in the repo is the code that ran.
- **val_167, not run.** 13 objects remembered; none eligible to move. Llama 4 had the same outcome in `final3`.
- **val_300, moved arm not run.** The evaluator found no legal destination for the dresser.
- **val_893, val_322, val_359, val_330:** abstentions or no answer in time, as listed above.

## Provenance

- The box ran a copy of the working tree without git history, so the manifests have no commit. Checksums of the
  files that decide behaviour, taken after the run and matching the copy on the box:
  `src/scout/student/trust.py` 401af5fc, `assets/brains/student_trust.npz` 3c6860f3, `src/scout/env/agent.py`
  53478690, `src/scout/env/episode.py` 7517a595, `src/scout/brains.py` 8ee9ef1f (MD5, first 8 hex).
- One change was made to the student before any episode ran: `decide` skips objects the agent refuses to re-check
  (the one just checked, any checked twice). It was pushed to the box at 16:38 UTC; the run started at 16:40.
- Records: manifest, log, memory and evaluator of every arm,
  [final3_student-records.tar.gz](https://github.com/tayyab415/scout/releases/download/final3-evidence/final3_student-records.tar.gz)
  (1.5 MB, SHA-256 in `records.sha256`). Rescore: `python scripts/rescore_records.py <dir> frozen_final3_student.json`.
