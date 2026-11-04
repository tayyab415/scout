# Evidence index

Every file below is public and opens without an account. They are attached to the GitHub release
[`final3-evidence`](https://github.com/tayyab415/scout/releases/tag/final3-evidence). `SHA256SUMS` in the release
hashes each file.

## Videos

| File | What it shows |
| --- | --- |
| [demo_final_silent.mp4](https://github.com/tayyab415/scout/releases/download/final3-evidence/demo_final_silent.mp4) | The 2:43 demo cut before narration (`scripts/demo_cut.py`). |
| [demo_final_silent_student.mp4](https://github.com/tayyab415/scout/releases/download/final3-evidence/demo_final_silent_student.mp4) | The 2:52 cut: the 2:43 cut with the student card (`scripts/student_card.py`, 9 s) inserted at 2:17.067. |
| [branch_terra_752.mp4](https://github.com/tayyab415/scout/releases/download/final3-evidence/branch_terra_752.mp4) | The branch shot: one saved state split into two worlds, the sofa moved in one (`scripts/branch_video.py`). |
| [social_20s.mp4](https://github.com/tayyab415/scout/releases/download/final3-evidence/social_20s.mp4) | 20 s clip for posting. |

## Failures

| File | What it shows |
| --- | --- |
| [failure_no_target_llama4_val_167.mp4](https://github.com/tayyab415/scout/releases/download/final3-evidence/failure_no_target_llama4_val_167.mp4) | Llama 4, house val_167: exploration ends with no object eligible to move, so neither arm runs. Counted as "not run". |
| [failure_wrong_target_random_val_893_moved.mp4](https://github.com/tayyab415/scout/releases/download/final3-evidence/failure_wrong_target_random_val_893_moved.mp4) | Random brain, house val_893, moved arm: asked about the sofa, it checks the sink. Counted as "wrong target". |

## Tests and checks

| File | What it shows |
| --- | --- |
| [tests_73_short_suite.mp4](https://github.com/tayyab415/scout/releases/download/final3-evidence/tests_73_short_suite.mp4) | Terminal recording of 73 of the 74 tests passing. |
| [tests_render_end_to_end.mp4](https://github.com/tayyab415/scout/releases/download/final3-evidence/tests_render_end_to_end.mp4) | The remaining test: renders the video panels end to end from a recorded episode (about 3 min). |
| [replay_check.mp4](https://github.com/tayyab415/scout/releases/download/final3-evidence/replay_check.mp4) | Branch replay on an AWS A10G: 120 steps from one snapshot, run twice; maximum qpos difference 5.8e-12. Saved output: `docs/results/evidence/replay_check.txt`. |

The test recordings were made by the independent reviewer on 4 November, against the commit in this repo at that time.

## Records: reproduce the scores

[final3-records.tar.gz](https://github.com/tayyab415/scout/releases/download/final3-evidence/final3-records.tar.gz)
(5.6 MB) holds the manifest, log, memory and evaluator file of each arm of all 32 planned episodes.
[record-hashes.json](https://github.com/tayyab415/scout/releases/download/final3-evidence/record-hashes.json) hashes
each record.

```bash
curl -LO https://github.com/tayyab415/scout/releases/download/final3-evidence/final3-records.tar.gz
mkdir records && tar xzf final3-records.tar.gz -C records
python scripts/rescore_records.py records                                  # every outcome, all 64 arms, plus quiz denominators
python scripts/score_episode.py records/terra_procthor-10k-val_val_752_s0  # one episode
```

Expected output is committed in `docs/results/final3/rescore_from_release.txt`:

- all arms: 16 correct, 10 abstain, 0 wrong, 16 wrong target, 0 ineligible, 8 no answer, 14 not run;
- memory quiz (questions / correct / errors): terra 112 / 92 / 0, rules 112 / 93 / 0, llama4 96 / 48 / 16,
  random 96 / 16 / 0.

Floor coverage needs the house layouts, so the script skips it. Everything else in the leaderboard is recomputed.
Provenance per episode: `docs/results/final3/provenance.json` (see `provenance_reconstruction.md` for the fields
filled after the run).

## Records: the live student run

[final3_student-records.tar.gz](https://github.com/tayyab415/scout/releases/download/final3-evidence/final3_student-records.tar.gz)
(1.5 MB) holds the records of the 8 student episodes run after `final3` was frozen
(`docs/results/final3_student/leaderboard.md`). `SHA256SUMS_student` in the release covers this archive and the
2:52 cut.

```bash
curl -LO https://github.com/tayyab415/scout/releases/download/final3-evidence/final3_student-records.tar.gz
mkdir records_student && tar xzf final3_student-records.tar.gz -C records_student
python scripts/rescore_records.py records_student frozen_final3_student.json
```

Expected output: `docs/results/final3_student/rescore.txt`.
