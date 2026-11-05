| brain | episodes planned | completed | target frozen | verify unchanged (correct / abstain / wrong / other) | verify moved (correct / abstain / wrong / other) | memory-persistence rule, unchanged / moved (correct) | quiz accuracy (errors) | affirmative answers with valid evidence | own checks (correct / abstain / wrong) | floor seen (median) | falls / got up |
|---|---|---|---|---|---|---|---|---|---|---|---|
| llama4 | 8 | 8 | 6 | 2 / 2 / 0 / 4 | 1 / 2 / 0 / 5 | 3 / 0 | 0.50 (16) | 54/64 | 19 / 18 / 1 | 96% | 15 / 15 |
| random | 8 | 8 | 6 | 0 / 0 / 0 / 8 | 0 / 0 / 0 / 8 | 0 / 0 | 0.17 (0) | 2/64 | 8 / 12 / 1 | 83% | 0 / 0 |
| rules | 8 | 8 | 7 | 4 / 1 / 0 / 3 | 2 / 2 / 0 / 4 | 5 / 0 | 0.83 (0) | 61/87 | 3 / 8 / 0 | 98% | 17 / 17 |
| spark | 8 | 8 | 7 | 3 / 2 / 0 / 3 | 4 / 1 / 0 / 3 | 7 / 0 | 0.82 (0) | 61/82 | 29 / 28 / 6 | 93% | 3 / 3 |

Verification outcomes per arm (physical check; 'not run' = no target frozen, crashed, or moved arm skipped):

| brain | arm | correct | abstain | wrong | wrong target | ineligible | no answer | not run |
|---|---|---|---|---|---|---|---|---|
| llama4 | unchanged | 2 | 2 | 0 | 2 | 0 | 0 | 2 |
| llama4 | moved | 1 | 2 | 0 | 0 | 0 | 2 | 3 |
| random | unchanged | 0 | 0 | 0 | 6 | 0 | 0 | 2 |
| random | moved | 0 | 0 | 0 | 4 | 0 | 1 | 3 |
| rules | unchanged | 4 | 1 | 0 | 2 | 0 | 0 | 1 |
| rules | moved | 2 | 2 | 0 | 2 | 0 | 1 | 1 |
| spark | unchanged | 3 | 2 | 0 | 0 | 0 | 2 | 1 |
| spark | moved | 4 | 1 | 0 | 0 | 0 | 2 | 1 |

Falls, recoveries and own checks are counted on the unchanged arm (it shares the exploration with the moved arm).
