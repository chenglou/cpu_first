# Relaunch of the five cut runs (SCALE.md §6), 2026-10-24 14:43–14:45 PDT

Each launch came after `integration/ablation_s3/spend.py 3750` printed LAUNCH OK ($3,004.28 billed this month at
21:40 UTC). The commands are the five in §6 verbatim, with SCALE_CPU=16 SCALE_MEM=65536, from `cpu_first`. No code
under `library/`, `bar3/*.py`, `bar3/breadth/` or `bar3/scale/` had changed since the original launches.

| app | function call | runs | store replayed |
|---|---|---|---|
| ap-SHsQUnYAtcg9UncO2DSK6d | fc-01M5QXX45BBAS0A4MKCCAWRKJR | seq_lp_clean_all3_targets_L1, seq_ct2a2_clean_all3_targets_L1, seq_ct2a2_clean_new_targets_L1 | store_seq_lp_clean_all3_targets_L1.jsonl (18.9 MiB) |
| ap-WDpWaqTrimEyrsY22qBtWy | fc-01M5QXY5PHQ7BPDQBSKWDX3RW9 | seq_ct2a2_clean_targets_L1 (--share) | store_seq_ct2a2_clean_targets_L1.jsonl (7.6 MiB) |
| ap-ZScpEOAvKYFffh56hZlJRd | fc-01M5QXYSFDBX2W3MJYPN1CF403 | branch_ct2a2_clean_new_targets_L1_after_cube_fib_L1 (--share) | store_branch_ct2a2_clean_new_targets_L1_after_cube_fib_L1.jsonl (251 KiB) |
| ap-FuPTikqMsqsSH7mrhKcVWc | fc-01M5QXZPRQMJSBHZPETX1VC02V | branch_ct2a2_clean_targets_L1_after_cube_fib_L1 (--share) | store_branch_ct2a2_clean_targets_L1_after_cube_fib_L1.jsonl (847 KiB) |
| ap-tOeQGZ8hBRvHghZbdpz5xi | fc-01M5QY09GX0B95J3F8F0BV1AXC | lp_clean_new_k26_fib_N1 (--share) | store_lp_clean_new_k26_fib_N1.jsonl (104 KiB) |

The first round each launch printed matched its record (the replay is exact up to the cut). The container logs are
in `app_<app id>.log` beside this file, written by `modal app logs <app id>` until each app stopped.

## How it ended (2026-10-25)

All five apps stopped on their own, each after logging `group done`; a log watcher reported failure signs only after
each log's last preemption, and found none.

| app | stopped (PDT) | preemptions | outcome |
|---|---|---|---|
| ap-SHsQUnYAtcg9UncO2DSK6d | Oct 25 02:51 | 3 | hardest first: 10·x² + 1 no in 5 rounds, x³ round 0, Montgomery's round 3, Fibonacci round 1; both anchored-pairs sequences: Montgomery's and Fibonacci no in 5 rounds |
| ap-WDpWaqTrimEyrsY22qBtWy | Oct 24 17:28 | 2 | Montgomery's and Fibonacci no in 5 rounds |
| ap-ZScpEOAvKYFffh56hZlJRd | Oct 24 15:27 | 0 | Fibonacci no in 5 rounds |
| ap-FuPTikqMsqsSH7mrhKcVWc | Oct 24 15:32 | 1 | Fibonacci no in 5 rounds |
| ap-tOeQGZ8hBRvHghZbdpz5xi | Oct 24 15:34 | 0 | Fibonacci at N ≤ 26 round 1 |

Each preemption restarted its function, which replayed from its store. Every record matches its pre-cut version
(git `5d108a7`) in each round they share (`compare.py --old`: 12, 11, 11, 12, 3 and 1 rounds; the new-sources branch
had no record before). New rounds: 38, 184 search-hours. Cost: $21.92, the app `bar3-scale`'s billing from 21:00 UTC
Oct 24 to its stop. Each program found was rechecked on every pair up to N = 120 (7,260 pairs; Montgomery's 3,600,
odd N). The records and their `.state.<target>.json` files were fetched with
`modal volume get --force bar3-scale v1/<file> bar3/scale/records/<file>`.
