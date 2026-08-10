# Per-episode evaluation results

One directory per replication. Each contains eight `shard_*/shard_summary.json`
files with the full `evaluations` array (200 per run), plus `aggregate.json`.

| Directory | Policy | Table II row | Genuine / 195 |
|---|---|---|---|
| repl-20260527T210514 | bc_v1 | v1 (orig.) | 10 |
| repl-20260729T163244 | bc_v1_seed1 | v1 seed1 | 23 |
| repl-20260729T163259 | bc_v1_seed2 | v1 seed2 | 5 |
| repl-20260528T224750 | bc_v2a | v2a | 0 |
| repl-20260529T221603 | bc_v2b | v2b | 0 |
| repl-20260530T085526 | bc_v2b_bc | v2b-bc (8,4) | 0 |
| repl-20260720T124231 | bc_v2b_bc | v2b-bc (8,1) | 0 |
| repl-20260729T163317 | bc_v2b_bc_seed1 | v2b-bc seed1 | 0 |
| repl-20260729T163332 | bc_v2b_bc_seed2 | v2b-bc seed2 | 0 |

The two `bc_v2b_bc` runs share a checkpoint and differ only in `replan_k`
(4 vs 1), recorded in each run's `aggregate.json` config block. The K=1 run is
also the source of the cost figures in Table I.

Not in Table II: `repl-20260526T213700` is the pre-fix v1 run that scored
0/195 due to the serving-time normalization failure described in the paper.
Recordings are excluded here (~88 MB each); a few are attached to the release.
