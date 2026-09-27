# Closed-Loop Manipulation Evaluation at a Commodity Price

Code and artifacts for running Toyota Research Institute's Drake-based
[`lbm_eval`](https://github.com/ToyotaResearchInstitute/lbm_eval) benchmark at
200-rollout scale on serverless GPUs, at roughly \$3.50 and 38 minutes of
wall-clock per replication.

This repository accompanies the paper *Closed-Loop Physics-Based Manipulation
Evaluation at a Commodity Price: What 4.4 L4 GPU hours at \$3.50 Buys You*.

**What this is not:** a policy-learning contribution. The trained policies here
are deliberately small and their success rates are low. The point is the
evaluation infrastructure and what running it carefully reveals.

---

## Quick start

```bash
pip install modal
modal setup                      # one-time auth

# 1. verify the container image builds and Drake renders headless
modal run lbm_phase0.py::run_sample_eval

# 2. build the training cache from the TRI demonstration data
modal run lbm_phase2_train.py::prepare_data

# 3. train the baseline policy (~26 min, ~$0.35)
modal run lbm_phase2_train.py::train --seed 1 \
    --out-path /data/phase2_models/bc_v1_seed1.pt

# 4. smoke-test the serving path on 3 scenarios
modal run lbm_phase3.py::test_shard \
    --checkpoint /data/phase2_models/bc_v1_seed1.pt

# 5. full 200-rollout replication (~38 min wall-clock, ~$3.49)
modal run --detach lbm_phase3.py::launch \
    --checkpoint /data/phase2_models/bc_v1_seed1.pt

# 6. screen and aggregate
modal run lbm_phase3.py::aggregate --run-id=repl-YYYYMMDDTHHMMSS
modal run lbm_phase3.py::screen    --run-id=repl-YYYYMMDDTHHMMSS
```

Total cost to reproduce every row of the paper's ablation table: approximately
\$34, assuming clean execution.

---

## Prerequisites

- A [Modal](https://modal.com) account. The free tier is enough to start, but
  note the default concurrency cap — one replication uses 8 GPUs, so running
  several at once will queue.
- The TRI demonstration data for `PickAndPlaceBox`, downloaded from
  `https://tri-ml-public.s3.amazonaws.com/datasets/lbm-eval-v1.1-sim-training-data/`
  and extracted to the Modal volume under `phase2_training_data/`.
  **This data is TRI's and is not redistributed here.**
- No local GPU required. Everything runs in Modal containers.

---

## Repository layout

```
bc_model.py                   shared model definition (encoder + decoder heads)
bc_policy_server.py           gRPC policy server; loads a checkpoint and serves actions

lbm_phase0.py                 environment de-risking: wheels install, headless render
lbm_phase1.py                 fan-out and aggregation, validated with the sample policy
lbm_phase2_probe.py           data-format probes (see "Verified interfaces" below)
lbm_phase2_train.py           v1: BC regressor, H=1, plus prepare_data
lbm_phase2_train_v2a.py       v2a: single-step DDPM
lbm_phase2_train_v2b.py       v2b: chunked DDPM (H=16), plus prepare_data_v2b
lbm_phase2_train_v2b_bc.py    v2b-bc: chunked BC regressor (H=8)
lbm_phase3.py                 replication driver, screening, Wilson-CI aggregation
lbm_phase3_diag.py            serving-time diagnostic probes

results/                      per-episode JSON and aggregates for every run in the paper
results/trivial_indices.json  the five screened scenario indices
```

Trained checkpoints are attached to the GitHub release rather than committed,
since they are ~45 MB each.


---

## The reporting protocol

Results in the paper follow three rules, implemented in `lbm_phase3.py`:

- **R1 — trivial-scene screening.** Completions recorded in under one second
  are excluded. Five of the 200 scenarios (indices 6, 94, 116, 140, 146) report
  success at 0.2–0.3 s before the policy meaningfully acts, under every policy
  tested including a random-motion sample policy. All rates are reported over
  the resulting 195-scene denominator. Omitting this rule reports a 2.5%
  success rate for a policy that does nothing.
- **R2 — recording verification.** Claimed successes are checked against their
  MeshCat recordings rather than trusted from the harness flag.
- **R3 — interval reporting and retention.** Rates are reported as N/195 with a
  95% Wilson score interval, and the full per-episode JSON is retained. All of
  it is in `results/`.

Note that all failures run to the 15-second episode limit, so the one-second
threshold separates two disjoint populations rather than dividing a continuum.

---

## Verified interfaces

The conventions needed to train against the released demonstration data are
documented, but spread across the benchmark README, `TRAINING_DATA_FORMAT.md`,
and per-episode `metadata.yaml`. Each was verified empirically before training
rather than assumed; `lbm_phase2_probe.py` contains the probes.

| Convention | Resolution |
|---|---|
| Action layout (20-dim) | `[R_xyz \| R_rot6d \| L_xyz \| L_rot6d \| RG \| LG]` |
| Training camera | serial `6CD146030E99` = harness key `scene_right_0` |
| Rotation representation | `rot6d` as stacked `[v1, v2]`, i.e. columns of R |
| Proprioception (18-dim) | `[R_xyz, R_rot6d, L_xyz, L_rot6d]` |
| Episode admissibility | episodes under 10 keyframes dropped as aborted |

Only the rotation convention was genuinely underdetermined — the format
document describes it as a truncated and flattened rotation matrix without
specifying rows versus columns. It was resolved by orthonormality testing.

---

## Environment

All runs execute in a Modal container built from the image definition in lbm_phase3.py: Debian slim, Python 3.12, torch 2.3.0, torchvision 0.18.0, numpy<2, and lbm_eval wheels 1.1.0, with one L4 GPU per shard. CUDA comes from torch's bundled build. The image also writes the EGL vendor ICD and warms the Drake asset cache at build time (see "Things that will bite you"). A small number of packages, including tqdm and the apt system libraries, are installed unpinned; a fully resolved lockfile is not currently provided.

---

## Things that will bite you

These cost real hours and are documented nowhere upstream:

1. **EGL vendor ICD.** Modal mounts the NVIDIA driver into GPU containers but
   not the userspace GL libraries. Drake's renderer looks for a vendor ICD JSON
   under `/usr/share/glvnd/egl_vendor.d/`; the image writes it at build time.
2. **Interpreter path.** Drake's asset-download script carries a
   `/usr/bin/python3` shebang, but the Debian-slim base puts Python elsewhere.
   The build creates and verifies a symlink.
3. **Checkpoint format drift.** The original trainer stored weights under
   `model_state` with one less level of module nesting than the later shared
   model module, which uses `model_state_dict`. Either difference alone
   prevents the server from loading the baseline checkpoint. The server now
   accepts both layouts and reports unmatched parameters at load time rather
   than silently proceeding with partially initialized weights.
4. **Local dispatcher fragility.** The fan-out driver originally ran on the
   operator's machine, so a replication died if the laptop slept — losing all
   in-flight shards with no partial results. It now dispatches from a
   cloud-resident function (`launch` / `run_replication_remote`), so a
   replication survives disconnection and can be aggregated afterward from the
   volume.

---

## Reproducing the ablation table

```bash
# baseline, three seeds
for S in 1 2 3; do
  modal run lbm_phase2_train.py::train --seed $S \
      --out-path /data/phase2_models/bc_v1_seed$S.pt
done

# diffusion variants
modal run lbm_phase2_train_v2a.py::train
modal run lbm_phase2_train_v2b.py::prepare_data_v2b
modal run lbm_phase2_train_v2b.py::train

# chunked BC control, three seeds
for S in 1 2 3; do
  modal run lbm_phase2_train_v2b_bc.py::train --seed $S \
      --out-path /data/phase2_models/bc_v2b_bc_seed$S.pt
done

# evaluate each; --replan-k 1 reproduces the K=1 isolation row
modal run --detach lbm_phase3.py::launch --checkpoint <path>
```

Seeds control network initialization and batch order only. The
train/validation split is generated from a separate fixed seed in both
trainers, so validation losses remain comparable across runs.

Expect training-seed variance to exceed the sampling variance a Wilson interval
captures: the three baseline checkpoints in the paper span 2.6% to 11.8% with
non-overlapping intervals. A single closed-loop number in this regime is not a
property of the method.

---

## Known limitations

- Recordings are sampled by shard position, not by outcome, so the retained set
  is unlikely to contain the successes you want to inspect. Selecting
  recordings by success would be a straightforward improvement.
- R2 verifies claimed successes but does not audit claimed failures.
- `lbm_phase2_serve.py` is superseded by `bc_policy_server.py` and kept only
  for reference.

---

## License and attribution

The LBM Eval benchmark, its wheels, and the demonstration data are Toyota Research Institute's. The benchmark software (v1.1.0) is dual-licensed under MIT and Apache 2.0; see LICENSE-MIT and LICENSE-APACHE in TRI's repository. The demonstration data is publicly distributed by TRI at the S3 URL above and is not redistributed here. This repository contains only wrapper code, trained checkpoints, and results.

Note that TRI's current release is a Fall 2025 snapshot, newer than the Spring 2025 version used in their published LBM paper; TRI states it will not necessarily reproduce that paper's results. This work replicates the benchmark harness at scale, not TRI's published numbers.
