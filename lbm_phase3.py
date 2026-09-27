"""
LBM Eval on Modal — Phase 3: scale the trained BC policy to a real number
=========================================================================
Phase 2 closed the train -> serve -> eval loop with ONE in-process evaluation
(`evaluate_one(policy=BCPolicy(), use_rpc=False)`). One sample is uninformative
by design — Wilson CI ~[0%,79%]. Phase 3 runs BC v1 over ~200 distinct
`scenario_index` values, fanned across Modal containers, to get its real
`pick_and_place_box` success rate with a tight 95% Wilson interval. That is
the first concrete result for goal 1, and the baseline the v2 diffusion policy
(goal 3) will be measured against.

DESIGN — gRPC, reusing Phase 1's machinery unchanged
----------------------------------------------------
Phase 1's `run_eval_shard` is already a gRPC *client*: it starts a policy
server subprocess and calls `evaluate_many` (no `policy=` arg), which connects
to that server on localhost:50051. Phase 1 used the bundled `wave_around_
policy_server`. Phase 3 changes exactly one thing — it starts our own
`bc_policy_server.py` instead. Everything else (disjoint `scenario_index`
slices, the aggregator, the Wilson CIs) is carried over verbatim.

This is the gRPC fork from the Phase 3 kickoff note, chosen over the in-process
path: BC v1 shares the L4 happily, so over loopback the server's transport cost
is negligible — and doing gRPC now means Phase 3 doubles as the shakedown of
the gRPC harness on a simple, fast, well-understood policy, instead of
debugging transport and a new diffusion policy simultaneously at v2.

TWO FILES
---------
  bc_policy_server.py  — the standalone gRPC server (no Modal). Added into the
                         image and launched as a subprocess inside each shard.
  lbm_phase3.py        — this file: the Modal fan-out + aggregation.

DEFERRED PHASE 1 CONFIG CHANGES — applied here
----------------------------------------------
 1. Aggregate JSON written to the Volume. Aggregation reads each shard's
    `shard_summary.json` FROM the Volume (`aggregate_from_volume`), so a
    `--detach` run whose client disconnects can still be aggregated after the
    fact: `modal run lbm_phase3.py::aggregate --run-id=...`.
 2. Per-episode data retained. The aggregate keeps, per skill, the failed and
    errored `scenario_index` lists, a total_time summary, and the full
    `evaluations` array — not just counts.
 3. Per-skill timing spread — N/A here: Phase 3 is a single skill, and
    `pick_and_place_box` is the fast one (~85s/eval).

Run order:
  modal run lbm_phase3.py::test_shard          # smoke-test the gRPC BC path
  modal run --detach lbm_phase3.py::run_replication   # the real scaled run
  modal run lbm_phase3.py::aggregate --run-id=repl-... # (re)aggregate a run
"""

import modal

APP_NAME = "lbm-eval-phase3"
app = modal.App(APP_NAME)

# --- Persistent storage (same Volume as Phases 0-2) -------------------------
volume = modal.Volume.from_name("lbm-eval-data", create_if_missing=True)
VOL_PATH = "/data"

# --- BC checkpoint + policy server ------------------------------------------
# The checkpoint is loaded from the Volume at runtime (no image rebuild on
# retrain). One switch selects which policy this run evaluates; the server
# auto-detects the architecture from the checkpoint's policy_type field, so
# nothing else changes between v1 and v2a.
#   bc_v1.pt  -> the BC baseline (5.1% [2.8%, 9.2%] on 195 scenes)
#   bc_v2a.pt -> the diffusion policy (lbm_phase2_train_v2a.py)
# Each run gets its own run_id, so v1 and v2a aggregates never collide and the
# comparison is on identical scenes + serving code.
# No default checkpoint: every entrypoint requires --checkpoint explicitly so
# a run can never silently evaluate the wrong policy.
CKPT_PATH = ""
# Human label for run metadata, derived from the checkpoint filename.
import os as _os
# POLICY_LABEL = _os.path.splitext(_os.path.basename(CKPT_PATH))[0]
BC_SERVER_REMOTE_PATH = "/root/bc_policy_server.py"
BC_MODEL_REMOTE_PATH = "/root/bc_model.py"
SERVER_URI = "localhost:50051"   # bc_policy_server default == evaluate default

DRAKE_CACHE_DIR = "/opt/drake_cache"

WHEEL_URLS = [
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/robot_gym-1.1.0-py3-none-any.whl",
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/lbm_eval-1.1.0-py3-none-any.whl",
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/lbm_eval_models-1.1.0-py3-none-any.whl",
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/lbm_eval_scenarios-1.1.0-py3-none-any.whl",
]

SYSTEM_LIBS = [
    "libegl1", "libgl1", "libgles2", "libglib2.0-0", "libx11-6", "libxext6",
    "libsm6", "libglu1-mesa", "libopengl0", "xvfb", "xauth", "wget",
    "ca-certificates",
]

# --- Image: Phase 1's verified image + torch + the BC server file -----------
# Phase 1's recipe (EGL ICD, python3 symlink, drake_models baked at build
# time) is reused exactly — fan-out still needs the cache baked so containers
# do not re-download assets. torch/torchvision are added because the BC server
# subprocess runs the checkpoint. The versions are left unpinned to match the
# Phase 2 environment that produced bc_v1.pt; pin them before the goal-4
# release for full reproducibility.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*SYSTEM_LIBS)
    .pip_install(*WHEEL_URLS)
    .pip_install("torch==2.3.0", "torchvision==0.18.0", "numpy<2", "tqdm")
    .run_commands(
        "mkdir -p /usr/share/glvnd/egl_vendor.d",
        'echo \'{"file_format_version":"1.0.0",'
        '"ICD":{"library_path":"libEGL_nvidia.so.0"}}\' '
        "> /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        "set -e; REAL=$(readlink -f /usr/local/bin/python3); "
        'ln -sf "$REAL" /usr/bin/python3; /usr/bin/python3 --version',
    )
    .env(
        {
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "MESA_GL_VERSION_OVERRIDE": "4.5",
            "XDG_CACHE_HOME": DRAKE_CACHE_DIR,
        }
    )
    .run_commands(
        f"mkdir -p {DRAKE_CACHE_DIR}",
        "echo '--- Phase 3: warming drake_models cache ---'",
        "evaluate --help > /dev/null && echo 'evaluate --help: OK'",
        f"du -sh {DRAKE_CACHE_DIR} || true",
        f'test -n "$(find {DRAKE_CACHE_DIR} -type f -print -quit)" '
        f'|| (echo "ERROR: {DRAKE_CACHE_DIR} is empty after warm-up" '
        f'&& exit 1)',
        "echo '--- Phase 3: drake_models cache baked into image ---'",
    )
    # The standalone gRPC server + the shared model module, launched as a
    # subprocess inside each shard. bc_model.py sits next to the server in
    # /root so `import bc_model` resolves for the subprocess regardless of cwd.
    .add_local_file("bc_policy_server.py", BC_SERVER_REMOTE_PATH)
    .add_local_file("bc_model.py", BC_MODEL_REMOTE_PATH)
)


# ---------------------------------------------------------------------------
# Subprocess / display helpers
# ---------------------------------------------------------------------------
def _start_xvfb(display=":99"):
    """Start an Xvfb virtual X server and point DISPLAY at it."""
    import os
    import subprocess
    import time

    proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1280x1024x24", "-ac",
         "+extension", "GLX"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    os.environ["DISPLAY"] = display
    time.sleep(3)
    if proc.poll() is not None:
        out = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"Xvfb failed to start:\n{out}")
    print(f"  Xvfb running on DISPLAY={display}")
    return proc


def _start_bc_policy_server(checkpoint: str, server_uri=SERVER_URI,
                            *, replan_k=0, ready_timeout=240):
    """Launch bc_policy_server.py as a subprocess and wait until it is ready.

    Replaces Phase 1's `_start_policy_server` (which ran the wave_around
    binary). Phase 1's helper did `sleep(8)` then a single `poll()` — fine for
    the trivial wave_around binary, but the BC server imports torch, inits
    CUDA, and loads the checkpoint, taking tens of seconds. A fixed sleep would
    either be wastefully long or return while the server is still loading.

    Instead this polls the gRPC port for connectability. The server binds the
    port only AFTER the model is fully loaded (see bc_policy_server.main), so
    a successful TCP connect means the server is genuinely ready. The loop
    also checks `poll()` each iteration: if the server dies during model load
    (e.g. a bad checkpoint), this raises immediately with a clear message
    rather than letting `evaluate_many` wait forever on a dead endpoint.

    The server's stdout/stderr are inherited (not piped), so its logs stream
    into the Modal container log and there is no pipe-buffer deadlock risk.
    """
    import os
    import socket
    import subprocess
    import sys
    import time
    host, port = server_uri.split(":")
    port = int(port)
    print(f"Starting BC policy server: {checkpoint} on {server_uri} "
          f"(replan_k={replan_k or 'ckpt default'}) ...")
    server = subprocess.Popen(
        [sys.executable, BC_SERVER_REMOTE_PATH,
         "--checkpoint", checkpoint, "--server-uri", server_uri],
        env={**os.environ, "BC_REPLAN_K": str(replan_k)},
    )

    t0 = time.time()
    while True:
        if server.poll() is not None:
            raise SystemExit(
                f"BC policy server exited with code {server.returncode} "
                f"during startup — see the server log above.")
        try:
            with socket.create_connection((host, port), timeout=1):
                break
        except OSError:
            pass
        if time.time() - t0 > ready_timeout:
            server.kill()
            raise SystemExit(
                f"BC policy server did not open {server_uri} within "
                f"{ready_timeout}s — see the server log above.")
        time.sleep(1)

    print(f"  BC policy server ready on {server_uri} "
          f"({time.time() - t0:.0f}s to load).")
    return server


def _stop(proc, timeout=10):
    """Terminate a subprocess, escalating to kill if it does not exit."""
    import subprocess

    if proc is None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def _result_to_dict(e):
    """SingleEvaluationResult -> plain dict (same schema as Phases 0-1)."""
    return {
        k: getattr(e, k, None)
        for k in (
            "skill_type",
            "scenario_index",
            "is_pending",
            "total_time",
            "is_success",
            "failure_message",
        )
    }


# ===========================================================================
# FAN-OUT UNIT — one shard = the BC policy over a disjoint scenario_index slice
# ===========================================================================
@app.function(image=image, gpu="L4", volumes={VOL_PATH: volume}, timeout=7200)
def run_eval_shard(
    skill: str,
    index_start: int,
    index_end: int,
    run_id: str,
    checkpoint: str,
    n_recordings: int = 1,
    replan_k: int = 0,
):
    """Evaluate the BC policy on `skill` over scenario indices
    [index_start, index_end).

    Identical to Phase 1's `run_eval_shard` except it starts the BC gRPC
    server instead of the wave_around sample. One shard = one container = one
    L4; `num_processes=1` so evaluations run serially inside the shard and the
    parallelism comes from running many shards. The disjoint index slice is
    what makes the union of shards reproducible and non-overlapping.

    Heavy per-rollout artifacts (~88 MB recording.html files) stay in
    container-local /tmp; only the small results JSON, this shard's
    `shard_summary.json`, and `n_recordings` sampled recordings go to the
    Volume. timeout=7200s: pick_and_place_box is ~85s/eval, so a 25-eval shard
    is ~40 min — well inside the limit, with headroom for the server cold
    start and any per-eval variance.
    """
    import glob
    import json
    import os
    import shutil
    import time
    from pathlib import Path

    n_evals = index_end - index_start
    policy_label = os.path.splitext(os.path.basename(checkpoint))[0]
    print(f"SHARD  skill={skill}  indices=[{index_start}:{index_end})  "
          f"({n_evals} evals)  run_id={run_id}  policy={policy_label}")

    if not os.path.exists(checkpoint):
        raise SystemExit(
            f"{checkpoint} missing on the Volume — train it and confirm "
            f"volume.commit() ran before evaluating.")

    # Container-local scratch for evaluate's heavy artifacts. evaluate_many
    # requires output_directory to be a pathlib.Path.
    local_out = Path("/tmp/shard_output")
    if local_out.exists():
        shutil.rmtree(local_out)
    local_out.mkdir(parents=True, exist_ok=True)

    xvfb = None
    server = None
    try:
        xvfb = _start_xvfb()
        server = _start_bc_policy_server(checkpoint, replan_k=replan_k)

        from lbm_eval.evaluate import evaluate_many

        # Each dict is kwargs for evaluate_one; the disjoint scenario_index
        # slice is what makes shards non-overlapping. No `policy=` arg ->
        # evaluate_many runs as a gRPC client against the server above.
        evaluations = [
            {
                "skill_type": skill,
                "scenario_index": i,
                "output_directory": local_out,
            }
            for i in range(index_start, index_end)
        ]

        def _progress(results):
            done = sum(1 for e in results.evaluations if not e.is_pending)
            print(f"  progress: {done}/{len(results.evaluations)} done",
                  flush=True)

        t0 = time.time()
        results = evaluate_many(
            evaluations=evaluations,
            output_directory=local_out,
            num_processes=1,
            progress_callback=_progress,
        )
        dt = time.time() - t0
    finally:
        # Always tear the subprocesses down, even if evaluate_many raised.
        _stop(server)
        _stop(xvfb)

    # --- Persist: small JSON + sampled recordings to the Volume -------------
    vol_dir = os.path.join(
        VOL_PATH, "phase3_runs", run_id, skill,
        f"shard_{index_start:04d}_{index_end:04d}",
    )
    os.makedirs(vol_dir, exist_ok=True)

    for j in glob.glob(os.path.join(local_out, "results-*.json")):
        shutil.copy(j, vol_dir)
        print(f"  saved {os.path.basename(j)} -> {vol_dir}")

    rec_dirs = sorted(glob.glob(os.path.join(local_out, skill,
                                             "demonstration_*")))
    for d in rec_dirs[:max(0, n_recordings)]:
        html = os.path.join(d, "recording.html")
        if os.path.exists(html):
            dst = os.path.join(vol_dir, f"{os.path.basename(d)}.recording.html")
            shutil.copy(html, dst)
            print(f"  saved sample recording {os.path.basename(dst)} "
                  f"({os.path.getsize(dst):,} bytes)")

    evals = [_result_to_dict(e) for e in results.evaluations]
    n_success = sum(1 for e in evals if e.get("is_success"))
    n_done = sum(1 for e in evals if not e.get("is_pending"))

    summary = {
        "skill": skill,
        "index_start": index_start,
        "index_end": index_end,
        "run_id": run_id,
        "policy": policy_label,
        "elapsed_time": dt,
        "evaluations": evals,
    }
    # The shard writes its OWN summary to the Volume. aggregate_from_volume
    # reads these back, so aggregation never depends on the driver process
    # surviving a long --detach run.
    with open(os.path.join(vol_dir, "shard_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    volume.commit()

    print(f"SHARD DONE  {n_done}/{n_evals} completed, {n_success} success, "
          f"{dt:.0f}s wall")
    return summary


# ===========================================================================
# AGGREGATION — pure functions
# ===========================================================================
def _wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n. Behaves well
    near 0 and 1, where the normal approximation breaks down — exactly the
    regime a manipulation success-rate replication lives in."""
    import math

    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def aggregate_results(shard_summaries, config, run_id):
    """Concatenate every shard's `evaluations` array, group by skill, and
    compute per-skill + overall success rates with 95% Wilson intervals.

    Beyond Phase 1's counts, the per-skill block retains (deferred config
    change 2): the failed and errored `scenario_index` lists, a total_time
    summary, and the full `evaluations` array. `failure_message` being
    non-null marks an episode that errored/crashed rather than failing the
    task cleanly — surfaced separately so a misbehaving run is not silently
    folded into the honest 'failed the task' count."""
    import collections
    import datetime

    by_skill = collections.defaultdict(list)
    for s in shard_summaries:
        for e in s["evaluations"]:
            by_skill[e["skill_type"]].append(e)

    per_skill = {}
    tot_completed = tot_success = 0
    for skill, evals in sorted(by_skill.items()):
        evals = sorted(evals, key=lambda e: e.get("scenario_index", -1))
        completed = [e for e in evals if not e.get("is_pending")]
        n_completed = len(completed)
        n_success = sum(1 for e in completed if e.get("is_success"))
        errored = [e for e in completed
                   if not e.get("is_success") and e.get("failure_message")]
        failed = [e for e in completed
                  if not e.get("is_success") and not e.get("failure_message")]
        rate = (n_success / n_completed) if n_completed else 0.0
        lo, hi = _wilson_ci(n_success, n_completed)
        times = [e["total_time"] for e in completed
                 if e.get("total_time") is not None]
        per_skill[skill] = {
            "n_evaluated": len(evals),
            "n_completed": n_completed,
            "n_pending": sum(1 for e in evals if e.get("is_pending")),
            "n_success": n_success,
            "n_errored": len(errored),
            "success_rate": rate,
            "ci95_low": lo,
            "ci95_high": hi,
            "failed_indices": [e["scenario_index"] for e in failed],
            "errored_indices": [e["scenario_index"] for e in errored],
            "total_time": {
                "min": min(times) if times else None,
                "mean": (sum(times) / len(times)) if times else None,
                "max": max(times) if times else None,
            },
            "evaluations": evals,
        }
        tot_completed += n_completed
        tot_success += n_success

    o_lo, o_hi = _wilson_ci(tot_success, tot_completed)
    return {
        "run_id": run_id,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "overall": {
            "n_completed": tot_completed,
            "n_success": tot_success,
            "success_rate": (tot_success / tot_completed)
            if tot_completed else 0.0,
            "ci95_low": o_lo,
            "ci95_high": o_hi,
        },
        "per_skill": per_skill,
        "shards": [
            {k: s[k] for k in ("skill", "index_start", "index_end",
                               "elapsed_time")}
            for s in shard_summaries
        ],
    }


def print_aggregate_table(agg):
    """Human-readable per-skill table with 95% confidence intervals."""
    print()
    print("=" * 78)
    print(f"BC v1 REPLICATION AGGREGATE — run_id={agg['run_id']}")
    print("=" * 78)
    print(f"{'skill':<34}{'n':>5}{'succ':>6}{'rate':>9}{'95% CI':>14}"
          f"{'mean s':>10}")
    print("-" * 78)
    for skill, st in agg["per_skill"].items():
        ci = f"[{st['ci95_low']*100:.0f},{st['ci95_high']*100:.0f}]%"
        rate = f"{st['success_rate']*100:.1f}%"
        mean_t = st["total_time"]["mean"]
        mean_s = f"{mean_t:.0f}" if mean_t is not None else "-"
        flags = ""
        if st["n_pending"]:
            flags += f"  !{st['n_pending']}pending"
        if st["n_errored"]:
            flags += f"  !{st['n_errored']}errored"
        print(f"{skill:<34}{st['n_completed']:>5}{st['n_success']:>6}"
              f"{rate:>9}{ci:>14}{mean_s:>10}{flags}")
    print("-" * 78)
    o = agg["overall"]
    o_ci = f"[{o['ci95_low']*100:.0f},{o['ci95_high']*100:.0f}]%"
    print(f"{'OVERALL':<34}{o['n_completed']:>5}{o['n_success']:>6}"
          f"{o['success_rate']*100:>8.1f}%{o_ci:>14}")
    print("=" * 78)
    for skill, st in agg["per_skill"].items():
        if st["errored_indices"]:
            print(f"  {skill} errored scenario_index: {st['errored_indices']}")


@app.function(image=image, volumes={VOL_PATH: volume}, timeout=600)
def aggregate_from_volume(run_id: str, config: dict | None = None):
    """Read every shard's `shard_summary.json` for `run_id` from the Volume,
    aggregate, write `aggregate.json` back to the Volume, and return the
    aggregate dict.

    Because this reads shard data straight from the Volume, it works whether
    or not the original driver process survived — so a `--detach` run can be
    aggregated (or re-aggregated) at any time."""
    import glob
    import json
    import os

    volume.reload()
    run_dir = os.path.join(VOL_PATH, "phase3_runs", run_id)
    summary_paths = sorted(
        glob.glob(os.path.join(run_dir, "*", "shard_*", "shard_summary.json")))
    if not summary_paths:
        raise SystemExit(f"no shard_summary.json found under {run_dir}")

    shard_summaries = []
    for p in summary_paths:
        with open(p) as f:
            shard_summaries.append(json.load(f))
    print(f"aggregating {len(shard_summaries)} shard summaries for {run_id}")

    if not config:
        # Re-aggregation without a config: recover what we can from the shards
        # so provenance is never destroyed by a second aggregate run.
        labels = {s.get("policy") for s in shard_summaries if s.get("policy")}
        config = {"policy": sorted(labels)[0] if len(labels) == 1 else sorted(labels),
                  "recovered_from_shards": True}

    agg = aggregate_results(shard_summaries, config or {}, run_id)

    out_path = os.path.join(run_dir, "aggregate.json")
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    volume.commit()
    print(f"aggregate written to Volume: phase3_runs/{run_id}/aggregate.json")
    return agg

TRIVIAL_TIME_S = 1.0   # R1: completions faster than this are benchmark artifacts

@app.function(image=modal.Image.debian_slim(), volumes={VOL_PATH: volume},
              timeout=600)
def screen_from_volume(run_id: str):
    """R1 screening: separate genuine successes from benchmark artifacts.

    Five of the 200 pick_and_place_box scenarios report success in ~0.2-0.3s,
    before the policy has meaningfully acted. They do so under every policy
    tested, including a random-motion sample, so they are a property of the
    benchmark's initial conditions rather than of any policy. All rates in the
    paper are reported over the resulting 195-scene denominator.
    """
    import glob, json, os
    volume.reload()
    pattern = os.path.join(VOL_PATH, "phase3_runs", run_id,
                           "*", "shard_*", "shard_summary.json")
    evals, policy = [], "?"
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            d = json.load(f)
        evals += d["evaluations"]
        policy = d.get("policy", policy)
    if not evals:
        raise SystemExit(f"no shard summaries found for {run_id}")

    succ = [e for e in evals if e.get("is_success")]
    trivial = sorted(e["scenario_index"] for e in succ
                     if (e.get("total_time") or 0) < TRIVIAL_TIME_S)
    genuine = sorted(e["scenario_index"] for e in succ
                     if (e.get("total_time") or 0) >= TRIVIAL_TIME_S)
    denom = len(evals) - len(trivial)
    lo, hi = _wilson_ci(len(genuine), denom)

    print(f"\nR1 SCREENING — run_id={run_id}  policy={policy}")
    print(f"  evaluated            : {len(evals)}")
    print(f"  trivial ({len(trivial)}, <{TRIVIAL_TIME_S}s) : {trivial}")
    print(f"  genuine successes    : {len(genuine)} {genuine}")
    print(f"  screened rate        : {len(genuine)}/{denom} = "
          f"{100*len(genuine)/denom:.1f}%  95% Wilson [{lo*100:.1f}, {hi*100:.1f}]")
    return {"run_id": run_id, "policy": policy, "n_evaluated": len(evals),
            "trivial_indices": trivial, "genuine_indices": genuine,
            "denominator": denom, "n_genuine": len(genuine),
            "rate": len(genuine)/denom, "ci95": [lo, hi]}


@app.local_entrypoint()
def screen(run_id: str):
    """`modal run lbm_phase3.py::screen --run-id=repl-...` — apply R1."""
    screen_from_volume.remote(run_id)

# ===========================================================================
# ENTRYPOINTS
# ===========================================================================
@app.local_entrypoint()
def test_shard(checkpoint: str, replan_k: int = 0):
    """`modal run lbm_phase3.py::test_shard` — smoke-test the gRPC BC path.

    Phase 2 only ran BC in-process; the gRPC-served BC path has never run.
    This runs ONE small shard (pick_and_place_box, 3 evals) through the BC
    policy server and confirms: the server starts and is reachable, all evals
    complete with no errors, and the scenario indices are exactly [0,1,2].
    Run this green before fanning out."""
    import datetime

    run_id = "test-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"\n### Phase 3 gRPC BC smoke test  run_id={run_id} ###")
    out = run_eval_shard.remote(
        skill="pick_and_place_box",
        index_start=0,
        index_end=3,
        run_id=run_id,
        checkpoint=checkpoint,
        n_recordings=2,
        replan_k=replan_k,
    )

    evals = out["evaluations"]
    n_success = sum(1 for e in evals if e.get("is_success"))
    n_done = sum(1 for e in evals if not e.get("is_pending"))
    n_errored = sum(1 for e in evals
                    if e.get("failure_message") and not e.get("is_success"))
    idxs = sorted(e["scenario_index"] for e in evals)
    print("\n" + "-" * 60)
    print(f"shard {out['skill']} [{out['index_start']}:{out['index_end']}]")
    print(f"  scenario_index values : {idxs}   <- should be [0, 1, 2]")
    print(f"  completed             : {n_done}/{len(evals)}")
    print(f"  errored episodes      : {n_errored}   <- should be 0")
    print(f"  successes             : {n_success}")
    print(f"  wall time             : {out['elapsed_time']:.0f}s")
    print("-" * 60)
    if idxs == [0, 1, 2] and n_done == 3 and n_errored == 0:
        print("gRPC BC path verified — ready to fan out with run_replication.")
    else:
        print("Something is off — inspect the shard output above before "
              "scaling.")


@app.local_entrypoint()
def run_replication(checkpoint: str, replan_k: int = 0):
    """`modal run lbm_phase3.py::run_replication` — the real scaled run.

    Fans the BC policy across Modal containers over disjoint scenario_index
    slices, then aggregates from the Volume. For the real run, launch detached
    so a local disconnect cannot kill it:
        modal run --detach lbm_phase3.py::run_replication

    The CONFIG below is the actual Phase 3 target: BC v1 on pick_and_place_box,
    200 evaluations (the README's recommended statistical floor), in 8 shards
    of 25. ~85s/eval -> ~35-40 min/shard; shards run in parallel, so wall time
    is roughly one shard. Lowering SHARD_SIZE buys wall-time, not L4-hours.
    """
    import datetime

    # ---- CONFIG -----------------------------------------------------------
    SKILLS = ["pick_and_place_box"]
    NUM_EVALUATIONS = 200      # per skill; README recommends 200+
    SHARD_SIZE = 25            # evals per container
    # -----------------------------------------------------------------------

    run_id = "repl-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    policy_label = _os.path.splitext(_os.path.basename(checkpoint))[0]
    config = {
        "skills": SKILLS,
        "num_evaluations": NUM_EVALUATIONS,
        "shard_size": SHARD_SIZE,
        "policy": policy_label,
        "checkpoint": checkpoint,
        "replan_k": replan_k,
    }
    N_RECORDINGS_PER_SHARD = 1
    shard_args = []
    for skill in SKILLS:
        for start in range(0, NUM_EVALUATIONS, SHARD_SIZE):
            end = min(start + SHARD_SIZE, NUM_EVALUATIONS)
            shard_args.append((skill, start, end, run_id, checkpoint, N_RECORDINGS_PER_SHARD, replan_k))

    print(f"\n### Phase 3 replication  run_id={run_id} ###")
    print(f"  checkpoint   : {checkpoint}")
    print(f"  replan_k     : {replan_k or 'ckpt default'}")
    print(f"  skills       : {SKILLS}")
    print(f"  shards       : {len(shard_args)} ({SHARD_SIZE} evals each, "
          f"one L4 per shard)")
    print(f"  total evals  : {len(SKILLS) * NUM_EVALUATIONS}")
    print("  fanning out across Modal containers ...\n")

    # return_exceptions=True: one failed shard must not sink the whole run.
    results = list(run_eval_shard.starmap(shard_args, return_exceptions=True))

    ok, bad = [], []
    for args, res in zip(shard_args, results):
        if isinstance(res, dict):
            ok.append(res)
        else:
            bad.append((args, res))

    if bad:
        print(f"\n  {len(bad)} shard(s) FAILED:")
        for args, exc in bad:
            print(f"    {args[0]} [{args[1]}:{args[2]}] -> "
                  f"{type(exc).__name__}: {exc}")
    if not ok:
        raise SystemExit("all shards failed — nothing to aggregate.")

    # Aggregate from the Volume (each ok shard committed its shard_summary.json
    # there). This path is identical to the standalone `aggregate` entrypoint,
    # so a --detach run that loses its client can be aggregated later the same
    # way.
    agg = aggregate_from_volume.remote(run_id, config)
    if bad:
        agg["failed_shards"] = [
            {"skill": a[0], "index_start": a[1], "index_end": a[2],
             "error": f"{type(e).__name__}: {e}"}
            for a, e in bad
        ]
    print_aggregate_table(agg)
    print(f"\nFull aggregate (with per-episode data) on the Volume: "
          f"phase3_runs/{run_id}/aggregate.json")
    if bad:
        print(f"NOTE: {len(bad)} shard(s) failed; rerun those index ranges "
              f"and re-aggregate with `::aggregate --run-id={run_id}`.")


@app.local_entrypoint()
def aggregate(run_id: str):
    """`modal run lbm_phase3.py::aggregate --run-id=repl-...` — (re)build the
    aggregate for a run from the shard summaries already on the Volume. Use
    this if a --detach run's driver disconnected before aggregating, or to
    re-aggregate after rerunning failed shards."""
    agg = aggregate_from_volume.remote(run_id)
    print_aggregate_table(agg)
    print(f"\nAggregate on the Volume: phase3_runs/{run_id}/aggregate.json")

@app.function(image=image, volumes={VOL_PATH: volume}, timeout=14400)
def run_replication_remote(run_id: str, checkpoint: str, replan_k: int = 0):
    """Server-side fan-out driver. Runs the starmap inside Modal so the
    replication does not depend on a local client surviving ~35 minutes."""
    import os as _o
    SKILLS = ["pick_and_place_box"]
    NUM_EVALUATIONS, SHARD_SIZE = 200, 25
    policy_label = _o.path.splitext(_o.path.basename(checkpoint))[0]
    config = {
        "skills": SKILLS, "num_evaluations": NUM_EVALUATIONS,
        "shard_size": SHARD_SIZE, "policy": policy_label,
        "checkpoint": checkpoint, "replan_k": replan_k,
    }
    N_RECORDINGS_PER_SHARD = 1
    shard_args = [
        (s, st, min(st + SHARD_SIZE, NUM_EVALUATIONS), run_id, checkpoint, N_RECORDINGS_PER_SHARD, replan_k)
        for s in SKILLS
        for st in range(0, NUM_EVALUATIONS, SHARD_SIZE)
    ]
    results = list(run_eval_shard.starmap(shard_args, return_exceptions=True))
    ok = [r for r in results if isinstance(r, dict)]
    print(f"{len(ok)}/{len(shard_args)} shards ok")
    if not ok:
        raise SystemExit("all shards failed")
    return aggregate_from_volume.remote(run_id, config)


@app.local_entrypoint()
def launch(checkpoint: str, replan_k: int = 0):
    import datetime
    run_id = "repl-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    run_replication_remote.spawn(run_id, checkpoint, replan_k)
    print(f"spawned run_id={run_id}  — safe to close the laptop")
    print(f"later:  modal run lbm_phase3.py::aggregate --run-id={run_id}")