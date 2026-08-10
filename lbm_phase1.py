"""
LBM Eval on Modal — Phase 1: replicate at statistical scale
===========================================================
Phase 0 (de-risking) is done. Phase 1, task 1 (cache drake_models into the
image) is done and verified. This file now adds the fan-out unit.

The shard design — and why it is built on `evaluate_many`, not the CLI
---------------------------------------------------------------------
`evaluate_one(skill_type, scenario_index, ...)` derives the scene's
`random_seed` deterministically from `scenario_index`
(evaluate.py:300 -> `get_demonstration_seed(scenario_index, use_eval_seed)`).
So `scenario_index` IS the sharding handle: index i always yields the same
initial condition.

The `evaluate` CLI hardwires `for scenario_index in range(num_evaluations)`
(evaluate.py:461) — it can only ever run indices 0..N-1, with no offset.
Handing K containers `num_evaluations=N` each would run the SAME N scenes K
times: a fake "K*N-sample" run. So the fan-out does NOT use the CLI.

Instead each shard calls `evaluate_many(evaluations=[...])`, where we pass an
explicit list of per-evaluation kwargs dicts, each with its own
`scenario_index`. Shards get DISJOINT index slices (shard 0 -> 0..24,
shard 1 -> 25..49, ...); the union is 0..199, every scene distinct, and the
whole run is exactly reproducible.

Run order:
  modal run lbm_phase1.py::verify_drake_cache   # smoke test (cache + 1 eval)
  modal run lbm_phase1.py::test_shard           # 1 small shard, 4 evals
  (fan-out driver + aggregator come next, once test_shard is green)
"""

import modal

APP_NAME = "lbm-eval-phase1"
app = modal.App(APP_NAME)

# --- Persistent storage (same Volume as Phase 0) ----------------------------
volume = modal.Volume.from_name("lbm-eval-data", create_if_missing=True)
VOL_PATH = "/data"

# --- Explicit Drake cache location ------------------------------------------
# Drake's cache directory is rooted at XDG_CACHE_HOME. Pinning it means the
# build-time warm-up writes here AND every runtime container reads from here.
DRAKE_CACHE_DIR = "/opt/drake_cache"

# --- Wheel URLs (from the 1.1.0 GitHub release) -----------------------------
WHEEL_URLS = [
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/robot_gym-1.1.0-py3-none-any.whl",
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/lbm_eval-1.1.0-py3-none-any.whl",
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/lbm_eval_models-1.1.0-py3-none-any.whl",
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/lbm_eval_scenarios-1.1.0-py3-none-any.whl",
]

# --- EGL / OpenGL userspace libraries Drake's headless renderer needs -------
SYSTEM_LIBS = [
    "libegl1",
    "libgl1",
    "libgles2",
    "libglib2.0-0",
    "libx11-6",
    "libxext6",
    "libsm6",
    "libglu1-mesa",
    "libopengl0",
    # Xvfb: virtual X framebuffer. Drake's RenderEngineGl uses GLX and needs a
    # DISPLAY; Xvfb provides one headlessly. With the NVIDIA driver present,
    # GLX still routes to the GPU for hardware acceleration.
    "xvfb",
    "xauth",
    "wget",
    "ca-certificates",
]

# --- Image (verified in Phase 1 task 1) -------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*SYSTEM_LIBS)
    .pip_install(*WHEEL_URLS)
    # NVIDIA EGL vendor ICD + the /usr/bin/python3 symlink fix (Phase 0).
    .run_commands(
        "mkdir -p /usr/share/glvnd/egl_vendor.d",
        'echo \'{"file_format_version":"1.0.0",'
        '"ICD":{"library_path":"libEGL_nvidia.so.0"}}\' '
        "> /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        # Drake's PackageMap downloader has a `#!/usr/bin/python3` shebang;
        # debian_slim's Python is at /usr/local/bin. Symlink it. MUST precede
        # the cache warm-up below (the download invokes python3).
        "set -e; "
        "REAL=$(readlink -f /usr/local/bin/python3); "
        'echo "real python3 -> $REAL"; '
        "ln -sf \"$REAL\" /usr/bin/python3; "
        "/usr/bin/python3 --version",
    )
    .env(
        {
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            "MESA_GL_VERSION_OVERRIDE": "4.5",
            # Root of Drake's cache dir. Set before the warm-up below; inherited
            # by every runtime container so the drake_models lookup hits.
            "XDG_CACHE_HOME": DRAKE_CACHE_DIR,
        }
    )
    # Bake drake_models into the image at BUILD time. `evaluate --help` builds
    # evaluate's argparser -> get_known_skill_types() -> GetPath("drake_models")
    # -> downloads via the exact code path evaluate uses at runtime.
    .run_commands(
        f"mkdir -p {DRAKE_CACHE_DIR}",
        "echo '--- Phase 1: warming drake_models cache ---'",
        "evaluate --help > /dev/null && echo 'evaluate --help: OK'",
        f"du -sh {DRAKE_CACHE_DIR} || true",
        f'test -n "$(find {DRAKE_CACHE_DIR} -type f -print -quit)" '
        f'|| (echo "ERROR: {DRAKE_CACHE_DIR} is empty after warm-up" && exit 1)',
        "echo '--- Phase 1: drake_models cache baked into image ---'",
    )
)


# --- Headless display helper ------------------------------------------------
def _start_xvfb(display=":99"):
    """Start an Xvfb virtual X server and point DISPLAY at it. Returns the
    subprocess handle (terminate it before the function returns)."""
    import os
    import subprocess
    import time

    proc = subprocess.Popen(
        ["Xvfb", display, "-screen", "0", "1280x1024x24", "-ac", "+extension", "GLX"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    os.environ["DISPLAY"] = display
    time.sleep(3)
    if proc.poll() is not None:
        out = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"Xvfb failed to start:\n{out}")
    print(f"  Xvfb running on DISPLAY={display}")
    return proc


def _start_policy_server():
    """Start the bundled wave_around sample policy server on localhost:50051.
    Returns the subprocess handle. (Phase 2 will swap this for a real policy.)"""
    import os
    import subprocess
    import sys
    import time

    bindir = os.path.dirname(sys.executable)
    server_bin = os.path.join(bindir, "wave_around_policy_server")
    print("Starting wave_around_policy_server ...")
    server = subprocess.Popen(
        [server_bin], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    time.sleep(8)
    if server.poll() is not None:
        out = server.stdout.read() if server.stdout else ""
        raise SystemExit(f"policy server died on startup:\n{out}")
    print("  server process is up")
    return server


def _stop(proc, timeout=10):
    """Terminate a subprocess, escalating to kill if it does not exit."""
    import subprocess

    proc.terminate()
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()


def _result_to_dict(e):
    """SingleEvaluationResult -> plain dict matching the Phase 0 JSON schema.
    Done by attribute access (not dataclasses.asdict) so it is robust to the
    exact class definition."""
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
# SMOKE TEST — drake_models cache is baked and used (Phase 1 task 1)
# ===========================================================================
@app.function(image=image, gpu="L4", volumes={VOL_PATH: volume}, timeout=1800)
def verify_drake_cache():
    """Confirm the drake_models assets are baked into the image and that a
    real evaluation no longer downloads anything from GitHub."""
    import os
    import subprocess
    import time
    from pathlib import Path

    print("=" * 64)
    print("SIGNAL A — drake_models cache baked into the image")
    print("=" * 64)
    cache = os.environ.get("XDG_CACHE_HOME", "(unset)")
    print(f"  XDG_CACHE_HOME = {cache}")
    os.system(f"du -sh {cache} 2>/dev/null || echo '  (cache dir missing!)'")
    n_files = subprocess.run(
        f"find {cache} -type f 2>/dev/null | wc -l",
        shell=True, capture_output=True, text=True,
    ).stdout.strip()
    print(f"  total files under cache: {n_files}")
    cache_ok = n_files.isdigit() and int(n_files) > 0

    print()
    print("=" * 64)
    print("SIGNAL B — a real evaluation re-downloads nothing")
    print("=" * 64)
    # evaluate_many / evaluate_one require output_directory to be a Path.
    out_dir = Path(VOL_PATH) / "phase1_cache_verify"
    out_dir.mkdir(parents=True, exist_ok=True)

    xvfb = _start_xvfb()
    server = _start_policy_server()

    from lbm_eval.evaluate import evaluate_many

    print("\nRunning one evaluation (pick_and_place_box, scenario_index=0) ...")
    t0 = time.time()
    # Capture stdout to scan for the PackageMap download line.
    import io
    import contextlib

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        results = evaluate_many(
            evaluations=[
                {
                    "skill_type": "pick_and_place_box",
                    "scenario_index": 0,
                    "output_directory": out_dir,
                }
            ],
            output_directory=out_dir,
            num_processes=1,
        )
    eval_stdout = buf.getvalue()
    dt = time.time() - t0
    print(eval_stdout)
    print(f"  evaluate_many finished in {dt:.1f}s")

    _stop(server)
    xvfb.terminate()
    volume.commit()

    downloads = [ln for ln in eval_stdout.splitlines()
                 if "PackageMap: Downloading" in ln]
    no_download = len(downloads) == 0
    if downloads:
        print("\n  CACHE MISS — evaluate still downloaded:")
        for ln in downloads:
            print(f"    {ln.strip()}")

    eval_ok = len(results.evaluations) == 1 and not results.evaluations[0].is_pending
    overall = cache_ok and eval_ok and no_download
    print()
    print("-" * 64)
    print(f"  SIGNAL A  cache baked & non-empty : {'PASS' if cache_ok else 'FAIL'}")
    print(f"  ----      evaluation completed    : {'PASS' if eval_ok else 'FAIL'}")
    print(f"  SIGNAL B  zero runtime downloads  : {'PASS' if no_download else 'FAIL'}")
    print("-" * 64)
    print("CACHE SMOKE TEST:", "PASS" if overall else "FAIL")
    if not overall:
        raise SystemExit("cache not verified — inspect output above.")
    return {"seconds": dt, "downloads": downloads}


# ===========================================================================
# FAN-OUT UNIT — one shard = one skill over a disjoint scenario_index slice
# ===========================================================================
@app.function(image=image, gpu="L4", volumes={VOL_PATH: volume}, timeout=3600)
def run_eval_shard(
    skill: str,
    index_start: int,
    index_end: int,
    run_id: str,
    n_recordings: int = 1,
):
    """Evaluate `skill` over scenario indices [index_start, index_end).

    One shard runs in one container on one L4 with num_processes=1 (serial
    inside the container; parallelism comes from running many shards). Keep a
    shard at roughly <= 30 evals so it finishes well inside the 1h timeout
    (~75s/eval => 30 evals ~= 38 min).

    Heavy per-rollout artifacts (the ~88 MB recording.html files) are written
    to container-local /tmp and discarded with the container. Only the small
    results JSON and `n_recordings` sampled recordings are copied to the
    Volume, so a 200-eval run does not put ~18 GB on the Volume.

    Returns a summary dict (also written as results-*.json on the Volume).
    """
    import glob
    import os
    import shutil
    import time
    from pathlib import Path

    n_evals = index_end - index_start
    print(f"SHARD  skill={skill}  indices=[{index_start}:{index_end})  "
          f"({n_evals} evals)  run_id={run_id}")

    # Container-local scratch for evaluate's heavy artifacts. evaluate_many /
    # evaluate_one require output_directory to be a pathlib.Path (they apply
    # the `/` operator to it), so this is a Path, not a str.
    local_out = Path("/tmp/shard_output")
    if local_out.exists():
        shutil.rmtree(local_out)
    local_out.mkdir(parents=True, exist_ok=True)

    xvfb = _start_xvfb()
    server = _start_policy_server()

    from lbm_eval.evaluate import evaluate_many

    # Each dict is kwargs for evaluate_one. The disjoint scenario_index slice
    # is what makes shards non-overlapping and the union reproducible.
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
        print(f"  progress: {done}/{len(results.evaluations)} done", flush=True)

    t0 = time.time()
    results = evaluate_many(
        evaluations=evaluations,
        output_directory=local_out,
        num_processes=1,
        progress_callback=_progress,
    )
    dt = time.time() - t0

    _stop(server)
    xvfb.terminate()

    # --- Persist: small JSON + sampled recordings to the Volume -------------
    vol_dir = os.path.join(
        VOL_PATH, "phase1_runs", run_id, skill,
        f"shard_{index_start:04d}_{index_end:04d}",
    )
    os.makedirs(vol_dir, exist_ok=True)

    for j in glob.glob(os.path.join(local_out, "results-*.json")):
        shutil.copy(j, vol_dir)
        print(f"  saved {os.path.basename(j)} -> {vol_dir}")

    # Sample a few recordings (first n_recordings by scenario_index) so there
    # is something to eyeball without keeping all of them.
    rec_dirs = sorted(glob.glob(os.path.join(local_out, skill, "demonstration_*")))
    for d in rec_dirs[:max(0, n_recordings)]:
        html = os.path.join(d, "recording.html")
        if os.path.exists(html):
            dst = os.path.join(vol_dir, f"{os.path.basename(d)}.recording.html")
            shutil.copy(html, dst)
            print(f"  saved sample recording {os.path.basename(dst)} "
                  f"({os.path.getsize(dst):,} bytes)")

    volume.commit()

    evals = [_result_to_dict(e) for e in results.evaluations]
    n_success = sum(1 for e in evals if e.get("is_success"))
    n_done = sum(1 for e in evals if not e.get("is_pending"))
    print(f"SHARD DONE  {n_done}/{n_evals} completed, {n_success} success, "
          f"{dt:.0f}s wall")

    return {
        "skill": skill,
        "index_start": index_start,
        "index_end": index_end,
        "run_id": run_id,
        "elapsed_time": dt,
        "evaluations": evals,
    }


# ===========================================================================
# AGGREGATION — pure functions, run locally in the driver (no GPU, no Modal)
# ===========================================================================
def _wilson_ci(k, n, z=1.96):
    """95% Wilson score interval for a binomial proportion k/n. Behaves well
    near 0 and 1, where the normal approximation breaks down — which is
    exactly the regime a manipulation success-rate replication lives in."""
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

    `failure_message` being non-null means the episode errored/crashed rather
    than failing the task cleanly — surfaced separately so a misbehaving skill
    is not silently folded into the "failed the task" count."""
    import collections
    import datetime

    by_skill = collections.defaultdict(list)
    for s in shard_summaries:
        for e in s["evaluations"]:
            by_skill[e["skill_type"]].append(e)

    per_skill = {}
    tot_completed = tot_success = 0
    for skill, evals in sorted(by_skill.items()):
        n_pending = sum(1 for e in evals if e.get("is_pending"))
        completed = [e for e in evals if not e.get("is_pending")]
        n_completed = len(completed)
        n_success = sum(1 for e in completed if e.get("is_success"))
        n_errored = sum(1 for e in completed
                        if not e.get("is_success") and e.get("failure_message"))
        rate = (n_success / n_completed) if n_completed else 0.0
        lo, hi = _wilson_ci(n_success, n_completed)
        per_skill[skill] = {
            "n_evaluated": len(evals),
            "n_completed": n_completed,
            "n_pending": n_pending,
            "n_success": n_success,
            "n_errored": n_errored,
            "success_rate": rate,
            "ci95_low": lo,
            "ci95_high": hi,
        }
        tot_completed += n_completed
        tot_success += n_success

    o_rate = (tot_success / tot_completed) if tot_completed else 0.0
    o_lo, o_hi = _wilson_ci(tot_success, tot_completed)
    return {
        "run_id": run_id,
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "config": config,
        "overall": {
            "n_completed": tot_completed,
            "n_success": tot_success,
            "success_rate": o_rate,
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
    print("=" * 74)
    print(f"REPLICATION AGGREGATE — run_id={agg['run_id']}")
    print("=" * 74)
    print(f"{'skill':<40}{'n':>5}{'succ':>6}{'rate':>9}{'95% CI':>14}")
    print("-" * 74)
    for skill, st in agg["per_skill"].items():
        ci = f"[{st['ci95_low']*100:.0f},{st['ci95_high']*100:.0f}]%"
        rate = f"{st['success_rate']*100:.1f}%"
        flags = ""
        if st["n_pending"]:
            flags += f"  !{st['n_pending']}pending"
        if st["n_errored"]:
            flags += f"  !{st['n_errored']}errored"
        print(f"{skill:<40}{st['n_completed']:>5}{st['n_success']:>6}"
              f"{rate:>9}{ci:>14}{flags}")
    print("-" * 74)
    o = agg["overall"]
    o_ci = f"[{o['ci95_low']*100:.0f},{o['ci95_high']*100:.0f}]%"
    print(f"{'OVERALL':<40}{o['n_completed']:>5}{o['n_success']:>6}"
          f"{o['success_rate']*100:>8.1f}%{o_ci:>14}")
    print("=" * 74)


# ===========================================================================
# ENTRYPOINTS
# ===========================================================================
@app.local_entrypoint()
def main():
    """`modal run lbm_phase1.py` — run the cache smoke test."""
    print("\n### cache smoke test ###")
    verify_drake_cache.remote()
    print("\nDone. Review output above.")


@app.local_entrypoint()
def test_shard():
    """`modal run lbm_phase1.py::test_shard` — smoke-test the shard function
    on ONE small shard (pick_and_place_box, 4 evals) before fanning out."""
    import datetime

    run_id = "test-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    print(f"\n### shard smoke test  run_id={run_id} ###")
    out = run_eval_shard.remote(
        skill="pick_and_place_box",
        index_start=0,
        index_end=4,
        run_id=run_id,
        n_recordings=2,
    )

    evals = out["evaluations"]
    n_success = sum(1 for e in evals if e.get("is_success"))
    n_done = sum(1 for e in evals if not e.get("is_pending"))
    idxs = sorted(e["scenario_index"] for e in evals)
    print("\n" + "-" * 56)
    print(f"shard {out['skill']} [{out['index_start']}:{out['index_end']}]")
    print(f"  scenario_index values : {idxs}   <- should be [0, 1, 2, 3]")
    print(f"  completed             : {n_done}/{len(evals)}")
    print(f"  successes             : {n_success}   (wave_around -> expect 0)")
    print(f"  wall time             : {out['elapsed_time']:.0f}s")
    print("-" * 56)
    print("If scenario_index is [0,1,2,3] and all 4 completed, the shard "
          "unit works — disjoint slices are ready to fan out.")


@app.local_entrypoint()
def run_replication():
    """`modal run lbm_phase1.py::run_replication` — fan a replication run
    across Modal containers and aggregate the results.

    The CONFIG below is a SMALL verification run (2 skills x 20 evals => 4
    shards, ~15 min, ~1 L4-hour) to confirm the driver + aggregator path end
    to end. Scale up by editing CONFIG — see the comment for full numbers.

    NOTE: the policy is still the `wave_around` sample, so every success_rate
    will be ~0%. That is expected — this run verifies the fan-out and
    aggregation MACHINERY. Real numbers arrive once a trained policy is
    plugged into the server (next phase).
    """
    import datetime
    import json

    # ---- CONFIG (edit to scale up) ----------------------------------------
    SKILLS = ["pick_and_place_box", "put_mug_on_saucer"]
    NUM_EVALUATIONS = 20      # per skill;  full replication: 200+
    SHARD_SIZE = 10           # evals per container; keep <= ~30 (1h timeout)
    # Full-replication example:
    #   SKILLS = [<representative subset of the 49 skills>]
    #   NUM_EVALUATIONS = 200
    #   SHARD_SIZE = 25       # => 8 shards/skill
    # For long runs, launch detached so a local disconnect does not kill it:
    #   modal run --detach lbm_phase1.py::run_replication
    # -----------------------------------------------------------------------

    run_id = "repl-" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    config = {
        "skills": SKILLS,
        "num_evaluations": NUM_EVALUATIONS,
        "shard_size": SHARD_SIZE,
        "policy": "wave_around_sample",
    }

    # Build disjoint (skill, index_start, index_end, run_id) shards.
    shard_args = []
    for skill in SKILLS:
        for start in range(0, NUM_EVALUATIONS, SHARD_SIZE):
            end = min(start + SHARD_SIZE, NUM_EVALUATIONS)
            shard_args.append((skill, start, end, run_id))

    print(f"\n### replication run {run_id} ###")
    print(f"  skills       : {len(SKILLS)} -> {SKILLS}")
    print(f"  evals/skill  : {NUM_EVALUATIONS}")
    print(f"  shards       : {len(shard_args)} "
          f"({SHARD_SIZE} evals each, one L4 container per shard)")
    print(f"  total evals  : {len(SKILLS) * NUM_EVALUATIONS}")
    print("  fanning out across Modal containers ...\n")

    # return_exceptions=True: one failed shard must not sink the whole run.
    results = list(run_eval_shard.starmap(shard_args, return_exceptions=True))

    ok, bad = [], []
    for args, res in zip(shard_args, results):
        (ok if isinstance(res, dict) else bad).append(
            res if isinstance(res, dict) else (args, res)
        )

    if bad:
        print(f"\n  {len(bad)} shard(s) FAILED:")
        for args, exc in bad:
            print(f"    {args[0]} [{args[1]}:{args[2]}] -> "
                  f"{type(exc).__name__}: {exc}")
    if not ok:
        raise SystemExit("all shards failed — nothing to aggregate.")

    agg = aggregate_results(ok, config, run_id)
    if bad:
        agg["failed_shards"] = [
            {"skill": a[0], "index_start": a[1], "index_end": a[2],
             "error": f"{type(e).__name__}: {e}"}
            for a, e in bad
        ]
    print_aggregate_table(agg)

    out_path = f"aggregate-{run_id}.json"
    with open(out_path, "w") as f:
        json.dump(agg, f, indent=2)
    print(f"\naggregate written to ./{out_path}")
    print(f"per-shard JSONs + sample recordings on the Volume under "
          f"phase1_runs/{run_id}/")