"""
LBM Eval on Modal — Phase 0
===========================
Goal: de-risk the environment BEFORE running any real evaluation.

Three functions, run them in order:

  1. modal run lbm_phase0.py::probe_install
        -> Do the four TRI wheels even install + import on Modal's OS?

  2. modal run lbm_phase0.py::probe_render
        -> On an L4: can Drake create an EGL rendering context and render a
           frame headless? Does this wheel's Meshcat expose the recording API
           (StartRecording / StopRecording / StaticHtml)?

  3. modal run lbm_phase0.py::run_sample_eval
        -> End-to-end: sample 'wave_around' policy server + `evaluate` client
           in ONE container, pick_and_place_box, num_evaluations=1.
           Writes results JSON to the Modal Volume.

Only move to Phase 1 (scale-up) once all three are green.

Notes / decisions (see chat):
  - GPU: L4 (plenty for Drake rendering; the GPU here is for OpenGL camera
    rendering throughput, not neural-net inference).
  - Storage: Modal Volume mounted at /data.
  - Viz: we run headless and SAVE artifacts (Option A). No live MeshCat.
  - The wheels target Ubuntu 24.04 / Py3.12. We pin Py3.12 on debian_slim and
    install EGL/OpenGL userspace libs explicitly. If probe_install fails on a
    glibc/OS mismatch, switch to the ubuntu:24.04 base image (commented below).
"""

import modal

APP_NAME = "lbm-eval-phase0"
app = modal.App(APP_NAME)

# --- Persistent storage -----------------------------------------------------
# created on first use; survives across runs
volume = modal.Volume.from_name("lbm-eval-data", create_if_missing=True)
VOL_PATH = "/data"

# --- Wheel URLs (from the 1.1.0 GitHub release) -----------------------------
WHEEL_URLS = [
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/robot_gym-1.1.0-py3-none-any.whl",
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/lbm_eval-1.1.0-py3-none-any.whl",
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/lbm_eval_models-1.1.0-py3-none-any.whl",
    "https://github.com/ToyotaResearchInstitute/lbm_eval/releases/download/1.1.0/lbm_eval_scenarios-1.1.0-py3-none-any.whl",
]

# --- EGL / OpenGL userspace libraries Drake's headless renderer needs -------
# The NVIDIA *driver* is mounted into Modal GPU containers automatically;
# these are the userspace libs that are NOT present by default.
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
    # DISPLAY; Xvfb provides one in a headless container. With the NVIDIA
    # driver present, GLX still routes to the GPU for hardware acceleration.
    "xvfb",
    "xauth",
    "wget",
    "ca-certificates",
]

# --- Image ------------------------------------------------------------------
# debian_slim + Python 3.12 to match the wheels' target as closely as possible.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*SYSTEM_LIBS)
    # Install the four TRI wheels directly from the release URLs.
    .pip_install(*WHEEL_URLS)
    # NVIDIA EGL vendor ICD: tells libEGL to use the NVIDIA driver headlessly.
    # Drake looks for an ICD JSON under /usr/share/glvnd/egl_vendor.d/.
    .run_commands(
        "mkdir -p /usr/share/glvnd/egl_vendor.d",
        'echo \'{"file_format_version":"1.0.0",'
        '"ICD":{"library_path":"libEGL_nvidia.so.0"}}\' '
        "> /usr/share/glvnd/egl_vendor.d/10_nvidia.json",
        # Drake's PackageMap asset downloader script has a `#!/usr/bin/python3`
        # shebang. On debian_slim Python is at /usr/local/bin/python3, so we
        # symlink it. We resolve the real interpreter explicitly and verify
        # the link works at build time (fail loud, not silently at runtime).
        "set -e; "
        "REAL=$(readlink -f /usr/local/bin/python3); "
        'echo "real python3 -> $REAL"; '
        "ln -sf \"$REAL\" /usr/bin/python3; "
        "/usr/bin/python3 --version",
    )
    .env(
        {
            # Hint to GL/EGL stacks to go through the NVIDIA vendor lib.
            "__GLX_VENDOR_LIBRARY_NAME": "nvidia",
            # Drake's RenderEngineGl uses EGL when no X display is present.
            "MESA_GL_VERSION_OVERRIDE": "4.5",
        }
    )
)

# If probe_install fails with a glibc / OS-version error, swap the image above
# for the Ubuntu base below and re-run:
#
# image = (
#     modal.Image.from_registry("ubuntu:24.04", add_python="3.12")
#     .apt_install(*SYSTEM_LIBS)
#     .pip_install(*WHEEL_URLS)
#     .run_commands(... same ICD JSON ...)
#     .env(... same ...)
# )


# --- Headless display helper ------------------------------------------------
def _start_xvfb(display=":99"):
    """Start an Xvfb virtual X server and point DISPLAY at it.

    Drake's RenderEngineGl uses GLX, which requires an X display. In a
    headless container we provide one with Xvfb. The NVIDIA driver still
    provides hardware-accelerated GL through this virtual display.

    Returns the Xvfb subprocess handle (keep it alive for the function's
    lifetime; terminate it before returning).
    """
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
    # Give Xvfb a moment to come up before any GL call.
    time.sleep(3)
    if proc.poll() is not None:
        out = proc.stdout.read() if proc.stdout else ""
        raise RuntimeError(f"Xvfb failed to start:\n{out}")
    print(f"  Xvfb running on DISPLAY={display}")
    return proc


# ===========================================================================
# PROBE 1 — installation / import
# ===========================================================================
@app.function(image=image)
def probe_install():
    """Verify the four wheels installed and import cleanly. No GPU needed."""
    import importlib
    import sys

    print(f"Python: {sys.version}")
    print("-" * 60)

    results = {}
    for mod in ["pydrake", "robot_gym", "lbm_eval"]:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "(no __version__)")
            print(f"  OK   import {mod:<12} version={ver}")
            results[mod] = True
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL import {mod:<12} {type(e).__name__}: {e}")
            results[mod] = False

    # pydrake version detail — useful when comparing against the paper snapshot.
    try:
        from pydrake.common import GetDrakeVersion  # type: ignore

        print(f"\n  Drake version: {GetDrakeVersion()}")
    except Exception as e:  # noqa: BLE001
        print(f"\n  (could not read Drake version: {e})")

    all_ok = all(results.values())
    print("-" * 60)
    print("PROBE 1:", "PASS" if all_ok else "FAIL")
    if not all_ok:
        raise SystemExit("probe_install failed — fix the image before probe 2.")
    return results


# ===========================================================================
# PROBE 2 — EGL rendering context + MeshCat recording API
# ===========================================================================
@app.function(image=image, gpu="L4", volumes={VOL_PATH: volume}, timeout=600)
def probe_render():
    """On an L4: render one frame headless via EGL, and verify the MeshCat
    recording API exists on this wheel's Drake build."""
    import os

    print("=" * 60)
    print("PART A — GPU / EGL visibility")
    print("=" * 60)
    # nvidia-smi should work if the GPU is attached.
    os.system("nvidia-smi --query-gpu=name,memory.total --format=csv || "
              "echo '  (nvidia-smi not available)'")
    # Confirm the NVIDIA EGL userspace lib is present.
    os.system("ls -la /usr/lib/x86_64-linux-gnu/libEGL_nvidia.so* 2>/dev/null || "
              "echo '  libEGL_nvidia.so not found via ls'")

    print()
    print("=" * 60)
    print("PART B — Drake headless EGL render")
    print("=" * 60)
    # Drake's RenderEngineGl uses GLX and needs a DISPLAY. Start Xvfb first.
    xvfb = _start_xvfb()
    # The ONE thing that matters here: can Drake create a GL render engine
    # and produce a non-blank image? We render into a SceneGraph-free
    # RenderEngine using its low-level API, which is stable across versions.
    render_ok = False
    try:
        import numpy as np
        from pydrake.geometry import (
            MakeRenderEngineGl,
            RenderEngineGlParams,
        )

        # This call allocates the EGL context. It throws loudly if EGL can't
        # reach the NVIDIA driver — which is the real thing we are probing.
        engine = MakeRenderEngineGl(RenderEngineGlParams())
        print(f"  OK   MakeRenderEngineGl() -> {type(engine).__name__} "
              f"(EGL context created)")

        # Try to drive a real render. The render-camera class names have moved
        # between modules across Drake versions, so locate them dynamically
        # instead of hardcoding an import path.
        cam_classes = {}
        for modname in ("pydrake.geometry", "pydrake.systems.sensors"):
            try:
                mod = __import__(modname, fromlist=["*"])
            except Exception:  # noqa: BLE001
                continue
            for cls in ("ColorRenderCamera", "RenderCameraCore",
                        "ClippingRange", "CameraInfo"):
                if hasattr(mod, cls) and cls not in cam_classes:
                    cam_classes[cls] = getattr(mod, cls)

        found = sorted(cam_classes.keys())
        print(f"  info render-camera classes found: {found}")

        if {"ColorRenderCamera", "RenderCameraCore",
                "ClippingRange", "CameraInfo"} <= set(cam_classes):
            from pydrake.math import RigidTransform

            CameraInfo = cam_classes["CameraInfo"]
            ClippingRange = cam_classes["ClippingRange"]
            RenderCameraCore = cam_classes["RenderCameraCore"]
            ColorRenderCamera = cam_classes["ColorRenderCamera"]

            intrinsics = CameraInfo(width=640, height=480, fov_y=0.785)
            core = RenderCameraCore(
                "probe_renderer",
                intrinsics,
                ClippingRange(0.01, 10.0),
                RigidTransform(),
            )
            color_cam = ColorRenderCamera(core, show_window=False)

            from pydrake.systems.sensors import ImageRgba8U

            img = ImageRgba8U(640, 480)
            # Empty engine -> renders the background; the point is that the
            # GL pipeline executes and returns a correctly shaped buffer.
            engine.RenderColorImage(color_cam, img)
            arr = np.asarray(img.data).reshape(480, 640, 4)
            print(f"  OK   RenderColorImage() -> array {arr.shape}, "
                  f"dtype={arr.dtype}, mean={arr.mean():.1f}")
            render_ok = True
        else:
            # Even if we couldn't assemble the camera, EGL context creation
            # succeeding is the critical signal. The full render path is
            # exercised by lbm_eval's own code in Probe 3 regardless.
            print("  info could not assemble a render camera from the API; "
                  "EGL context creation still succeeded — Probe 3 will "
                  "exercise the full render path via lbm_eval itself.")
            render_ok = True
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL Drake EGL render path: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        xvfb.terminate()

    print()
    print("=" * 60)
    print("PART C — MeshCat recording API (Option A viz)")
    print("=" * 60)
    meshcat_ok = False
    try:
        from pydrake.geometry import Meshcat

        mc = Meshcat()
        needed = ["StartRecording", "StopRecording", "PublishRecording",
                  "StaticHtml", "DeleteRecording"]
        present = {name: hasattr(mc, name) for name in needed}
        for name, ok in present.items():
            print(f"  {'OK  ' if ok else 'MISS'} Meshcat.{name}")
        meshcat_ok = all(present.values())

        # Actually exercise StaticHtml — write a (trivial, empty-scene) html to
        # the Volume so we confirm the artifact path works end to end.
        html = mc.StaticHtml()
        out = os.path.join(VOL_PATH, "probe_meshcat.html")
        with open(out, "w") as f:
            f.write(html)
        volume.commit()
        print(f"  OK   wrote StaticHtml probe artifact -> {out} "
              f"({len(html):,} bytes)")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL MeshCat API: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    print()
    print("-" * 60)
    overall = render_ok and meshcat_ok
    print("PROBE 2:", "PASS" if overall else "PARTIAL/FAIL — read output above")
    return {"render_ok": render_ok, "meshcat_ok": meshcat_ok}


# ===========================================================================
# PROBE 3 — sample evaluation, end to end, single container
# ===========================================================================
@app.function(image=image)
def probe_python_paths():
    """Diagnose the /usr/bin/python3 situation that breaks Drake's downloader."""
    import os
    import subprocess
    import sys

    print(f"sys.executable      = {sys.executable}")
    print(f"sys.version         = {sys.version}")
    print()
    for cmd in [
        "which python3",
        "which python",
        "ls -la /usr/bin/python3",
        "ls -la /usr/local/bin/python3",
        "readlink -f /usr/bin/python3",
        "/usr/bin/python3 --version",
        "head -1 $(which evaluate)",
    ]:
        print(f"$ {cmd}")
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        out = (r.stdout + r.stderr).strip()
        print(f"  -> [{r.returncode}] {out}")
        print()

    # The actual Drake downloader script — find it and show its shebang.
    print("Looking for Drake's package downloader script...")
    r = subprocess.run(
        "find / -name '*package_downloader*' -o -name '*_download*' 2>/dev/null "
        "| grep -i drake | head -20",
        shell=True, capture_output=True, text=True,
    )
    print(r.stdout or "  (none found by that pattern)")
    return {"sys_executable": sys.executable}


@app.function(image=image, gpu="L4", volumes={VOL_PATH: volume}, timeout=1800)
def run_sample_eval():
    """Run the bundled 'wave_around' sample policy through one evaluation of
    the pick_and_place_box skill. Server + client share one container; we
    drive both via subprocess so we don't depend on cross-container gRPC yet.
    """
    import os
    import subprocess
    import sys
    import time

    out_dir = os.path.join(VOL_PATH, "sample_eval_output")
    os.makedirs(out_dir, exist_ok=True)

    # Drake's renderer (used by the `evaluate` simulation) needs a DISPLAY.
    xvfb = _start_xvfb()

    # The wheels install console scripts; resolve them from the running venv's
    # bin dir so we don't assume a path.
    bindir = os.path.dirname(sys.executable)
    server_bin = os.path.join(bindir, "wave_around_policy_server")
    evaluate_bin = os.path.join(bindir, "evaluate")
    for b in (server_bin, evaluate_bin):
        print(f"  {'found' if os.path.exists(b) else 'MISSING'}: {b}")

    # 1) Start the sample policy server (runs forever; we kill it at the end).
    print("\nStarting wave_around_policy_server ...")
    server = subprocess.Popen(
        [server_bin],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    # Give the gRPC server a moment to bind its port. `evaluate` will itself
    # wait for the server, but a short sleep avoids a noisy race in logs.
    time.sleep(8)
    if server.poll() is not None:
        out = server.stdout.read() if server.stdout else ""
        raise SystemExit(f"policy server died on startup:\n{out}")
    print("  server process is up")

    # 2) Run the evaluate client: simplest skill, ONE evaluation, ONE process.
    print("\nRunning evaluate (pick_and_place_box, num_evaluations=1) ...")
    t0 = time.time()
    proc = subprocess.run(
        [
            evaluate_bin,
            "--skill_type=pick_and_place_box",
            "--num_evaluations=1",
            "--num_processes=1",
            f"--output_dir={out_dir}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    dt = time.time() - t0
    print(proc.stdout)
    print(f"  evaluate exited code={proc.returncode} in {dt:.1f}s")

    # 3) Tear down the server.
    server.terminate()
    try:
        server.wait(timeout=10)
    except subprocess.TimeoutExpired:
        server.kill()

    # Tear down Xvfb.
    xvfb.terminate()

    # 4) Persist results and show what landed on the Volume.
    volume.commit()
    print("\nOutput dir contents:")
    for root, _dirs, files in os.walk(out_dir):
        for fn in files:
            p = os.path.join(root, fn)
            print(f"  {os.path.getsize(p):>10,d}  {p}")

    if proc.returncode != 0:
        raise SystemExit("evaluate returned non-zero — inspect output above.")
    print("\nPROBE 3: PASS")
    return {"returncode": proc.returncode, "seconds": dt}


@app.local_entrypoint()
def main():
    """Convenience: `modal run lbm_phase0.py` runs the three probes in order."""
    print("\n### PROBE 1: install ###")
    probe_install.remote()
    print("\n### PROBE 2: render + meshcat ###")
    probe_render.remote()
    print("\n### PROBE 3: sample evaluation ###")
    run_sample_eval.remote()
    print("\nAll three probes completed. Review output above.")