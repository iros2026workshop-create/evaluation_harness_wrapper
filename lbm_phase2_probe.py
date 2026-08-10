"""
LBM Eval on Modal — Phase 2 data probe
======================================
Before writing any data loader or training code, confirm the ACTUAL shape of
the training data. TRAINING_DATA_FORMAT.md describes the 1.0 software and one
example episode; we train on 1.1 data, so we verify against ground truth
rather than against the doc.

Task chosen for the first trained policy: pick_and_place_box (pipeline
shakedown). Architecture: plain BC regression. This probe only inspects data.

What this does:
  1. Download PickAndPlaceBox.tar from the TRI public S3 bucket to the Volume
     (cached there so later loader/training runs don't re-download).
  2. Extract it, locate episodes (the nested tasks/.../episode_N layout).
  3. For a few episodes, load actions.npz / observations.npz / summary.npz and
     print shapes, dtypes, image resolution, episode length, success flag.
  4. Report the episode-length distribution across all episodes.

Run:
  modal run lbm_phase2_probe.py::probe_training_data

Decisions this resolves before the loader is written:
  - exact action array shape/dtype (doc says (T, 20))
  - which observations.npz keys exist, image resolution, RGB dtype/layout
  - episode count and length distribution (-> batching, action chunking)
  - how many episodes have episode_success=True (-> filter to good demos)
"""

import modal

APP_NAME = "lbm-eval-phase2-probe"
app = modal.App(APP_NAME)

# Reuse the same Volume as Phases 0/1.
volume = modal.Volume.from_name("lbm-eval-data", create_if_missing=True)
VOL_PATH = "/data"

# TRI public dataset. Per-skill tarballs; we only need the box task here.
DATA_BASE_URL = (
    "https://tri-ml-public.s3.amazonaws.com/datasets/"
    "lbm-eval-v1.1-sim-training-data/"
)
SKILL_TARBALL = "PickAndPlaceBox.tar"

# Where probe_training_data extracts the tarball (also used by the
# camera-match probe below, and matches lbm_phase2_train.py's EXTRACT_DIR).
EXTRACT_DIR = f"{VOL_PATH}/phase2_training_data/PickAndPlaceBox"

# A light image — this probe needs no GPU, no Drake, just numpy + a downloader.
probe_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "requests", "pyyaml")
)


@app.function(image=probe_image, volumes={VOL_PATH: volume}, timeout=3600)
def probe_training_data():
    import os
    import tarfile
    import time
    import numpy as np

    raw_dir = os.path.join(VOL_PATH, "phase2_training_data", "raw")
    extract_dir = os.path.join(VOL_PATH, "phase2_training_data", "PickAndPlaceBox")
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(extract_dir, exist_ok=True)
    tar_path = os.path.join(raw_dir, SKILL_TARBALL)

    # -- 1. Download (skip if already cached on the Volume) ----------------
    print("=" * 66)
    print("STEP 1 — download tarball")
    print("=" * 66)
    if os.path.exists(tar_path) and os.path.getsize(tar_path) > 0:
        print(f"  already cached: {tar_path} "
              f"({os.path.getsize(tar_path):,} bytes)")
    else:
        import requests

        url = DATA_BASE_URL + SKILL_TARBALL
        print(f"  GET {url}")
        t0 = time.time()
        with requests.get(url, stream=True, timeout=120) as r:
            print(f"  HTTP {r.status_code}")
            r.raise_for_status()
            total = 0
            with open(tar_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
                    total += len(chunk)
            print(f"  downloaded {total:,} bytes in {time.time()-t0:.1f}s")
        volume.commit()

    # -- 2. Extract --------------------------------------------------------
    print()
    print("=" * 66)
    print("STEP 2 — extract + locate episodes")
    print("=" * 66)
    already = [r for r, _d, f in os.walk(extract_dir) if "processed" in r]
    if already:
        print(f"  already extracted ({len(already)} processed dirs found)")
    else:
        with tarfile.open(tar_path) as tf:
            members = tf.getmembers()
            print(f"  tar contains {len(members)} members")
            tf.extractall(extract_dir)
        volume.commit()
        print("  extracted.")

    # An episode is any dir that directly contains a 'processed' subdir.
    episode_dirs = []
    for root, dirs, _files in os.walk(extract_dir):
        if "processed" in dirs:
            episode_dirs.append(root)
    episode_dirs.sort()
    print(f"  found {len(episode_dirs)} episode directories")
    if not episode_dirs:
        # Show the tree so we can see what the layout actually is.
        print("  !! no episodes found — dumping directory tree (depth 6):")
        os.system(f"find {extract_dir} -maxdepth 6 | head -60")
        raise SystemExit("could not locate episodes — inspect layout above.")

    # -- 3. Inspect a few episodes in detail -------------------------------
    print()
    print("=" * 66)
    print("STEP 3 — per-episode contents (first 3 episodes)")
    print("=" * 66)
    for ep in episode_dirs[:3]:
        proc = os.path.join(ep, "processed")
        print(f"\n  episode: {os.path.relpath(ep, extract_dir)}")
        print(f"    files: {sorted(os.listdir(proc))}")

        # actions.npz — doc says (T, 20): [R_xyz|R_rot6d|L_xyz|L_rot6d|RG|LG]
        ap = os.path.join(proc, "actions.npz")
        if os.path.exists(ap):
            a = np.load(ap, allow_pickle=True)
            print(f"    actions.npz keys: {list(a.keys())}")
            if "actions" in a:
                arr = a["actions"]
                print(f"      actions: shape={arr.shape} dtype={arr.dtype}")
                print(f"      actions[0]: {np.asarray(arr[0]).round(4)}")

        # summary.npz — episode_success is the demo-quality filter.
        sp = os.path.join(proc, "summary.npz")
        if os.path.exists(sp):
            s = np.load(sp, allow_pickle=True)
            print(f"    summary.npz keys: {list(s.keys())}")
            if "episode_success" in s:
                print(f"      episode_success: {s['episode_success']}")

        # observations.npz — find the RGB camera keys + resolution.
        op = os.path.join(proc, "observations.npz")
        if os.path.exists(op):
            o = np.load(op, allow_pickle=True)
            keys = list(o.keys())
            print(f"    observations.npz: {len(keys)} keys")
            # RGB camera keys: no _depth / _label suffix, and ndim==4 (T,H,W,C).
            rgb_keys, pose_keys = [], []
            for k in keys:
                if k.endswith("_depth") or k.endswith("_label"):
                    continue
                try:
                    arr = o[k]
                except Exception:
                    continue
                if getattr(arr, "ndim", 0) == 4 and arr.shape[-1] in (3, 4):
                    rgb_keys.append((k, arr.shape, arr.dtype))
                if "poses" in k and "xyz" in k:
                    pose_keys.append((k, arr.shape, arr.dtype))
            print("      RGB camera arrays:")
            for k, shp, dt in rgb_keys:
                print(f"        {k:<28} shape={shp} dtype={dt}")
            print("      end-effector pose arrays:")
            for k, shp, dt in pose_keys:
                print(f"        {k:<40} shape={shp} dtype={dt}")

    # -- 4. Episode-length + success distribution across ALL episodes ------
    print()
    print("=" * 66)
    print("STEP 4 — distribution across all episodes")
    print("=" * 66)
    lengths, n_success, n_with_flag = [], 0, 0
    for ep in episode_dirs:
        proc = os.path.join(ep, "processed")
        ap = os.path.join(proc, "actions.npz")
        sp = os.path.join(proc, "summary.npz")
        try:
            a = np.load(ap, allow_pickle=True)
            lengths.append(int(a["actions"].shape[0]))
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {ep}: {e})")
            continue
        try:
            s = np.load(sp, allow_pickle=True)
            if "episode_success" in s:
                n_with_flag += 1
                if bool(np.asarray(s["episode_success"]).ravel()[0]):
                    n_success += 1
        except Exception:  # noqa: BLE001
            pass

    if lengths:
        L = np.array(lengths)
        print(f"  episodes inspected : {len(L)}")
        print(f"  episode length     : min={L.min()} max={L.max()} "
              f"mean={L.mean():.1f} median={int(np.median(L))}")
        print(f"  total keyframes    : {L.sum():,}  "
              f"(~training samples for a per-step BC policy)")
        print(f"  episode_success    : {n_success}/{n_with_flag} True "
              f"(filter to these for BC)")

    print()
    print("-" * 66)
    print("PROBE COMPLETE — paste this output; loader is designed against it.")
    return {
        "n_episodes": len(episode_dirs),
        "lengths": lengths,
        "n_success": n_success,
    }


# ===========================================================================
# CAMERA-MATCH PROBE — match training-camera SERIALS to live SEMANTIC names
# ===========================================================================
# The live eval harness keys cameras by semantic name (scene_right_0, ...);
# the training data keys them by hardware serial (6CD146030E99, ...). To train
# the BC policy on a camera we can identify with certainty at eval time, we
# match by intrinsics K (and extrinsics X_TC) — a physical fingerprint, not a
# guessed name.
#
# Live values observed in the visuo probe, for reference when reading output:
#   scene_right_0 : K fx=616.129 fy=615.758 cx=321.269 cy=247.864  (480x640)
#                   X_TC p=[0.4925, 0.2626, 0.9073]
# Paste this probe's output next to those to read off the correspondence.

# Hard-coded live camera fingerprints from the visuo probe dump. Extend this
# dict if a future visuo probe prints K/X_TC for the other 5 cameras.
LIVE_CAMERAS = {
    "scene_right_0": {
        "K": [[616.129, 0.0, 321.269],
              [0.0, 615.758, 247.864],
              [0.0, 0.0, 1.0]],
        "p": [0.4925198648813639, 0.26256949292330367, 0.9073075406104864],
    },
}


@app.function(image=probe_image, volumes={VOL_PATH: volume}, timeout=900)
def probe_camera_match():
    """Load one training episode's intrinsics.npz / extrinsics.npz, print each
    serial-numbered camera's K and pose, and score them against the known live
    cameras so we can pick the correct training camera for the BC policy."""
    import os
    import numpy as np

    if not os.path.isdir(EXTRACT_DIR):
        raise SystemExit(f"{EXTRACT_DIR} missing — run probe_training_data first.")

    # Use the first episode that has both npz files.
    ep = None
    for root, dirs, _f in os.walk(EXTRACT_DIR):
        if "processed" in dirs:
            proc = os.path.join(root, "processed")
            if (os.path.exists(os.path.join(proc, "intrinsics.npz"))
                    or os.path.exists(os.path.join(proc, "extrinsics.npz"))):
                ep = proc
                break
    if ep is None:
        raise SystemExit("no episode with intrinsics/extrinsics found.")
    print(f"using episode: {ep}\n")

    def dump_npz(path):
        if not os.path.exists(path):
            print(f"  ({os.path.basename(path)} not present)")
            return {}
        z = np.load(path, allow_pickle=True)
        print(f"  {os.path.basename(path)} keys: {list(z.keys())}")
        out = {}
        for k in z.keys():
            v = z[k]
            out[k] = v
            # numpy may have wrapped a dict/object in a 0-d array.
            if getattr(v, "shape", None) == () and v.dtype == object:
                inner = v.item()
                print(f"    {k}: 0-d object -> {type(inner).__name__}")
                if isinstance(inner, dict):
                    print(f"      inner keys: {list(inner.keys())}")
                out[k] = inner
            else:
                print(f"    {k}: shape={getattr(v,'shape',None)} "
                      f"dtype={getattr(v,'dtype',None)}")
        return out

    print("=" * 66)
    print("intrinsics.npz")
    print("=" * 66)
    intr = dump_npz(os.path.join(ep, "intrinsics.npz"))
    print()
    print("=" * 66)
    print("extrinsics.npz")
    print("=" * 66)
    extr = dump_npz(os.path.join(ep, "extrinsics.npz"))

    # -- pull a per-camera K out of whatever structure intrinsics has ------
    # Try: dict keyed by camera; or parallel arrays; or object array.
    def extract_per_camera(d):
        """Return {camera_name: np.ndarray} from a loosely-typed npz dict."""
        cams = {}
        for k, v in d.items():
            if isinstance(v, dict):
                for ck, cv in v.items():
                    cams[str(ck)] = np.asarray(cv)
            elif hasattr(v, "shape") and v.ndim >= 2:
                cams[str(k)] = np.asarray(v)
        return cams

    intr_cams = extract_per_camera(intr)
    extr_cams = extract_per_camera(extr)
    print()
    print("=" * 66)
    print("per-camera values found")
    print("=" * 66)
    print(f"  intrinsics cameras: {list(intr_cams.keys())}")
    print(f"  extrinsics cameras: {list(extr_cams.keys())}")
    for name, K in intr_cams.items():
        if K.shape[-2:] == (3, 3):
            fx, fy = K[..., 0, 0], K[..., 1, 1]
            cx, cy = K[..., 0, 2], K[..., 1, 2]
            print(f"  {name}: fx={np.ravel(fx)[0]:.3f} fy={np.ravel(fy)[0]:.3f} "
                  f"cx={np.ravel(cx)[0]:.3f} cy={np.ravel(cy)[0]:.3f}")
        else:
            print(f"  {name}: K shape {K.shape} (not 3x3 — inspect manually)")

    # -- score each training camera against each known live camera --------
    print()
    print("=" * 66)
    print("MATCH SCORING — training serial vs live semantic camera")
    print("=" * 66)
    for live_name, live in LIVE_CAMERAS.items():
        liveK = np.array(live["K"])
        print(f"\n  live '{live_name}'  fx={liveK[0,0]:.3f} cx={liveK[0,2]:.3f}")
        best = None
        for name, K in intr_cams.items():
            if K.shape[-2:] != (3, 3):
                continue
            K2 = np.asarray(K).reshape(-1, 3, 3)[0]
            kdiff = float(np.abs(K2 - liveK).max())
            line = f"    {name:<20} |K diff|max={kdiff:.4f}"
            if best is None or kdiff < best[1]:
                best = (name, kdiff)
            print(line)
        if best is not None:
            verdict = "STRONG MATCH" if best[1] < 1.0 else "no close match"
            print(f"    -> closest: {best[0]}  ({verdict})")

    print()
    print("-" * 66)
    print("Read the closest-match camera; that serial is what bc_v1 should")
    print("train on so the eval-time camera is provably the same one.")
    return {"intr_cameras": list(intr_cams.keys()),
            "extr_cameras": list(extr_cams.keys())}


@app.local_entrypoint()
def main():
    probe_training_data.remote()


@app.local_entrypoint()
def camera_match():
    probe_camera_match.remote()


# ===========================================================================
# ROT6D CONVENTION PROBE — columns-first vs rows-first
# ===========================================================================
# The trainer read precomputed `rot_6d` arrays from observations.npz. The BC
# SERVER must recompute rot6d from a live RigidTransform the SAME way. The 6D
# representation has a convention choice (first two COLUMNS of R vs first two
# ROWS). This probe finds a rotation-matrix source and the matching `rot_6d`
# for the same frames, and reports which convention reproduces the stored
# values — so the server is built on fact, not a 50/50 guess.
@app.function(image=probe_image, volumes={VOL_PATH: volume}, timeout=900)
def probe_rot6d_convention():
    import os
    import numpy as np

    if not os.path.isdir(EXTRACT_DIR):
        raise SystemExit(f"{EXTRACT_DIR} missing — run probe_training_data first.")

    ep = None
    for root, dirs, _f in os.walk(EXTRACT_DIR):
        if "processed" in dirs:
            ep = os.path.join(root, "processed")
            break
    if ep is None:
        raise SystemExit("no episode found.")
    print(f"episode: {ep}\n")

    obs = np.load(os.path.join(ep, "observations.npz"), allow_pickle=True)
    keys = list(obs.keys())

    # Find the stored rot_6d and a rotation source for the SAME arm/frames.
    rot6d_keys = [k for k in keys if "rot_6d" in k]
    print(f"rot_6d keys: {rot6d_keys}")
    # Candidate rotation-matrix or quaternion keys for the same poses.
    rot_src_keys = [k for k in keys
                    if ("poses" in k and ("rot" in k or "quat" in k
                                          or "matrix" in k))
                    and "rot_6d" not in k]
    print(f"other rotation keys on poses: {rot_src_keys}")
    print(f"\nall pose-related keys:")
    for k in keys:
        if "pose" in k.lower():
            print(f"  {k}: shape={np.asarray(obs[k]).shape}")

    if not rot6d_keys:
        raise SystemExit("no rot_6d key found — inspect key list above.")

    # Take the right arm's stored rot_6d.
    r6_key = next((k for k in rot6d_keys if "right" in k), rot6d_keys[0])
    stored = np.asarray(obs[r6_key], dtype=np.float64)   # (T,6)
    print(f"\nusing rot_6d key: {r6_key}  shape={stored.shape}")
    print(f"stored rot_6d[0] = {stored[0].round(4)}")

    # Look for a 3x3 / 4x4 / quaternion companion for the same arm.
    arm = "right" if "right" in r6_key else "left"
    companions = [k for k in keys if arm in k and "pose" in k.lower()
                  and "rot_6d" not in k and "xyz" not in k]
    print(f"\ncompanion rotation keys for arm '{arm}': {companions}")
    for k in companions:
        v = np.asarray(obs[k])
        print(f"  {k}: shape={v.shape} dtype={v.dtype}")
        # If it's a 3x3 or 4x4 per frame, test both 6D conventions.
        if v.ndim == 3 and v.shape[-2:] in ((3, 3), (4, 4)):
            R = v[0][:3, :3].astype(np.float64)
            cols = np.concatenate([R[:, 0], R[:, 1]])
            rows = np.concatenate([R[0, :], R[1, :]])
            d_cols = float(np.abs(cols - stored[0]).max())
            d_rows = float(np.abs(rows - stored[0]).max())
            print(f"    columns-first vs stored: |diff|max={d_cols:.5f}")
            print(f"    rows-first    vs stored: |diff|max={d_rows:.5f}")
            if min(d_cols, d_rows) < 1e-3:
                winner = "COLUMNS-first" if d_cols < d_rows else "ROWS-first"
                print(f"    -> 6D convention is {winner}")
            else:
                print("    -> neither matches; inspect this key's layout")

    print()
    print("-" * 60)
    print("Use the matching convention in the server's _rot6d_from_matrix /")
    print("_matrix_from_rot6d. If no 3x3 companion exists, the server can")
    print("instead read the same observation keys the trainer used.")
    return {"rot6d_keys": rot6d_keys, "companions": companions}


@app.local_entrypoint()
def rot6d():
    probe_rot6d_convention.remote()


# ===========================================================================
# TRANSFORMS PROBE — resolve the 6D convention via transforms.npz
# ===========================================================================
# observations.npz stores rotation ONLY as rot_6d (no matrix companion), so
# the convention can't be checked there. transforms.npz (present in every
# episode, never opened) likely holds the end-effector poses as full matrices
# / quaternions for the same frames. This probe opens it and, if it carries
# per-frame rotations, tests both 6D conventions against the known
# right-arm rot_6d[0] = [0.6624, 0.7465, 0.0628, 0.7485, -0.663, -0.0143].
KNOWN_RIGHT_ROT6D_0 = [0.6624, 0.7465, 0.0628, 0.7485, -0.663, -0.0143]


@app.function(image=probe_image, volumes={VOL_PATH: volume}, timeout=900)
def probe_transforms():
    import os
    import numpy as np

    if not os.path.isdir(EXTRACT_DIR):
        raise SystemExit(f"{EXTRACT_DIR} missing — run probe_training_data first.")

    ep = None
    for root, dirs, _f in os.walk(EXTRACT_DIR):
        if "processed" in dirs:
            ep = os.path.join(root, "processed")
            break
    if ep is None:
        raise SystemExit("no episode found.")
    tpath = os.path.join(ep, "transforms.npz")
    if not os.path.exists(tpath):
        raise SystemExit(f"{tpath} not present — list processed/ contents.")
    print(f"episode: {ep}\n")

    z = np.load(tpath, allow_pickle=True)
    keys = list(z.keys())
    print("=" * 64)
    print(f"transforms.npz — {len(keys)} keys")
    print("=" * 64)

    def unwrap(v):
        """Unwrap a 0-d object array if numpy boxed a dict/object."""
        if getattr(v, "shape", None) == () and getattr(v, "dtype", None) == object:
            return v.item()
        return v

    # Dump structure.
    flat = {}
    for k in keys:
        v = unwrap(z[k])
        if isinstance(v, dict):
            print(f"  {k}: dict keys={list(v.keys())}")
            for ik, iv in v.items():
                iv = np.asarray(iv)
                print(f"    [{ik}]: shape={iv.shape} dtype={iv.dtype}")
                flat[f"{k}/{ik}"] = iv
        else:
            v = np.asarray(v)
            print(f"  {k}: shape={v.shape} dtype={v.dtype}")
            flat[k] = v

    # Find a per-frame 3x3/4x4 transform for the RIGHT end-effector.
    print()
    print("=" * 64)
    print("6D CONVENTION TEST — against known right-arm rot_6d[0]")
    print("=" * 64)
    known = np.array(KNOWN_RIGHT_ROT6D_0, dtype=np.float64)
    print(f"  known rot_6d[0] = {known.round(4)}\n")

    tested = False
    for name, v in flat.items():
        v = np.asarray(v)
        # Per-frame 3x3 or 4x4 matrices.
        is_mat = (v.ndim == 3 and v.shape[-2:] in ((3, 3), (4, 4)))
        if not is_mat:
            continue
        # Prefer keys that look like the right end-effector / panda.
        hint = any(t in name.lower()
                   for t in ("right", "panda", "ee", "tcp", "gripper",
                             "hand", "actual"))
        R = v[0][:3, :3].astype(np.float64)
        cols = np.concatenate([R[:, 0], R[:, 1]])
        rows = np.concatenate([R[0, :], R[1, :]])
        d_cols = float(np.abs(cols - known).max())
        d_rows = float(np.abs(rows - known).max())
        tag = "  <- right-arm-ish" if hint else ""
        print(f"  {name}: |diff|cols={d_cols:.5f}  |diff|rows={d_rows:.5f}{tag}")
        if min(d_cols, d_rows) < 1e-3:
            tested = True
            winner = "COLUMNS-first" if d_cols < d_rows else "ROWS-first"
            print(f"    *** MATCH — 6D convention is {winner} ***")

    if not tested:
        print("\n  No per-frame matrix matched. transforms.npz may store")
        print("  something else (camera/world transforms). Full key dump is")
        print("  above — we pick the resolution path from it.")

    print()
    print("-" * 64)
    print("If a convention matched: set the server's _rot6d_from_matrix /")
    print("_matrix_from_rot6d to that convention. Done.")
    return {"keys": keys, "resolved": tested}


@app.local_entrypoint()
def transforms():
    probe_transforms.remote()


# ===========================================================================
# ORTHONORMALITY PROBE — resolve 6D convention with NO new files
# ===========================================================================
# transforms.npz is empty and observations.npz has no matrix companion, so we
# cannot match rot_6d against a known matrix. But a valid 6D rotation encoding
# is orthonormal in its INTENDED reading: the two stored 3-vectors are a
# unit-length, mutually-orthogonal pair. Under the WRONG (transposed) reading
# they generally are not. This probe reads many stored rot_6d vectors and
# measures, for each interpretation, how close the two 3-vectors are to
# orthonormal. The clean one is the real convention.
@app.function(image=probe_image, volumes={VOL_PATH: volume}, timeout=900)
def probe_rot6d_orthonormality():
    import os
    import numpy as np

    if not os.path.isdir(EXTRACT_DIR):
        raise SystemExit(f"{EXTRACT_DIR} missing — run probe_training_data first.")

    # Gather rot_6d from several episodes for a robust measurement.
    procs = []
    for root, dirs, _f in os.walk(EXTRACT_DIR):
        if "processed" in dirs:
            procs.append(os.path.join(root, "processed"))
    procs.sort()

    vecs = []
    for proc in procs[:20]:
        try:
            o = np.load(os.path.join(proc, "observations.npz"),
                        allow_pickle=True)
            for k in o.keys():
                if "rot_6d" in k and "actual" in k:
                    vecs.append(np.asarray(o[k], dtype=np.float64))
        except Exception:  # noqa: BLE001
            pass
    if not vecs:
        raise SystemExit("no rot_6d arrays found.")
    V = np.concatenate(vecs, axis=0)        # (M, 6)
    print(f"measuring {V.shape[0]} stored rot_6d vectors\n")

    def orthonormality_error(pairs):
        """pairs: (M,2,3). Returns mean |unit-norm error| and |dot| — both
        near 0 for a genuine orthonormal pair."""
        a, b = pairs[:, 0, :], pairs[:, 1, :]
        na = np.abs(np.linalg.norm(a, axis=1) - 1.0).mean()
        nb = np.abs(np.linalg.norm(b, axis=1) - 1.0).mean()
        dot = np.abs((a * b).sum(axis=1)).mean()
        return (na + nb) / 2, dot

    # Interpretation A — the 6 numbers are two stacked 3-vectors as given
    # (this is what both "columns" and "rows" reduce to: the stored layout IS
    # [v1(3), v2(3)]; the question is only whether v1,v2 are orthonormal).
    pairs = V.reshape(-1, 2, 3)
    norm_err, dot_err = orthonormality_error(pairs)
    print("stored layout [v1(3), v2(3)] as-is:")
    print(f"  mean unit-norm error : {norm_err:.6f}")
    print(f"  mean |v1 . v2|       : {dot_err:.6f}")
    as_is_ok = norm_err < 1e-3 and dot_err < 1e-3

    # Interpretation B — the 6 numbers are interleaved [x1,x2,x3 | y1,y2,y3]
    # vs [x1,y1 | x2,y2 | x3,y3]. Test the interleaved reading too.
    inter = np.stack([V[:, 0::2], V[:, 1::2]], axis=1)   # (M,2,3)
    norm_err_i, dot_err_i = orthonormality_error(inter)
    print("\ninterleaved layout [v[0::2], v[1::2]]:")
    print(f"  mean unit-norm error : {norm_err_i:.6f}")
    print(f"  mean |v1 . v2|       : {dot_err_i:.6f}")
    inter_ok = norm_err_i < 1e-3 and dot_err_i < 1e-3

    print()
    print("-" * 64)
    if as_is_ok and not inter_ok:
        print("VERDICT: stored rot_6d is [v1(3), v2(3)] — two orthonormal")
        print("3-vectors stacked. The server's _rot6d_from_matrix (concatenate")
        print("R[:,0], R[:,1]) and _matrix_from_rot6d (Gram-Schmidt) are")
        print("CORRECT as written. Proceed to bc_eval.")
    elif inter_ok and not as_is_ok:
        print("VERDICT: stored rot_6d is INTERLEAVED. The server must")
        print("de/interleave — _rot6d_from_matrix and _matrix_from_rot6d")
        print("need adjusting before bc_eval.")
    elif as_is_ok and inter_ok:
        print("VERDICT: ambiguous (both readings orthonormal) — unlikely;")
        print("inspect the raw numbers.")
    else:
        print("VERDICT: neither reading is cleanly orthonormal. The 6D may")
        print("be a non-Gram-Schmidt encoding; inspect raw values.")
    print("Note: column-vs-row of R is moot for the proprio channel as long")
    print("as the server uses ONE consistent map; what matters is the stacked")
    print("vs interleaved layout, which this probe resolves.")
    return {"as_is_ok": bool(as_is_ok), "inter_ok": bool(inter_ok)}


@app.local_entrypoint()
def ortho():
    probe_rot6d_orthonormality.remote()