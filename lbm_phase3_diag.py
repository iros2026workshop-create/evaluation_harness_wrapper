"""
LBM Eval on Modal — Phase 3 diagnostic: why does served BC v1 stutter in place?
===============================================================================
Phase 3 measured BC v1 at 0/195 task-competent successes. The failure
recordings show the right arm stuttering near its start pose FROM THE FIRST
FRAME, visually similar across different scenes. Two hypotheses explain a low
rollout success rate, and they demand opposite responses:

  (H1) Genuine covariate shift / data scarcity. The model learned the training
       distribution but can't generalize. -> Expected; diffusion (v2) is the
       answer; 0/195 is a real, publishable finding.
  (H2) A serve-side input mismatch feeds the net out-of-distribution
       observations, so it mean-regresses. -> A BUG; 0/195 is an artifact and
       the run must be repeated after the fix.

The training number already argues against H1's "mean-regression" form: the
val MSE of 0.0149 is in NORMALIZED action space, where a mean-ignoring-inputs
predictor scores ~1.0. 0.0149 is ~67x better than the mean predictor, so on
its TRAINING distribution the net clearly conditions on inputs. Combined with
first-frame collapse in sim (the first served frame is the one guaranteed
on-distribution state), that points at H2. These probes confirm it and
localize the fault.

PROBE 1 — probe_model_on_cache  (no sim, no Xvfb)
  Runs the checkpoint on the cached TRAINING tensors (exactly what training
  fed it). Reports, in normalized space:
    - val MSE          : should reproduce ~0.0149 (validates the probe is
                         loading + normalizing faithfully)
    - mean-baseline MSE: predicting the mean; ~1.0 by construction
    - per-dim pred std vs target std: the direct mean-regression test
  If the model reproduces ~0.0149 and predicts with input-driven variance,
  the model is FINE on training inputs -> the fault is serve-side -> run
  PROBE 2. If it mean-regresses on its OWN cached inputs, H1 holds and 0/195
  is real.

PROBE 2 — probe_serve_inputs  (sim + Xvfb, in-process; gRPC not needed for a
  diagnostic, and serve-time extraction is identical in-process vs gRPC)
  Runs ONE eval with an instrumented policy that, on the first few steps,
  captures the SERVED proprio vector and SERVED image channel means using the
  EXACT serve-time extraction code, and compares them against the cached
  TRAINING distribution (proprio_mean/std, per-channel image means). Flags:
    - served proprio dims outside training [mean +/- 4 std]  -> proprio OOD
      (prime suspect: rot6d convention mismatch, train reads stored rot_6d,
       serve recomputes columns-of-R)
    - served image channel-mean order swapped vs training    -> RGB/BGR swap

Run:
  modal run lbm_phase3_diag.py::probe_model_on_cache
  modal run lbm_phase3_diag.py::probe_serve_inputs
"""

import modal

APP_NAME = "lbm-eval-phase3-diag"
app = modal.App(APP_NAME)

volume = modal.Volume.from_name("lbm-eval-data", create_if_missing=True)
VOL_PATH = "/data"

CACHE_PATH = f"{VOL_PATH}/phase2_training_data/processed_bc_v1.pt"
CKPT_PATH = f"{VOL_PATH}/phase2_models/bc_v1.pt"

# Serve-time interface constants (from lbm_phase2_serve.py).
SCENE_CAMERA_LIVE = "scene_right_0"   # = training serial 6CD146030E99
ARM_KEYS = ["right::panda", "left::panda"]

# Wheels/libs only PROBE 2 needs (the sim). PROBE 1 uses the light image.
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

# PROBE 1 image: just torch (load cache + checkpoint, run forward).
light_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy", "torch==2.3.0", "torchvision==0.18.0")
)

# PROBE 2 image: the full sim image (same recipe as Phase 2/3).
sim_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install(*SYSTEM_LIBS)
    .pip_install(*WHEEL_URLS)
    .pip_install("numpy", "torch==2.3.0", "torchvision==0.18.0")
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
        "evaluate --help > /dev/null && echo 'evaluate --help: OK'",
    )
)


def _build_model(proprio_dim, action_dim):
    """Same architecture as lbm_phase2_train._build_model (weights=None; the
    state_dict is loaded from the checkpoint)."""
    import torch.nn as nn
    import torchvision

    class BCHead(nn.Module):
        def __init__(self, in_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, action_dim),
            )

        def forward(self, feats):
            return self.net(feats)

    class BCPolicyNet(nn.Module):
        def __init__(self, proprio_dim, action_dim):
            super().__init__()
            backbone = torchvision.models.resnet18(weights=None)
            self.img_dim = backbone.fc.in_features
            backbone.fc = nn.Identity()
            self.encoder = backbone
            self.proprio_mlp = nn.Sequential(
                nn.Linear(proprio_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
            )
            self.head = BCHead(self.img_dim + 128, action_dim)

        def forward(self, img, proprio):
            import torch
            feats = self.encoder(img)
            p = self.proprio_mlp(proprio)
            return self.head(torch.cat([feats, p], dim=1))

    return BCPolicyNet(proprio_dim, action_dim)


# ===========================================================================
# PROBE 1 — model on its own cached training inputs (no sim)
# ===========================================================================
@app.function(image=light_image, gpu="L4", volumes={VOL_PATH: volume},
              timeout=1800)
def probe_model_on_cache():
    import os
    import pickle
    import numpy as np
    import torch

    for p in (CACHE_PATH, CKPT_PATH):
        if not os.path.exists(p):
            raise SystemExit(f"{p} missing.")

    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    model = _build_model(ckpt["proprio_dim"], ckpt["action_dim"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Rebuild the EXACT training tensors + the EXACT train/val split (seed 0,
    # val_frac 0.15) so the val MSE here is directly comparable to the 0.0149
    # reported by lbm_phase2_train.train.
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs = torch.from_numpy(data["images"]).float().permute(0, 3, 1, 2) / 255.0
    imgs = (imgs - imagenet_mean) / imagenet_std
    p_mean = torch.from_numpy(data["proprio_mean"])
    p_std = torch.from_numpy(data["proprio_std"])
    a_mean = torch.from_numpy(data["action_mean"])
    a_std = torch.from_numpy(data["action_std"])
    proprio = (torch.from_numpy(data["proprios"]) - p_mean) / p_std
    action = (torch.from_numpy(data["actions"]) - a_mean) / a_std   # normalized

    n = imgs.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * 0.15)
    val_idx = perm[:n_val]
    print(f"samples={n}  val={n_val}  proprio_dim={ckpt['proprio_dim']}")

    # Forward over val in minibatches.
    preds = []
    with torch.no_grad():
        for i in range(0, len(val_idx), 256):
            b = val_idx[i:i + 256]
            out = model(imgs[b].to(device), proprio[b].to(device)).cpu()
            preds.append(out)
    preds = torch.cat(preds, 0)                       # normalized predictions
    targ = action[val_idx]                            # normalized targets

    val_mse = torch.mean((preds - targ) ** 2).item()
    # Mean predictor in normalized space = predict 0 -> MSE = E[targ^2].
    mean_mse = torch.mean(targ ** 2).item()
    # Mean-regression test: per-dim std of predictions vs targets.
    pred_std = preds.std(0)
    targ_std = targ.std(0)
    std_ratio = (pred_std / (targ_std + 1e-8))

    print("\n" + "=" * 64)
    print("PROBE 1 — model on cached TRAINING inputs (normalized space)")
    print("=" * 64)
    print(f"  val MSE (reproduce ~0.0149) : {val_mse:.4f}")
    print(f"  mean-predictor MSE (~1.0)   : {mean_mse:.4f}")
    print(f"  variance explained vs mean  : {100*(1-val_mse/mean_mse):.1f}%")
    print(f"  pred std / target std       : "
          f"mean={std_ratio.mean():.2f}  min={std_ratio.min():.2f}  "
          f"max={std_ratio.max():.2f}")
    print("  per-dim pred/target std ratio:")
    labels = (["R_xyz"] * 3 + ["R_rot6d"] * 6 + ["L_xyz"] * 3
              + ["L_rot6d"] * 6 + ["RG", "LG"])
    for d in range(preds.shape[1]):
        print(f"    [{d:2d}] {labels[d]:8s} ratio={std_ratio[d]:.2f}")

    collapses = bool(std_ratio.mean() < 0.25)
    reproduces = bool(val_mse < 3 * 0.0149)
    print("-" * 64)
    if reproduces and not collapses:
        print("VERDICT: model reproduces training-val MSE and predicts with "
              "input-driven variance.\n  -> The model is FINE on training "
              "inputs. The sim collapse is therefore a SERVE-SIDE input "
              "mismatch.\n  -> Run probe_serve_inputs to localize it.")
    elif collapses:
        print("VERDICT: model output has near-zero variance on its OWN "
              "training inputs.\n  -> H1 holds: the policy genuinely "
              "mean-regressed in training. 0/195 is real;\n     MSE-BC on ~4k "
              "samples can't solve the task. Diffusion (v2) is the path.")
    else:
        print("VERDICT: model does NOT reproduce the reported val MSE on the "
              "cache.\n  -> The probe, checkpoint, or cache disagree; resolve "
              "that before trusting either hypothesis.")
    print("=" * 64)
    return {"val_mse": val_mse, "mean_mse": mean_mse,
            "std_ratio_mean": float(std_ratio.mean())}


# ===========================================================================
# PROBE 2 — fidelity of SERVED inputs vs the training distribution (sim)
# ===========================================================================
@app.function(image=sim_image, gpu="L4", volumes={VOL_PATH: volume},
              timeout=1800)
def probe_serve_inputs(scenario_index: int = 0, n_steps: int = 3,
                       dump_npz: bool = False):
    import os
    import pickle
    import numpy as np

    if not os.path.exists(CACHE_PATH):
        raise SystemExit(f"{CACHE_PATH} missing.")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    # Training distribution references (numpy).
    p_mean = np.asarray(data["proprio_mean"], dtype=np.float32)
    p_std = np.asarray(data["proprio_std"], dtype=np.float32)
    # Per-channel mean of the cached TRAINING images (uint8 RGB), for the
    # channel-order / brightness comparison.
    train_img_chan_mean = data["images"].reshape(-1, 3).mean(0)  # (3,) R,G,B

    # --- serve-time extraction, copied verbatim from lbm_phase2_serve.py ----
    imagenet_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    imagenet_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def _rot6d_from_matrix(R):
        R = np.asarray(R, dtype=np.float32)
        return np.concatenate([R[:, 0], R[:, 1]]).astype(np.float32)

    def _extract_proprio(observation):
        poses = observation.robot.actual.poses
        parts = []
        for arm in ARM_KEYS:
            X = poses[arm]
            xyz = np.asarray(X.translation(), dtype=np.float32)
            rot6d = _rot6d_from_matrix(X.rotation().matrix())
            parts.append(xyz)
            parts.append(rot6d)
        return np.concatenate(parts).astype(np.float32)

    def _extract_image_raw(observation):
        rgb = observation.visuo[SCENE_CAMERA_LIVE].rgb.array  # (480,640,3) u8
        return rgb[::2, ::2, :]                                # (240,320,3) u8

    def _start_xvfb(display=":99"):
        import subprocess
        import time

        proc = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1280x1024x24", "-ac",
             "+extension", "GLX"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        os.environ["DISPLAY"] = display
        time.sleep(3)
        if proc.poll() is not None:
            raise RuntimeError("Xvfb failed to start")
        return proc

    captured = []

    from robot_gym.policy import Policy, PolicyMetadata
    from robot_gym.multiarm_spaces import PosesAndGrippers

    class CaptureProbePolicy(Policy):
        """Captures served proprio + served image channel means for the first
        n_steps, then holds the arms still so the episode completes."""

        def __init__(self):
            self._step = 0

        def reset(self, seed=None, options=None):
            self._step = 0

        def get_policy_metadata(self):
            return PolicyMetadata(
                name="CaptureProbe", skill_type="pick_and_place_box",
                checkpoint_path="None", git_repo="lbm_eval", git_sha="diag")

        def step(self, observation):
            import copy

            if self._step < n_steps:
                proprio = _extract_proprio(observation)
                raw_img = _extract_image_raw(observation)
                captured.append({
                    "step": self._step,
                    "proprio": proprio,
                    "img_chan_mean": raw_img.reshape(-1, 3).mean(0),
                    "img_shape": raw_img.shape,
                    "img_dtype": str(raw_img.dtype),
                })
            self._step += 1
            poses = copy.deepcopy(observation.robot.actual.poses)
            grippers = copy.deepcopy(observation.robot.actual.grippers)
            return PosesAndGrippers(poses=poses, grippers=grippers)

    xvfb = _start_xvfb()
    from lbm_eval.evaluate import evaluate_one
    from pathlib import Path

    out_dir = Path(VOL_PATH) / "phase3_diag_serve"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"running one eval (scenario_index={scenario_index}), capturing "
          f"first {n_steps} served observations ...")
    evaluate_one(
        skill_type="pick_and_place_box",
        scenario_index=scenario_index,
        output_directory=out_dir,
        policy=CaptureProbePolicy(),
        use_rpc=False,
    )
    xvfb.terminate()

    # --- compare served inputs to the training distribution ----------------
    labels = (["R_xyz0", "R_xyz1", "R_xyz2"]
              + [f"R_r6d{i}" for i in range(6)]
              + ["L_xyz0", "L_xyz1", "L_xyz2"]
              + [f"L_r6d{i}" for i in range(6)])

    if dump_npz:
        raw = np.stack([c["proprio"] for c in captured])            # (T, 18)
        std_naive = (data["proprios"].std(axis=0) + 1e-6).astype(np.float32)
        npz_path = os.path.join(
            VOL_PATH, "phase3_diag_serve",
            f"serve_proprio_raw_s{scenario_index}.npz")
        np.savez(npz_path, raw=raw, proprio_mean=p_mean,
                 std_floored=p_std, std_naive=std_naive,
                 labels=np.array(labels))
        volume.commit()
        print(f"[dump] wrote {npz_path}  raw={raw.shape}")

    print("\n" + "=" * 72)
    print("PROBE 2 — served inputs vs training distribution")
    print("=" * 72)
    print(f"training image channel mean (R,G,B): "
          f"[{train_img_chan_mean[0]:.1f}, {train_img_chan_mean[1]:.1f}, "
          f"{train_img_chan_mean[2]:.1f}]")
    if not captured:
        print("  !! no observations captured (episode produced no steps?)")
        volume.commit()
        return {"captured": 0}

    c0 = captured[0]
    sm = c0["img_chan_mean"]
    print(f"served   image channel mean (0,1,2): "
          f"[{sm[0]:.1f}, {sm[1]:.1f}, {sm[2]:.1f}]   "
          f"shape={c0['img_shape']} dtype={c0['img_dtype']}")
    # Channel-order heuristic: is served closer to train as-is, or reversed?
    as_is = float(np.abs(sm - train_img_chan_mean).sum())
    rev = float(np.abs(sm[::-1] - train_img_chan_mean).sum())
    print(f"  |served - train| as-is={as_is:.1f}   reversed={rev:.1f}  "
          f"-> {'CHANNEL ORDER LIKELY SWAPPED' if rev + 5 < as_is else 'order consistent'}")

    print("\nserved proprio (step 0) vs training [mean +/- std], z = "
          "(served-mean)/std:")
    print(f"  {'dim':8s}{'served':>10}{'train_mean':>12}{'train_std':>11}"
          f"{'z':>8}")
    proprio0 = c0["proprio"]
    n_oob = 0
    for d in range(len(proprio0)):
        z = (proprio0[d] - p_mean[d]) / p_std[d]
        oob = abs(z) > 4
        n_oob += int(oob)
        flag = "  <-- OOD" if oob else ""
        print(f"  {labels[d]:8s}{proprio0[d]:>10.3f}{p_mean[d]:>12.3f}"
              f"{p_std[d]:>11.3f}{z:>8.1f}{flag}")

    print("-" * 72)
    rot_oob = sum(
        1 for d in range(len(proprio0))
        if "r6d" in labels[d] and abs((proprio0[d] - p_mean[d]) / p_std[d]) > 4)
    xyz_oob = sum(
        1 for d in range(len(proprio0))
        if "xyz" in labels[d] and abs((proprio0[d] - p_mean[d]) / p_std[d]) > 4)
    print(f"served proprio dims out-of-distribution (|z|>4): {n_oob}/18  "
          f"(rot6d {rot_oob}/12, xyz {xyz_oob}/6)")
    if rot_oob >= 6 and xyz_oob == 0:
        print("  -> rot6d dims OOD but xyz fine: SERVE/TRAIN ROT6D CONVENTION "
              "MISMATCH is the likely fault.\n     (train reads stored "
              "rot_6d; serve recomputes columns-of-R via _rot6d_from_matrix.)")
    elif n_oob == 0 and not (rev + 5 < as_is):
        print("  -> served inputs are IN distribution and channel order is "
              "consistent.\n     Inputs are clean; investigate ACTION "
              "semantics next (absolute vs delta pose targets).")
    print("=" * 72)
    volume.commit()
    return {
        "captured": len(captured),
        "n_proprio_oob": n_oob,
        "rot_oob": rot_oob,
        "xyz_oob": xyz_oob,
        "img_chan_as_is": as_is,
        "img_chan_reversed": rev,
    }


@app.local_entrypoint()
def cache():
    probe_model_on_cache.remote()


@app.local_entrypoint()
def serve(scenario_index: int = 0):
    probe_serve_inputs.remote(scenario_index=scenario_index)