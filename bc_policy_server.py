"""
LBM Eval — BC / Diffusion policy gRPC server (bc_policy_server.py)
===================================================================
Standalone server that wraps the trained policy checkpoint as a gRPC service
compatible with the lbm_eval harness.  `lbm_phase3.py` launches this as a
subprocess inside each Modal shard.

Supported checkpoint types (auto-detected from checkpoint's `policy_type` key):
  "bc"                 — v1 BC MSE regressor, stateless step()
  "diffusion"          — v2a single-step DDPM, stateless step()
  "diffusion_chunked"  — v2b chunked DDPM, **stateful** step() with chunk buffer

The architecture is defined in bc_model.py (imported from /root/ where
lbm_phase3.py ships it alongside this file).

v2b stateful serving — receding-horizon execution
--------------------------------------------------
For "diffusion_chunked" checkpoints, BCPolicy.step() is stateful:

  - The policy maintains a chunk_buffer (deque of H action vectors) and a
    steps_since_resample counter.
  - Every K steps (exec_cadence from checkpoint), the DDPM sampler is called
    once and fills the buffer with H actions.  The first action is immediately
    popped and returned.
  - On subsequent steps (within the same chunk), the next buffered action is
    popped without re-sampling.
  - BCPolicy.reset() MUST flush the buffer before each new episode — stale
    actions from episode N bleeding into episode N+1 would be silent corruption.
    BCPolicyBatch.reset_batch() creates a fresh BCPolicy per episode, which
    guarantees this.

Correctness invariant: at most one denoising call per K steps; exactly one
action returned per step(); buffer state never crosses episode boundaries.
"""

import argparse
import sys
import uuid
from collections import deque
import os

import numpy as np
import torch

sys.path.insert(0, "/root")
import bc_model

# lbm_eval imports are deferred to main() — this file is also imported by the
# Modal shard process before the subprocess is launched, and lbm_eval may not
# be on that path. The classes are only needed inside the running server.
Policy = None
PolicyMetadata = None
PosesAndGrippers = None
RigidTransform = None
RotationMatrix = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SCENE_CAMERA_LIVE = "scene_right_0"
ARM_KEYS = ("right::panda", "left::panda")
DEFAULT_CHECKPOINT = "/data/phase2_models/bc_v2b.pt"

# ImageNet normalisation (must match prepare_data)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


# ---------------------------------------------------------------------------
# Rotation helpers (serve-only; not part of bc_model)
# ---------------------------------------------------------------------------
def _rot6d_from_matrix(R: np.ndarray) -> np.ndarray:
    """3×3 rotation matrix → 6D representation (first two columns, flattened)."""
    R = np.asarray(R, dtype=np.float32)
    return np.concatenate([R[:, 0], R[:, 1]])


def _matrix_from_rot6d(v: np.ndarray) -> np.ndarray:
    """6D representation → valid 3×3 rotation matrix (Gram-Schmidt)."""
    v = np.asarray(v, dtype=np.float64)
    a1, a2 = v[:3], v[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 /= np.linalg.norm(b2) + 1e-8
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=1)


# ---------------------------------------------------------------------------
# Shared inference context (built once in main, shared across episodes)
# ---------------------------------------------------------------------------
class _InferenceContext:
    """Model + normalization constants. Built once; BCPolicy instances share it."""

    def __init__(self, model: torch.nn.Module, device: str,
                 ckpt: dict, checkpoint_path: str):
        self.model = model
        self.device = device
        self.checkpoint_path = checkpoint_path

        # Normalization (numpy, applied per-step)
        self.a_mean = np.asarray(ckpt["action_mean"], dtype=np.float32)   # (20,)
        self.a_std  = np.asarray(ckpt["action_std"],  dtype=np.float32)   # (20,)
        self.p_mean = np.asarray(ckpt["proprio_mean"], dtype=np.float32)  # (18,)
        self.p_std  = np.asarray(ckpt["proprio_std"],  dtype=np.float32)  # (18,)

        # v2b-specific serving config (harmless defaults for v1/v2a)
        self.policy_type  = ckpt.get("policy_type", "bc")
        self.chunk_size   = int(ckpt.get("chunk_size",   1))
        self.exec_cadence = int(ckpt.get("exec_cadence", 1))
        _k = int(os.environ.get("BC_REPLAN_K", "0"))   # 0 = use checkpoint default
        if _k > 0:
            self.exec_cadence = _k
            print(f"[bc_policy_server] exec_cadence overridden to K={_k} "
                  f"via BC_REPLAN_K", flush=True)

# ---------------------------------------------------------------------------
# BCPolicy — per-episode inference; stateful for v2b
# ---------------------------------------------------------------------------
class BCPolicy:
    """Serves any supported checkpoint type against the lbm_eval harness.

    For v1/v2a: step() is stateless — call model every step, return action.
    For v2b:    step() is stateful — maintain a chunk buffer, sample every K
                steps, return one action per call from the buffer.
    """

    _logged_obs: bool = False  # class-level: log first observation once

    def __init__(self, ctx: _InferenceContext):
        self._ctx = ctx
        self._step_count = 0

        # v2b chunk buffer — always present; unused for v1/v2a
        self._chunk_buffer: deque[np.ndarray] = deque()
        self._steps_since_resample: int = 0

    def reset(self, seed=None, options=None):
        """Called at episode start. Flush chunk buffer — required for v2b."""
        self._step_count = 0
        self._chunk_buffer.clear()
        self._steps_since_resample = 0

    def get_policy_metadata(self) -> PolicyMetadata:
        return PolicyMetadata(
            name=f"BC_{self._ctx.policy_type}",
            skill_type="pick_and_place_box",
            checkpoint_path=self._ctx.checkpoint_path,
            git_repo="lbm_eval",
            git_sha="phase3",
        )

    # ------------------------------------------------------------------
    # Observation preprocessing
    # ------------------------------------------------------------------
    def _extract_proprio(self, observation) -> np.ndarray:
        """18-dim proprio: right-arm [xyz, rot6d] then left-arm [xyz, rot6d]."""
        poses = observation.robot.actual.poses
        parts = []
        for arm in ARM_KEYS:
            X = poses[arm]
            parts.append(np.asarray(X.translation(), dtype=np.float32))
            parts.append(_rot6d_from_matrix(X.rotation().matrix()))
        return np.concatenate(parts)  # (18,)

    def _extract_image(self, observation) -> np.ndarray:
        """CHW float32, ImageNet-normalised, 2× downsampled (240×320)."""
        rgb = observation.visuo[SCENE_CAMERA_LIVE].rgb.array  # (480,640,3)
        rgb = rgb[::2, ::2, :]                                # (240,320,3)
        img = rgb.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))                    # (3,240,320)
        return ((img - IMAGENET_MEAN) / IMAGENET_STD).astype(np.float32)

    # ------------------------------------------------------------------
    # Action decode
    # ------------------------------------------------------------------
    def _decode_action(self, action_n: np.ndarray) -> PosesAndGrippers:
        """Un-normalise a (20,) vector and convert to PosesAndGrippers."""
        ctx = self._ctx
        action = action_n * ctx.a_std + ctx.a_mean   # (20,)

        r_xyz,   r_rot6d = action[0:3],   action[3:9]
        l_xyz,   l_rot6d = action[9:12],  action[12:18]
        r_grip,  l_grip  = float(action[18]), float(action[19])

        poses = {
            "right::panda": RigidTransform(
                RotationMatrix(_matrix_from_rot6d(r_rot6d)),
                r_xyz.astype(np.float64),
            ),
            "left::panda": RigidTransform(
                RotationMatrix(_matrix_from_rot6d(l_rot6d)),
                l_xyz.astype(np.float64),
            ),
        }
        grippers = {
            "right::panda_hand": r_grip,
            "left::panda_hand":  l_grip,
        }
        return PosesAndGrippers(poses=poses, grippers=grippers)

    # ------------------------------------------------------------------
    # Chunk sampling (v2b)
    # ------------------------------------------------------------------
    def _fill_chunk_buffer(self, img_t: torch.Tensor,
                           prop_t: torch.Tensor) -> None:
        """Query the model for a full chunk and fill the buffer.

        For bc_chunked: model.forward() returns (1, H, 20) directly.
        For diffusion_chunked: model.sample() returns (1, H, 20).
        Each entry in the buffer is a (20,) float32 normalised action vector.
        """
        ctx = self._ctx
        with torch.no_grad():
            if ctx.policy_type == "bc_chunked":
                chunk_n = ctx.model(img_t, prop_t).cpu().numpy()[0]   # (H, 20)
            else:
                # model.sample() returns (1, H, 20) normalised
                chunk_n = ctx.model.sample(img_t, prop_t).cpu().numpy()[0]  # (H, 20)

        self._chunk_buffer.clear()
        for h in range(chunk_n.shape[0]):
            self._chunk_buffer.append(chunk_n[h])   # each: (20,)
        self._steps_since_resample = 0

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------
    def step(self, observation) -> PosesAndGrippers:
        ctx = self._ctx

        # One-time observation structure guard log
        if not BCPolicy._logged_obs:
            BCPolicy._logged_obs = True
            try:
                keys = list(observation.visuo.keys())
                rgb  = observation.visuo[SCENE_CAMERA_LIVE].rgb.array
                print(f"[bc_policy_server] first obs OK — "
                      f"visuo keys={keys}, "
                      f"'{SCENE_CAMERA_LIVE}' rgb shape={rgb.shape} "
                      f"dtype={rgb.dtype}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[bc_policy_server] WARNING inspecting first obs: "
                      f"{type(e).__name__}: {e}", flush=True)

        # --- build network inputs ---
        proprio = self._extract_proprio(observation)
        img     = self._extract_image(observation)

        # Normalise + clip (guard against parked-arm std-floor edge cases)
        proprio_n = np.clip(
            (proprio - ctx.p_mean) / ctx.p_std, -5.0, 5.0
        ).astype(np.float32)

        img_t  = torch.from_numpy(img).unsqueeze(0).to(ctx.device)
        prop_t = torch.from_numpy(proprio_n).unsqueeze(0).to(ctx.device)

        # --- dispatch by policy type ---
        if ctx.policy_type in ("diffusion_chunked", "bc_chunked"):
            # Receding-horizon chunk buffer: re-query every exec_cadence steps
            # or when the buffer is empty (episode start / after reset).
            if (not self._chunk_buffer or
                    self._steps_since_resample >= ctx.exec_cadence):
                self._fill_chunk_buffer(img_t, prop_t)

            action_n = self._chunk_buffer.popleft()     # (20,) normalised
            self._steps_since_resample += 1

        else:
            # v1 (bc) and v2a (diffusion): single-step, no buffer
            with torch.no_grad():
                out = ctx.model(img_t, prop_t).cpu().numpy()

            if ctx.policy_type == "bc":
                action_n = out[0]                   # (20,)
            else:
                # v2a: DiffusionPolicy.forward() returns (1, 1, 20)
                action_n = out[0, 0]                # (20,)

        self._step_count += 1
        return self._decode_action(action_n)


# ---------------------------------------------------------------------------
# BCPolicyBatch — batch interface wrapper required by run_policy_server
# ---------------------------------------------------------------------------
class BCPolicyBatch:
    """Wraps BCPolicy with the reset_batch/step_batch interface.

    One BCPolicy per client UUID (= per concurrent episode). Constructing a
    fresh BCPolicy per reset_batch call is cheap — the model lives in the
    shared _InferenceContext; no ResNet rebuild. Critically, it guarantees the
    chunk buffer is always fresh at episode start (BCPolicy.__init__ clears it).
    """

    def __init__(self, ctx: _InferenceContext):
        self._ctx = ctx
        self._internal_uuid = uuid.uuid4()
        self._sub_policies: dict[uuid.UUID, BCPolicy] = {}

    def reset(self, seed=None, options=None):
        self.reset_batch({self._internal_uuid: seed}, options)

    def reset_batch(self, seeds, options=None):
        # Fresh BCPolicy per episode — clears chunk buffer, resets counters.
        # The scene seed is handled by the harness (scenario_index), not here.
        for uid in seeds:
            self._sub_policies[uid] = BCPolicy(self._ctx)

    def get_policy_metadata(self) -> PolicyMetadata:
        return BCPolicy(self._ctx).get_policy_metadata()

    def step(self, observation) -> PosesAndGrippers:
        return self.step_batch({self._internal_uuid: observation})[self._internal_uuid]

    def step_batch(self, observations: dict) -> dict:
        return {
            uid: self._sub_policies[uid].step(obs)
            for uid, obs in observations.items()
        }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    # Deferred imports — only available inside the server subprocess where
    # lbm_eval is installed. Assigned to globals so BCPolicy/BCPolicyBatch
    # class bodies (defined at module level) can reference them at call time.
    global Policy, PolicyMetadata, PosesAndGrippers
    global RigidTransform, RotationMatrix
    global LbmPolicyServerConfig, run_policy_server
    from pydrake.math import RigidTransform as _RT, RotationMatrix as _RM
    from robot_gym.policy import Policy as _P, PolicyMetadata as _PM
    from robot_gym.multiarm_spaces import PosesAndGrippers as _PAG
    from grpc_workspace.lbm_policy_server import (LbmPolicyServerConfig as _LPSC,
                                                   run_policy_server as _RPS)
    Policy = _P
    PolicyMetadata = _PM
    PosesAndGrippers = _PAG
    RigidTransform = _RT
    RotationMatrix = _RM
    LbmPolicyServerConfig = _LPSC
    run_policy_server = _RPS

    parser = argparse.ArgumentParser(
        description="BC/Diffusion policy gRPC server for the lbm_eval benchmark.")
    LbmPolicyServerConfig.add_argparse_arguments(parser)
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to the policy checkpoint (.pt). Default: %(default)s")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[bc_policy_server] loading {args.checkpoint} (device={device}) …",
          flush=True)

    # weights_only=False: checkpoint contains numpy arrays (normalization stats).
    # This file is produced by our own trainer — trusted.
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    policy_type = ckpt.get("policy_type", "bc")   # default "bc" for old v1 ckpts
    chunk_size  = int(ckpt.get("chunk_size", 1))
    n_timesteps = int(ckpt.get("n_timesteps", 100))
    proprio_dim = int(ckpt.get("proprio_dim", bc_model.PROPRIO_DIM))
    action_dim  = int(ckpt.get("action_dim",  bc_model.ACTION_DIM))

    print(f"[bc_policy_server] policy_type={policy_type!r}  "
          f"chunk_size={chunk_size}  n_timesteps={n_timesteps}", flush=True)

    model = bc_model.build_model(
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        policy_type=policy_type,
        n_timesteps=n_timesteps,
        chunk_size=chunk_size,
        pretrained=False,   # weights come from checkpoint
    ).to(device)

    def _remap_legacy_state(state):
        """lbm_phase2_train.py's inline BCPolicy vs bc_model.BCModel: same network,
        different nesting. Remap so the released server loads both."""
        out = {}
        for k, v in state.items():
            if k.startswith("proprio_mlp."):
                out["encoder." + k] = v
            elif k.startswith("head.net."):
                out["head." + k[len("head.net."):]] = v
            elif k.startswith("encoder."):
                out["encoder." + k] = v
            else:
                out[k] = v
        return out


    state = ckpt.get("model_state_dict") or ckpt.get("model_state")
    if state is None:
        raise SystemExit(f"no state dict; keys: {sorted(ckpt.keys())}")
    if "head.net.0.weight" in state:
        print("[bc_policy_server] legacy v1 layout detected — remapping")
        state = _remap_legacy_state(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[bc_policy_server] missing={len(missing)} unexpected={len(unexpected)}")
    if missing:
        print(f"  missing: {missing[:5]}")
        
    model.load_state_dict(state)
    # model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print("[bc_policy_server] model loaded OK", flush=True)

    ctx = _InferenceContext(model, device, ckpt, args.checkpoint)
    policy = BCPolicyBatch(ctx)

    run_policy_server(policy, args)


if __name__ == "__main__":
    main()