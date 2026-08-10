"""
LBM Eval on Modal — Phase 2: BC v1 trainer
==========================================
Trains a plain behavior-cloning policy on the pick_and_place_box teleop data,
as a PIPELINE SHAKEDOWN (first trained policy; judged on "loop runs + beats
0%", not on being a strong policy). Diffusion Policy is the planned v2 — the
policy head here is a swappable nn.Module so v2 changes only the head.

Design (decided in chat):
  - Encoder: ImageNet-pretrained ResNet-18 (pretrained is the real small-data
    mitigation — only ~4k frames available).
  - Input: ONE 480x640 scene camera (a 6CD... view) + the `actual` end-effector
    pose vector. Action target: 20-dim, z-scored per-dimension.
  - Two steps so hyperparameter iteration never re-decodes the 16 GB tarball:
      prepare_data : episodes on the Volume -> one processed .pt cache
      train        : reads the .pt cache -> trains -> writes a checkpoint
  - Episode filters: episode_success=True (all 104 pass) AND a min-length
    cutoff. prepare_data PRINTS the length histogram; set MIN_EPISODE_LEN
    against it (default 10).

Run order:
  modal run lbm_phase2_train.py::prepare_data
  modal run lbm_phase2_train.py::train

Carry-over for the policy server (next file): the checkpoint dict contains
`action_mean` / `action_std` — the server MUST un-normalize the network's
output with these before returning PosesAndGrippers.
"""

import modal

APP_NAME = "lbm-eval-phase2-train"
app = modal.App(APP_NAME)

volume = modal.Volume.from_name("lbm-eval-data", create_if_missing=True)
VOL_PATH = "/data"

# Paths on the Volume.
EXTRACT_DIR = f"{VOL_PATH}/phase2_training_data/PickAndPlaceBox"
CACHE_PATH = f"{VOL_PATH}/phase2_training_data/processed_bc_v1.pt"
CKPT_PATH = f"{VOL_PATH}/phase2_models/bc_v1.pt"

# --- Data spec --------------------------------------------------------------
SCENE_CAMERA = "6CD146030E99"   # one 480x640 scene camera
IMG_H, IMG_W = 240, 320         # ResNet input after 2x downscale of 480x640
PROPRIO_KEYS = [                # `actual` end-effector pose, both arms
    "robot__actual__poses__right::panda__xyz",
    "robot__actual__poses__right::panda__rot_6d",
    "robot__actual__poses__left::panda__xyz",
    "robot__actual__poses__left::panda__rot_6d",
]
ACTION_DIM = 20
MIN_EPISODE_LEN = 10            # set against the histogram prepare_data prints

# --- Images -----------------------------------------------------------------
# prepare_data only needs numpy; train needs torch + torchvision (GPU).
prep_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.3.0", "torchvision==0.18.0", "numpy", "tqdm")
)
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.3.0", "torchvision==0.18.0", "numpy", "tqdm")
)


# ===========================================================================
# STEP 1 — prepare_data: episodes -> one processed .pt cache
# ===========================================================================
@app.function(image=prep_image, volumes={VOL_PATH: volume}, timeout=3600)
def prepare_data():
    """Decode every usable episode into flat per-keyframe samples and cache
    them as one file. Prints the episode-length histogram first so the
    MIN_EPISODE_LEN cutoff is set against real counts."""
    import os
    import pickle
    import numpy as np

    if not os.path.isdir(EXTRACT_DIR):
        raise SystemExit(f"{EXTRACT_DIR} missing — run lbm_phase2_probe.py first.")

    episode_dirs = []
    for root, dirs, _f in os.walk(EXTRACT_DIR):
        if "processed" in dirs:
            episode_dirs.append(os.path.join(root, "processed"))
    episode_dirs.sort()
    print(f"found {len(episode_dirs)} episodes")

    def _floored_std(x, frac=0.1):
        """Per-dimension std with a floor proportional to the typical spread.
        The original `+1e-6` floor was a divide-by-zero guard, but on a dim
        that is constant in the data (the parked left arm in this task) it
        produced ~1e6 amplification of any serve-time deviation — the Phase 3
        first-frame stutter. Flooring at a fraction of the median std keeps
        constant dims contributing ~0 even when serve inputs deviate slightly,
        instead of exploding. Applied to BOTH proprio and action, read by BOTH
        train and serve (same stored stats), so train/serve stay consistent."""
        s = x.std(axis=0)
        return np.maximum(s, frac * np.median(s) + 1e-6)

    # -- length histogram (decide the cutoff against this) -----------------
    lengths = []
    for proc in episode_dirs:
        try:
            a = np.load(os.path.join(proc, "actions.npz"), allow_pickle=True)
            lengths.append((proc, int(a["actions"].shape[0])))
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {proc}: {e})")
    L = np.array([n for _p, n in lengths])
    print("\nepisode-length histogram:")
    for lo in range(0, int(L.max()) + 10, 10):
        c = int(((L >= lo) & (L < lo + 10)).sum())
        print(f"  [{lo:3d},{lo+10:3d}) {'#' * c} {c}")
    keep = [(p, n) for p, n in lengths if n >= MIN_EPISODE_LEN]
    print(f"\nMIN_EPISODE_LEN={MIN_EPISODE_LEN} -> keep {len(keep)}/{len(lengths)} "
          f"episodes, drop {len(lengths) - len(keep)}")

    # -- decode kept episodes into flat samples ----------------------------
    images, proprios, actions = [], [], []
    for proc, n in keep:
        try:
            o = np.load(os.path.join(proc, "observations.npz"), allow_pickle=True)
            a = np.load(os.path.join(proc, "actions.npz"), allow_pickle=True)
            if SCENE_CAMERA not in o:
                print(f"  (skip {proc}: no camera {SCENE_CAMERA})")
                continue
            rgb = o[SCENE_CAMERA]                       # (T,480,640,3) uint8
            act = a["actions"].astype(np.float32)        # (T,20)
            proprio = np.concatenate(
                [np.asarray(o[k], dtype=np.float32) for k in PROPRIO_KEYS],
                axis=1,
            )                                           # (T, proprio_dim)
            T = min(rgb.shape[0], act.shape[0], proprio.shape[0])
            # 2x nearest-neighbour downscale 480x640 -> 240x320 (cheap, no PIL).
            rgb = rgb[:T, ::2, ::2, :]
            images.append(rgb.astype(np.uint8))
            proprios.append(proprio[:T])
            actions.append(act[:T])
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {proc}: {e})")

    images = np.concatenate(images, axis=0)             # (N,240,320,3) uint8
    proprios = np.concatenate(proprios, axis=0)         # (N, proprio_dim)
    actions = np.concatenate(actions, axis=0)           # (N, 20)
    print(f"\ndecoded {images.shape[0]} samples  "
          f"img={images.shape} proprio={proprios.shape} act={actions.shape}")

    # -- per-dimension normalization stats (saved; the SERVER inverts these) -
    action_mean = actions.mean(axis=0)
    action_std = _floored_std(actions)
    proprio_mean = proprios.mean(axis=0)
    proprio_std = _floored_std(proprios)
    # action_mean = actions.mean(axis=0)
    # action_std = actions.std(axis=0) + 1e-6
    # proprio_mean = proprios.mean(axis=0)
    # proprio_std = proprios.std(axis=0) + 1e-6

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(
            {
                "images": images,
                "proprios": proprios.astype(np.float32),
                "actions": actions.astype(np.float32),
                "action_mean": action_mean.astype(np.float32),
                "action_std": action_std.astype(np.float32),
                "proprio_mean": proprio_mean.astype(np.float32),
                "proprio_std": proprio_std.astype(np.float32),
                "scene_camera": SCENE_CAMERA,
                "proprio_keys": PROPRIO_KEYS,
                "min_episode_len": MIN_EPISODE_LEN,
            },
            f,
            protocol=4,
        )
    volume.commit()
    print(f"cache written -> {CACHE_PATH} "
          f"({os.path.getsize(CACHE_PATH):,} bytes)")
    return {"n_samples": int(images.shape[0]), "n_episodes": len(keep)}


# ===========================================================================
# STEP 2 — model: ResNet-18 encoder + proprio MLP + swappable PolicyHead
# ===========================================================================
def _build_model(proprio_dim):
    """Returns (model, BCHead-info). Defined inside the train image where torch
    is importable. The head is its own module so v2 swaps only the head."""
    import torch.nn as nn
    import torchvision

    class BCHead(nn.Module):
        """v1 policy head: MLP regressor, trained with MSE. v2 will be a
        DiffusionHead behind this same (features -> action) interface."""

        def __init__(self, in_dim, action_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(),
                nn.Linear(256, 256), nn.ReLU(),
                nn.Linear(256, action_dim),
            )

        def forward(self, feats):
            return self.net(feats)

    class BCPolicy(nn.Module):
        def __init__(self, proprio_dim, action_dim):
            super().__init__()
            # ImageNet-pretrained ResNet-18, classifier head removed.
            backbone = torchvision.models.resnet18(
                weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            )
            self.img_dim = backbone.fc.in_features      # 512
            backbone.fc = nn.Identity()
            self.encoder = backbone
            self.proprio_mlp = nn.Sequential(
                nn.Linear(proprio_dim, 128), nn.ReLU(),
                nn.Linear(128, 128), nn.ReLU(),
            )
            self.head = BCHead(self.img_dim + 128, action_dim)

        def forward(self, img, proprio):
            feats = self.encoder(img)
            p = self.proprio_mlp(proprio)
            import torch
            return self.head(torch.cat([feats, p], dim=1))

    return BCPolicy(proprio_dim, ACTION_DIM)


@app.function(image=train_image, gpu="L4", volumes={VOL_PATH: volume},
              timeout=7200)
def train(epochs: int = 60, batch_size: int = 64, lr: float = 1e-4,
          val_frac: float = 0.15, seed: int = 0, out_path: str = CKPT_PATH):
    """Train BC v1 from the processed cache. Writes a checkpoint to the Volume
    that includes the normalization stats the policy server needs."""
    import os
    import pickle
    import numpy as np
    import torch
    import torch.nn as nn

    if not os.path.exists(CACHE_PATH):
        raise SystemExit("processed cache missing — run prepare_data first.")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  samples={data['images'].shape[0]}")
    import random
    volume.reload()
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f"seed={seed}  out_path={out_path}")

    # ImageNet normalization constants for the pretrained encoder.
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

    # Tensors. Images -> (N,3,H,W) float in [0,1]; proprio + action z-scored.
    imgs = torch.from_numpy(data["images"]).float().permute(0, 3, 1, 2) / 255.0
    imgs = (imgs - imagenet_mean) / imagenet_std
    a_mean = torch.from_numpy(data["action_mean"])
    a_std = torch.from_numpy(data["action_std"])
    p_mean = torch.from_numpy(data["proprio_mean"])
    p_std = torch.from_numpy(data["proprio_std"])
    proprio = (torch.from_numpy(data["proprios"]) - p_mean) / p_std
    action = (torch.from_numpy(data["actions"]) - a_mean) / a_std

    # Train/val split.
    n = imgs.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"train={len(tr_idx)}  val={len(val_idx)}")

    model = _build_model(proprio.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    def batches(idx, shuffle):
        order = idx[torch.randperm(len(idx))] if shuffle else idx
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            yield (imgs[b].to(device), proprio[b].to(device),
                   action[b].to(device))

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        model.train()
        tr_loss = nb = 0.0
        for img, p, act in batches(tr_idx, shuffle=True):
            opt.zero_grad()
            loss = loss_fn(model(img, p), act)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            nb += 1
        model.eval()
        v_loss = vb = 0.0
        with torch.no_grad():
            for img, p, act in batches(val_idx, shuffle=False):
                v_loss += loss_fn(model(img, p), act).item()
                vb += 1
        tr, va = tr_loss / max(nb, 1), v_loss / max(vb, 1)
        flag = ""
        if va < best_val:
            best_val = va
            flag = "  <- best"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "proprio_dim": proprio.shape[1],
                    "action_dim": ACTION_DIM,
                    # The server MUST invert these on the network output:
                    "action_mean": data["action_mean"],
                    "action_std": data["action_std"],
                    "proprio_mean": data["proprio_mean"],
                    "proprio_std": data["proprio_std"],
                    "scene_camera": data["scene_camera"],
                    "proprio_keys": data["proprio_keys"],
                    "img_hw": [IMG_H, IMG_W],
                    "seed": seed,
                },
                out_path,
            )
            volume.commit()
        if ep % 5 == 0 or ep == 1 or flag:
            print(f"  epoch {ep:3d}  train MSE={tr:.4f}  val MSE={va:.4f}{flag}")

    print(f"\nbest val MSE={best_val:.4f}")
    print(f"checkpoint -> {out_path}")
    print("NOTE: this is a shakedown policy on ~4k samples — judge it by "
          "whether the eval loop runs and beats 0%, not by absolute MSE.")
    return {"best_val_mse": best_val}


@app.local_entrypoint()
def main():
    print("### prepare_data ###")
    prepare_data.remote()
    print("\n### train ###")
    train.remote()