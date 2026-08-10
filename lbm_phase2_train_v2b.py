"""
LBM Eval on Modal — Phase 2 v2b: chunked diffusion trainer
===========================================================
Trains the v2b policy: DDPM over a H-step action chunk, executed K steps at a
time (receding-horizon). This directly targets both failure modes observed so
far:

  - v1 (BC regressor, 5.1%): drift / compounding error between single-step
    re-plans. Chunking commits to a multi-step plan, suppressing drift.
  - v2a (single-step diffusion, 0/195): per-step stochastic re-sampling
    destroys trajectory coherence. Chunking samples once per K steps — no
    mid-chunk re-roll.

Default hyperparameters match the original Diffusion Policy paper (Chi et al.):
  H = 16   (chunk length: predict 16 future actions in one denoising pass)
  K = 8    (execute cadence: execute 8 actions open-loop before re-sampling)

Data path change vs v2a
-----------------------
v2a reused the v1 single-step cache (processed_bc_v1.pt). v2b needs temporal
windows: each training sample is

    (image_t, proprio_t) → (action_t, action_{t+1}, …, action_{t+H-1})

so prepare_data_v2b() builds a NEW cache (processed_bc_v2b.pt) from the raw
episodes already decoded in phase2_training_data/PickAndPlaceBox/.

Window policy:
  - Windows are within-episode only (no cross-episode boundaries).
  - Short tails (< H frames remaining to end of episode) are DROPPED — they
    would require padding with a repeated final action, which risks the model
    learning to stall. With mean ~40 keyframes/episode and H=16 this drops
    at most the last 15 frames of each episode; the rest are clean.

Model change vs v2a
-------------------
DiffusionPolicy(chunk_size=H) in bc_model.py: the denoiser's output dimension
is H*20 instead of 20. The encoder and conditioning path are unchanged.
Checkpoint stores policy_type="diffusion_chunked" and chunk_size=H so the
server auto-detects the architecture.

Checkpoint selection
--------------------
Training loss is noise-prediction MSE (not comparable to v1's 0.0149). Val
metric is sampled-action MSE: run the full DDPM sampler on each val observation,
compute action-space MSE vs ground truth on EACH step of the chunk, average.
This is comparable in spirit to v1's val metric and is what governs checkpoint
selection.

Run order:
  1. modal run lbm_phase2_train_v2b.py::prepare_data_v2b   # ~15 min
  2. modal run lbm_phase2_train_v2b.py::train               # ~2–3 h on L4
"""

import modal

APP_NAME = "lbm-eval-phase2-train-v2b"
app = modal.App(APP_NAME)

volume = modal.Volume.from_name("lbm-eval-data", create_if_missing=True)
VOL_PATH = "/data"

EPISODES_DIR = f"{VOL_PATH}/phase2_training_data/PickAndPlaceBox"
V2B_CACHE_PATH = f"{VOL_PATH}/phase2_training_data/processed_bc_v2b.pt"
V1_CACHE_PATH = f"{VOL_PATH}/phase2_training_data/processed_bc_v1.pt"  # kept intact
CKPT_OUT = f"{VOL_PATH}/phase2_models/bc_v2b.pt"

# v2b hyperparameters
CHUNK_SIZE = 16   # H: predict 16 future actions
EXEC_CADENCE = 8  # K: execute 8 before re-sampling (used at serve time only)
N_TIMESTEPS = 100

# ---------------------------------------------------------------------------
# Image
# ---------------------------------------------------------------------------
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.3.0", "torchvision==0.18.0", "numpy", "tqdm")
    .add_local_file("bc_model.py", "/root/bc_model.py")
)


# ---------------------------------------------------------------------------
# prepare_data_v2b — build temporal-window cache
# ---------------------------------------------------------------------------
@app.function(image=train_image, volumes={VOL_PATH: volume}, timeout=3600,
              memory=16384)
def prepare_data_v2b(chunk_size: int = CHUNK_SIZE, force: bool = False):
    """Build processed_bc_v2b.pt with (image_t, proprio_t) → chunk_t windows.

    Reads normalisation constants from the v1 cache (same raw data, same
    statistics — recomputing them would give the same values). Writes a
    separate file so the v1 cache is preserved for back-comparison.

    Cache schema (pickle dict):
      images      np.uint8   (N, H_img, W_img, 3)   raw uint8, CHW at serve
      proprios    np.float32 (N, 18)                 raw, not normalised
      actions     np.float32 (N, chunk_size, 20)     raw, not normalised
      proprio_mean/std  (18,)   from v1 cache (identical)
      action_mean/std   (20,)   from v1 cache (identical, per action dim)
      chunk_size  int
      exec_cadence int
    """
    import os, sys, pickle
    import numpy as np

    sys.path.insert(0, "/root")
    volume.reload()

    if os.path.exists(V2B_CACHE_PATH) and not force:
        print(f"Cache already exists at {V2B_CACHE_PATH}. Pass force=True to rebuild.")
        return

    # Load normalization stats from v1 cache (don't recompute — same data)
    if not os.path.exists(V1_CACHE_PATH):
        raise SystemExit(
            f"v1 cache not found at {V1_CACHE_PATH}. "
            "Run lbm_phase2_train.py::prepare_data first."
        )
    print("Loading normalization stats from v1 cache …")
    with open(V1_CACHE_PATH, "rb") as f:
        v1 = pickle.load(f)
    action_mean = v1["action_mean"]   # (20,)
    action_std  = v1["action_std"]    # (20,)
    proprio_mean = v1["proprio_mean"] # (18,)
    proprio_std  = v1["proprio_std"]  # (18,)
    del v1

    # Discover episode directories — same walk as lbm_phase2_train.py::prepare_data.
    # Each episode is a directory that contains a "processed/" subdirectory with
    # observations.npz and actions.npz inside it.
    ep_proc_dirs = []
    for root, dirs, _files in os.walk(EPISODES_DIR):
        if "processed" in dirs:
            ep_proc_dirs.append(os.path.join(root, "processed"))
    ep_proc_dirs.sort()
    if not ep_proc_dirs:
        raise SystemExit(
            f"No episode 'processed/' directories found under {EPISODES_DIR}. "
            "Expected layout: <episode_dir>/processed/{{observations,actions}}.npz"
        )
    print(f"Found {len(ep_proc_dirs)} episode directories.")

    # These must match lbm_phase2_train.py exactly — keys come from observations.npz.
    SCENE_CAMERA = "6CD146030E99"   # serial-keyed in observations.npz; matches live "scene_right_0"
    PROPRIO_KEYS = [
        "robot__actual__poses__right::panda__xyz",
        "robot__actual__poses__right::panda__rot_6d",
        "robot__actual__poses__left::panda__xyz",
        "robot__actual__poses__left::panda__rot_6d",
    ]
    MIN_EPISODE_LEN = 10  # same floor as v1; very short episodes are noise

    images_list, proprios_list, actions_list = [], [], []
    dropped_tail, total_frames, skipped = 0, 0, 0

    for proc in ep_proc_dirs:
        try:
            o = np.load(os.path.join(proc, "observations.npz"), allow_pickle=True)
            a = np.load(os.path.join(proc, "actions.npz"),      allow_pickle=True)
        except Exception as e:  # noqa: BLE001
            print(f"  (skip {proc}: {e})")
            skipped += 1
            continue

        if SCENE_CAMERA not in o:
            print(f"  (skip {proc}: no camera key '{SCENE_CAMERA}')")
            skipped += 1
            continue

        rgb  = o[SCENE_CAMERA]                                     # (T,480,640,3) uint8
        acts = np.asarray(a["actions"], dtype=np.float32)          # (T,20)
        prop_parts = [np.asarray(o[k], dtype=np.float32) for k in PROPRIO_KEYS
                      if k in o]
        if len(prop_parts) != len(PROPRIO_KEYS):
            missing = [k for k in PROPRIO_KEYS if k not in o]
            print(f"  (skip {proc}: missing proprio keys {missing})")
            skipped += 1
            continue
        proprios_ep = np.concatenate(prop_parts, axis=1)           # (T,18)

        T = min(rgb.shape[0], acts.shape[0], proprios_ep.shape[0])
        if T < MIN_EPISODE_LEN:
            skipped += 1
            continue

        # 2× nearest-neighbour downscale 480×640 → 240×320 (matches v1 prepare_data)
        imgs = rgb[:T, ::2, ::2, :]      # (T,240,320,3) — row/col stride, matches v1
        acts = acts[:T]
        proprios_ep = proprios_ep[:T]

        total_frames += T

        # Build within-episode windows; drop tail shorter than chunk_size
        n_windows = T - chunk_size + 1
        if n_windows <= 0:
            dropped_tail += T
            continue

        for t in range(n_windows):
            images_list.append(imgs[t])                              # (240,320,3)
            proprios_list.append(proprios_ep[t])                     # (18,)
            actions_list.append(acts[t : t + chunk_size])            # (chunk_size, 20)

        # Last (chunk_size-1) frames of each episode can't anchor a full window
        dropped_tail += chunk_size - 1

    print(f"Episodes: {len(ep_proc_dirs)} found, {skipped} skipped  |  "
          f"Total frames: {total_frames}  |  "
          f"Dropped tail frames: {dropped_tail}  |  "
          f"Windows: {len(images_list)}")

    images  = np.stack(images_list,   axis=0)  # (N, H_img, W_img, 3)
    proprios = np.stack(proprios_list, axis=0)  # (N, 18)
    actions  = np.stack(actions_list,  axis=0)  # (N, chunk_size, 20)

    cache = {
        "images":       images,
        "proprios":     proprios,
        "actions":      actions,
        "action_mean":  action_mean,
        "action_std":   action_std,
        "proprio_mean": proprio_mean,
        "proprio_std":  proprio_std,
        "chunk_size":   chunk_size,
        "exec_cadence": EXEC_CADENCE,
    }
    os.makedirs(os.path.dirname(V2B_CACHE_PATH), exist_ok=True)
    with open(V2B_CACHE_PATH, "wb") as f:
        pickle.dump(cache, f)
    volume.commit()
    print(f"Wrote {V2B_CACHE_PATH}  ({images.nbytes // 1024**2} MB images, "
          f"{actions.nbytes // 1024**2} MB actions)")


# ---------------------------------------------------------------------------
# train — DDPM noise-prediction + sampled-action val metric
# ---------------------------------------------------------------------------
@app.function(image=train_image, gpu="L4", volumes={VOL_PATH: volume},
              timeout=10800)
def train(epochs: int = 200, batch_size: int = 64, lr: float = 1e-4,
          val_frac: float = 0.15, n_val_samples: int = 4, resume: bool = True):
    """Train bc_v2b.pt with the chunked diffusion policy.

    epochs: total epochs to train (200 default; cosine schedule spans the full run).
    n_val_samples: stochastic samples averaged for the val metric (4 reduces noise).
    resume: if True and bc_v2b.pt exists, load weights and continue training
            with a fresh cosine schedule over the remaining epochs.
    """
    import os, sys, pickle
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, TensorDataset, random_split

    sys.path.insert(0, "/root")
    import bc_model

    volume.reload()

    if not os.path.exists(V2B_CACHE_PATH):
        raise SystemExit(
            f"v2b cache not found at {V2B_CACHE_PATH}. "
            "Run prepare_data_v2b first."
        )

    with open(V2B_CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    chunk_size = data["chunk_size"]
    assert chunk_size == CHUNK_SIZE, (
        f"Cache chunk_size={chunk_size} != CHUNK_SIZE={CHUNK_SIZE}"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  N={data['images'].shape[0]}  "
          f"chunk_size={chunk_size}  epochs={epochs}")

    # Normalization constants
    a_mean = torch.from_numpy(data["action_mean"]).float()     # (20,)
    a_std  = torch.from_numpy(data["action_std"]).float()      # (20,)
    p_mean = torch.from_numpy(data["proprio_mean"]).float()    # (18,)
    p_std  = torch.from_numpy(data["proprio_std"]).float()     # (18,)

    # Tensors
    # Images: uint8 → float, ImageNet normalise, NCHW
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs = torch.from_numpy(data["images"]).float().permute(0, 3, 1, 2) / 255.0
    imgs = (imgs - imagenet_mean) / imagenet_std                # (N, 3, H, W)

    proprios = (torch.from_numpy(data["proprios"]).float() - p_mean) / p_std  # (N,18)

    # Actions: normalise per action dim across the chunk.
    # data["actions"] is (N, chunk_size, 20); a_mean/a_std are (20,).
    acts_raw = torch.from_numpy(data["actions"]).float()        # (N, H, 20)
    acts = (acts_raw - a_mean.unsqueeze(0).unsqueeze(0)) / \
           a_std.unsqueeze(0).unsqueeze(0)                      # (N, H, 20)

    dataset = TensorDataset(imgs, proprios, acts)
    n_val = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    # Build model — optionally resume from existing checkpoint
    model = bc_model.build_model(
        proprio_dim=data["proprios"].shape[1],
        action_dim=bc_model.ACTION_DIM,
        policy_type="diffusion_chunked",
        n_timesteps=N_TIMESTEPS,
        chunk_size=chunk_size,
        pretrained=True,
    ).to(device)

    start_epoch = 1
    if resume and os.path.exists(CKPT_OUT):
        print(f"Resuming from {CKPT_OUT} …")
        ckpt_resume = torch.load(CKPT_OUT, map_location=device, weights_only=False)
        model.load_state_dict(ckpt_resume["model_state_dict"])
        start_epoch = ckpt_resume.get("epoch", 0) + 1
        print(f"  loaded epoch={start_epoch-1}  "
              f"val_sampled_mse={ckpt_resume.get('val_sampled_mse', '?'):.4f}")
    else:
        print("Training from scratch (pretrained ResNet-18 encoder).")

    # Fresh cosine schedule over the epochs remaining from start_epoch.
    # T_max = remaining epochs so lr reaches eta_min at the final epoch
    # regardless of whether we resumed.
    remaining = max(epochs - start_epoch + 1, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=remaining, eta_min=lr * 1e-2)

    # -----------------------------------------------------------------------
    # Val metric: sampled-action MSE (action-space, not noise-space)
    # -----------------------------------------------------------------------
    def _val_sampled_mse(loader, n_samples: int) -> float:
        """Average action-space MSE over n_samples stochastic draws per batch.

        Runs the full DDPM sampler on each val observation; compares all
        chunk_size steps against ground truth. This is the metric to watch
        for checkpoint selection.
        """
        model.eval()
        total_mse, count = 0.0, 0
        with torch.no_grad():
            for img_b, prop_b, act_b in loader:
                img_b  = img_b.to(device)
                prop_b = prop_b.to(device)
                act_b  = act_b.to(device)   # (B, H, 20) normalised

                mse_sample = 0.0
                for _ in range(n_samples):
                    pred_n = model.sample(img_b, prop_b)  # (B, H, 20) normalised
                    mse_sample += (pred_n - act_b).pow(2).mean().item()
                mse_sample /= n_samples

                total_mse += mse_sample * img_b.shape[0]
                count      += img_b.shape[0]
        model.train()
        return total_mse / count if count > 0 else float("inf")

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_mse = float("inf")
    best_epoch   = -1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        train_loss = 0.0
        for img_b, prop_b, act_b in train_loader:
            img_b  = img_b.to(device)
            prop_b = prop_b.to(device)
            act_b  = act_b.to(device)          # (B, H, 20) normalised

            optimizer.zero_grad()
            loss = model.loss_ddpm(act_b, img_b, prop_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * img_b.shape[0]

        scheduler.step()
        train_loss /= n_train

        # Val every 5 epochs; always on the last epoch
        if epoch % 5 == 0 or epoch == epochs:
            val_mse = _val_sampled_mse(val_loader, n_val_samples)
            marker = ""
            if val_mse < best_val_mse:
                best_val_mse = val_mse
                best_epoch   = epoch
                marker = "  ← best"
                # Save checkpoint
                os.makedirs(os.path.dirname(CKPT_OUT), exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "policy_type":      "diffusion_chunked",
                    "chunk_size":       chunk_size,
                    "exec_cadence":     EXEC_CADENCE,
                    "n_timesteps":      N_TIMESTEPS,
                    "proprio_dim":      data["proprios"].shape[1],
                    "action_dim":       bc_model.ACTION_DIM,
                    "action_mean":      data["action_mean"],
                    "action_std":       data["action_std"],
                    "proprio_mean":     data["proprio_mean"],
                    "proprio_std":      data["proprio_std"],
                    "epoch":            epoch,
                    "val_sampled_mse":  val_mse,
                }, CKPT_OUT)
                volume.commit()
            print(f"epoch {epoch:3d}/{epochs}  "
                  f"train_loss={train_loss:.4f}  "
                  f"val_sampled_mse={val_mse:.4f}{marker}")
        else:
            print(f"epoch {epoch:3d}/{epochs}  train_loss={train_loss:.4f}")

    print(f"\nDone. Best val_sampled_mse={best_val_mse:.4f} at epoch {best_epoch}.")
    print(f"Checkpoint: {CKPT_OUT}")