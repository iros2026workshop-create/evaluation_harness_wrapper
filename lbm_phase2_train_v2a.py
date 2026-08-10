"""
LBM Eval on Modal — Phase 2 v2a: diffusion-policy trainer
=========================================================
Trains the v2a diffusion policy on the SAME cached pick_and_place_box data the
v1 BC policy used (processed_bc_v1.pt), and writes a SEPARATE checkpoint
(bc_v2a.pt) so the v1 baseline stays intact for comparison.

This is the clean ablation half of the diffusion plan: data path frozen (reuses
the exact cache — no prepare_data rerun), eval/fan-out machinery untouched, only
the decoder changes (MLP regressor -> DDPM epsilon-predictor). The architecture
lives in the shared bc_model.py, imported by both this trainer and the server,
so train and serve cannot drift. 

WHAT CHANGES vs the v1 trainer
------------------------------
  - model: bc_model.build_model(..., policy_type="diffusion") instead of the
    inline MLP-head model.
  - loss: DDPM noise-prediction. Sample a timestep t, add noise to the clean
    (normalized) action, predict the noise, MSE on the NOISE. This is NOT
    comparable to v1's 0.0149 — it's error on noise, a different quantity.
  - checkpoint selection: because the training loss is no longer an
    action-space quantity, selection uses a SAMPLED-ACTION val metric: actually
    run the sampler on each val observation and measure action-space MSE vs the
    ground-truth action (averaged over n_val_samples draws, since sampling is
    stochastic). This is comparable in spirit to v1's val MSE and is the number
    to watch.
  - checkpoint: writes bc_v2a.pt with policy_type="diffusion" and n_timesteps,
    which the server reads to rebuild the right architecture.

Run (prepare_data from lbm_phase2_train.py must already have produced the cache):
  modal run lbm_phase2_train_v2a.py::train
"""

import modal

APP_NAME = "lbm-eval-phase2-train-v2a"
app = modal.App(APP_NAME)

volume = modal.Volume.from_name("lbm-eval-data", create_if_missing=True)
VOL_PATH = "/data"

# Reuse the v1 cache (data path frozen); write a NEW checkpoint.
CACHE_PATH = f"{VOL_PATH}/phase2_training_data/processed_bc_v1.pt"
CKPT_PATH = f"{VOL_PATH}/phase2_models/bc_v2a.pt"

ACTION_DIM = 20
IMG_H, IMG_W = 240, 320
N_TIMESTEPS = 100        # DDPM steps; vector denoiser -> full-depth is cheap

# bc_model is imported INSIDE the remote train() function, so the module must
# be importable in the container: add it as local python source.
train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.3.0", "torchvision==0.18.0", "numpy", "tqdm")
    .add_local_python_source("bc_model")
)


@app.function(image=train_image, gpu="L4", volumes={VOL_PATH: volume},
              timeout=10800)
def train(epochs: int = 150, batch_size: int = 64, lr: float = 1e-4,
          val_frac: float = 0.15, n_val_samples: int = 2):
    """Train the v2a diffusion policy from the v1 cache; write bc_v2a.pt."""
    import os
    import pickle
    import numpy as np
    import torch
    import torch.nn as nn

    import bc_model

    if not os.path.exists(CACHE_PATH):
        raise SystemExit(
            f"{CACHE_PATH} missing — run lbm_phase2_train.py::prepare_data "
            f"first (v2a reuses the v1 cache).")
    with open(CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  samples={data['images'].shape[0]}  "
          f"n_timesteps={N_TIMESTEPS}")

    # --- tensors: identical preprocessing to v1 ---------------------------
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs = torch.from_numpy(data["images"]).float().permute(0, 3, 1, 2) / 255.0
    imgs = (imgs - imagenet_mean) / imagenet_std
    a_mean = torch.from_numpy(data["action_mean"])
    a_std = torch.from_numpy(data["action_std"])
    p_mean = torch.from_numpy(data["proprio_mean"])
    p_std = torch.from_numpy(data["proprio_std"])
    proprio = (torch.from_numpy(data["proprios"]) - p_mean) / p_std
    action = (torch.from_numpy(data["actions"]) - a_mean) / a_std   # clean tgt

    # --- same split (seed 0, val_frac 0.15) as v1 ------------------------
    n = imgs.shape[0]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = int(n * val_frac)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"train={len(tr_idx)}  val={len(val_idx)}")

    model = bc_model.build_model(
        proprio.shape[1], ACTION_DIM, policy_type="diffusion",
        n_timesteps=N_TIMESTEPS, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()

    def batches(idx, shuffle):
        order = idx[torch.randperm(len(idx))] if shuffle else idx
        for i in range(0, len(order), batch_size):
            b = order[i:i + batch_size]
            yield (imgs[b].to(device), proprio[b].to(device),
                   action[b].to(device))

    best_val = float("inf")
    for ep in range(1, epochs + 1):
        # --- train: DDPM noise-prediction --------------------------------
        model.train()
        tr_loss = nb = 0.0
        for img, p, act in batches(tr_idx, shuffle=True):
            opt.zero_grad()
            cond = model.encode(img, p)
            t = torch.randint(0, N_TIMESTEPS, (img.shape[0],), device=device)
            noise = torch.randn_like(act)
            noisy = model.q_sample(act, t, noise)
            pred = model.predict_noise(noisy, t, cond)
            loss = mse(pred, noise)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            nb += 1

        # --- val: SAMPLED-ACTION MSE (the selection metric) --------------
        model.eval()
        v_loss = vb = 0.0
        with torch.no_grad():
            for img, p, act in batches(val_idx, shuffle=False):
                acc = 0.0
                for _ in range(n_val_samples):
                    sampled = model.sample(img, p)       # (B,20) normalized
                    acc += mse(sampled, act).item()
                v_loss += acc / n_val_samples
                vb += 1
        tr = tr_loss / max(nb, 1)                         # noise-pred MSE
        va = v_loss / max(vb, 1)                          # action-space MSE

        flag = ""
        if va < best_val:
            best_val = va
            flag = "  <- best"
            os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "proprio_dim": proprio.shape[1],
                    "action_dim": ACTION_DIM,
                    "policy_type": "diffusion",     # server branches on this
                    "n_timesteps": N_TIMESTEPS,
                    "action_mean": data["action_mean"],
                    "action_std": data["action_std"],
                    "proprio_mean": data["proprio_mean"],
                    "proprio_std": data["proprio_std"],
                    "scene_camera": data["scene_camera"],
                    "proprio_keys": data["proprio_keys"],
                    "img_hw": [IMG_H, IMG_W],
                },
                CKPT_PATH,
            )
            volume.commit()
        if ep % 5 == 0 or ep == 1 or flag:
            print(f"  epoch {ep:3d}  train noise-MSE={tr:.4f}  "
                  f"val action-MSE={va:.4f}{flag}")

    print(f"\nbest val action-MSE={best_val:.4f}")
    print(f"checkpoint -> {CKPT_PATH}")
    print("Compare against v1's 5.1% [2.8%, 9.2%] on the identical 195 scenes "
          "by pointing lbm_phase3.py's CKPT_PATH at bc_v2a.pt and rerunning "
          "run_replication. (val action-MSE here is NOT comparable to v1's "
          "0.0149 — v1's was a direct regression MSE; this is sampled-action "
          "MSE. Judge v2a by the de-trivialized Phase 3 success rate.)")
    return {"best_val_action_mse": best_val}


@app.local_entrypoint()
def main():
    train.remote()