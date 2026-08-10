"""
LBM Eval — Phase 2 v2b-bc: chunked BC regressor trainer
=========================================================
Chunked BC: predict H future actions in one forward pass with MSE loss,
execute K open-loop before re-querying. Same receding-horizon execution as
v2b diffusion, but with a deterministic MLP regressor head instead of DDPM.

This is the control experiment for v2b:
  - v1  (bc, H=1):  5.1%  — single-step MSE regressor
  - v2a (diffusion, H=1):  0%   — single-step DDPM, broken by per-step noise
  - v2b (diffusion, H=16): 0%   — chunked DDPM, diffusion still not fitting
  - v2b-bc (bc, H=8):      ???  — chunked MSE regressor

H=8, K=4 (half of v2b's H=16, K=8) for two reasons:
  1. The v2b kickoff note flagged H=8,K=4 as the natural follow-up ablation.
  2. With a regressor the prediction target is a flat (H×20)-dim vector — a
     deterministic MLP regressing 160 dims is much more tractable than a DDPM
     over 320 dims on 2718 samples.

Data: reuses processed_bc_v2b.pt (chunk_size=16 windows). We slice the first
H=8 steps of each 16-step window rather than rebuilding the cache — the window
already covers at least 8 steps for every kept episode, so no data is lost.

Architecture: same ResNet-18 + proprio-MLP encoder (640-dim), MLP head outputs
(H×20) flattened, reshaped to (H,20) before loss. policy_type="bc_chunked".

Checkpoint: bc_v2b_bc.pt — separate file, all previous checkpoints preserved.
"""

import modal

APP_NAME = "lbm-eval-phase2-train-v2b-bc"
app = modal.App(APP_NAME)

volume = modal.Volume.from_name("lbm-eval-data", create_if_missing=True)
VOL_PATH = "/data"

V2B_CACHE_PATH = f"{VOL_PATH}/phase2_training_data/processed_bc_v2b.pt"
CKPT_OUT       = f"{VOL_PATH}/phase2_models/bc_v2b_bc.pt"

CHUNK_SIZE   = 8   # H: predict 8 future actions (sliced from 16-step cache)
EXEC_CADENCE = 4   # K: execute 4 before re-querying (serve-time only)

train_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch==2.3.0", "torchvision==0.18.0", "numpy", "tqdm")
    .add_local_file("bc_model.py", "/root/bc_model.py")
)


@app.function(image=train_image, gpu="L4", volumes={VOL_PATH: volume},
              timeout=7200)
def train(epochs: int = 120, batch_size: int = 64, lr: float = 1e-4,
          val_frac: float = 0.15, seed: int = 0, out_path: str = CKPT_OUT):
    import os, sys, pickle
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset, random_split

    sys.path.insert(0, "/root")
    import bc_model

    volume.reload()
    import random
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    print(f"seed={seed}  out_path={out_path}")

    if not os.path.exists(V2B_CACHE_PATH):
        raise SystemExit(f"v2b cache not found at {V2B_CACHE_PATH}. "
                         "Run lbm_phase2_train_v2b.py::prepare_data_v2b first.")

    with open(V2B_CACHE_PATH, "rb") as f:
        data = pickle.load(f)

    cache_chunk = data["chunk_size"]   # 16
    assert cache_chunk >= CHUNK_SIZE, (
        f"Cache chunk_size={cache_chunk} < requested CHUNK_SIZE={CHUNK_SIZE}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    N = data["images"].shape[0]
    print(f"device={device}  N={N}  chunk_size={CHUNK_SIZE} "
          f"(sliced from cache chunk_size={cache_chunk})  epochs={epochs}")

    # Normalization constants
    a_mean = torch.from_numpy(data["action_mean"]).float()   # (20,)
    a_std  = torch.from_numpy(data["action_std"]).float()    # (20,)
    p_mean = torch.from_numpy(data["proprio_mean"]).float()  # (18,)
    p_std  = torch.from_numpy(data["proprio_std"]).float()   # (18,)

    # Images: uint8 → float, ImageNet normalise, NCHW
    imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    imgs = torch.from_numpy(data["images"]).float().permute(0, 3, 1, 2) / 255.0
    imgs = (imgs - imagenet_mean) / imagenet_std              # (N, 3, H, W)

    proprios = (torch.from_numpy(data["proprios"]).float() - p_mean) / p_std

    # Slice first CHUNK_SIZE steps from each 16-step window
    acts_raw = torch.from_numpy(data["actions"]).float()     # (N, 16, 20)
    acts_raw = acts_raw[:, :CHUNK_SIZE, :]                   # (N, 8, 20)
    acts = (acts_raw - a_mean.unsqueeze(0).unsqueeze(0)) / \
           a_std.unsqueeze(0).unsqueeze(0)                   # (N, 8, 20)

    dataset = TensorDataset(imgs, proprios, acts)
    n_val   = max(1, int(len(dataset) * val_frac))
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size,
                              shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size,
                              shuffle=False, num_workers=2, pin_memory=True)

    model = bc_model.build_model(
        proprio_dim=data["proprios"].shape[1],
        action_dim=bc_model.ACTION_DIM,
        policy_type="bc_chunked",
        chunk_size=CHUNK_SIZE,
        pretrained=True,
    ).to(device)

    loss_fn   = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=lr * 1e-2)

    best_val, best_epoch = float("inf"), -1

    for epoch in range(1, epochs + 1):
        model.train()
        tr_loss = 0.0
        for img_b, prop_b, act_b in train_loader:
            img_b  = img_b.to(device)
            prop_b = prop_b.to(device)
            act_b  = act_b.to(device)           # (B, 8, 20)

            optimizer.zero_grad()
            pred = model(img_b, prop_b)         # (B, 8, 20)
            loss = loss_fn(pred, act_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item() * img_b.shape[0]

        scheduler.step()
        tr_loss /= n_train

        if epoch % 5 == 0 or epoch == epochs:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for img_b, prop_b, act_b in val_loader:
                    pred = model(img_b.to(device), prop_b.to(device))
                    val_loss += loss_fn(pred, act_b.to(device)).item() * img_b.shape[0]
            val_loss /= n_val
            model.train()

            marker = ""
            if val_loss < best_val:
                best_val, best_epoch = val_loss, epoch
                marker = "  ← best"
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "policy_type":      "bc_chunked",
                    "chunk_size":       CHUNK_SIZE,
                    "exec_cadence":     EXEC_CADENCE,
                    "proprio_dim":      data["proprios"].shape[1],
                    "action_dim":       bc_model.ACTION_DIM,
                    "action_mean":      data["action_mean"],
                    "action_std":       data["action_std"],
                    "proprio_mean":     data["proprio_mean"],
                    "proprio_std":      data["proprio_std"],
                    "epoch":            epoch,
                    "val_mse":          val_loss,
                    "seed":             seed,
                }, out_path)
                volume.commit()
            print(f"epoch {epoch:3d}/{epochs}  "
                  f"train_mse={tr_loss:.4f}  val_mse={val_loss:.4f}{marker}")
        else:
            print(f"epoch {epoch:3d}/{epochs}  train_mse={tr_loss:.4f}")

    print(f"\nDone. Best val_mse={best_val:.4f} at epoch {best_epoch}.")
    print(f"Checkpoint: {out_path}")


@app.local_entrypoint()
def main():
    train.remote()