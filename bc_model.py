"""
LBM Eval — shared policy architecture (bc_model.py)
====================================================
SINGLE SOURCE OF TRUTH for all policy architectures, imported by BOTH the
trainer (lbm_phase2_train*.py) and the policy server (bc_policy_server.py).

Policy types and their checkpoints
-----------------------------------
  "bc"                 bc_v1.pt   — BC MSE regressor (v1 baseline, 5.1%)
  "diffusion"          bc_v2a.pt  — single-step DDPM, no chunking (v2a, 0/195)
  "diffusion_chunked"  bc_v2b.pt  — chunked DDPM, H actions / K execute (v2b)

All three share the same encoder: ResNet-18 backbone + 2-layer proprio MLP,
producing a 640-dim conditioning vector. The decoder (head) is what differs.

build_model() is the public factory. The server calls it after reading
`policy_type` (and optional `chunk_size`) from the checkpoint, so adding a new
type here is the only change needed for the server to load it.

Backward compatibility: bc_v1.pt checkpoints produced before this file existed
have no `policy_type` key. build_model defaults to "bc" in that case — the
server inserts the default before calling build_model (see bc_policy_server.py).
"""

import math
import torch
import torch.nn as nn
import torchvision.models as tvm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTION_DIM = 20   # 3+6 right arm xyz+rot6d, 3+6 left, 1+1 grippers
PROPRIO_DIM = 18  # 3+6 right, 3+6 left (no gripper in proprio)

# Encoder output sizes
_IMG_FEAT_DIM = 512   # ResNet-18 penultimate
_PROP_FEAT_DIM = 128  # proprio MLP output
COND_DIM = _IMG_FEAT_DIM + _PROP_FEAT_DIM  # 640


# ---------------------------------------------------------------------------
# Shared encoder
# ---------------------------------------------------------------------------
class _Encoder(nn.Module):
    """ResNet-18 visual backbone + 2-layer proprio MLP → 640-dim feature."""

    def __init__(self, proprio_dim: int, pretrained: bool = True):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = tvm.resnet18(weights=weights)
        backbone.fc = nn.Identity()
        self.encoder = backbone                     # (B,3,H,W) → (B,512)
        self.proprio_mlp = nn.Sequential(
            nn.Linear(proprio_dim, 128), nn.ReLU(),
            nn.Linear(128, _PROP_FEAT_DIM), nn.ReLU(),
        )                                           # (B,D) → (B,128)

    def forward(self, img: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        img_feat = self.encoder(img)                # (B,512)
        prp_feat = self.proprio_mlp(proprio)        # (B,128)
        return torch.cat([img_feat, prp_feat], dim=1)  # (B,640)


# ---------------------------------------------------------------------------
# v1: BC MSE regressor
# ---------------------------------------------------------------------------
class BCModel(nn.Module):
    """Deterministic MLP regressor head.  forward() → (B, action_dim)."""

    def __init__(self, proprio_dim: int, action_dim: int = ACTION_DIM,
                 pretrained: bool = True):
        super().__init__()
        self.encoder = _Encoder(proprio_dim, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Linear(COND_DIM, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, img: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(img, proprio))


# ---------------------------------------------------------------------------
# bc_chunked: deterministic MLP regressor over a chunk of H actions
# ---------------------------------------------------------------------------
class ChunkedBCModel(nn.Module):
    """Chunked BC regressor: forward() → (B, chunk_size, action_dim).

    Same encoder as BCModel; the head outputs chunk_size * action_dim dims
    flattened, then reshaped. Training uses MSE over the full chunk.
    At serve time the chunk buffer pops one action per step (see server).
    """

    def __init__(self, proprio_dim: int, action_dim: int = ACTION_DIM,
                 chunk_size: int = 8, pretrained: bool = True):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.encoder = _Encoder(proprio_dim, pretrained=pretrained)
        self.head = nn.Sequential(
            nn.Linear(COND_DIM, 512), nn.ReLU(),
            nn.Linear(512, 512), nn.ReLU(),
            nn.Linear(512, action_dim * chunk_size),
        )

    def forward(self, img: torch.Tensor,
                proprio: torch.Tensor) -> torch.Tensor:
        """Returns (B, chunk_size, action_dim)."""
        feats = self.encoder(img, proprio)           # (B, 640)
        flat  = self.head(feats)                     # (B, chunk_size*action_dim)
        return flat.reshape(-1, self.chunk_size, self.action_dim)


# ---------------------------------------------------------------------------
# v2a / v2b: DDPM ε-predictor denoiser
# ---------------------------------------------------------------------------
class _SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal timestep embedding."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32,
                                             device=t.device) / (half - 1)
        )
        emb = t.float().unsqueeze(1) * freqs.unsqueeze(0)  # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)   # (B, dim)


class _DenoiseMLP(nn.Module):
    """MLP denoiser: (noisy_action, cond, t_emb) → predicted noise.

    Input layout: concat([noisy_action, cond, t_emb]) at every layer is the
    simplest conditioning; avoids FiLM complexity for this small action dim.
    action_flat_dim is chunk_size * action_dim (scalar action_dim for v2a,
    H*action_dim for v2b).
    """
    def __init__(self, action_flat_dim: int, cond_dim: int = COND_DIM,
                 t_emb_dim: int = 64, hidden: int = 256):
        super().__init__()
        self.t_emb = _SinusoidalPosEmb(t_emb_dim)
        in_dim = action_flat_dim + cond_dim + t_emb_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, action_flat_dim),
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, action_flat_dim) — noisy action
        cond: (B, cond_dim)
        t:    (B,) int64 or float — diffusion timestep index
        """
        t_emb = self.t_emb(t)
        h = torch.cat([x, cond, t_emb], dim=-1)
        return self.net(h)


class DiffusionPolicy(nn.Module):
    """Conditional DDPM over a flat action vector.

    For v2a: action_dim=20, chunk_size=1 (default).
    For v2b: action_dim=20, chunk_size=H  → predicts H*20 dims in one pass.

    forward(img, proprio) runs the FULL reverse diffusion chain and returns
    a (B, chunk_size, action_dim) tensor (chunk_size=1 → same as before for
    v2a compatibility; callers squeeze dim 1 if they want (B,20)).

    Training: use loss_ddpm() instead of forward() to get the noise-prediction
    loss on a batch of (noisy_action, cond, t) triples.
    """

    def __init__(self, proprio_dim: int, action_dim: int = ACTION_DIM,
                 n_timesteps: int = 100, chunk_size: int = 1,
                 pretrained: bool = True):
        super().__init__()
        self.action_dim = action_dim
        self.n_timesteps = n_timesteps
        self.chunk_size = chunk_size
        self.action_flat_dim = action_dim * chunk_size  # 20 (v2a) or 320 (v2b H=16)

        self.encoder = _Encoder(proprio_dim, pretrained=pretrained)
        self.denoiser = _DenoiseMLP(self.action_flat_dim)

        # DDPM schedule (linear beta, cosine would also work)
        betas = torch.linspace(1e-4, 2e-2, n_timesteps)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Register as buffers so .to(device) moves them automatically
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", alpha_bar.sqrt())
        self.register_buffer("sqrt_one_minus_alpha_bar", (1 - alpha_bar).sqrt())

    # ------------------------------------------------------------------
    # Training helper
    # ------------------------------------------------------------------
    def loss_ddpm(self, clean_action: torch.Tensor, img: torch.Tensor,
                  proprio: torch.Tensor) -> torch.Tensor:
        """Compute noise-prediction MSE for a batch.

        clean_action: (B, chunk_size, action_dim) — normalised, from DataLoader.
        Returns scalar loss.
        """
        B = clean_action.shape[0]
        device = clean_action.device

        # Flatten chunk dimension for the MLP
        x0 = clean_action.reshape(B, self.action_flat_dim)  # (B, H*20)

        # Sample random timesteps
        t = torch.randint(0, self.n_timesteps, (B,), device=device)

        # Add noise: x_t = sqrt(ā_t) * x0 + sqrt(1-ā_t) * ε
        eps = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alpha_bar[t].unsqueeze(1)        # (B,1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].unsqueeze(1)
        x_t = sqrt_ab * x0 + sqrt_1mab * eps

        # Condition
        with torch.no_grad():
            cond = self.encoder(img, proprio)               # (B, 640)
        # Allow encoder grads during training by NOT using no_grad here;
        # the with block above is wrong for training — compute cond outside:
        cond = self.encoder(img, proprio)

        # Predict noise and compute loss
        eps_hat = self.denoiser(x_t, cond, t)
        return nn.functional.mse_loss(eps_hat, eps)

    # ------------------------------------------------------------------
    # Sampling (inference)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, img: torch.Tensor,
               proprio: torch.Tensor) -> torch.Tensor:
        """Run full DDPM reverse chain conditioned on (img, proprio).

        Returns (B, chunk_size, action_dim).  B is inferred from img.
        """
        B = img.shape[0]
        device = img.device
        cond = self.encoder(img, proprio)                    # (B, 640)

        x = torch.randn(B, self.action_flat_dim, device=device)
        for t_idx in reversed(range(self.n_timesteps)):
            t_batch = torch.full((B,), t_idx, device=device, dtype=torch.long)
            eps_hat = self.denoiser(x, cond, t_batch)

            alpha_t = self.alphas[t_idx]
            alpha_bar_t = self.alpha_bar[t_idx]
            sqrt_1mab = self.sqrt_one_minus_alpha_bar[t_idx]

            # DDPM mean
            x0_hat = (x - sqrt_1mab * eps_hat) / alpha_bar_t.sqrt()
            x0_hat = x0_hat.clamp(-3.0, 3.0)

            mean = (x - (1 - alpha_t) / sqrt_1mab * eps_hat) / alpha_t.sqrt()

            if t_idx > 0:
                beta_t = self.betas[t_idx]
                # Posterior variance (simplified: β_t)
                noise = torch.randn_like(x)
                x = mean + beta_t.sqrt() * noise
            else:
                x = mean

        return x.reshape(B, self.chunk_size, self.action_dim)  # (B, H, 20)

    def forward(self, img: torch.Tensor,
                proprio: torch.Tensor) -> torch.Tensor:
        """Alias for sample() so the server can call model(img, prop) uniformly.

        Returns (B, chunk_size, action_dim).
        """
        return self.sample(img, proprio)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def build_model(proprio_dim: int, action_dim: int = ACTION_DIM,
                policy_type: str = "bc", n_timesteps: int = 100,
                chunk_size: int = 1, pretrained: bool = True) -> nn.Module:
    """Build a policy network from its type string.

    policy_type values (stored in checkpoint under 'policy_type'):
      "bc"                 — BCModel, MSE regressor (v1)
      "diffusion"          — DiffusionPolicy, chunk_size=1 (v2a)
      "diffusion_chunked"  — DiffusionPolicy, chunk_size=H (v2b)

    The server reads policy_type (defaulting to "bc" for old checkpoints) and
    chunk_size (defaulting to 1) from the checkpoint before calling this.
    """
    if policy_type == "bc":
        return BCModel(proprio_dim, action_dim, pretrained=pretrained)
    if policy_type == "bc_chunked":
        return ChunkedBCModel(proprio_dim, action_dim,
                              chunk_size=chunk_size, pretrained=pretrained)
    if policy_type in ("diffusion", "diffusion_chunked"):
        return DiffusionPolicy(
            proprio_dim, action_dim,
            n_timesteps=n_timesteps,
            chunk_size=chunk_size,
            pretrained=pretrained,
        )
    raise ValueError(f"unknown policy_type: {policy_type!r}")