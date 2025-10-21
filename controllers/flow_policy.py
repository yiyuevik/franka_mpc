# controllers/flow_policy.py
import torch
import torch.nn as nn
import numpy as np

from models_flow.transformer import TransformerFlow  # same backbone as training


def _to_tensor(x):
    return x if isinstance(x, torch.Tensor) else torch.as_tensor(x, dtype=torch.float32)


class _Normalizer:
    """Normalizer that is device-safe and supports numpy/torch & batched inputs."""
    def __init__(self, stats, device):
        self.mode = stats.get('mode', 'none')
        for k, v in stats.items():
            if k != 'mode':
                tensor = torch.as_tensor(v, dtype=torch.float32).to(device)
                setattr(self, k, tensor)

    def _align(self, x):
        x = _to_tensor(x).float()
        # pick a reference tensor based on mode (avoid boolean ctx on tensors)
        ref = None
        if self.mode == 'gaussian' and hasattr(self, 'mean'):
            ref = self.mean
        elif self.mode == 'limits' and hasattr(self, 'min'):
            ref = self.min
        # fallbacks if above not present
        if ref is None:
            for name in ('mean', 'min', 'std', 'range'):
                if hasattr(self, name):
                    ref = getattr(self, name)
                    break
        if ref is not None and x.device != ref.device:
            x = x.to(ref.device)
        return x

    def norm(self, x):
        x = self._align(x)
        if self.mode == 'gaussian':
            return (x - self.mean) / (self.std + 1e-8)
        if self.mode == 'limits':
            z = (x - self.min) / (self.range + 1e-8)
            return z * 2 - 1
        return x

    def unnorm(self, x):
        x = self._align(x)
        if self.mode == 'gaussian':
            return x * (self.std + 1e-8) + self.mean
        if self.mode == 'limits':
            z = (x + 1) / 2
            return z * self.range + self.min
        return x


class FlowPolicy(nn.Module):
    """
    Batched Euler sampler (no guidance, no constraints).
    Dynamics in latent trajectory space:
      - X ∈ R^{B×H×(A+S)}, dX/dt = v_theta(X, t)
      - Keep the step-0 state fixed by zeroing its derivative each Euler step
      - AMP (fp16/bf16) speeds up GPU inference
    """
    def __init__(
        self,
        ckpt_path: str,
        norm_stats_path: str,
        state_dim: int,
        action_dim: int,
        horizon: int,
        hidden_size: int,
        depth: int,
        num_heads: int,
        device: str = 'cuda',
        euler_steps: int = 100,               # #Euler steps on t∈[0,1]
        use_amp: bool = True,                 # enable AMP
        amp_dtype: torch.dtype = torch.float16,  # or torch.bfloat16
        matmul_precision: str = 'high',       # PyTorch 2.x: 'medium'/'high'
    ):
        super().__init__()
        self.device = torch.device(device)
        self.S = state_dim
        self.A = action_dim
        self.H = horizon
        self.euler_steps = int(euler_steps)
        self.use_amp = use_amp and (self.device.type == 'cuda')
        self.amp_dtype = amp_dtype

        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision(matmul_precision)

        # 1) Model (must match training)
        self.model = TransformerFlow(
            seq_len=horizon,
            in_channels=state_dim + action_dim,
            out_channels=None,
            hidden_size=hidden_size,
            depth=depth,
            num_heads=num_heads,
        ).to(self.device)
        sd = torch.load(ckpt_path, map_location='cpu')
        if isinstance(sd, dict) and 'state_dict' in sd:
            sd = sd['state_dict']
        if isinstance(sd, dict) and 'module' in sd:
            sd = sd['module']
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()

        # 2) Normalizers
        ns = torch.load(norm_stats_path, map_location='cpu')
        self.state_norm = _Normalizer(ns['state_norm'], device=self.device)
        self.action_norm = _Normalizer(ns['action_norm'], device=self.device)

    # ---------- public APIs ----------

    @torch.inference_mode()
    def act(self, q_curr_np: np.ndarray) -> np.ndarray:
        """Single-env API: (S,) -> (A,)"""
        return self.act_batch(q_curr_np[None, :])[0]

    @torch.inference_mode()
    def act_batch(self, q_batch_np: np.ndarray) -> np.ndarray:
        """
        Batched API:
          q_batch_np: (B, S) numpy
          returns:    (B, A) numpy (in env action space)
        """
        q = torch.from_numpy(q_batch_np).to(self.device, dtype=torch.float32)  # (B, S)
        qn = self.state_norm.norm(q)                                           # (B, S)
        B = qn.shape[0]

        # X(0) ~ N(0, I), then write the step-0 state token with the condition
        X = torch.randn(B, self.H, self.A + self.S, device=self.device)        # (B, H, A+S)
        X[:, 0, self.A:] = qn                                                  # fix value at step-0

        # Euler over t ∈ [0,1]; t_i = i/(N-1), dt = 1/(N-1)
        if self.euler_steps <= 1:
            dt = 1.0
            ts = [torch.tensor(0.0, device=self.device)]
        else:
            dt = 1.0 / (self.euler_steps - 1 + 1e-8)
            ts = [torch.tensor(i * dt, device=self.device) for i in range(self.euler_steps)]

        for t in ts:
            t_b = t.expand(B)  # (B,)
            if self.use_amp:
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                    dX = self.model(X, t_b)  # (B, H, A+S)
            else:
                dX = self.model(X, t_b)

            # zero derivative on step-0 state to keep the condition exact
            dX[:, 0, self.A:] = 0.0

            # Euler step
            X = X + dX * dt

        # Action = step-0 action at t=1, unnormalize to env space
        a_norm = X[:, 0, :self.A]                      # (B, A) on CUDA
        a = self.action_norm.unnorm(a_norm).detach().cpu().numpy()
        return a
