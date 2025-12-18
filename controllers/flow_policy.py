# controllers/flow_policy.py
import torch
import torch.nn as nn
import numpy as np
import configs
from utils.helpers import compute_end_effector_position
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
        n_samples: int = 100,                   # number of samples per state for cost selection
        cost_type: str = 'target',            # 'none' | 'target' | 'obstacle' | 'smoothness'
    ):
        super().__init__()
        self.device = torch.device(device)
        self.S = state_dim
        self.A = action_dim
        self.H = horizon
        self.euler_steps = int(euler_steps)
        self.use_amp = use_amp and (self.device.type == 'cuda')
        self.amp_dtype = amp_dtype
        self.n_samples = n_samples
        self.cost_type = cost_type

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

        # 3) Cost function setup
        self.target_pos = configs.target_position
        self.obstacle_pos = configs.Obstacle_Position
        self.obstacle_scale = configs.Obstacle_Scale
        self.fk_fn = compute_end_effector_position

        if self.cost_type != 'none' and self.n_samples > 1:
            if self.target_pos is None:
                print(f"[WARN] cost_type='{self.cost_type}' but no target_pos provided, disabling cost selection")
                self.cost_type = 'none'
            elif self.fk_fn is None:
                print(f"[WARN] cost_type='{self.cost_type}' but forward kinematics not available, disabling cost selection")
                self.cost_type = 'none'
            else:
                if self.cost_type == 'obstacle' and self.obstacle_pos is not None:
                    print(f"   Obstacle: pos={self.obstacle_pos}, radius={1/self.obstacle_scale:.3f}")

    def _compute_cost(self, state_traj: np.ndarray, action_traj: np.ndarray) -> float:
        """
        Compute cost for a single trajectory.
        
        Args:
            state_traj: (H, S) - predicted joint angle trajectory
            
        Returns:
            scalar cost
        """
        if self.cost_type == 'none' or self.fk_fn is None or self.target_pos is None:
            return 0.0
        
        total_cost = 0.0
        pos_traj = [self.fk_fn(state) for state in state_traj]
        dists = np.linalg.norm(pos_traj - self.target_pos, axis=1)
        total_cost += 10 *  configs.Q_pos[0,0] * dists[:-1].sum() + configs.P_pos[0,0] * dists[-1]
        total_cost += configs.R[0,0] * np.sum(action_traj ** 2) * 10000
        if self.cost_type == 'target':
            return total_cost
        
        # 2. Obstacle avoidance cost
        if self.cost_type in ['obstacle'] and self.obstacle_pos is not None and self.obstacle_radius is not None:
            for q in state_traj:
                pos = self.fk_fn(q)
                dist_to_obstacle = np.linalg.norm(pos - self.obstacle_pos)
                
                # Penetration penalty
                if dist_to_obstacle < self.obstacle_radius:
                    penetration = self.obstacle_radius - dist_to_obstacle
                    total_cost += 1000 * penetration
            return total_cost

        # 3. Smoothness cost
        if self.cost_type in ['smoothness'] and len(state_traj) > 2:
            q_diff2 = np.diff(state_traj, n=2, axis=0)  # (H-2, S) second-order difference
            smoothness_cost = np.sum(q_diff2 ** 2)
            total_cost += 0.01 * smoothness_cost
        
        return total_cost
    
    # ---------- public APIs ----------
    @torch.inference_mode()
    def _sample_trajectories(self, q_batch_np: np.ndarray, n_samples: int = 1):
        """
        Internal method to sample trajectories.
        
        Args:
            q_batch_np: (B, S) numpy - batch of states
            n_samples: int - number of samples per state
            
        Returns:
            actions: (B, n_samples, A) numpy - sampled actions
            state_trajs: (B, n_samples, H, S) numpy - predicted state trajectories
        """
        q = torch.from_numpy(q_batch_np).to(self.device, dtype=torch.float32)  # (B, S)
        qn = self.state_norm.norm(q)                                           # (B, S)
        B = qn.shape[0]
        
        # Expand: each state gets n_samples
        # Shape becomes (B*n_samples, S)
        qn_expanded = qn.unsqueeze(1).expand(B, n_samples, self.S).reshape(B * n_samples, self.S)
        
        # X(0) ~ N(0, I), then write the step-0 state token with the condition
        X = torch.randn(B * n_samples, self.H, self.A + self.S, device=self.device)  # (B*n_samples, H, A+S)
        X[:, 0, self.A:] = qn_expanded                                                # fix value at step-0

        # Euler over t ∈ [0,1]; t_i = i/(N-1), dt = 1/(N-1)
        if self.euler_steps <= 1:
            dt = 1.0
            ts = [torch.tensor(0.0, device=self.device)]
        else:
            dt = 1.0 / (self.euler_steps - 1 + 1e-8)
            ts = [torch.tensor(i * dt, device=self.device) for i in range(self.euler_steps)]

        for t in ts:
            t_b = t.expand(B * n_samples)  # (B*n_samples,)
            if self.use_amp:
                with torch.autocast(device_type='cuda', dtype=self.amp_dtype):
                    dX = self.model(X, t_b)  # (B*n_samples, H, A+S)
            else:
                dX = self.model(X, t_b)

            # zero derivative on step-0 state to keep the condition exact
            dX[:, 0, self.A:] = 0.0

            # Euler step
            X = X + dX * dt

        # Extract actions and state trajectories
        # Actions: step-0 action at t=1
        a_norm = X[:, 0, :self.A]  # (B*n_samples, A)
        a = self.action_norm.unnorm(a_norm).detach()  # keep on GPU for now
        
        a_traj_norm = X[:, :, :self.A]  # (B*n_samples, A)
        a_traj = self.action_norm.unnorm(a_traj_norm).detach()  # keep on GPU for now
        # State trajectories: full horizon state predictions
        s_traj_norm = X[:, :, self.A:]  # (B*n_samples, H, S)
        s_traj = self.state_norm.unnorm(s_traj_norm).detach()  # keep on GPU for now
        
        # Reshape to (B, n_samples, A) and (B, n_samples, H, S)
        a = a.reshape(B, n_samples, self.A).cpu().numpy()
        a_traj = a_traj.reshape(B, n_samples, self.H, self.A).cpu().numpy()
        s_traj = s_traj.reshape(B, n_samples, self.H, self.S).cpu().numpy()
        
        return a, a_traj, s_traj
    
    @torch.inference_mode()
    def act(self, q_curr_np: np.ndarray) -> np.ndarray:
        """Single-env API: (S,) -> (A,)"""
        return self.act_batch(q_curr_np[None, :])[0]

    @torch.inference_mode()
    def act_batch(self, q_batch_np: np.ndarray) -> np.ndarray:
        """
        Batched API with optional multi-sampling and cost selection.
        
        Args:
          q_batch_np: (B, S) numpy
          
        Returns:
          (B, A) numpy (in env action space)
          
        Behavior:
          - If n_samples == 1 or cost_type == 'none': single sample per state
          - If n_samples > 1 and cost_type != 'none': 
              * Sample n_samples trajectories per state (total B*n_samples)
              * Evaluate cost for each trajectory using built-in cost function
              * Return action with minimum cost for each state
        """
        if self.n_samples <= 1 or self.cost_type == 'none':
            # Original behavior: single sample
            return self._sample_trajectories(q_batch_np, n_samples=1)[0][:, 0, :]
        
        # Multi-sampling mode with cost-based selection
        B = q_batch_np.shape[0]
        
        # Sample n_samples trajectories per state
        actions, action_trajs, state_trajs = self._sample_trajectories(q_batch_np, n_samples=self.n_samples)
        # actions: (B, n_samples, A)
        # action_trajs: (B, n_samples, H, A)
        # state_trajs: (B, n_samples, H, S)
        
        # Evaluate cost for each trajectory
        costs = np.zeros((B, self.n_samples))
        for i in range(B):
            for j in range(self.n_samples):
                # state_trajs[i, j]: (H, S) - predicted state trajectory
                costs[i, j] = self._compute_cost(state_trajs[i, j], action_trajs[i, j])
        
        # Select action with minimum cost for each state
        min_indices = np.argmin(costs, axis=1)  # (B,)
        selected_actions = actions[np.arange(B), min_indices]  # (B, A)
        
        return selected_actions