# tiny_dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from typing import Optional, Dict

Batch = namedtuple('Batch', 'trajectories conditions')

class SimpleNormalizer:
    """三种模式：'none' | 'limits'([-1,1]) | 'gaussian'(0均值1方差)"""
    def __init__(self, X, mode='limits', eps=1e-8):
        self.mode = mode
        X = X.float()
        if mode == 'limits':
            self.min = X.amin(dim=0)
            self.max = X.amax(dim=0)
            # 防止除0
            self.range = torch.clamp(self.max - self.min, min=eps)
        elif mode == 'gaussian':
            self.mean = X.mean(dim=0)
            self.std  = X.std(dim=0).clamp_min(eps)

    def normalize(self, X):
        if self.mode == 'none':
            return X
        if self.mode == 'limits':
            # [0,1] -> [-1,1]
            z = (X - self.min) / self.range
            return z * 2 - 1
        if self.mode == 'gaussian':
            return (X - self.mean) / self.std
        raise ValueError(self.mode)

    def unnormalize(self, X):
        if self.mode == 'none':
            return X
        if self.mode == 'limits':
            z = (X + 1) / 2
            return z * self.range + self.min
        if self.mode == 'gaussian':
            return X * self.std + self.mean
        raise ValueError(self.mode)


class PandaSequenceDataset(Dataset):
    """
    states:  (N, H, 7)  —— 关节状态（你说的“状态和速度”）
    actions: (N, H, 7)  —— 控制量（比如力矩/位置指令）
    x0:      (N, 7)     —— 第0时刻状态（等同于 states[:,0]，若已给就直接用）
    """
    def __init__(
        self,
        states: torch.Tensor,      # (N,H,7)
        actions: torch.Tensor,     # (N,H,7)
        x0: torch.Tensor,          # (N,7)
        norm_mode_state: str = 'limits',   # 'none' | 'limits' | 'gaussian'
        norm_mode_action: str = 'limits',
    ):
        assert states.ndim == actions.ndim == 3
        assert states.shape == actions.shape
        assert x0.shape == states[:,0].shape
        self.states = states.float().contiguous()
        self.actions = actions.float().contiguous()
        self.x0 = x0.float().contiguous()
        self.N, self.H, self.dim = self.states.shape

        # 计算归一化器（按所有步展平到 (N*H, dim)）
        S = self.states.reshape(-1, self.dim)
        A = self.actions.reshape(-1, self.dim)
        self.state_norm = SimpleNormalizer(S, mode=norm_mode_state)
        self.action_norm = SimpleNormalizer(A, mode=norm_mode_action)

        # 预先归一化副本（避免反复算）
        self.norm_states  = self.state_norm.normalize(self.states)
        self.norm_actions = self.action_norm.normalize(self.actions)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # (H,7), (H,7)
        obs = self.norm_states[idx]
        act = self.norm_actions[idx]

        # 模型期望：按最后一维拼接 -> (H, 14)
        trajectories = torch.cat([act, obs], dim=-1)

        # 条件：第0步状态；注意这里用“归一化空间”的状态，和训练保持一致
        cond0 = obs[0]                  # (7,)
        conditions: Dict[int, torch.Tensor] = {0: cond0}

        return Batch(trajectories, conditions)


def collate_batch(list_of_batches):
    """
    把若干个 Batch 合并为一个大 batch：
    - trajectories: (B,H,14)
    - conditions:   {0: (B,7)}
    DataLoader 默认的 collate 对 dict 会产生 list，这里手动栈起来。
    """
    trajs = torch.stack([b.trajectories for b in list_of_batches], dim=0)  # (B,H,14)

    # 收集所有键（一般只有 key=0）
    all_keys = set().union(*[b.conditions.keys() for b in list_of_batches])
    cond = {}
    for k in sorted(all_keys):
        cond[k] = torch.stack([b.conditions[k] for b in list_of_batches], dim=0)  # (B,7)

    return Batch(trajs, cond)


def make_loader(
    states_pt, actions_pt, x0_pt,
    batch_size=256, shuffle=True,
    num_workers=None, pin_memory=None, persistent_workers=True,
    norm_mode_state='gaussian', norm_mode_action='gaussian',  # ← 新增
):
    def load(x):
        if isinstance(x, str):
            return torch.load(x)
        return x

    states  = load(states_pt)
    actions = load(actions_pt)
    x0      = load(x0_pt)

    # 这里把归一化模式传进去（原来用的 **norm_kwargs 会报 NameError）
    ds = PandaSequenceDataset(
        states, actions, x0,
        norm_mode_state=norm_mode_state,
        norm_mode_action=norm_mode_action,
    )

    if num_workers is None:
        import os
        num_workers = max(1, (os.cpu_count() or 2) // 2) if torch.cuda.is_available() else 0
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_batch,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
    )
    return loader, ds
