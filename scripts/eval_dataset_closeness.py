#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_dataset_closeness.py
=========================

用法（在仓库根目录执行，已训练好并有 ckpt 与 norm_stats）：
    python scripts/eval_dataset_closeness.py \
        --cfg-train configs/flow_train.yaml \
        --cfg-eval  configs/flow_eval.yaml \
        --num-pairs 20000 \
        --samples-per-state 32 \
        --batch-size 256 \
        --exclude-t0

功能
----
1) **条件动作一致性（one-step）**：在随机抽取的 (s_t, a_t) 对上，
   多次采样策略输出，统计 Top-1 MSE 与 Best-of-N MSE；支持排除 t=0。
2) （可选）**t=0 分布 MMD**：若设置 --mmd-s0，会在完全相同 s0 条件下，
   计算数据集 a0 分布与策略采样分布的 RBF-MMD（越接近 0 越好）。

注意
----
- 本脚本默认 **在“环境空间”比较动作误差**（与 FlowPolicy.act_batch 输出一致）。
- 数据集 .pt 文件路径从 `configs/flow_train.yaml` 里读取；
  策略和归一化从 `configs/flow_eval.yaml` 里读取。
- 你可以用 --only-t0 或 --exclude-t0 控制是否只评估/排除第 0 步。

输出
----
- 一个 JSON 指标文件：runs/eval/<timestamp>/metrics.json
- 两张直方图：Top1_MSE_hist.png、BestOfN_MSE_hist.png
- （可选）MMD_s0.txt

"""

import os
import sys
import json
import yaml
import time
import math
import argparse
from datetime import datetime

import numpy as np
import torch

# 允许从仓库根或 scripts/ 运行
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(REPO_ROOT) == "scripts":
    REPO_ROOT = os.path.dirname(REPO_ROOT)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from controllers.flow_policy import FlowPolicy  # 按你仓库里的实现
# --------- 小工具 ----------

def set_seed(seed: int | None):
    if seed is None:
        return
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def now_ts():
    return datetime.now().strftime("%Y%m%d_%H%M")

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_pt(path: str) -> torch.Tensor:
    # 允许 .pt 用 map_location=cpu
    return torch.load(path, map_location="cpu")

# RBF-MMD
def _pdist2_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # x:(n,d), y:(m,d) -> (n,m) pairwise squared distances
    x2 = (x ** 2).sum(dim=1, keepdim=True)
    y2 = (y ** 2).sum(dim=1, keepdim=True).T
    return x2 + y2 - 2.0 * x @ y.T

def mmd_rbf_torch(X: np.ndarray, Y: np.ndarray, gammas=(0.1, 1.0, 10.0)) -> float:
    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)
    Dxx = _pdist2_torch(X_t, X_t)
    Dyy = _pdist2_torch(Y_t, Y_t)
    Dxy = _pdist2_torch(X_t, Y_t)
    Kxx = 0.0
    Kyy = 0.0
    Kxy = 0.0
    for g in gammas:
        Kxx = Kxx + torch.exp(-g * Dxx).mean()
        Kyy = Kyy + torch.exp(-g * Dyy).mean()
        Kxy = Kxy + torch.exp(-g * Dxy).mean()
    mmd2 = Kxx - 2.0 * Kxy + Kyy
    return float(mmd2.item())

# --------- 主逻辑 ----------

def sample_pairs(states: torch.Tensor, actions: torch.Tensor, only_t0: bool, exclude_t0: bool,
                 num_pairs: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从 (N,H,D) 的 states/actions 里随机抽样 (s_t, a_t) 对。
    返回：s_sel:(M,S), a_sel:(M,A), 还返回一个 idx_s0_mask:(M,) 用于标记是否 t==0（用来分析）
    """
    assert states.ndim == 3 and actions.ndim == 3
    N, H, S = states.shape
    _, _, A = actions.shape

    if only_t0 and exclude_t0:
        raise ValueError("only_t0 与 exclude_t0 不能同时设置。")

    # 展平为 (N*H, D)
    states_flat = states.reshape(N * H, S)
    actions_flat = actions.reshape(N * H, A)

    # 对应的 t 索引
    t_idx = np.tile(np.arange(H), reps=N)  # (N*H,)

    # 构造可选的索引集合
    if only_t0:
        pool_idx = np.nonzero(t_idx == 0)[0]
    elif exclude_t0:
        pool_idx = np.nonzero(t_idx != 0)[0]
    else:
        pool_idx = np.arange(N * H)

    if len(pool_idx) == 0:
        raise RuntimeError("符合条件的样本为空，请检查 only_t0 / exclude_t0 与数据尺寸。")

    # 随机抽样
    sel = rng.choice(pool_idx, size=min(num_pairs, len(pool_idx)), replace=False)
    s_sel = states_flat[sel].numpy()
    a_sel = actions_flat[sel].numpy()
    is_t0 = (t_idx[sel] == 0).astype(np.int32)  # 1 表示 t=0
    return s_sel, a_sel, is_t0

def batched_eval(policy: FlowPolicy, S_np: np.ndarray, A_np: np.ndarray,
                 samples_per_state: int = 32, batch_size: int = 256) -> dict:
    """
    对一批 (s,a) 做 Top-1 / Best-of-N 评估。
    S_np:(M,S), A_np:(M,A)
    返回：字典含 top1/bestN 各向量与均值等。
    """
    M, S = S_np.shape
    A_dim = A_np.shape[1]

    top1_list = []
    bestN_list = []

    # 分块处理，避免爆显存
    for start in range(0, M, batch_size):
        end = min(M, start + batch_size)
        s_chunk = S_np[start:end]                       # (b,S)
        a_gt_chunk = A_np[start:end]                   # (b,A)
        b = s_chunk.shape[0]

        # 复制 N 次 -> (b*N, S)
        s_rep = np.repeat(s_chunk, repeats=samples_per_state, axis=0)

        # 推理 (b*N, A)
        a_samps = policy.act_batch(s_rep)              # numpy (b*N, A)

        # 组织成 (b, N, A)
        a_samps = a_samps.reshape(b, samples_per_state, A_dim)

        # 计算 (b, N) 的 MSE（欧氏平方）
        diffs = ((a_samps - a_gt_chunk[:, None, :]) ** 2).sum(axis=2)

        # Top-1: N 次的平均（近似“随机采一次”的期望误差）
        top1_vec = diffs.mean(axis=1)                  # (b,)
        # Best-of-N: 取 min
        bestN_vec = diffs.min(axis=1)                  # (b,)

        top1_list.append(top1_vec)
        bestN_list.append(bestN_vec)

    top1_all = np.concatenate(top1_list, axis=0)
    bestN_all = np.concatenate(bestN_list, axis=0)

    def agg(x):
        return float(np.mean(x)), float(np.std(x))

    return {
        "top1_mse": top1_all,
        "bestN_mse": bestN_all,
        "top1_mse_mean": agg(top1_all)[0],
        "top1_mse_std":  agg(top1_all)[1],
        "bestN_mse_mean": agg(bestN_all)[0],
        "bestN_mse_std":  agg(bestN_all)[1],
    }

def maybe_plot_hist(out_dir: str, vec: np.ndarray, title: str, fname: str, bins: int = 80):
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib 不可用，跳过绘图：{e}")
        return
    import os
    ensure_dir(out_dir)
    plt.figure()
    plt.hist(vec, bins=bins, alpha=0.9)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    plt.grid(True, linestyle="--", alpha=0.4)
    path = os.path.join(out_dir, fname)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVE] {path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate dataset closeness (one-step, optional s0-MMD).")
    parser.add_argument("--cfg-train", type=str, default="configs/flow_train.yaml")
    parser.add_argument("--cfg-eval",  type=str, default="configs/flow_eval.yaml")
    parser.add_argument("--num-pairs", type=int, default=20000, help="随机抽取 (s_t,a_t) 的对数")
    parser.add_argument("--samples-per-state", type=int, default=32, help="每个状态采样次数 N")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only-t0", action="store_true", help="只评估 t=0 样本")
    parser.add_argument("--exclude-t0", action="store_true", help="评估时排除 t=0 样本")
    parser.add_argument("--mmd-s0", action="store_true", help="额外计算 t=0 条件下的数据动作分布 vs 策略分布的 MMD")
    parser.add_argument("--save-dir", type=str, default=None, help="保存目录（默认 runs/eval/<timestamp>）")
    args = parser.parse_args()

    set_seed(args.seed)

    # 1) 读取配置
    cfg_train = load_yaml(args.cfg_train)
    cfg_eval  = load_yaml(args.cfg_eval)

    # 数据路径（相对仓库根）
    data_dir  = cfg_train["data"]["dir"]
    state_pt  = os.path.join(data_dir, cfg_train["data"]["state_pt"])
    action_pt = os.path.join(data_dir, cfg_train["data"]["action_pt"])

    # 模型参数
    mcfg = cfg_eval["model"]
    state_dim  = int(mcfg["state_dim"])
    action_dim = int(mcfg["action_dim"])
    horizon    = int(mcfg["horizon"])
    hidden_size = int(mcfg["hidden_size"])
    depth       = int(mcfg["depth"])
    num_heads   = int(mcfg["num_heads"])

    pcfg = cfg_eval["policy"]
    ckpt_path     = pcfg["ckpt"]
    norm_stats_pt = pcfg["norm_stats"]
    ode_steps     = int(pcfg.get("ode_steps", 100))

    # 2) 加载数据张量（未归一化）
    print(f"[LOAD] {state_pt}")
    states = load_pt(state_pt)    # (N,H,S)
    print(f"[LOAD] {action_pt}")
    actions = load_pt(action_pt)  # (N,H,A)

    # 3) 初始化策略（会内部做归一化/反归一化）
    dev = cfg_eval.get("device", "auto")
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device={dev}")
    policy = FlowPolicy(
        ckpt_path=ckpt_path,
        norm_stats_path=norm_stats_pt,
        state_dim=state_dim,
        action_dim=action_dim,
        horizon=horizon,
        hidden_size=hidden_size,
        depth=depth,
        num_heads=num_heads,
        device=dev,
        euler_steps=ode_steps,
    )

    # 4) 抽样 (s_t,a_t)
    rng = np.random.default_rng(args.seed or 0)
    S_np, A_np, is_t0 = sample_pairs(states, actions,
                                     only_t0=args.only_t0,
                                     exclude_t0=args.exclude_t0,
                                     num_pairs=args.num_pairs,
                                     rng=rng)
    print(f"[INFO] sampled pairs: {S_np.shape} / {A_np.shape}  (t0_ratio={is_t0.mean():.3f})")

    # 5) 评估 Top-1 / Best-of-N
    results = batched_eval(policy, S_np, A_np,
                           samples_per_state=args.samples_per_state,
                           batch_size=args.batch_size)

    # 6) 保存结果
    out_dir = args.save_dir or os.path.join("data", "eval", now_ts())
    ensure_dir(out_dir)
    metrics = {
        "config": {
            "cfg_train": args.cfg_train,
            "cfg_eval": args.cfg_eval,
            "num_pairs": int(args.num_pairs),
            "samples_per_state": int(args.samples_per_state),
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "only_t0": bool(args.only_t0),
            "exclude_t0": bool(args.exclude_t0),
            "mmd_s0": bool(args.mmd_s0),
        },
        "summary": {
            "top1_mse_mean": results["top1_mse_mean"],
            "top1_mse_std":  results["top1_mse_std"],
            "bestN_mse_mean": results["bestN_mse_mean"],
            "bestN_mse_std":  results["bestN_mse_std"],
        }
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {os.path.join(out_dir, 'metrics.json')}")

    # 7) 绘图（直方图）
    maybe_plot_hist(out_dir, results["top1_mse"],   "Top-1 MSE", f"Top1_MSE_hist.png")
    maybe_plot_hist(out_dir, results["bestN_mse"],  f"Best-of-{args.samples_per_state} MSE", f"BestOfN_MSE_hist.png")

    # 8) （可选）在 s0 上做 MMD
    if args.mmd_s0:
        # 收集数据集的 a0
        a0_data = actions[:, 0, :].numpy()  # (N, A)
        # 策略在相同 s0 上采样
        s0 = states[:, 0, :].numpy()        # (N, S)
        # 采样数量做个下界，避免过大
        Nprobe = min(len(s0), max(2000, len(s0)//2))
        idx = np.random.default_rng(args.seed or 0).choice(len(s0), size=Nprobe, replace=False)
        s0_probe = s0[idx]
        # 为每个 s0 采样 1 次（也可以多次，拼起来）
        a0_model = FlowPolicy.act_batch(policy, s0_probe)  # (Nprobe, A)
        mmd = mmd_rbf_torch(a0_data[idx], a0_model, gammas=(0.1, 1.0, 10.0))
        with open(os.path.join(out_dir, "MMD_s0.txt"), "w") as f:
            f.write(f"MMD(s0): {mmd:.6f}\n")
        print(f"[SAVE] MMD(s0) -> {os.path.join(out_dir, 'MMD_s0.txt')}  value={mmd:.6f}")

    print("[DONE]")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
