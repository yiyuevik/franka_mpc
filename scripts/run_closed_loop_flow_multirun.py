# scripts/run_closed_loop_flow_multirun.py
import csv
import os, yaml, random
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import plotly.graph_objects as go  # NEW

import configs
from simulators.mujoco_simulator import MuJoCoSimulator
from controllers.flow_policy import FlowPolicy  # Euler-batched version
from utils.helpers import compute_end_effector_position

# ---------- helpers ----------

def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _segments_from_traj(traj_2d: np.ndarray):
    """(T,2) -> (T-1,2,2) segments for LineCollection."""
    return np.stack([traj_2d[:-1], traj_2d[1:]], axis=1)

def compute_trajectory_cost(q_traj: np.ndarray, target_pos: np.ndarray) -> dict:
    """
    计算单条轨迹的成本
    
    Args:
        q_traj: (T, 7) - 关节角度轨迹
        target_pos: (3,) - 目标位置
        
    Returns:
        dict with cost breakdown and total
    """
    T = len(q_traj)
    
    # 计算末端位置轨迹
    pos_traj = np.array([compute_end_effector_position(q) for q in q_traj])  # (T, 3)
    
    # 计算到目标的距离
    dists = np.linalg.norm(pos_traj - target_pos, axis=1)  # (T,)
    
    # 位置成本: Q_pos * sum(dists[:-1]) + P_pos * dists[-1]
    stage_cost = configs.Q_pos[0, 0] * np.sum(dists[:-1]) *10
    terminal_cost = configs.P_pos[0, 0] * dists[-1]
    position_cost = stage_cost + terminal_cost
    
    # 动作成本: R * sum(action^2)
    # 这里 action 是相邻状态的差分（速度）
    action_traj = np.diff(q_traj, axis=0)  # (T-1, 7)
    action_cost = configs.R[0, 0] * np.sum(action_traj ** 2) * 10000
    
    # 总成本
    total_cost = position_cost + action_cost
    
    return {
        'total': total_cost,
        'position': position_cost,
        'stage': stage_cost,
        'terminal': terminal_cost,
        'action': action_cost,
        'final_distance': dists[-1],
        'mean_distance': np.mean(dists),
    }

def save_sim_trajs_csv(poss_list, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for k, pos in enumerate(poss_list):
        path = os.path.join(out_dir, f"sim_rollout_{k+1}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "x", "y", "z"])
            for t, p in enumerate(pos):
                w.writerow([t, float(p[0]), float(p[1]), float(p[2])])
        print(f"💾 saved: {path}")

def save_sim_q_trajs_ros_format(q_trajs_list, out_dir, dt=0.1):
    """
    保存和 ROS 真实运行时类似格式的关节轨迹：
    每个文件：q_now_1.csv, q_now_2.csv, ...
    列为: stamp_sec, q1, ..., q7

    q_trajs_list: list of arrays, 每个 (T, 7)
    dt: 假定的时间间隔（秒），仅用于 stamp_sec；可根据需要改。
    """
    os.makedirs(out_dir, exist_ok=True)
    for k, q_traj in enumerate(q_trajs_list):
        path = os.path.join(out_dir, f"q_now_{k+1}.csv")
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            header = ["stamp_sec"] + [f"q{i}" for i in range(1, 8)]
            w.writerow(header)
            for t, q in enumerate(q_traj):
                stamp = t * dt
                w.writerow([stamp] + [float(v) for v in q.tolist()])
        print(f"💾 saved ROS-style q traj: {path}")


# def plot_multimodal_trajectories_3d_html(
#     poss_list,
#     out_html='multi_trajs_3d.html',
#     target_pos=None,
#     title='Flow (multi-rollouts, 3D)'
# ):
#     """
#     Make a simple 3D interactive HTML with multiple trajectories.
#     - poss_list: list of arrays, each (T, 3)
#     - target_pos: optional (3,)
#     """
#     fig = go.Figure()

#     # add each rollout as a 3D line
#     for k, pos in enumerate(poss_list):
#         fig.add_trace(go.Scatter3d(
#             x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
#             mode='lines',
#             line=dict(width=3),
#             opacity=0.8,
#             showlegend=False,           # avoid a huge legend
#             name=f'rollout {k}',
#             hoverinfo='none',
#         ))

#     # start point (use the first rollout's start)
#     if len(poss_list) > 0:
#         s0 = poss_list[0][0]
#         fig.add_trace(go.Scatter3d(
#             x=[s0[0]], y=[s0[1]], z=[s0[2]],
#             mode='markers',
#             marker=dict(color='green', size=5),
#             name='start',
#             showlegend=True
#         ))

#     # target point (optional)
#     if target_pos is not None:
#         target_pos = np.array(target_pos).reshape(-1)
#         fig.add_trace(go.Scatter3d(
#             x=[target_pos[0]], y=[target_pos[1]], z=[target_pos[2]],
#             mode='markers',
#             marker=dict(color='red', size=6),
#             name='target',
#             showlegend=True
#         ))

#     fig.update_layout(
#         scene=dict(
#             xaxis_title='X (m)',
#             yaxis_title='Y (m)',
#             zaxis_title='Z (m)',
#             aspectmode='data',
#         ),
#         title=title,
#         margin=dict(l=0, r=0, b=0, t=30),
#         template='plotly_dark',
#     )

#     os.makedirs(os.path.dirname(out_html), exist_ok=True)
#     fig.write_html(out_html, include_plotlyjs='cdn')
#     print(f"✅ saved 3D interactive HTML to: {out_html}")

def plot_multimodal_trajectories_3d_html(
    poss_list,
    out_html='multi_trajs_3d.html',
    target_pos=None,
    title='Flow (multi-rollouts, 3D)'
):
    """
    Make a 3D interactive HTML with multiple trajectories + optional spherical obstacle.
    - poss_list: list of arrays, each (T, 3)
    - target_pos: optional (3,)
    """
    fig = go.Figure()

    # add each rollout as a 3D line
    for k, pos in enumerate(poss_list):
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='lines',
            line=dict(width=3),
            opacity=0.85,
            showlegend=False,
            name=f'rollout {k}',
            hoverinfo='none',
        ))
    
    # start point (use the first rollout's start)
    if len(poss_list) > 0:
        s0 = poss_list[0][0]
        fig.add_trace(go.Scatter3d(
            x=[s0[0]], y=[s0[1]], z=[s0[2]],
            mode='markers',
            marker=dict(color='green', size=6),
            name='start',
            showlegend=True
        ))

    # target point (optional)
    if target_pos is not None:
        target_pos = np.array(target_pos).reshape(-1)
        fig.add_trace(go.Scatter3d(
            x=[target_pos[0]], y=[target_pos[1]], z=[target_pos[2]],
            mode='markers',
            marker=dict(color='red', size=7),
            name='target',
            showlegend=True
        ))

    # --- optional spherical obstacle from configs ---
    # Expect:
    #   configs.Obstacle_Avoidance: bool
    #   configs.Obstacle_Position: 3-vector (x, y, z)
    #   configs.Obstacle_Scale: 3-vector; radius = 1 / min(scale) (same as your old code)
    if getattr(configs, 'Obstacle_Avoidance', False) \
       and hasattr(configs, 'Obstacle_Position') \
       and hasattr(configs, 'Obstacle_Scale'):

        center = np.array(configs.Obstacle_Position)
        scale = np.array(configs.Obstacle_Scale)
        radius = 1 / (np.min(scale)) 

        # sphere mesh
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        fig.add_trace(go.Surface(
            x=x_sphere, y=y_sphere, z=z_sphere,
            colorscale=[[0, 'red'], [1, 'darkred']],
            opacity=0.7,
            showscale=False,
            name='Obstacle',
            hovertemplate='<b>Obstacle</b><br>Center: (%.3f, %.3f, %.3f)<br>Radius: %.3f<extra></extra>' % (*center, radius)
        ))


    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
        ),
        title=title,
        margin=dict(l=0, r=0, b=0, t=30),
        # template='fixed_axes',
        uirevision="fixed_axes"
    )

    os.makedirs(os.path.dirname(out_html), exist_ok=True)
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"✅ saved 3D interactive HTML to: {out_html}")

def save_trajectory_costs(costs_list, out_dir):
    """
    保存所有轨迹的成本信息到 CSV
    
    Args:
        costs_list: list of dicts - 每条轨迹的成本字典
        out_dir: str - 输出目录
    """
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "trajectory_costs.csv")
    
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        # 写表头
        w.writerow([
            "trajectory_id", 
            "total_cost", 
            "position_cost", 
            "stage_cost", 
            "terminal_cost",
            "action_cost",
            "final_distance",
            "mean_distance"
        ])
        
        # 写每条轨迹的成本
        for k, cost_dict in enumerate(costs_list):
            w.writerow([
                k + 1,
                float(cost_dict['total']),
                float(cost_dict['position']),
                float(cost_dict['stage']),
                float(cost_dict['terminal']),
                float(cost_dict['action']),
                float(cost_dict['final_distance']),
                float(cost_dict['mean_distance']),
            ])
    
    print(f"💾 saved trajectory costs: {path}")
# def plot_multimodal_trajectories_2d(
#     poss_list, plane='xy', out_path='multi_trajs.png',
#     target_pos=None, start_pos=None, title='Flow (multi-rollouts)'
# ):
#     # select plane
#     idx_map = {'xy': (0, 1), 'xz': (0, 2), 'yz': (1, 2)}
#     i, j = idx_map[plane]

#     # collect all 2d points for bounds & heat
#     all_pts = np.concatenate([p[:, [i, j]] for p in poss_list], axis=0)
#     xmin, ymin = all_pts.min(axis=0)
#     xmax, ymax = all_pts.max(axis=0)
#     pad_x = 0.05 * (xmax - xmin + 1e-6)
#     pad_y = 0.05 * (ymax - ymin + 1e-6)
#     extent = (xmin - pad_x, xmax + pad_x, ymin - pad_y, ymax + pad_y)

#     plt.style.use('dark_background')
#     fig, ax = plt.subplots(figsize=(3.0, 6.0))  # tall style
#     ax.set_facecolor('#2b0a2b')

#     # simple density background
#     H, xedges, yedges = np.histogram2d(
#         all_pts[:, 0], all_pts[:, 1],
#         bins=180,
#         range=[[extent[0], extent[1]], [extent[2], extent[3]]]
#     )
#     H = np.sqrt(H)
#     ax.imshow(
#         H.T, origin='lower',
#         extent=extent, cmap='magma', alpha=0.35, aspect='auto'
#     )

#     # plot every trajectory with time-gradient color
#     for pos in poss_list:
#         traj2d = pos[:, [i, j]]
#         segs = _segments_from_traj(traj2d)
#         T = traj2d.shape[0]
#         colors = np.linspace(0.0, 1.0, T - 1)
#         lc = LineCollection(segs, cmap='viridis', array=colors, linewidths=2.0, alpha=0.85)
#         ax.add_collection(lc)

#     if start_pos is not None:
#         ax.scatter(start_pos[i], start_pos[j], s=30, c='yellow', edgecolors='k', linewidths=0.5, zorder=5)
#     if target_pos is not None:
#         ax.scatter(target_pos[i], target_pos[j], s=30, c='red', edgecolors='k', linewidths=0.5, zorder=5)

#     ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
#     ax.set_xticks([]); ax.set_yticks([])
#     ax.set_title(title, fontsize=11, pad=8)
#     plt.tight_layout()
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.savefig(out_path, dpi=220)
#     plt.close(fig)

# ---------- main ----------

def main(cfg_path="configs/flow_eval.yaml", n_rollouts=64, save_dir="data/models/flow/multimodal", 
         batch_size=None, n_samples = 100, cost_type = 'target'):
    # load cfg
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # device (GPU if available)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # create N simulators (CPU side is cheap; network forward will be batched on GPU)
    xml_path = cfg['env'].get('xml', None)
    sims = [MuJoCoSimulator(xml_path=xml_path) for _ in range(n_rollouts)]

    # init states for each sim
    init_q = np.array(configs.x0, dtype=np.float64)
    init_qd = np.zeros(7, dtype=np.float64)
    for sim in sims:
        sim.reset(q_init=init_q, qd_init=init_qd)


    # policy (single model) — GPU batched
    md = cfg['model']
    pol = cfg['policy']
    policy = FlowPolicy(
        ckpt_path      = pol['ckpt'],
        norm_stats_path= pol['norm_stats'],
        state_dim      = md['state_dim'],
        action_dim     = md['action_dim'],
        horizon        = md['horizon'],
        hidden_size    = md['hidden_size'],
        depth          = md['depth'],
        num_heads      = md['num_heads'],
        device         = device,
        euler_steps    = pol.get('ode_steps', 100),   # reuse cfg name
        use_amp        = pol.get('use_amp', True),
        amp_dtype      = torch.float16 if pol.get('amp_dtype', 'fp16') == 'fp16' else torch.bfloat16,
        matmul_precision = pol.get('matmul_precision', 'high'),
        cost_type        = cost_type,
        n_samples      = n_samples,
    )

    # rollout length
    T = int(cfg['eval']['steps'])
    os.makedirs(save_dir, exist_ok=True)

    # seed (network noise is inside policy, we just fix global seeds)
    set_all_seeds(cfg.get('seed', 1))

    # storage
    poss_list = []
    q_trajs_list = [] 
    for sim in sims:
        pos = sim.get_end_effector_pos()
        poss_list.append([pos.copy()])  # will append per-step
        q0 = sim.data.qpos[:md['state_dim']].copy()
        q_trajs_list.append([q0])

    # optional chunking (VRAM control)
    B_total = n_rollouts
    B_chunk = batch_size or B_total

    # closed-loop
    for t in range(T):
        # gather q in chunks -> batched actions on GPU
        for start in range(0, B_total, B_chunk):
            end = min(start + B_chunk, B_total)
            q_batch = np.stack([sims[k].data.qpos[:md['state_dim']].copy() for k in range(start, end)], axis=0)  # (B, S)
            a_batch = policy.act_batch(q_batch)  # (B, A) numpy

            # apply to sims
            for idx, k in enumerate(range(start, end)):
                u = a_batch[idx]
                sims[k].step(u)
                pos = sims[k].get_end_effector_pos()
                poss_list[k].append(pos.copy())

                q_now = sims[k].data.qpos[:md['state_dim']].copy()
                q_trajs_list[k].append(q_now)

    # to ndarray (T+1, 3)
    poss_list = [np.array(p) for p in poss_list]
    q_trajs_list = [np.array(q) for q in q_trajs_list]

    # ========== 计算所有轨迹的成本 ==========
    print(f"\n📊 Computing trajectory costs...")
    target_pos = np.array(configs.target_position)
    costs_list = []
    
    for k, q_traj in enumerate(q_trajs_list):
        cost_dict = compute_trajectory_cost(q_traj, target_pos)
        costs_list.append(cost_dict)
        
        # 每10条轨迹打印一次进度
        if (k + 1) % 10 == 0:
            print(f"   Computed costs for {k+1}/{n_rollouts} trajectories...")
    
    # 统计信息
    total_costs = np.array([c['total'] for c in costs_list])
    final_dists = np.array([c['final_distance'] for c in costs_list])
    position_costs = np.array([c['position'] for c in costs_list])
    action_costs = np.array([c['action'] for c in costs_list])
    
    print(f"\n📈 Cost Statistics:")
    print(f"   Total Cost     - Mean: {np.mean(total_costs):.4f}, Std: {np.std(total_costs):.4f}, Min: {np.min(total_costs):.4f}, Max: {np.max(total_costs):.4f}")
    print(f"   Position Cost  - Mean: {np.mean(position_costs):.4f}, Std: {np.std(position_costs):.4f}")
    print(f"   Action Cost    - Mean: {np.mean(action_costs):.4f}, Std: {np.std(action_costs):.4f}")
    print(f"   Final Distance - Mean: {np.mean(final_dists):.4f}, Std: {np.std(final_dists):.4f}, Min: {np.min(final_dists):.4f}, Max: {np.max(final_dists):.4f}")
    
    # 找出最好和最差的轨迹
    best_idx = np.argmin(total_costs)
    worst_idx = np.argmax(total_costs)
    print(f"\n🏆 Best trajectory: #{best_idx+1} (total_cost={total_costs[best_idx]:.4f}, final_dist={final_dists[best_idx]:.4f})")
    print(f"⚠️  Worst trajectory: #{worst_idx+1} (total_cost={total_costs[worst_idx]:.4f}, final_dist={final_dists[worst_idx]:.4f})")

    # 保存成本到 CSV
    save_trajectory_costs(costs_list, save_dir)


    csv_dir = os.path.join(save_dir, "sim_trajs_csv")
    save_sim_trajs_csv(poss_list, csv_dir)

    q_csv_dir = os.path.join(save_dir, "sim_q_trajs_ros")
    save_sim_q_trajs_ros_format(q_trajs_list, q_csv_dir, dt=0.1)
    
    # --- 3D interactive HTML (multi trajectories) ---
    out_html = os.path.join(save_dir, "multi_trajs_3d.html")
    plot_multimodal_trajectories_3d_html(
        poss_list,
        out_html=out_html,
        target_pos=getattr(configs, 'target_position', None),
        title=f"Flow (N={n_rollouts})"
    )

    # (optional) 2D overlay if you still want a static PNG:
    # out_png = os.path.join(save_dir, f"multi_trajs_{plane}.png")
    # plot_multimodal_trajectories_2d(
    #     poss_list,
    #     plane=plane,
    #     out_path=out_png,
    #     target_pos=getattr(configs, 'target_position', None),
    #     start_pos=poss_list[0][0] if len(poss_list) > 0 else None,
    #     title=f"Flow (N={n_rollouts})"
    # )

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", type=str, default="configs/flow_eval.yaml")
    p.add_argument("--n", type=int, default=64, help="number of rollouts")
    p.add_argument("--plane", type=str, default="xz", choices=["xy","xz","yz"])
    p.add_argument("--save_dir", type=str, default="data/models/flow2/multimodal")
    p.add_argument("--batch", type=int, default=None, help="GPU batch size per forward; default=n")
    p.add_argument("--n_samples", type=int, default=1, help="samples per state for cost selection")
    p.add_argument("--cost_type", type=str, default="none", 
                   choices=["none", "target", "obstacle", "smoothness"],
                   help="cost function type")
    args = p.parse_args()
    main(args.cfg, args.n, args.save_dir, args.batch, args.n_samples, args.cost_type)