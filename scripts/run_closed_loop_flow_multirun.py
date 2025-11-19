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

def main(cfg_path="configs/flow_eval.yaml", n_rollouts=64, plane='xy', save_dir="data/models/flow/multimodal", batch_size=None):
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
    )

    # rollout length
    T = int(cfg['eval']['steps'])
    os.makedirs(save_dir, exist_ok=True)

    # seed (network noise is inside policy, we just fix global seeds)
    set_all_seeds(cfg.get('seed', 1))

    # storage
    poss_list = []
    for sim in sims:
        pos = sim.get_end_effector_pos()
        poss_list.append([pos.copy()])  # will append per-step

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

    # to ndarray (T+1, 3)
    poss_list = [np.array(p) for p in poss_list]
    

    csv_dir = os.path.join(save_dir, "sim_trajs_csv")
    save_sim_trajs_csv(poss_list, csv_dir)

    
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
    p.add_argument("--save_dir", type=str, default="data/models/flow/multimodal")
    p.add_argument("--batch", type=int, default=None, help="GPU batch size per forward; default=n")
    args = p.parse_args()
    main(args.cfg, args.n, args.plane, args.save_dir, args.batch)
