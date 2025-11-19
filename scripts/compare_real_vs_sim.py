#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, glob, argparse
import numpy as np
import plotly.graph_objects as go

from utils.helpers import compute_end_effector_position


import configs

# ---------------- IO ----------------

def load_real_positions(real_dir):
    """
    读取真机 CSV（stamp_sec,q1..q7），用 compute_end_effector_position(q) 得到 (T,3)
    返回 list of (name, (T,3) ndarray)
    """
    files = sorted(glob.glob(os.path.join(real_dir, "*.csv")))
    all_trajs = []
    for fp in files:
        arr = np.genfromtxt(fp, delimiter=",", names=True)
        q_cols = [f"q{i}" for i in range(1, 8)]
        Q = np.stack([arr[c] for c in q_cols], axis=1).astype(float)
        P = np.array([compute_end_effector_position(q) for q in Q], dtype=float)
        all_trajs.append((os.path.basename(fp), P))
        print(f"✅ loaded real traj {fp} | steps={len(P)}")
    return all_trajs

def load_sim_positions(sim_dir):
    """
    读取模拟 CSV：sim_rollout_*.csv（列: step,x,y,z），返回 list of (name, (T,3))
    """
    files = sorted(glob.glob(os.path.join(sim_dir, "sim_rollout_*.csv")))
    all_trajs = []
    for fp in files:
        arr = np.genfromtxt(fp, delimiter=",", names=True)
        P = np.stack([arr["x"], arr["y"], arr["z"]], axis=1).astype(float)
        all_trajs.append((os.path.basename(fp), P))
        print(f"✅ loaded sim  traj {fp} | steps={len(P)}")
    return all_trajs

# --------------- 绘图（复刻你之前 HTML 风格） ---------------

def _add_optional_spherical_obstacle(fig):
    """
    读取 configs 中的 Obstacle_Avoidance/Position/Scale，与之前函数一致画球面。
    半径: radius = 1 / min(scale)
    """
    if getattr(configs, 'Obstacle_Avoidance', False) \
       and hasattr(configs, 'Obstacle_Position') \
       and hasattr(configs, 'Obstacle_Scale'):
        center = np.array(configs.Obstacle_Position, dtype=float)
        scale  = np.array(configs.Obstacle_Scale, dtype=float)
        radius = 1.0 / float(np.min(scale))

        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 20)
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))

        fig.add_trace(go.Surface(
            x=x, y=y, z=z,
            colorscale=[[0, 'red'], [1, 'darkred']],
            opacity=0.7,
            showscale=False,
            name='Obstacle',
            hovertemplate='<b>Obstacle</b><br>' +
                          'Center: (%.3f, %.3f, %.3f)<br>Radius: %.3f<extra></extra>' %
                          (center[0], center[1], center[2], radius)
        ))

def plot_real_vs_sim_3d_html(
    real_list,            # list of (name, (T,3))
    sim_list,             # list of (name, (T,3))
    out_html='compare_real_vs_sim.html',
    target_pos=None,
    title='Real (yellow) vs Sim (blue)'
):
    fig = go.Figure()

    # 模拟：蓝线（隐藏图例，宽度与不透明度对齐你之前的风格）
    for name, pos in sim_list:
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='lines',
            line=dict(width=3, color='blue'),
            opacity=0.85,
            showlegend=False,
            name=f'Sim {name}',
            hoverinfo='none',
        ))

    # 真机：红线
    for name, pos in real_list:
        fig.add_trace(go.Scatter3d(
            x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
            mode='lines',
            line=dict(width=3, color='yellow'),
            opacity=0.95,
            showlegend=False,
            name=f'Real {name}',
            hoverinfo='none',
        ))

    # 起点（用第一条轨迹的第一个点；优先真机，没有就用模拟）
    s0 = None
    if len(real_list) > 0 and len(real_list[0][1]) > 0:
        s0 = real_list[0][1][0]
    elif len(sim_list) > 0 and len(sim_list[0][1]) > 0:
        s0 = sim_list[0][1][0]

    if s0 is not None:
        fig.add_trace(go.Scatter3d(
            x=[s0[0]], y=[s0[1]], z=[s0[2]],
            mode='markers',
            marker=dict(color='green', size=6),
            name='start',
            showlegend=True
        ))

    # 目标点（来自 configs）
    if target_pos is not None:
        tp = np.array(target_pos, dtype=float).reshape(-1)
        fig.add_trace(go.Scatter3d(
            x=[tp[0]], y=[tp[1]], z=[tp[2]],
            mode='markers',
            marker=dict(color='red', size=7),
            name='target',
            showlegend=True
        ))

    # 可选障碍
    _add_optional_spherical_obstacle(fig)

    fig.update_layout(
        scene=dict(
            xaxis_title='X (m)',
            yaxis_title='Y (m)',
            zaxis_title='Z (m)',
            aspectmode='data',
        ),
        title=title,
        margin=dict(l=0, r=0, b=0, t=30),
        uirevision="fixed_axes"
    )

    os.makedirs(os.path.dirname(out_html) or ".", exist_ok=True)
    fig.write_html(out_html, include_plotlyjs='cdn')
    print(f"✅ saved 3D interactive HTML to: {out_html}")

# ---------------- main ----------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--real_dir", type=str, default="data/models/flow/multimodal/batch_runs", help="真机 CSV 目录（列: stamp_sec,q1..q7）")
    p.add_argument("--sim_dir",  type=str, default="data/models/flow/multimodal/sim_trajs_csv", help="模拟 CSV 目录（含 sim_rollout_*.csv）")
    p.add_argument("--out_html", type=str, default="real_vs_sim.html")
    p.add_argument("--title",    type=str, default="Real (red) vs Sim (blue)")
    args = p.parse_args()

    real_trajs = load_real_positions(args.real_dir)
    sim_trajs  = load_sim_positions(args.sim_dir)

    target = getattr(configs, 'target_position', None)
    plot_real_vs_sim_3d_html(
        real_trajs, sim_trajs,
        out_html=args.out_html,
        target_pos=target,
        title=args.title
    )

if __name__ == "__main__":
    main()
