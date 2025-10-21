# scripts/run_closed_loop_flow.py
import os, yaml
import numpy as np
import torch
import configs
from simulators.mujoco_simulator import MuJoCoSimulator
from controllers.flow_policy import FlowPolicy
from utils.plotting import plot_trajectories


def main(cfg_path="configs/flow_eval.yaml"):
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    dev = cfg.get('device', 'auto')
    device = 'cuda' if (dev=='auto' and torch.cuda.is_available()) else (dev if dev!='auto' else 'cpu')

    # 1) MuJoCo 仿真器
    xml_path = cfg['env'].get('xml', None)
    sim = MuJoCoSimulator(xml_path=xml_path)

    # 初始状态
    init_q = np.array(configs.x0, dtype=np.float64)
    init_qd = np.zeros(7, dtype=np.float64)
    sim.reset(q_init=init_q, qd_init=init_qd)

    # 2) Flow 策略
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
        ode_steps      = pol.get('ode_steps', 100),
        action_clip    = pol.get('action_clip', None),
    )

    # 3) 闭环滚动
    T = int(cfg['eval']['steps'])
    qs, us, poss = [], [], []
    q = sim.data.qpos[:md['state_dim']].copy()
    pos = sim.get_end_effector_pos()
    qs.append(q.copy()); poss.append(pos.copy())

    for t in range(T):
        u = policy.act(q)          # 关节速度(7,)
        q = sim.step(u)            # 你的 simulator 会把速度积分成位置并发给 ctrl
        pos = sim.get_end_effector_pos()

        qs.append(q.copy())
        us.append(u.copy())
        poss.append(pos.copy())

    plot_trajectories(np.array(qs), np.array(us), np.array(poss), target_position=configs.target_position)
    # 4) 保存
    os.makedirs(cfg['eval']['save_dir'], exist_ok=True)
    out = os.path.join(cfg['eval']['save_dir'], 'flow_rollout.npz')
    np.savez(out, q=np.array(qs), u=np.array(us), pos=np.array(poss))
    print(f"✅ saved rollout to {out}")

if __name__ == "__main__":
    main()
