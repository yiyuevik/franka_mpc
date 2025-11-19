# scripts/run_closed_loop_flow.py
import os, yaml, time
import numpy as np
import torch
import configs
from simulators.mujoco_simulator import MuJoCoSimulator
from controllers.flow_policy import FlowPolicy
from utils.plotting import plot_trajectories


def main(cfg_path="configs/flow_eval.yaml", warmup= 10):
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
        euler_steps    = pol.get('ode_steps', 50),   # reuse cfg name
        use_amp        = pol.get('use_amp', True),
        amp_dtype      = torch.float16 if pol.get('amp_dtype', 'fp16') == 'fp16' else torch.bfloat16,
        matmul_precision = pol.get('matmul_precision', 'high'),
    )


    # 3) 闭环滚动
    T = int(cfg['eval']['steps'])
    qs, us = [], []
    poss = []
    q = sim.data.qpos[:md['state_dim']].copy()
    pos = sim.get_end_effector_pos()
    times = []
    qs.append(q.copy())
    poss.append(pos.copy())
    # =============== 先预热几次，去掉第一次加载的开销 ===============
    # for _ in range(warmup):
    #     _ = policy.act(q)
    # print(f"warmup {warmup} steps done.")

    for t in range(T):
        t0 = time.time()
        u = policy.act(q)          # 关节速度(7,)
        t1 = time.time()
        times.append(t1 - t0)
        q = sim.step(u)            # 你的 simulator 会把速度积分成位置并发给 ctrl
        pos = sim.get_end_effector_pos()

        qs.append(q.copy())
        us.append(u.copy())
        poss.append(pos.copy())
    times = np.array(times)
    print("=============== FLOW INFERENCE TIMING ===============")
    print(f"device        : {device}")
    print(f"ode_steps     : {pol.get('ode_steps', 100)}")
    print(f"steps measured: {len(times)}")
    print(f"avg per step  : {times.mean()*1000:.3f} ms")
    print(f"min per step  : {times.min()*1000:.3f} ms")
    print(f"max per step  : {times.max()*1000:.3f} ms")
    print(f"10 Hz OK?     : {'YES' if times.max() < 0.1 else 'MAYBE'}")
    print("======================================================")
    plot_trajectories(np.array(qs), np.array(us), np.array(poss), target_position=configs.target_position)
    # print 最后position 和 target position的差距 (cm)
    print(f"Final end-effector position: {poss[-1]}")
    print(f"Target end-effector position: {configs.target_position}")
    print(f"Position error: {np.linalg.norm(poss[-1] - configs.target_position) * 100} cm")
    # 4) 保存
    os.makedirs(cfg['eval']['save_dir'], exist_ok=True)
    out = os.path.join(cfg['eval']['save_dir'], 'flow_rollout.npz')
    np.savez(out, q=np.array(qs), u=np.array(us), pos=np.array(poss))
    print(f"✅ saved rollout to {out}")

if __name__ == "__main__":
    main()
