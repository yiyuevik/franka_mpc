import os, yaml, torch, random
from time import time
import numpy as np
from itertools import cycle
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models_flow.transformer import TransformerFlow
from models_flow.flow_matcher import FlowMatcher
from dataset.sequence_dataset import make_loader, Batch

CFG_PATH = 'configs/flow_train.yaml'

def load_cfg(path=CFG_PATH):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    dev = cfg['train'].get('device', 'auto')
    if dev == 'auto':
        cfg['train']['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg

def set_seed(seed: int):
    if seed is None: return
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def build_scheduler(optimizer, save_freq: int):
    return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=save_freq * 2)

def pack_norm_stats(ds):
    def pack(n):
        d = {'mode': n.mode}
        if n.mode == 'gaussian':
            d.update({'mean': n.mean.cpu(), 'std': n.std.cpu()})
        elif n.mode == 'limits':
            d.update({'min': n.min.cpu(), 'max': n.max.cpu(), 'range': n.range.cpu()})
        return d
    return {
        'state_norm': pack(ds.state_norm),
        'action_norm': pack(ds.action_norm),
    }

def main():
    cfg = load_cfg()
    cfg_data  = cfg['data']
    cfg_model = cfg['model']
    cfg_train = cfg['train']
    cfg_flow  = cfg['flow']

    set_seed(cfg_train.get('seed', 0))
    device = torch.device(cfg_train['device'])
    os.makedirs(cfg_train['save_dir'], exist_ok=True)

    # ----------------
    # 1) DataLoader（使用你的 tiny_dataset，默认 gaussian 归一化）
    # ----------------
    data_dir   = cfg_data['dir']
    state_pt   = os.path.join(data_dir, cfg_data['state_pt'])
    action_pt  = os.path.join(data_dir, cfg_data['action_pt'])
    x0_pt      = os.path.join(data_dir, cfg_data['x0_pt'])

    norm_cfg = (cfg_data.get('normalize') or {})
    norm_state  = norm_cfg.get('state',  'gaussian')
    norm_action = norm_cfg.get('action', 'gaussian')

    loader, ds = make_loader(
    state_pt, action_pt, x0_pt,
    batch_size=int(cfg_train['batch_size']),
    shuffle=True,
    norm_mode_state=norm_state,
    norm_mode_action=norm_action,
    num_workers=4,
    pin_memory=(device.type == 'cuda'),
    )
    base_loader   = loader
    train_loader  = cycle(base_loader)
    steps_per_ep  = len(base_loader) if len(base_loader) > 0 else 1

    # ----------------
    # 2) 模型 / FlowMatcher / 优化器 / 调度器 / EMA
    # ----------------
    state_dim  = int(cfg_model['state_dim'])
    action_dim = int(cfg_model['action_dim'])
    horizon    = int(cfg_model['horizon'])
    token_dim  = state_dim + action_dim

    model = TransformerFlow(
        seq_len=horizon,
        in_channels=token_dim,
        out_channels=None,                 # = in_channels
        hidden_size=int(cfg_model['hidden_size']),
        depth=int(cfg_model['depth']),
        num_heads=int(cfg_model['num_heads']),
        # 默认 mlp_ratio=4.0, x_emb_proj='conv', x_emb_proj_conv_k=1
    ).to(device)

    fm = FlowMatcher(model, action_dim=action_dim, flow_matching_type=str(cfg_flow['type']))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg_train['lr']))
    scheduler = build_scheduler(optimizer, int(cfg_train['save_freq']))

    ema_cfg   = cfg_train.get('ema', {}) or {}
    use_ema   = bool(ema_cfg.get('use', True))
    ema_decay = float(ema_cfg.get('decay', 0.995))
    ema_model = torch.optim.swa_utils.AveragedModel(
        model,
        avg_fn=lambda avg, new, num: ema_decay * avg + (1.0 - ema_decay) * new
    ) if use_ema else None

    writer = SummaryWriter(log_dir=os.path.join(cfg_train['save_dir'], 'tb'))

    # ----------------
    # 3) 训练（step-based，对齐导师；step==0 也保存）
    # ----------------
    n_train_steps = int(cfg_train['n_train_steps'])
    save_freq     = int(cfg_train['save_freq'])
    log_every     = int(cfg_train['log_every'])
    save_dir      = cfg_train['save_dir']

    for step in range(n_train_steps):
        batch: Batch = next(train_loader)                # Batch(trajectories, conditions)
        trajs = batch.trajectories.to(device, non_blocking=True)   # (B,H,14)
        cond  = {k: v.to(device, non_blocking=True) for k, v in batch.conditions.items()}  # {0: (B,7)}

        loss, infos = fm.loss(trajs, cond)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()  # 每步更新

        if use_ema:
            ema_model.update_parameters(model)

        # log
        if step % log_every == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f'S{step:08d} | lr {cur_lr:.6e} | loss {infos["loss"]:.6f}')
            writer.add_scalar('train/loss', float(infos['loss']), step)
            writer.add_scalar('train/lr',   float(cur_lr),       step)
            writer.add_scalar('train/epoch_progress', step / max(1, steps_per_ep), step)

        # save（包含 step==0）
        if step % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(save_dir, f'model_{step // save_freq}.pth'))
            if use_ema:
                torch.save(ema_model.module.state_dict(), os.path.join(save_dir, f'model_ema_{step // save_freq}.pth'))
            # 保存归一化统计量（用于推理反归一化）
            torch.save(pack_norm_stats(ds), os.path.join(save_dir, f'norm_stats_{step // save_freq}.pt'))

    # 末尾再存一份
    torch.save(model.state_dict(), os.path.join(save_dir, 'final.pth'))
    if use_ema:
        torch.save(ema_model.module.state_dict(), os.path.join(save_dir, 'final_ema.pth'))
    torch.save(pack_norm_stats(ds), os.path.join(save_dir, 'norm_stats_final.pt'))

    writer.close()
    print('✅ done.')

if __name__ == '__main__':
    time_start = time.time()
    main()
    print(f"Total training time: {time.time() - time_start:.2f} seconds")
