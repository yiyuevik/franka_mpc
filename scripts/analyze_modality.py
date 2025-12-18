#!/usr/bin/env python3
"""
分析给定数据集中所有 group 的轨迹模态数量
"""
import numpy as np
import os
import glob
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_trajectories_from_groups(data_dir):
    """
    从数据目录加载所有 group 的主轨迹
    
    Args:
        data_dir: 数据目录路径，例如 'data/multimodality/parallel_rollout_20251105_1431'
    
    Returns:
        x_traj_all: list of arrays, 每个元素是一个 group 的轨迹 [N_step+1, num_state]
        group_ids: list of int, 对应的 group ID
    """
    group_dirs = sorted(glob.glob(os.path.join(data_dir, "group_*")))
    x_traj_all = []
    group_ids = []
    
    print(f"找到 {len(group_dirs)} 个 group 文件夹")
    
    for group_dir in group_dirs:
        main_simX_path = os.path.join(group_dir, "main_simX.npy")
        if not os.path.exists(main_simX_path):
            print(f"警告: {group_dir} 中没有找到 main_simX.npy")
            continue
        
        # 加载轨迹数据: shape [N_step, N_horizon+1, num_state]
        # 我们需要提取实际执行的轨迹，即每步的第一个状态
        simX = np.load(main_simX_path)
        
        # 提取实际执行的轨迹: 取每步预测轨迹的第一个状态
        # shape: [N_step, num_state]
        if simX.ndim == 3:
            # simX: [N_step, N_horizon+1, num_state]
            actual_traj = simX[:, 0, :]  # [N_step, num_state]
        else:
            # 如果已经是 [N_step, num_state] 格式
            actual_traj = simX
        
        x_traj_all.append(actual_traj)
        
        # 提取 group ID
        group_name = os.path.basename(group_dir)
        gid = int(group_name.split('_')[1])
        group_ids.append(gid)
    
    print(f"成功加载 {len(x_traj_all)} 个轨迹")
    return x_traj_all, group_ids


def cluster_trajectories(x_traj_all, threshold=1.0):
    """
    使用与 data_collect.py 相同的聚类算法对轨迹进行聚类
    
    Args:
        x_traj_all: list of arrays, 每个元素是一个轨迹 [N_step, num_state]
        threshold: 聚类阈值
    
    Returns:
        clusters: list of lists, 每个子列表包含属于该簇的轨迹索引
        cluster_assignment: list, 每个轨迹的簇 ID
        rep_traj_list: list of arrays, 每个簇的代表轨迹
    """
    clusters = []
    rep_traj_list = []
    cluster_assignment = [-1] * len(x_traj_all)
    threshold_sq = threshold ** 2
    
    for i, traj in enumerate(x_traj_all):
        assigned = False
        for cid, rep_traj in enumerate(rep_traj_list):
            # 首先检查终点距离
            if np.sum((rep_traj[-1] - traj[-1])**2) > threshold_sq:
                continue
            # 检查整条轨迹的最大距离
            max_sq = np.max(np.sum((rep_traj - traj) ** 2, axis=1))
            if max_sq <= threshold_sq:
                clusters[cid].append(i)
                cluster_assignment[i] = cid
                assigned = True
                break
        
        if not assigned:
            clusters.append([i])
            cluster_assignment[i] = len(clusters) - 1
            rep_traj_list.append(traj)
    
    return clusters, cluster_assignment, rep_traj_list


def visualize_clusters(x_traj_all, cluster_assignment, rep_traj_list, save_path=None):
    """
    可视化聚类结果（2D和3D）
    
    Args:
        x_traj_all: list of arrays, 所有轨迹
        cluster_assignment: list, 每个轨迹的簇 ID
        rep_traj_list: list of arrays, 代表轨迹
        save_path: 保存路径
    """
    n_clusters = len(rep_traj_list)
    colors = plt.cm.tab10(np.linspace(0, 1, n_clusters))
    
    # 创建包含多个子图的图形
    fig = plt.figure(figsize=(20, 10))
    
    # 子图1: 终点分布（前两个关节）
    ax1 = fig.add_subplot(2, 3, 1)
    for i, traj in enumerate(x_traj_all):
        cid = cluster_assignment[i]
        ax1.scatter(traj[-1, 0], traj[-1, 1], c=[colors[cid]], alpha=0.5, s=20)
    
    for cid, rep_traj in enumerate(rep_traj_list):
        ax1.scatter(rep_traj[-1, 0], rep_traj[-1, 1], c=[colors[cid]], 
                   marker='*', s=300, edgecolors='black', linewidths=2,
                   label=f'Cluster {cid}')
    ax1.set_xlabel('Joint 0 (rad)')
    ax1.set_ylabel('Joint 1 (rad)')
    ax1.set_title('终点分布 (Joint 0 vs Joint 1)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2: 终点分布（关节2和3）
    ax2 = fig.add_subplot(2, 3, 2)
    for i, traj in enumerate(x_traj_all):
        cid = cluster_assignment[i]
        ax2.scatter(traj[-1, 2], traj[-1, 3], c=[colors[cid]], alpha=0.5, s=20)
    
    for cid, rep_traj in enumerate(rep_traj_list):
        ax2.scatter(rep_traj[-1, 2], rep_traj[-1, 3], c=[colors[cid]], 
                   marker='*', s=300, edgecolors='black', linewidths=2)
    ax2.set_xlabel('Joint 2 (rad)')
    ax2.set_ylabel('Joint 3 (rad)')
    ax2.set_title('终点分布 (Joint 2 vs Joint 3)')
    ax2.grid(True, alpha=0.3)
    
    # 子图3: 终点分布（关节4和5）
    ax3 = fig.add_subplot(2, 3, 3)
    for i, traj in enumerate(x_traj_all):
        cid = cluster_assignment[i]
        ax3.scatter(traj[-1, 4], traj[-1, 5], c=[colors[cid]], alpha=0.5, s=20)
    
    for cid, rep_traj in enumerate(rep_traj_list):
        ax3.scatter(rep_traj[-1, 4], rep_traj[-1, 5], c=[colors[cid]], 
                   marker='*', s=300, edgecolors='black', linewidths=2)
    ax3.set_xlabel('Joint 4 (rad)')
    ax3.set_ylabel('Joint 5 (rad)')
    ax3.set_title('终点分布 (Joint 4 vs Joint 5)')
    ax3.grid(True, alpha=0.3)
    
    # 子图4: 完整轨迹（Joint 0）
    ax4 = fig.add_subplot(2, 3, 4)
    for cid, rep_traj in enumerate(rep_traj_list):
        ax4.plot(rep_traj[:, 0], c=colors[cid], linewidth=3, label=f'Cluster {cid}')
    ax4.set_xlabel('Time step')
    ax4.set_ylabel('Joint 0 (rad)')
    ax4.set_title('代表轨迹 - Joint 0')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # 子图5: 完整轨迹（Joint 1）
    ax5 = fig.add_subplot(2, 3, 5)
    for cid, rep_traj in enumerate(rep_traj_list):
        ax5.plot(rep_traj[:, 1], c=colors[cid], linewidth=3, label=f'Cluster {cid}')
    ax5.set_xlabel('Time step')
    ax5.set_ylabel('Joint 1 (rad)')
    ax5.set_title('代表轨迹 - Joint 1')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 子图6: 簇大小统计
    ax6 = fig.add_subplot(2, 3, 6)
    cluster_sizes = [len(clusters[cid]) for cid in range(n_clusters)]
    bars = ax6.bar(range(n_clusters), cluster_sizes, color=colors)
    ax6.set_xlabel('Cluster ID')
    ax6.set_ylabel('轨迹数量')
    ax6.set_title('每个簇的轨迹数量')
    ax6.grid(True, alpha=0.3, axis='y')
    
    # 在柱状图上标注数量
    for i, (bar, size) in enumerate(zip(bars, cluster_sizes)):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(size), ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可视化结果已保存到: {save_path}")
    
    plt.show()


def print_cluster_statistics(clusters, cluster_assignment, group_ids):
    """
    打印聚类统计信息
    """
    n_clusters = len(clusters)
    print(f"\n" + "="*60)
    print(f"聚类结果统计")
    print(f"="*60)
    print(f"总轨迹数: {len(group_ids)}")
    print(f"模态数量: {n_clusters}")
    print(f"\n各簇详情:")
    print(f"-"*60)
    
    for cid, cluster in enumerate(clusters):
        print(f"\nCluster {cid}:")
        print(f"  轨迹数量: {len(cluster)}")
        print(f"  包含的 group IDs: {[group_ids[i] for i in cluster]}")
    
    print(f"\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='分析轨迹模态数量')
    parser.add_argument('--data_dir', type=str, 
                       default='data/multimodality/parallel_rollout_20251105_1414',
                       help='数据目录路径')
    parser.add_argument('--threshold', type=float, default=1.0,
                       help='聚类阈值（默认: 1.0）')
    parser.add_argument('--visualize', action='store_true',
                       help='是否可视化结果')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='保存可视化结果的目录')
    
    args = parser.parse_args()
    
    # 加载轨迹数据
    print(f"从 {args.data_dir} 加载轨迹数据...")
    x_traj_all, group_ids = load_trajectories_from_groups(args.data_dir)
    
    if len(x_traj_all) == 0:
        print("错误: 没有找到任何轨迹数据")
        return
    
    # 聚类分析
    print(f"\n使用阈值 {args.threshold} 进行聚类分析...")
    clusters, cluster_assignment, rep_traj_list = cluster_trajectories(
        x_traj_all, threshold=args.threshold
    )
    
    # 打印统计信息
    print_cluster_statistics(clusters, cluster_assignment, group_ids)
    
    # 可视化（如果需要）
    if args.visualize:
        save_path = None
        if args.save_dir:
            os.makedirs(args.save_dir, exist_ok=True)
            save_path = os.path.join(args.save_dir, 'cluster_visualization.png')
        
        print("\n生成可视化...")
        visualize_clusters(x_traj_all, cluster_assignment, rep_traj_list, save_path)
    
    # 保存聚类结果
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        result_path = os.path.join(args.save_dir, 'cluster_results.npz')
        np.savez(result_path,
                 clusters=clusters,
                 cluster_assignment=cluster_assignment,
                 rep_traj_list=rep_traj_list,
                 group_ids=group_ids,
                 threshold=args.threshold)
        print(f"\n聚类结果已保存到: {result_path}")


if __name__ == "__main__":
    main()
