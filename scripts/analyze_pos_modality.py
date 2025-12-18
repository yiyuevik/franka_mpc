import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def load_position_trajectories(data_dir):
    """
    加载所有位置轨迹 CSV 文件
    
    Returns:
        pos_trajs: list of (T, 3) arrays - 位置轨迹
        filenames: list of str - 文件名
    """
    data_dir = Path(data_dir)
    csv_files = sorted(data_dir.glob("*.csv"))
    
    if len(csv_files) == 0:
        raise ValueError(f"在 {data_dir} 中没有找到 CSV 文件")
    
    pos_trajs = []
    filenames = []
    
    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        
        # 检查是否有 x, y, z 列
        if not all(col in df.columns for col in ['x', 'y', 'z']):
            print(f"[WARN] {csv_path.name} 缺少 x/y/z 列，跳过")
            continue
        
        pos = df[['x', 'y', 'z']].values  # (T, 3)
        pos_trajs.append(pos)
        filenames.append(csv_path.name)
    
    print(f"成功加载 {len(pos_trajs)} 个位置轨迹")
    return pos_trajs, filenames

def cluster_trajectories(pos_trajs, threshold=0.05):
    """
    使用简单的距离阈值聚类算法对轨迹进行分组
    参考 data_collect.py 中的聚类逻辑
    
    Args:
        pos_trajs: list of (T, 3) arrays
        threshold: float - 聚类阈值（米）
    
    Returns:
        clusters: list of lists - 每个簇包含的轨迹索引
        cluster_assignment: list of int - 每个轨迹的簇ID
        rep_traj_list: list of arrays - 每个簇的代表轨迹
    """
    clusters = []
    rep_traj_list = []
    cluster_assignment = [-1] * len(pos_trajs)
    threshold_sq = threshold ** 2
    
    for i, traj in enumerate(pos_trajs):
        assigned = False
        
        # 尝试分配到现有簇
        for cid, rep_traj in enumerate(rep_traj_list):
            # 首先检查终点距离
            if np.sum((rep_traj[-1] - traj[-1])**2) > threshold_sq:
                continue
            
            # 检查整条轨迹的最大距离
            # 对齐轨迹长度
            min_len = min(len(rep_traj), len(traj))
            rep_aligned = rep_traj[:min_len]
            traj_aligned = traj[:min_len]
            
            max_sq = np.max(np.sum((rep_aligned - traj_aligned) ** 2, axis=1))
            
            if max_sq <= threshold_sq:
                clusters[cid].append(i)
                cluster_assignment[i] = cid
                assigned = True
                break
        
        # 如果没有分配到现有簇，创建新簇
        if not assigned:
            clusters.append([i])
            cluster_assignment[i] = len(clusters) - 1
            rep_traj_list.append(traj)
    
    return clusters, cluster_assignment, rep_traj_list

def print_groups_dict(clusters, filenames):
    """
    打印 Python 字典格式的分组结果
    """
    print("\n" + "="*60)
    print("聚类结果（Python 字典格式）:")
    print("="*60)
    print("\nGROUPS = {")
    
    for cid, indices in enumerate(clusters):
        print(f"    {cid}: [")
        
        # 每行打印多个文件名
        files_in_cluster = [f'"{filenames[i]}"' for i in indices]
        
        # 格式化输出，每行最多 4 个文件
        for j in range(0, len(files_in_cluster), 4):
            line_files = files_in_cluster[j:j+4]
            line = ", ".join(line_files)
            if j + 4 < len(files_in_cluster):
                line += ","
            else:
                line += ","  # 最后一行也加逗号
            print(f"        {line}")
        
        print("    ],")
    
    print("}")
    print("="*60)

def main():
    parser = argparse.ArgumentParser(description="分析位置轨迹的模态数量")
    parser.add_argument("--data_dir", type=str, default='data/copy/sim_pos_trajs_ros',
                        help="包含位置轨迹 CSV 文件的目录")
    parser.add_argument("--threshold", type=float, default=0.05,
                        help="聚类距离阈值（米），默认 0.05")
    
    args = parser.parse_args()
    
    # 加载轨迹
    pos_trajs, filenames = load_position_trajectories(args.data_dir)
    
    # 聚类
    print(f"\n使用阈值 {args.threshold} 米进行聚类...")
    clusters, cluster_assignment, rep_traj_list = cluster_trajectories(
        pos_trajs, threshold=args.threshold
    )
    
    # 统计信息
    num_clusters = len(clusters)
    print(f"\n找到 {num_clusters} 个不同的模态")
    
    print("\n各簇详情:")
    for cid, indices in enumerate(clusters):
        print(f"  簇 {cid}: {len(indices)} 条轨迹")
        # 打印前几个文件名作为示例
        example_files = [filenames[i] for i in indices[:3]]
        print(f"    示例: {', '.join(example_files)}...")
    
    # 打印 Python 字典格式
    print_groups_dict(clusters, filenames)
    
    # 可选：保存结果
    output_file = Path(args.data_dir) / "cluster_result.txt"
    with open(output_file, "w") as f:
        f.write(f"聚类阈值: {args.threshold} 米\n")
        f.write(f"模态数量: {num_clusters}\n\n")
        f.write("GROUPS = {\n")
        for cid, indices in enumerate(clusters):
            f.write(f"    {cid}: [\n")
            files_in_cluster = [f'"{filenames[i]}"' for i in indices]
            for j in range(0, len(files_in_cluster), 4):
                line_files = files_in_cluster[j:j+4]
                line = ", ".join(line_files)
                if j + 4 < len(files_in_cluster):
                    line += ","
                else:
                    line += ","
                f.write(f"        {line}\n")
            f.write("    ],\n")
        f.write("}\n")
    
    print(f"\n✅ 聚类结果已保存到: {output_file}")

if __name__ == "__main__":
    main()