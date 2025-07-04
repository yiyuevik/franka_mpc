import numpy as np

def analyze_simulations(file_path):
   
    # 加载数据
    data = np.load(file_path)  # 假设形状为 (T, S, N) -> T: 样本数量, S: 状态数量, N: 模拟组数
    T, S, N = data.shape

    flat = data.reshape(T * S, N)

    # 计算每个模拟序列的方差
    variances = np.var(flat, axis=0)
    zero_var_indices = np.where(variances == 0)[0]
    nonzero_var_indices = np.where(variances != 0)[0]

    print(f"总模拟组数: {N}")
    print(f"常数序列数量 (方差=0): {len(zero_var_indices)}，索引: {zero_var_indices.tolist()}")
    print(f"非零方差序列数量: {len(nonzero_var_indices)}，索引: {nonzero_var_indices.tolist()}\n")

    # 仅对非零序列计算 Pearson 相关系数矩阵
    if len(nonzero_var_indices) > 1:
        # 提取非零序列数据
        subset = flat[:, nonzero_var_indices]
        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(subset, rowvar=False)

        # 取上三角（不含对角线）
        iu = np.triu_indices_from(corr_matrix, k=1)
        pairwise_corrs = corr_matrix[iu]
        r_min = pairwise_corrs.min()
        r_max = pairwise_corrs.max()
        r_mean = pairwise_corrs.mean()

        print("非零方差序列两两 Pearson 相关系数统计：")
        print(f"  最小相关系数: {r_min:.4f}")
        print(f"  最大相关系数: {r_max:.4f}")
        print(f"  平均相关系数: {r_mean:.4f}")
    else:
        print("非零方差序列不足以计算成对相关系数。")

# 示例调用：修改为你的文件路径
if __name__ == "__main__":
    file_path = "/mnt/c/Users/ASUS/Desktop/franka_mpc/simulation_results/all_simX_20250624_154558.npy"
    analyze_simulations(file_path)
