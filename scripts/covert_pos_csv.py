import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.helpers import compute_end_effector_position

# ===== 路径配置 =====
INPUT_DIR = Path("data/copy/sim_q_trajs_mpc")        # 输入：关节角度 CSV 文件夹
OUTPUT_DIR = Path("data/copy/sim_pos_trajs_mpc")     # 输出：末端位置 CSV 文件夹
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def convert_csv_file(input_csv_path, output_csv_path):
    """
    转换单个 CSV 文件：从关节角度到末端位置
    """
    # 读取关节角度 CSV
    df = pd.read_csv(input_csv_path)
    
    # 提取关节角度列
    q_cols = [f"q{i}" for i in range(1, 8)]
    if not all(col in df.columns for col in q_cols):
        raise ValueError(f"{input_csv_path} 缺少关节角度列")
    
    q_array = df[q_cols].values  # (T, 7)
    T = q_array.shape[0]
    
    # 转换为位置 - 使用 helpers 中的函数
    pos_array = np.zeros((T, 3))
    for t in range(T):
        pos_array[t] = compute_end_effector_position(q_array[t])
    
    # 创建新的 DataFrame
    df_out = pd.DataFrame({
        "stamp_sec": df["stamp_sec"].values,
        "x": pos_array[:, 0],
        "y": pos_array[:, 1],
        "z": pos_array[:, 2]
    })
    
    # 保存
    df_out.to_csv(output_csv_path, index=False)
    
    return T

# ===== 主循环：处理所有 CSV 文件 =====
csv_files = sorted(INPUT_DIR.glob("*.csv"))

if len(csv_files) == 0:
    print(f"[WARN] {INPUT_DIR} 中没有找到 CSV 文件")
else:
    print(f"找到 {len(csv_files)} 个 CSV 文件，开始转换...\n")
    
    for csv_path in csv_files:
        output_csv_path = OUTPUT_DIR / csv_path.name
        
        try:
            num_steps = convert_csv_file(csv_path, output_csv_path)
            print(f"✅ {csv_path.name} -> {output_csv_path.name}  ({num_steps} 步)")
        except Exception as e:
            print(f"❌ {csv_path.name} 转换失败: {e}")
    
    print(f"\n全部转换完成，输出目录: {OUTPUT_DIR}")