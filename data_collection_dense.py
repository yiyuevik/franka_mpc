"""
极密集数据收集脚本 - 内存优化版本
系统性收集不同初始猜测下的MPC控制数据
步长: 2.5 (中等密集模式)
总组合数: 17^3 = 4,913
内存优化: 每200个组合保存一次并清空内存
"""

import numpy as np
import json
import time
import os
import gc  # 垃圾回收
from datetime import datetime
from franka_ocp import create_ocp_solver, simulate_closed_loop
import config
import urdf2casadi.urdfparser as u2c

def GenerateGridInitialGuess(u4_min, u4_max):
    """
    生成网格化的初始猜测 - 分批次版本
    u4: [u4_min, u4_max], u5: [-20, 20], u7: [-20, 20]
    步长2.5
    """
    min_val = -20
    max_val = 20
    step_size = 2.5
    
    print(f"生成网格参数:")
    print(f"  u4 范围: [{u4_min}, {u4_max}]")
    print(f"  u5 范围: [{min_val}, {max_val}]")
    print(f"  u7 范围: [{min_val}, {max_val}]")
    print(f"  步长: {step_size}")
    
    # 生成网格点
    u4_range = np.arange(u4_min, u4_max + step_size, step_size)
    u5_range = np.arange(min_val, max_val + step_size, step_size)
    u7_range = np.arange(min_val, max_val + step_size, step_size)
    
    print(f"  u4 点数: {len(u4_range)}")
    print(f"  u5 点数: {len(u5_range)}")
    print(f"  u7 点数: {len(u7_range)}")
    print(f"  当前批次组合数: {len(u4_range)} × {len(u5_range)} × {len(u7_range)} = {len(u4_range) * len(u5_range) * len(u7_range)}")
    
    # 创建网格
    u4_grid, u5_grid, u7_grid = np.meshgrid(u4_range, u5_range, u7_range)
    
    # 展平成列表
    u4_flat = u4_grid.flatten()
    u5_flat = u5_grid.flatten()
    u7_flat = u7_grid.flatten()
    
    # 组合成初始猜测数组
    all_guesses = []
    for i in range(len(u4_flat)):
        u_guess = np.array([0, 0, 0, u4_flat[i], u5_flat[i], 0, u7_flat[i]])
        all_guesses.append(u_guess)
    
    return np.array(all_guesses)

def get_batch_ranges():
    """
    返回所有批次的u4范围
    步长2.5，分成3个批次
    """
    ranges = [
        (-20, -7.5),   # 批次1: u4 从 -20 到 -7.5 (6个点)
        (-5, 7.5),     # 批次2: u4 从 -5 到 7.5 (6个点) 
        (10, 20)       # 批次3: u4 从 10 到 20 (5个点)
    ]
    return ranges

def fk_position(T_fk_fun, q_row):
    """计算前向运动学位置"""
    T = T_fk_fun(q_row[:7])       
    p = T[:3, 3]               
    return np.array(p).reshape(3)

def collect_systematic_data_memory_optimized(batch_id=None):
    """
    内存优化的系统性数据收集 - 分批次版本
    """
    print("DEBUG: collect_systematic_data_memory_optimized 函数开始")
    
    # 获取所有批次范围
    batch_ranges = get_batch_ranges()
    
    if batch_id is None:
        # 让用户选择批次
        print("🎯 可用的批次:")
        for i, (u4_min, u4_max) in enumerate(batch_ranges, 1):
            u4_count = len(np.arange(u4_min, u4_max + 2.5, 2.5))
            total_combinations = u4_count * 17 * 17  # u5和u7各17个点
            print(f"  批次 {i}: u4 [{u4_min}, {u4_max}] ({u4_count}个点) - {total_combinations:,} 组合")
        
        print(f"\n📊 总计: {len(batch_ranges)} 个批次，总共约 17 × 17 × 17 = 4,913 组合")
        
        try:
            batch_choice = int(input("\n请选择要运行的批次 (1-3): "))
            if batch_choice < 1 or batch_choice > len(batch_ranges):
                print("❌ 无效的批次选择")
                return
            batch_id = batch_choice - 1
        except ValueError:
            print("❌ 请输入有效的数字")
            return
    
    u4_min, u4_max = batch_ranges[batch_id]
    
    # 生成当前批次的初始猜测组合
    print(f"\n🚀 开始生成批次 {batch_id + 1} 的网格化初始猜测...")
    all_initial_guesses = GenerateGridInitialGuess(u4_min, u4_max)
    total_combinations = len(all_initial_guesses)
    
    print(f"\n🎯 批次 {batch_id + 1} 启动:")
    print(f"   u4 范围: [{u4_min}, {u4_max}]")
    print(f"   当前批次组合数: {total_combinations:,}")
    print(f"   预计时间: ~{total_combinations * 0.1 / 3600:.1f} 小时")
    print(f"   模拟步数: 400 (保持不变)")
    
    # 固定初始状态
    x0 = np.array([0,0,0,0,0,0,0, 0,0,0,0,0,0,0])  # 与原代码一致
    
    # 仿真参数 - 保持与原代码一致
    N_sim = 400  # 保持原始设置
    
    # 初始化前向运动学
    franka = u2c.URDFparser()
    path_to_franka = os.path.dirname(os.path.abspath(__file__)) + '/urdf/panda_arm.urdf'
    franka.from_file(path_to_franka)
    fk_dict = franka.get_forward_kinematics(config.root, config.tip)
    T_fk_fun = fk_dict["T_fk"]
    
    # 创建保存目录 - 包含批次信息
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"mpc_data_batch{batch_id+1}_u4_{u4_min}to{u4_max}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 内存优化参数
    BATCH_SIZE = 200  # 每50个组合保存一次并清空内存
    batch_count = 0
    
    # 临时数据存储（小批量）
    batch_successful = {
        'initial_guesses': [],
        'trajectories_X': [],           # 包含末端位置的完整状态
        'trajectories_U': [],
        'costs': [],
        'solve_times': [],
        'success_flags': []
    }
    
    batch_failed = {
        'initial_guesses': [],
        'error_messages': [],
        'failure_step': []
    }
    
    # 全局统计（只保存数字，不保存大数组）
    global_stats = {
        'total_successful': 0,
        'total_failed': 0,
        'total_processed': 0,
        'total_solve_time': 0,
        'batch_files': [],
        'start_time': time.time()
    }
    
    print(f"\n📁 数据将保存到: {save_dir}")
    print(f"💾 批次大小: {BATCH_SIZE} (每{BATCH_SIZE}个组合保存一次)")
    print("=" * 60)
    
    # 开始数据收集
    for i, u_guess_array in enumerate(all_initial_guesses):
        progress = (i+1) / total_combinations * 100
        elapsed = time.time() - global_stats['start_time']
        eta = elapsed * (total_combinations - i - 1) / (i + 1) if i > 0 else 0
        
        print(f"\n📊 进度: {i+1:,}/{total_combinations:,} ({progress:.2f}%)")
        print(f"🎯 当前: u4={u_guess_array[3]:.1f}, u5={u_guess_array[4]:.1f}, u7={u_guess_array[6]:.1f}")
        print(f"⏱️  已用时: {elapsed/60:.1f}min, 预计剩余: {eta/60:.1f}min")
        
        try:
            # 创建求解器
            ocp, ocp_solver, integrator = create_ocp_solver(x0)
            
            # 设置初始猜测 - 与原代码逻辑一致
            ocp_solver.set(0, "x", x0)
            for j in range(config.Horizon):
                ocp_solver.set(j, "u", u_guess_array)
            
            # 记录开始时间
            start_time = time.time()
            
            # 运行仿真 - 使用N_sim=400
            t, simX, simU, simCost, success = simulate_closed_loop(
                ocp, ocp_solver, integrator, x0, N_sim=N_sim
            )
            
            end_time = time.time()
            solve_time = end_time - start_time
            
            if success:
                # 计算末端位置 - 与原代码逻辑一致
                pos = np.apply_along_axis(lambda q_i: fk_position(T_fk_fun, q_i), 1, simX)
                simX_with_pos = np.hstack((simX, pos))  # 添加末端位置
                
                # 存储到批次缓存
                batch_successful['initial_guesses'].append(u_guess_array)
                batch_successful['trajectories_X'].append(simX_with_pos)
                batch_successful['trajectories_U'].append(simU)
                batch_successful['costs'].append(simCost)
                batch_successful['solve_times'].append(solve_time)
                batch_successful['success_flags'].append(True)
                
                # 更新全局统计
                global_stats['total_successful'] += 1
                global_stats['total_solve_time'] += solve_time
                
                print(f"✅ 成功! 求解时间: {solve_time:.3f}s")
                
            else:
                # 记录失败信息
                batch_failed['initial_guesses'].append(u_guess_array)
                batch_failed['error_messages'].append("Simulation failed")
                batch_failed['failure_step'].append(-1)
                
                global_stats['total_failed'] += 1
                print(f"❌ 仿真失败")
                
        except Exception as e:
            # 记录异常
            batch_failed['initial_guesses'].append(u_guess_array)
            batch_failed['error_messages'].append(str(e))
            batch_failed['failure_step'].append(-1)
            
            global_stats['total_failed'] += 1
            print(f"❌ 异常: {str(e)}")
        
        global_stats['total_processed'] += 1
        
        # 🔥 关键：每BATCH_SIZE个组合保存一次并清空内存
        if (i + 1) % BATCH_SIZE == 0:
            batch_count += 1
            
            # 保存当前批次
            batch_filename = save_batch_and_clear_memory(
                batch_successful, batch_failed, save_dir, batch_count, i+1
            )
            global_stats['batch_files'].append(batch_filename)
            
            # 🚀 清空内存并强制垃圾回收
            batch_successful = {
                'initial_guesses': [],
                'trajectories_X': [],
                'trajectories_U': [],
                'costs': [],
                'solve_times': [],
                'success_flags': []
            }
            
            batch_failed = {
                'initial_guesses': [],
                'error_messages': [],
                'failure_step': []
            }
            
            # 强制垃圾回收
            gc.collect()
            
            print(f"💾 已保存批次 {batch_count}，内存已清空")
            print_memory_usage()
            print_progress_summary(global_stats, i+1, total_combinations)
    
    # 保存最后一个不完整的批次
    if batch_successful['initial_guesses'] or batch_failed['initial_guesses']:
        batch_count += 1
        batch_filename = save_batch_and_clear_memory(
            batch_successful, batch_failed, save_dir, batch_count, total_combinations
        )
        global_stats['batch_files'].append(batch_filename)
    
    # 保存全局统计
    save_global_statistics(global_stats, save_dir, timestamp, batch_id+1, u4_min, u4_max)
    
    # 打印最终统计
    print_final_statistics(global_stats, total_combinations, save_dir)

def save_batch_and_clear_memory(successful_data, failed_data, save_dir, batch_num, processed_count):
    """保存批次数据并返回文件名"""
    batch_filename = f"batch_{batch_num:04d}"
    
    # 保存成功的数据
    if successful_data['initial_guesses']:
        np.save(f"{save_dir}/{batch_filename}_successful_initial_guesses.npy", 
                np.array(successful_data['initial_guesses']))
        np.save(f"{save_dir}/{batch_filename}_successful_trajectories_X.npy", 
                np.array(successful_data['trajectories_X']))
        np.save(f"{save_dir}/{batch_filename}_successful_trajectories_U.npy", 
                np.array(successful_data['trajectories_U']))
        np.save(f"{save_dir}/{batch_filename}_successful_costs.npy", 
                np.array(successful_data['costs']))
        np.save(f"{save_dir}/{batch_filename}_successful_solve_times.npy", 
                np.array(successful_data['solve_times']))
        np.save(f"{save_dir}/{batch_filename}_successful_flags.npy", 
                np.array(successful_data['success_flags']))
    
    # 保存失败的数据
    if failed_data['initial_guesses']:
        np.save(f"{save_dir}/{batch_filename}_failed_initial_guesses.npy", 
                np.array(failed_data['initial_guesses']))
        np.save(f"{save_dir}/{batch_filename}_failed_error_messages.npy", 
                np.array(failed_data['error_messages'], dtype=object))
    
    # 保存这个批次的元数据
    batch_metadata = {
        'batch_number': batch_num,
        'processed_count': processed_count,
        'successful_count': len(successful_data['initial_guesses']),
        'failed_count': len(failed_data['initial_guesses']),
        'N_sim': 400,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(f"{save_dir}/{batch_filename}_metadata.json", 'w') as f:
        json.dump(batch_metadata, f, indent=2)
    
    success_count = batch_metadata['successful_count']
    failed_count = batch_metadata['failed_count']
    print(f"💾 批次 {batch_num} 已保存: {success_count} 成功, {failed_count} 失败")
    
    return batch_filename

def print_memory_usage():
    """打印内存使用情况"""
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"🧠 内存使用: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
    except ImportError:
        print("🧠 内存监控需要安装 psutil: pip install psutil")

def print_progress_summary(global_stats, processed, total):
    """打印进度摘要"""
    success_rate = global_stats['total_successful'] / processed * 100 if processed > 0 else 0
    avg_time = global_stats['total_solve_time'] / global_stats['total_successful'] if global_stats['total_successful'] > 0 else 0
    
    print(f"📈 当前统计:")
    print(f"   成功率: {success_rate:.1f}% ({global_stats['total_successful']}/{processed})")
    print(f"   平均求解时间: {avg_time:.3f}s")
    print(f"   剩余组合数: {total - processed:,}")

def save_global_statistics(global_stats, save_dir, timestamp, batch_id, u4_min, u4_max):
    """保存全局统计信息"""
    global_stats['timestamp'] = timestamp
    global_stats['batch_id'] = batch_id
    global_stats['u4_range'] = [u4_min, u4_max]
    global_stats['collection_completed'] = True
    global_stats['total_duration'] = time.time() - global_stats['start_time']
    global_stats['average_solve_time'] = (global_stats['total_solve_time'] / 
                                        global_stats['total_successful'] 
                                        if global_stats['total_successful'] > 0 else 0)
    
    # 移除不需要保存的临时数据
    stats_to_save = {k: v for k, v in global_stats.items() if k != 'start_time'}
    
    with open(f"{save_dir}/global_statistics.json", 'w') as f:
        json.dump(stats_to_save, f, indent=2)

def print_final_statistics(global_stats, total, save_dir):
    """打印最终统计信息"""
    duration = time.time() - global_stats['start_time']
    success_rate = global_stats['total_successful'] / total * 100
    
    print("\n" + "=" * 60)
    print("🎉 数据收集完成!")
    print("=" * 60)
    print(f"📊 总测试组合: {total:,}")
    print(f"✅ 总成功: {global_stats['total_successful']:,} ({success_rate:.1f}%)")
    print(f"❌ 总失败: {global_stats['total_failed']:,} ({100-success_rate:.1f}%)")
    print(f"⏱️  总耗时: {duration/3600:.2f} 小时")
    print(f"📁 保存的批次数: {len(global_stats['batch_files'])}")
    print(f"💾 数据目录: {save_dir}")
    
    if global_stats['total_successful'] > 0:
        avg_time = global_stats['total_solve_time'] / global_stats['total_successful']
        print(f"📈 平均求解时间: {avg_time:.3f}s")
    
    print("=" * 60)

def merge_all_batches(save_dir):
    """
    将所有批次文件合并成完整的数据集
    """
    print("🔗 开始合并所有批次数据...")
    
    # 找到所有批次文件
    batch_files = []
    for file in os.listdir(save_dir):
        if file.startswith('batch_') and file.endswith('_metadata.json'):
            batch_num = int(file.split('_')[1])
            batch_files.append(batch_num)
    
    batch_files.sort()
    print(f"找到 {len(batch_files)} 个批次文件")
    
    # 合并数据
    all_successful_guesses = []
    all_successful_X = []
    all_successful_U = []
    all_successful_costs = []
    all_successful_times = []
    all_successful_flags = []
    
    all_failed_guesses = []
    all_failed_messages = []
    
    for batch_num in batch_files:
        batch_prefix = f"batch_{batch_num:04d}"
        print(f"合并批次 {batch_num}...")
        
        # 加载成功数据
        try:
            guesses = np.load(f"{save_dir}/{batch_prefix}_successful_initial_guesses.npy")
            X = np.load(f"{save_dir}/{batch_prefix}_successful_trajectories_X.npy")
            U = np.load(f"{save_dir}/{batch_prefix}_successful_trajectories_U.npy")
            costs = np.load(f"{save_dir}/{batch_prefix}_successful_costs.npy")
            times = np.load(f"{save_dir}/{batch_prefix}_successful_solve_times.npy")
            flags = np.load(f"{save_dir}/{batch_prefix}_successful_flags.npy")
            
            all_successful_guesses.append(guesses)
            all_successful_X.append(X)
            all_successful_U.append(U)
            all_successful_costs.append(costs)
            all_successful_times.append(times)
            all_successful_flags.append(flags)
        except FileNotFoundError:
            pass
        
        # 加载失败数据
        try:
            failed_guesses = np.load(f"{save_dir}/{batch_prefix}_failed_initial_guesses.npy")
            failed_messages = np.load(f"{save_dir}/{batch_prefix}_failed_error_messages.npy", allow_pickle=True)
            
            all_failed_guesses.append(failed_guesses)
            all_failed_messages.append(failed_messages)
        except FileNotFoundError:
            pass
    
    # 合并并保存
    if all_successful_guesses:
        print("保存合并的成功数据...")
        np.save(f"{save_dir}/merged_successful_initial_guesses.npy", np.vstack(all_successful_guesses))
        np.save(f"{save_dir}/merged_successful_trajectories_X.npy", np.vstack(all_successful_X))
        np.save(f"{save_dir}/merged_successful_trajectories_U.npy", np.vstack(all_successful_U))
        np.save(f"{save_dir}/merged_successful_costs.npy", np.hstack(all_successful_costs))
        np.save(f"{save_dir}/merged_successful_solve_times.npy", np.hstack(all_successful_times))
        np.save(f"{save_dir}/merged_successful_flags.npy", np.hstack(all_successful_flags))
        
        print(f"✅ 成功数据: {len(np.vstack(all_successful_guesses))} 条记录")
    
    if all_failed_guesses:
        print("保存合并的失败数据...")
        np.save(f"{save_dir}/merged_failed_initial_guesses.npy", np.vstack(all_failed_guesses))
        np.save(f"{save_dir}/merged_failed_error_messages.npy", np.hstack(all_failed_messages))
        
        print(f"❌ 失败数据: {len(np.vstack(all_failed_guesses))} 条记录")
    
    print(f"✅ 数据合并完成! 合并文件保存在 {save_dir}/merged_*.npy")

def merge_all_batch_folders():
    """
    合并所有批次文件夹的数据成一个完整数据集
    """
    print("🔗 开始合并所有批次文件夹的数据...")
    
    # 找到所有批次文件夹
    batch_dirs = []
    for dir_name in os.listdir('.'):
        if dir_name.startswith('mpc_data_batch'):
            batch_dirs.append(dir_name)
    
    batch_dirs.sort()
    print(f"找到 {len(batch_dirs)} 个批次文件夹: {batch_dirs}")
    
    if len(batch_dirs) == 0:
        print("❌ 未找到任何批次文件夹")
        return
    
    # 创建合并后的保存目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_dir = f"merged_all_batches_{timestamp}"
    os.makedirs(merged_dir, exist_ok=True)
    
    # 合并数据
    all_successful_guesses = []
    all_successful_X = []
    all_successful_U = []
    all_successful_costs = []
    all_successful_times = []
    all_successful_flags = []
    
    all_failed_guesses = []
    all_failed_messages = []
    
    total_stats = {
        'total_successful': 0,
        'total_failed': 0,
        'total_solve_time': 0,
        'batch_info': []
    }
    
    for batch_dir in batch_dirs:
        print(f"处理批次文件夹: {batch_dir}")
        
        # 检查是否有合并文件
        merged_success_file = f"{batch_dir}/merged_successful_initial_guesses.npy"
        merged_failed_file = f"{batch_dir}/merged_failed_initial_guesses.npy"
        
        if os.path.exists(merged_success_file):
            # 直接加载合并文件
            guesses = np.load(merged_success_file)
            X = np.load(f"{batch_dir}/merged_successful_trajectories_X.npy")
            U = np.load(f"{batch_dir}/merged_successful_trajectories_U.npy")
            costs = np.load(f"{batch_dir}/merged_successful_costs.npy")
            times = np.load(f"{batch_dir}/merged_successful_solve_times.npy")
            flags = np.load(f"{batch_dir}/merged_successful_flags.npy")
            
            all_successful_guesses.append(guesses)
            all_successful_X.append(X)
            all_successful_U.append(U)
            all_successful_costs.append(costs)
            all_successful_times.append(times)
            all_successful_flags.append(flags)
            
            print(f"  成功数据: {len(guesses)} 条记录")
            total_stats['total_successful'] += len(guesses)
            total_stats['total_solve_time'] += np.sum(times)
        
        else:
            # 需要先合并该批次
            print(f"  正在合并批次 {batch_dir}...")
            merge_all_batches(batch_dir)
            
            # 然后加载合并文件
            if os.path.exists(merged_success_file):
                guesses = np.load(merged_success_file)
                X = np.load(f"{batch_dir}/merged_successful_trajectories_X.npy")
                U = np.load(f"{batch_dir}/merged_successful_trajectories_U.npy")
                costs = np.load(f"{batch_dir}/merged_successful_costs.npy")
                times = np.load(f"{batch_dir}/merged_successful_solve_times.npy")
                flags = np.load(f"{batch_dir}/merged_successful_flags.npy")
                
                all_successful_guesses.append(guesses)
                all_successful_X.append(X)
                all_successful_U.append(U)
                all_successful_costs.append(costs)
                all_successful_times.append(times)
                all_successful_flags.append(flags)
                
                print(f"  成功数据: {len(guesses)} 条记录")
                total_stats['total_successful'] += len(guesses)
                total_stats['total_solve_time'] += np.sum(times)
        
        # 处理失败数据
        if os.path.exists(f"{batch_dir}/merged_failed_initial_guesses.npy"):
            failed_guesses = np.load(f"{batch_dir}/merged_failed_initial_guesses.npy")
            failed_messages = np.load(f"{batch_dir}/merged_failed_error_messages.npy", allow_pickle=True)
            
            all_failed_guesses.append(failed_guesses)
            all_failed_messages.append(failed_messages)
            
            print(f"  失败数据: {len(failed_guesses)} 条记录")
            total_stats['total_failed'] += len(failed_guesses)
        
        # 读取批次统计信息
        global_stats_file = f"{batch_dir}/global_statistics.json"
        if os.path.exists(global_stats_file):
            with open(global_stats_file, 'r') as f:
                batch_stats = json.load(f)
                total_stats['batch_info'].append({
                    'folder': batch_dir,
                    'batch_id': batch_stats.get('batch_id', 'unknown'),
                    'u4_range': batch_stats.get('u4_range', 'unknown'),
                    'successful': batch_stats.get('total_successful', 0),
                    'failed': batch_stats.get('total_failed', 0)
                })
    
    # 最终合并并保存
    if all_successful_guesses:
        print("\n💾 保存最终合并的成功数据...")
        final_guesses = np.vstack(all_successful_guesses)
        final_X = np.vstack(all_successful_X)
        final_U = np.vstack(all_successful_U)
        final_costs = np.hstack(all_successful_costs)
        final_times = np.hstack(all_successful_times)
        final_flags = np.hstack(all_successful_flags)
        
        np.save(f"{merged_dir}/final_successful_initial_guesses.npy", final_guesses)
        np.save(f"{merged_dir}/final_successful_trajectories_X.npy", final_X)
        np.save(f"{merged_dir}/final_successful_trajectories_U.npy", final_U)
        np.save(f"{merged_dir}/final_successful_costs.npy", final_costs)
        np.save(f"{merged_dir}/final_successful_solve_times.npy", final_times)
        np.save(f"{merged_dir}/final_successful_flags.npy", final_flags)
        
        print(f"✅ 最终成功数据: {len(final_guesses):,} 条记录")
    
    if all_failed_guesses:
        print("💾 保存最终合并的失败数据...")
        final_failed_guesses = np.vstack(all_failed_guesses)
        final_failed_messages = np.hstack(all_failed_messages)
        
        np.save(f"{merged_dir}/final_failed_initial_guesses.npy", final_failed_guesses)
        np.save(f"{merged_dir}/final_failed_error_messages.npy", final_failed_messages)
        
        print(f"❌ 最终失败数据: {len(final_failed_guesses):,} 条记录")
    
    # 保存最终统计信息
    total_stats['merge_timestamp'] = timestamp
    total_stats['success_rate'] = (total_stats['total_successful'] / 
                                 (total_stats['total_successful'] + total_stats['total_failed']) * 100
                                 if (total_stats['total_successful'] + total_stats['total_failed']) > 0 else 0)
    total_stats['average_solve_time'] = (total_stats['total_solve_time'] / 
                                       total_stats['total_successful']
                                       if total_stats['total_successful'] > 0 else 0)
    
    with open(f"{merged_dir}/final_statistics.json", 'w') as f:
        json.dump(total_stats, f, indent=2)
    
    print(f"\n🎉 所有批次数据合并完成!")
    print(f"📊 最终统计:")
    print(f"   总成功: {total_stats['total_successful']:,}")
    print(f"   总失败: {total_stats['total_failed']:,}")
    print(f"   成功率: {total_stats['success_rate']:.1f}%")
    print(f"   平均求解时间: {total_stats['average_solve_time']:.3f}s")
    print(f"💾 合并数据保存在: {merged_dir}/")

def main():
    """主函数"""
    print("🚀 分批次数据收集脚本启动")
    print("=" * 60)
    print("📋 总任务划分:")
    print("   步长: 2.5 (中等密集模式)")
    print("   总任务: u4[-20,20] × u5[-20,20] × u7[-20,20] = 17×17×17 = 4,913 组合")
    print("   现在分成3个批次，每个批次约1,600组合")
    print("   每个批次预计耗时: ~27分钟")
    print("=" * 60)
    
    # 开始收集
    collect_systematic_data_memory_optimized()
    
    # 询问是否合并数据
    response = input("\n🤔 当前批次完成! 是否要合并当前批次的数据? (y/n): ")
    if response.lower() == 'y':
        # 获取最新的保存目录
        dirs = [d for d in os.listdir('.') if d.startswith('mpc_data_batch')]
        if dirs:
            latest_dir = sorted(dirs)[-1]
            merge_all_batches(latest_dir)
        else:
            print("❌ 未找到数据收集目录")
    
    print("\n" + "=" * 60)
    print("💡 提示: 要运行下一个批次，请再次运行脚本并选择不同的批次号")
    print("💡 所有3个批次完成后，可以使用合并工具将所有数据合并")
    print("=" * 60)

if __name__ == "__main__":
    main()
