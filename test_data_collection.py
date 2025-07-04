#!/usr/bin/env python3
"""
简化版数据收集脚本 - 用于测试
"""

import numpy as np
import json
import time
import os
import sys
from datetime import datetime

print("=" * 60)
print("🚀 简化版数据收集脚本启动")
print("=" * 60)

def get_batch_ranges():
    """返回所有批次的u4范围"""
    ranges = [
        (-20, -12),  # 批次1: u4 从 -20 到 -12
        (-11, -3),   # 批次2: u4 从 -11 到 -3
        (-2, 6),     # 批次3: u4 从 -2 到 6
        (7, 15),     # 批次4: u4 从 7 到 15
        (16, 20)     # 批次5: u4 从 16 到 20
    ]
    return ranges

def main():
    print("📋 可用的批次:")
    batch_ranges = get_batch_ranges()
    
    for i, (u4_min, u4_max) in enumerate(batch_ranges, 1):
        u4_count = int(u4_max - u4_min + 1)
        total_combinations = u4_count * 41 * 41
        print(f"  批次 {i}: u4 [{u4_min}, {u4_max}] ({u4_count}个点) - {total_combinations:,} 组合")
    
    print(f"\n📊 总计: {len(batch_ranges)} 个批次")
    
    # 直接提示用户输入
    print("\n请选择要运行的批次 (1-5): ", end="", flush=True)
    
    # 尝试不同的输入方法
    try:
        if sys.stdin.isatty():
            print("(检测到交互式终端)")
            choice = input()
        else:
            print("(检测到非交互式环境)")
            # 默认选择批次1
            choice = "1"
            print(f"自动选择批次: {choice}")
        
        batch_choice = int(choice)
        
        if batch_choice < 1 or batch_choice > len(batch_ranges):
            print("❌ 无效的批次选择")
            return
        
        u4_min, u4_max = batch_ranges[batch_choice - 1]
        
        print(f"\n✅ 已选择批次 {batch_choice}: u4 [{u4_min}, {u4_max}]")
        print("🎯 模拟开始数据收集...")
        
        # 模拟一些处理
        for i in range(3):
            print(f"  模拟处理组合 {i+1}/3...")
            time.sleep(1)
        
        print("✅ 测试完成!")
        
    except KeyboardInterrupt:
        print("\n❌ 用户取消")
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print(f"错误类型: {type(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
