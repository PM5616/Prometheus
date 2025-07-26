#!/usr/bin/env python3
"""
修复所有模块中的导入问题
将 from common.xxx import 改为 from src.common.xxx import
"""

import os
import re
from pathlib import Path

def fix_all_imports():
    """修复所有模块中的导入问题"""
    
    # 定义需要替换的模式
    patterns = [
        (r'from common\.', 'from src.common.'),
        (r'from datahub\.', 'from src.datahub.'),
        (r'from alpha_engine\.', 'from src.alpha_engine.'),
        (r'from portfolio_manager\.', 'from src.portfolio_manager.'),
        (r'from risk_sentinel\.', 'from src.risk_sentinel.'),
        (r'from execution\.', 'from src.execution.'),
        (r'from monitor\.', 'from src.monitor.'),
        (r'from backtest\.', 'from src.backtest.'),
        (r'from gateway\.', 'from src.gateway.'),
    ]
    
    # 获取 src 目录下的所有模块
    src_dir = Path('src')
    
    if not src_dir.exists():
        print(f"目录 {src_dir} 不存在")
        return
    
    # 遍历所有 Python 文件
    python_files = list(src_dir.rglob('*.py'))
    
    fixed_files = []
    
    for file_path in python_files:
        # 跳过 __pycache__ 目录
        if '__pycache__' in str(file_path):
            continue
            
        try:
            # 读取文件内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 应用所有替换模式
            for pattern, replacement in patterns:
                content = re.sub(pattern, replacement, content)
            
            # 如果内容有变化，写回文件
            if content != original_content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                fixed_files.append(str(file_path))
                print(f"已修复: {file_path}")
        
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    print(f"\n总共修复了 {len(fixed_files)} 个文件:")
    for file_path in fixed_files:
        print(f"  - {file_path}")

if __name__ == '__main__':
    fix_all_imports()