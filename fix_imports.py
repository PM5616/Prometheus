#!/usr/bin/env python3
"""
批量修复相对导入问题的脚本
"""

import os
import re
from pathlib import Path

def fix_relative_imports(file_path: str) -> bool:
    """修复单个文件的相对导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复相对导入模式
        patterns = [
            (r'from \.\.common\.', 'from common.'),
            (r'from \.\.datahub\.', 'from datahub.'),
            (r'from \.\.alpha_engine\.', 'from alpha_engine.'),
            (r'from \.\.portfolio_manager\.', 'from portfolio_manager.'),
            (r'from \.\.risk_sentinel\.', 'from risk_sentinel.'),
            (r'from \.\.execution\.', 'from execution.'),
            (r'from \.\.gateway\.', 'from gateway.'),
            (r'from \.\.monitor\.', 'from monitor.'),
            (r'from \.\.backtest\.', 'from backtest.'),
            (r'from \.\.strategies\.', 'from strategies.')
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✅ Fixed: {file_path}")
            return True
        else:
            return False
            
    except Exception as e:
        print(f"❌ Error fixing {file_path}: {e}")
        return False

def main():
    """主函数"""
    src_dir = Path('/Users/pm/work_ai/Prometheus/src')
    
    print("开始修复相对导入问题...")
    
    fixed_count = 0
    total_count = 0
    
    # 遍历所有Python文件
    for py_file in src_dir.rglob('*.py'):
        if py_file.name == '__init__.py' or 'test' in str(py_file):
            continue
            
        total_count += 1
        if fix_relative_imports(str(py_file)):
            fixed_count += 1
    
    print(f"\n修复完成！")
    print(f"总文件数: {total_count}")
    print(f"修复文件数: {fixed_count}")

if __name__ == '__main__':
    main()