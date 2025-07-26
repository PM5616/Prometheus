#!/usr/bin/env python3
"""
修复 backtest 模块中的相对导入问题
"""

import os
import re

def fix_relative_imports(file_path):
    """修复文件中的相对导入"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # 修复相对导入模式
    patterns = [
        (r'from \.\.common\.logging import', 'from common.logging import'),
        (r'from \.\.common\.exceptions\.backtest import', 'from common.exceptions.backtest import'),
        (r'from \.\.common\.models import', 'from common.models import'),
        (r'from \.\.datahub\.data_manager import', 'from datahub.data_manager import'),
        (r'from \.\.alpha_engine\.base import', 'from alpha_engine.base import'),
        (r'from \.\.alpha_engine\.signal import', 'from alpha_engine.signal import'),
        (r'from \.\.execution\.engine import', 'from execution.engine import'),
        (r'from \.\.risk_sentinel\.manager import', 'from risk_sentinel.manager import'),
    ]
    
    for pattern, replacement in patterns:
        content = re.sub(pattern, replacement, content)
    
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        return True
    return False

def main():
    """主函数"""
    backtest_dir = '/Users/pm/work_ai/Prometheus/src/backtest'
    fixed_files = []
    
    for root, dirs, files in os.walk(backtest_dir):
        for file in files:
            if file.endswith('.py'):
                file_path = os.path.join(root, file)
                if fix_relative_imports(file_path):
                    fixed_files.append(file_path)
                    print(f'Fixed: {file_path}')
    
    print(f'\nFixed {len(fixed_files)} files.')

if __name__ == '__main__':
    main()