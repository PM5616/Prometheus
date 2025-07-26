#!/usr/bin/env python3
"""
修复 gateway 模块中的导入问题
将 from common.xxx import 改为 from src.common.xxx import
"""

import os
import re
from pathlib import Path

def fix_gateway_imports():
    """修复 gateway 模块中的导入问题"""
    
    # 定义需要替换的模式
    patterns = [
        (r'from common\.logging\.logger import', 'from src.common.logging import'),
        (r'from common\.exceptions\.gateway_exceptions import', 'from src.common.exceptions.gateway_exceptions import'),
        (r'from common\.models import', 'from src.common.models import'),
        (r'from common\.logging import', 'from src.common.logging import'),
        (r'from common\.exceptions import', 'from src.common.exceptions import'),
        (r'from common\.config import', 'from src.common.config import'),
    ]
    
    # 获取 gateway 模块目录
    gateway_dir = Path('src/gateway')
    
    if not gateway_dir.exists():
        print(f"目录 {gateway_dir} 不存在")
        return
    
    # 遍历所有 Python 文件
    python_files = list(gateway_dir.glob('*.py'))
    
    fixed_files = []
    
    for file_path in python_files:
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
    fix_gateway_imports()