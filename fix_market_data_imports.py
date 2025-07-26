#!/usr/bin/env python3
"""
修复 market_data 导入路径问题
将 from common.models.market_data import 改为 from common.models.market import
"""

import os
import re

def fix_market_data_imports(file_path):
    """修复单个文件的 market_data 导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复导入语句
        patterns = [
            (r'from\s+common\.models\.market_data\s+import', 'from common.models.market import'),
            (r'from\s+\.{3}common\.models\.market_data\s+import', 'from ...common.models.market import'),
            (r'from\s+\.{2}common\.models\.market_data\s+import', 'from ..common.models.market import'),
            (r'from\s+\.common\.models\.market_data\s+import', 'from .common.models.market import'),
            (r'import\s+common\.models\.market_data', 'import common.models.market'),
        ]
        
        for pattern, replacement in patterns:
            content = re.sub(pattern, replacement, content)
        
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed: {file_path}")
            return True
        else:
            print(f"No changes needed: {file_path}")
            return False
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return False

def main():
    """主函数"""
    files_to_fix = [
        '/Users/pm/work_ai/Prometheus/src/datahub/processors/data_transformer.py',
        '/Users/pm/work_ai/Prometheus/src/datahub/processors/base.py',
        '/Users/pm/work_ai/Prometheus/src/datahub/data_manager.py',
        '/Users/pm/work_ai/Prometheus/src/alpha_engine/base_strategy.py',
        '/Users/pm/work_ai/Prometheus/src/alpha_engine/engine.py',
        '/Users/pm/work_ai/Prometheus/src/alpha_engine/strategy_manager.py',
        '/Users/pm/work_ai/Prometheus/src/alpha_engine/base.py'
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_market_data_imports(file_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nFixed {fixed_count} files.")

if __name__ == "__main__":
    main()