#!/usr/bin/env python3
"""
修复 common.logger 导入路径问题
将 from common.logger import 改为 from common.logging import
"""

import os
import re

def fix_logger_imports(file_path):
    """修复单个文件的 logger 导入"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 修复导入语句
        patterns = [
            (r'from\s+common\.logger\s+import', 'from common.logging import'),
            (r'import\s+common\.logger', 'import common.logging'),
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
        '/Users/pm/work_ai/Prometheus/src/risk_sentinel/monitor.py',
        '/Users/pm/work_ai/Prometheus/src/risk_sentinel/manager.py',
        '/Users/pm/work_ai/Prometheus/src/risk_sentinel/reporter.py',
        '/Users/pm/work_ai/Prometheus/src/risk_sentinel/controller.py',
        '/Users/pm/work_ai/Prometheus/src/risk_sentinel/analyzer.py'
    ]
    
    fixed_count = 0
    for file_path in files_to_fix:
        if os.path.exists(file_path):
            if fix_logger_imports(file_path):
                fixed_count += 1
        else:
            print(f"File not found: {file_path}")
    
    print(f"\nFixed {fixed_count} files.")

if __name__ == "__main__":
    main()