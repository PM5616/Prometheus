#!/usr/bin/env python3
"""测试所有模块导入"""

import sys
sys.path.append('.')

modules = [
    'src.alpha_engine',
    'src.backtest', 
    'src.common',
    'src.datahub',
    'src.execution',
    'src.gateway',
    'src.monitor',
    'src.portfolio_manager',
    'src.risk_sentinel',
    'src.strategies'
]

failed = []

for module in modules:
    try:
        exec(f'import {module}')
        print(f"{module}: OK")
    except Exception as e:
        print(f"{module}: FAILED - {e}")
        failed.append(module)

print(f'\nSummary: {len(modules)-len(failed)}/{len(modules)} modules imported successfully')
if failed:
    print(f'Failed modules: {failed}')
else:
    print('All modules imported successfully!')