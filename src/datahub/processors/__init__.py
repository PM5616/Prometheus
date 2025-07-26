"""DataHub Processors Module

数据处理模块，提供数据清洗、转换、分析和计算功能。

主要功能：
- 数据清洗：去除异常值、填充缺失值、数据标准化
- 数据转换：格式转换、时间序列重采样、数据聚合
- 技术指标：移动平均、RSI、MACD、布林带等
- 数据验证：数据完整性检查、格式验证
- 实时处理：流式数据处理、增量计算

主要组件：
- BaseProcessor: 处理器基类
- DataCleaner: 数据清洗器
- DataTransformer: 数据转换器
- TechnicalIndicators: 技术指标计算器
- DataValidator: 数据验证器
- StreamProcessor: 流式处理器
"""

__version__ = "1.0.0"
__author__ = "Prometheus Team"

# 导入处理器组件
from .base import BaseProcessor, ProcessorType
from .data_cleaner import DataCleaner
from .data_transformer import DataTransformer
from .technical_indicators import TechnicalIndicators
from .data_validator import DataValidator
from .stream_processor import StreamProcessor

# 导出的公共接口
__all__ = [
    'BaseProcessor',
    'ProcessorType',
    'DataCleaner',
    'DataTransformer',
    'TechnicalIndicators',
    'DataValidator',
    'StreamProcessor'
]