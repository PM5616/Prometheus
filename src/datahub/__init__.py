"""DataHub Module

数据中心模块，负责数据的接收、处理、存储和分发。

主要功能：
- 数据提供商接口管理
- 实时数据流处理
- 历史数据存储和查询
- 数据质量监控
- 数据缓存管理
"""

from .core.data_manager import DataManager
from .providers.base import BaseDataProvider
from .storage.base import BaseStorage
from .processors.base import BaseProcessor

__all__ = [
    'DataManager',
    'BaseDataProvider',
    'BaseStorage', 
    'BaseProcessor'
]

__version__ = '1.0.0'
__author__ = 'Prometheus Team'
__description__ = 'DataHub - 数据中心模块'