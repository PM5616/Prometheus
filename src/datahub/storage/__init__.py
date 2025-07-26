"""DataHub Storage Module

数据存储模块，提供多种数据存储解决方案。

支持的存储类型：
- InfluxDB: 时序数据库，用于存储行情、K线等时序数据
- Redis: 内存数据库，用于缓存和实时数据
- PostgreSQL: 关系型数据库，用于存储配置、策略等结构化数据
- File: 文件存储，用于数据备份和离线分析

主要组件：
- BaseStorage: 存储接口基类
- InfluxDBStorage: InfluxDB存储实现
- RedisStorage: Redis存储实现
- PostgreSQLStorage: PostgreSQL存储实现
- FileStorage: 文件存储实现
"""

__version__ = "1.0.0"
__author__ = "Prometheus Team"

# 导入存储组件
from .base import BaseStorage, StorageType
from .influxdb_storage import InfluxDBStorage
from .redis_storage import RedisStorage
from .postgresql_storage import PostgreSQLStorage
from .file_storage import FileStorage

# 导出的公共接口
__all__ = [
    'BaseStorage',
    'StorageType',
    'InfluxDBStorage',
    'RedisStorage', 
    'PostgreSQLStorage',
    'FileStorage'
]