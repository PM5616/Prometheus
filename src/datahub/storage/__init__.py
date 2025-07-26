"""Storage Module

数据存储模块，提供统一的数据存储接口。

支持的存储类型：
- InfluxDB: 时序数据存储
- Redis: 缓存和实时数据
- PostgreSQL: 关系型数据存储
- MongoDB: 文档型数据存储
- File: 文件存储
"""

from .base import BaseStorage, StorageType
from .influxdb_storage import InfluxDBStorage
from .redis_storage import RedisStorage
from .postgresql_storage import PostgreSQLStorage
from .mongodb_storage import MongoDBStorage

__all__ = [
    'BaseStorage',
    'StorageType',
    'InfluxDBStorage',
    'RedisStorage',
    'PostgreSQLStorage',
    'MongoDBStorage'
]