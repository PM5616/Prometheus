"""Data Utility Functions

数据转换和处理相关的工具函数。
"""

import json
import pickle
import base64
import hashlib
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal
import pandas as pd
import numpy as np
from loguru import logger


def serialize_data(data: Any, method: str = 'json') -> str:
    """序列化数据
    
    Args:
        data: 要序列化的数据
        method: 序列化方法 ('json', 'pickle')
        
    Returns:
        str: 序列化后的字符串
    """
    try:
        if method.lower() == 'json':
            return json.dumps(data, default=_json_serializer, ensure_ascii=False)
        elif method.lower() == 'pickle':
            pickled_data = pickle.dumps(data)
            return base64.b64encode(pickled_data).decode('utf-8')
        else:
            raise ValueError(f"Unsupported serialization method: {method}")
    except Exception as e:
        logger.error(f"Error serializing data: {e}")
        return ""


def deserialize_data(data_str: str, method: str = 'json') -> Any:
    """反序列化数据
    
    Args:
        data_str: 序列化的字符串
        method: 反序列化方法 ('json', 'pickle')
        
    Returns:
        Any: 反序列化后的数据
    """
    try:
        if method.lower() == 'json':
            return json.loads(data_str)
        elif method.lower() == 'pickle':
            pickled_data = base64.b64decode(data_str.encode('utf-8'))
            return pickle.loads(pickled_data)
        else:
            raise ValueError(f"Unsupported deserialization method: {method}")
    except Exception as e:
        logger.error(f"Error deserializing data: {e}")
        return None


def _json_serializer(obj: Any) -> Any:
    """JSON序列化器，处理特殊类型
    
    Args:
        obj: 要序列化的对象
        
    Returns:
        Any: 可序列化的对象
    """
    if isinstance(obj, Decimal):
        return float(obj)
    elif isinstance(obj, (pd.Timestamp, pd.DatetimeIndex)):
        return obj.isoformat()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    elif hasattr(obj, 'to_dict'):
        return obj.to_dict()
    elif hasattr(obj, '__dict__'):
        return obj.__dict__
    else:
        return str(obj)


def flatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """扁平化嵌套字典
    
    Args:
        data: 嵌套字典
        separator: 键分隔符
        
    Returns:
        Dict[str, Any]: 扁平化后的字典
    """
    def _flatten(obj: Any, parent_key: str = '') -> Dict[str, Any]:
        items = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{parent_key}{separator}{key}" if parent_key else key
                items.extend(_flatten(value, new_key).items())
        else:
            return {parent_key: obj}
        
        return dict(items)
    
    try:
        return _flatten(data)
    except Exception as e:
        logger.error(f"Error flattening dictionary: {e}")
        return data


def unflatten_dict(data: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """反扁平化字典
    
    Args:
        data: 扁平化的字典
        separator: 键分隔符
        
    Returns:
        Dict[str, Any]: 嵌套字典
    """
    try:
        result = {}
        
        for key, value in data.items():
            keys = key.split(separator)
            current = result
            
            for k in keys[:-1]:
                if k not in current:
                    current[k] = {}
                current = current[k]
            
            current[keys[-1]] = value
        
        return result
    except Exception as e:
        logger.error(f"Error unflattening dictionary: {e}")
        return data


def merge_dicts(*dicts: Dict[str, Any], deep: bool = True) -> Dict[str, Any]:
    """合并多个字典
    
    Args:
        *dicts: 要合并的字典
        deep: 是否深度合并
        
    Returns:
        Dict[str, Any]: 合并后的字典
    """
    try:
        result = {}
        
        for d in dicts:
            if not isinstance(d, dict):
                continue
            
            if deep:
                result = _deep_merge(result, d)
            else:
                result.update(d)
        
        return result
    except Exception as e:
        logger.error(f"Error merging dictionaries: {e}")
        return {}


def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """深度合并两个字典
    
    Args:
        dict1: 第一个字典
        dict2: 第二个字典
        
    Returns:
        Dict[str, Any]: 合并后的字典
    """
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result


def filter_dict(data: Dict[str, Any], 
               include_keys: Optional[List[str]] = None,
               exclude_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """过滤字典键值
    
    Args:
        data: 原始字典
        include_keys: 包含的键列表
        exclude_keys: 排除的键列表
        
    Returns:
        Dict[str, Any]: 过滤后的字典
    """
    try:
        result = data.copy()
        
        if include_keys:
            result = {k: v for k, v in result.items() if k in include_keys}
        
        if exclude_keys:
            result = {k: v for k, v in result.items() if k not in exclude_keys}
        
        return result
    except Exception as e:
        logger.error(f"Error filtering dictionary: {e}")
        return data


def convert_types(data: Any, type_mapping: Dict[str, type]) -> Any:
    """转换数据类型
    
    Args:
        data: 原始数据
        type_mapping: 类型映射字典
        
    Returns:
        Any: 转换后的数据
    """
    try:
        if isinstance(data, dict):
            result = {}
            for key, value in data.items():
                if key in type_mapping:
                    target_type = type_mapping[key]
                    try:
                        if target_type == Decimal:
                            result[key] = Decimal(str(value))
                        else:
                            result[key] = target_type(value)
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Failed to convert {key} to {target_type}: {e}")
                        result[key] = value
                else:
                    result[key] = convert_types(value, type_mapping)
            return result
        elif isinstance(data, list):
            return [convert_types(item, type_mapping) for item in data]
        else:
            return data
    except Exception as e:
        logger.error(f"Error converting types: {e}")
        return data


def validate_data_structure(data: Any, schema: Dict[str, Any]) -> bool:
    """验证数据结构
    
    Args:
        data: 要验证的数据
        schema: 数据结构模式
        
    Returns:
        bool: 验证结果
    """
    try:
        return _validate_recursive(data, schema)
    except Exception as e:
        logger.error(f"Error validating data structure: {e}")
        return False


def _validate_recursive(data: Any, schema: Any) -> bool:
    """递归验证数据结构
    
    Args:
        data: 数据
        schema: 模式
        
    Returns:
        bool: 验证结果
    """
    if isinstance(schema, type):
        return isinstance(data, schema)
    elif isinstance(schema, dict):
        if not isinstance(data, dict):
            return False
        for key, value_schema in schema.items():
            if key not in data:
                return False
            if not _validate_recursive(data[key], value_schema):
                return False
        return True
    elif isinstance(schema, list):
        if not isinstance(data, list):
            return False
        if len(schema) == 1:
            # 列表中所有元素都应该符合同一个模式
            item_schema = schema[0]
            return all(_validate_recursive(item, item_schema) for item in data)
        else:
            # 列表长度和每个位置的类型都要匹配
            if len(data) != len(schema):
                return False
            return all(_validate_recursive(data[i], schema[i]) for i in range(len(data)))
    else:
        return data == schema


def calculate_hash(data: Any, algorithm: str = 'md5') -> str:
    """计算数据哈希值
    
    Args:
        data: 要计算哈希的数据
        algorithm: 哈希算法 ('md5', 'sha1', 'sha256')
        
    Returns:
        str: 哈希值
    """
    try:
        # 序列化数据
        data_str = serialize_data(data, 'json')
        data_bytes = data_str.encode('utf-8')
        
        # 计算哈希
        if algorithm.lower() == 'md5':
            hash_obj = hashlib.md5(data_bytes)
        elif algorithm.lower() == 'sha1':
            hash_obj = hashlib.sha1(data_bytes)
        elif algorithm.lower() == 'sha256':
            hash_obj = hashlib.sha256(data_bytes)
        else:
            raise ValueError(f"Unsupported hash algorithm: {algorithm}")
        
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash: {e}")
        return ""


def chunk_list(data: List[Any], chunk_size: int) -> List[List[Any]]:
    """将列表分块
    
    Args:
        data: 原始列表
        chunk_size: 块大小
        
    Returns:
        List[List[Any]]: 分块后的列表
    """
    try:
        if chunk_size <= 0:
            return [data]
        
        return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
    except Exception as e:
        logger.error(f"Error chunking list: {e}")
        return [data]


def remove_duplicates(data: List[Any], key_func: Optional[callable] = None) -> List[Any]:
    """去除列表中的重复项
    
    Args:
        data: 原始列表
        key_func: 用于确定唯一性的键函数
        
    Returns:
        List[Any]: 去重后的列表
    """
    try:
        if key_func is None:
            # 简单去重
            seen = set()
            result = []
            for item in data:
                if item not in seen:
                    seen.add(item)
                    result.append(item)
            return result
        else:
            # 基于键函数去重
            seen = set()
            result = []
            for item in data:
                key = key_func(item)
                if key not in seen:
                    seen.add(key)
                    result.append(item)
            return result
    except Exception as e:
        logger.error(f"Error removing duplicates: {e}")
        return data


def safe_get(data: Dict[str, Any], 
            path: str, 
            default: Any = None, 
            separator: str = '.') -> Any:
    """安全获取嵌套字典的值
    
    Args:
        data: 字典数据
        path: 路径字符串
        default: 默认值
        separator: 路径分隔符
        
    Returns:
        Any: 获取的值或默认值
    """
    try:
        keys = path.split(separator)
        current = data
        
        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default
        
        return current
    except Exception as e:
        logger.error(f"Error getting value from path {path}: {e}")
        return default


def safe_set(data: Dict[str, Any], 
            path: str, 
            value: Any, 
            separator: str = '.') -> bool:
    """安全设置嵌套字典的值
    
    Args:
        data: 字典数据
        path: 路径字符串
        value: 要设置的值
        separator: 路径分隔符
        
    Returns:
        bool: 设置是否成功
    """
    try:
        keys = path.split(separator)
        current = data
        
        # 创建嵌套结构
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # 如果路径上的值不是字典，无法继续
                return False
            current = current[key]
        
        # 设置最终值
        current[keys[-1]] = value
        return True
    except Exception as e:
        logger.error(f"Error setting value at path {path}: {e}")
        return False


def convert_to_dataframe(data: Union[List[Dict], Dict[str, List]], 
                        index_col: Optional[str] = None) -> pd.DataFrame:
    """转换数据为DataFrame
    
    Args:
        data: 数据（字典列表或列表字典）
        index_col: 索引列名
        
    Returns:
        pd.DataFrame: 转换后的DataFrame
    """
    try:
        df = pd.DataFrame(data)
        
        if index_col and index_col in df.columns:
            df.set_index(index_col, inplace=True)
        
        return df
    except Exception as e:
        logger.error(f"Error converting to DataFrame: {e}")
        return pd.DataFrame()


def normalize_data(data: Union[List[float], np.ndarray], 
                  method: str = 'minmax') -> np.ndarray:
    """数据标准化
    
    Args:
        data: 原始数据
        method: 标准化方法 ('minmax', 'zscore')
        
    Returns:
        np.ndarray: 标准化后的数据
    """
    try:
        data_array = np.array(data)
        
        if method.lower() == 'minmax':
            # 最小-最大标准化
            min_val = np.min(data_array)
            max_val = np.max(data_array)
            if max_val == min_val:
                return np.zeros_like(data_array)
            return (data_array - min_val) / (max_val - min_val)
        elif method.lower() == 'zscore':
            # Z-score标准化
            mean_val = np.mean(data_array)
            std_val = np.std(data_array)
            if std_val == 0:
                return np.zeros_like(data_array)
            return (data_array - mean_val) / std_val
        else:
            raise ValueError(f"Unsupported normalization method: {method}")
    except Exception as e:
        logger.error(f"Error normalizing data: {e}")
        return np.array(data)


def validate_symbol(symbol: str) -> bool:
    """验证交易对符号格式
    
    Args:
        symbol: 交易对符号
        
    Returns:
        bool: 是否有效
    """
    try:
        if not symbol or not isinstance(symbol, str):
            return False
        
        # 基本格式检查
        symbol = symbol.strip().upper()
        
        # 检查长度
        if len(symbol) < 3 or len(symbol) > 20:
            return False
        
        # 检查字符（只允许字母和数字）
        if not symbol.replace('/', '').replace('-', '').replace('_', '').isalnum():
            return False
        
        return True
    except Exception as e:
        logger.error(f"Error validating symbol {symbol}: {e}")
        return False


def normalize_symbol(symbol: str, format_type: str = 'standard') -> str:
    """标准化交易对符号
    
    Args:
        symbol: 原始交易对符号
        format_type: 格式类型 ('standard', 'binance', 'okx')
        
    Returns:
        str: 标准化后的符号
    """
    try:
        if not symbol:
            return ''
        
        # 清理和大写
        symbol = symbol.strip().upper()
        
        # 移除常见分隔符
        symbol = symbol.replace('/', '').replace('-', '').replace('_', '')
        
        if format_type.lower() == 'standard':
            # 标准格式：BTC/USDT
            if len(symbol) >= 6:
                # 假设最后3-4个字符是计价货币
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    quote = symbol[-4:]
                elif symbol.endswith(('USD', 'BTC', 'ETH')):
                    base = symbol[:-3]
                    quote = symbol[-3:]
                else:
                    # 默认分割
                    mid = len(symbol) // 2
                    base = symbol[:mid]
                    quote = symbol[mid:]
                return f"{base}/{quote}"
        elif format_type.lower() == 'binance':
            # Binance格式：BTCUSDT
            return symbol
        elif format_type.lower() == 'okx':
            # OKX格式：BTC-USDT
            if len(symbol) >= 6:
                if symbol.endswith('USDT'):
                    base = symbol[:-4]
                    quote = symbol[-4:]
                elif symbol.endswith(('USD', 'BTC', 'ETH')):
                    base = symbol[:-3]
                    quote = symbol[-3:]
                else:
                    mid = len(symbol) // 2
                    base = symbol[:mid]
                    quote = symbol[mid:]
                return f"{base}-{quote}"
        
        return symbol
    except Exception as e:
        logger.error(f"Error normalizing symbol {symbol}: {e}")
        return symbol


def parse_timeframe(timeframe: str) -> Dict[str, Union[int, str]]:
    """解析时间周期
    
    Args:
        timeframe: 时间周期字符串 (如 '1m', '5m', '1h', '1d')
        
    Returns:
        Dict[str, Union[int, str]]: 解析结果
    """
    try:
        if not timeframe or not isinstance(timeframe, str):
            return {'value': 1, 'unit': 'm', 'seconds': 60}
        
        timeframe = timeframe.strip().lower()
        
        # 解析数字和单位
        import re
        match = re.match(r'^(\d+)([smhd])$', timeframe)
        
        if not match:
            # 默认返回1分钟
            return {'value': 1, 'unit': 'm', 'seconds': 60}
        
        value = int(match.group(1))
        unit = match.group(2)
        
        # 计算秒数
        unit_seconds = {
            's': 1,
            'm': 60,
            'h': 3600,
            'd': 86400
        }
        
        seconds = value * unit_seconds.get(unit, 60)
        
        return {
            'value': value,
            'unit': unit,
            'seconds': seconds,
            'original': timeframe
        }
    except Exception as e:
        logger.error(f"Error parsing timeframe {timeframe}: {e}")
        return {'value': 1, 'unit': 'm', 'seconds': 60}