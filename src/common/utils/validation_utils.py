"""Validation Utility Functions

数据验证相关的工具函数。
"""

import re
from typing import Any, Dict, List, Optional, Union
from decimal import Decimal
from datetime import datetime
from loguru import logger


def validate_email(email: str) -> bool:
    """验证邮箱格式
    
    Args:
        email: 邮箱地址
        
    Returns:
        bool: 验证结果
    """
    try:
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return bool(re.match(pattern, email))
    except Exception as e:
        logger.error(f"Error validating email: {e}")
        return False


def validate_phone(phone: str, country_code: str = 'CN') -> bool:
    """验证手机号格式
    
    Args:
        phone: 手机号
        country_code: 国家代码
        
    Returns:
        bool: 验证结果
    """
    try:
        if country_code.upper() == 'CN':
            # 中国手机号验证
            pattern = r'^1[3-9]\d{9}$'
        elif country_code.upper() == 'US':
            # 美国手机号验证
            pattern = r'^\+?1?[2-9]\d{2}[2-9]\d{2}\d{4}$'
        else:
            # 通用格式验证
            pattern = r'^\+?[1-9]\d{1,14}$'
        
        return bool(re.match(pattern, phone.replace('-', '').replace(' ', '')))
    except Exception as e:
        logger.error(f"Error validating phone: {e}")
        return False


def validate_symbol(symbol: str) -> bool:
    """验证交易对格式
    
    Args:
        symbol: 交易对符号
        
    Returns:
        bool: 验证结果
    """
    try:
        # 币安交易对格式：BTCUSDT, ETHBTC等
        pattern = r'^[A-Z]{2,10}[A-Z]{3,6}$'
        return bool(re.match(pattern, symbol.upper()))
    except Exception as e:
        logger.error(f"Error validating symbol: {e}")
        return False


def validate_price(price: Union[str, float, Decimal]) -> bool:
    """验证价格格式
    
    Args:
        price: 价格
        
    Returns:
        bool: 验证结果
    """
    try:
        price_decimal = Decimal(str(price))
        return price_decimal > 0
    except Exception as e:
        logger.error(f"Error validating price: {e}")
        return False


def validate_quantity(quantity: Union[str, float, Decimal]) -> bool:
    """验证数量格式
    
    Args:
        quantity: 数量
        
    Returns:
        bool: 验证结果
    """
    try:
        quantity_decimal = Decimal(str(quantity))
        return quantity_decimal > 0
    except Exception as e:
        logger.error(f"Error validating quantity: {e}")
        return False


def validate_order_side(side: str) -> bool:
    """验证订单方向
    
    Args:
        side: 订单方向
        
    Returns:
        bool: 验证结果
    """
    try:
        valid_sides = ['BUY', 'SELL']
        return side.upper() in valid_sides
    except Exception as e:
        logger.error(f"Error validating order side: {e}")
        return False


def validate_order_type(order_type: str) -> bool:
    """验证订单类型
    
    Args:
        order_type: 订单类型
        
    Returns:
        bool: 验证结果
    """
    try:
        valid_types = [
            'MARKET', 'LIMIT', 'STOP_LOSS', 'STOP_LOSS_LIMIT',
            'TAKE_PROFIT', 'TAKE_PROFIT_LIMIT', 'LIMIT_MAKER'
        ]
        return order_type.upper() in valid_types
    except Exception as e:
        logger.error(f"Error validating order type: {e}")
        return False


def validate_time_in_force(tif: str) -> bool:
    """验证订单有效期类型
    
    Args:
        tif: 订单有效期类型
        
    Returns:
        bool: 验证结果
    """
    try:
        valid_tifs = ['GTC', 'IOC', 'FOK']
        return tif.upper() in valid_tifs
    except Exception as e:
        logger.error(f"Error validating time in force: {e}")
        return False


def validate_api_key(api_key: str) -> bool:
    """验证API密钥格式
    
    Args:
        api_key: API密钥
        
    Returns:
        bool: 验证结果
    """
    try:
        # 币安API密钥通常是64位十六进制字符串
        if len(api_key) != 64:
            return False
        
        # 检查是否只包含十六进制字符
        pattern = r'^[a-fA-F0-9]{64}$'
        return bool(re.match(pattern, api_key))
    except Exception as e:
        logger.error(f"Error validating API key: {e}")
        return False


def validate_timestamp(timestamp: Union[int, str, datetime]) -> bool:
    """验证时间戳格式
    
    Args:
        timestamp: 时间戳
        
    Returns:
        bool: 验证结果
    """
    try:
        if isinstance(timestamp, datetime):
            return True
        elif isinstance(timestamp, (int, str)):
            timestamp_int = int(timestamp)
            # 检查是否为合理的时间戳范围（2000年到2100年）
            min_timestamp = 946684800000  # 2000-01-01 00:00:00 UTC (毫秒)
            max_timestamp = 4102444800000  # 2100-01-01 00:00:00 UTC (毫秒)
            return min_timestamp <= timestamp_int <= max_timestamp
        else:
            return False
    except Exception as e:
        logger.error(f"Error validating timestamp: {e}")
        return False


def validate_percentage(percentage: Union[str, float, Decimal]) -> bool:
    """验证百分比格式
    
    Args:
        percentage: 百分比
        
    Returns:
        bool: 验证结果
    """
    try:
        percentage_decimal = Decimal(str(percentage))
        return 0 <= percentage_decimal <= 100
    except Exception as e:
        logger.error(f"Error validating percentage: {e}")
        return False


def validate_ip_address(ip: str) -> bool:
    """验证IP地址格式
    
    Args:
        ip: IP地址
        
    Returns:
        bool: 验证结果
    """
    try:
        import ipaddress
        ipaddress.ip_address(ip)
        return True
    except Exception as e:
        logger.error(f"Error validating IP address: {e}")
        return False


def validate_url(url: str) -> bool:
    """验证URL格式
    
    Args:
        url: URL地址
        
    Returns:
        bool: 验证结果
    """
    try:
        pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:[\w.])*)?)?$'
        return bool(re.match(pattern, url))
    except Exception as e:
        logger.error(f"Error validating URL: {e}")
        return False


def validate_json_string(json_str: str) -> bool:
    """验证JSON字符串格式
    
    Args:
        json_str: JSON字符串
        
    Returns:
        bool: 验证结果
    """
    try:
        import json
        json.loads(json_str)
        return True
    except Exception as e:
        logger.error(f"Error validating JSON string: {e}")
        return False


def validate_required_fields(data: Dict[str, Any], required_fields: List[str]) -> tuple:
    """验证必填字段
    
    Args:
        data: 数据字典
        required_fields: 必填字段列表
        
    Returns:
        tuple: (验证结果, 缺失字段列表)
    """
    try:
        missing_fields = []
        
        for field in required_fields:
            if field not in data or data[field] is None or data[field] == "":
                missing_fields.append(field)
        
        return len(missing_fields) == 0, missing_fields
    except Exception as e:
        logger.error(f"Error validating required fields: {e}")
        return False, required_fields


def validate_field_types(data: Dict[str, Any], type_mapping: Dict[str, type]) -> tuple:
    """验证字段类型
    
    Args:
        data: 数据字典
        type_mapping: 类型映射字典
        
    Returns:
        tuple: (验证结果, 类型错误字段列表)
    """
    try:
        invalid_fields = []
        
        for field, expected_type in type_mapping.items():
            if field in data:
                if not isinstance(data[field], expected_type):
                    # 尝试类型转换
                    try:
                        if expected_type == Decimal:
                            Decimal(str(data[field]))
                        else:
                            expected_type(data[field])
                    except (ValueError, TypeError):
                        invalid_fields.append(field)
        
        return len(invalid_fields) == 0, invalid_fields
    except Exception as e:
        logger.error(f"Error validating field types: {e}")
        return False, list(type_mapping.keys())


def validate_field_ranges(data: Dict[str, Any], 
                         range_mapping: Dict[str, Dict[str, Union[int, float, Decimal]]]) -> tuple:
    """验证字段范围
    
    Args:
        data: 数据字典
        range_mapping: 范围映射字典，格式：{field: {'min': min_val, 'max': max_val}}
        
    Returns:
        tuple: (验证结果, 范围错误字段列表)
    """
    try:
        invalid_fields = []
        
        for field, range_config in range_mapping.items():
            if field in data:
                value = data[field]
                min_val = range_config.get('min')
                max_val = range_config.get('max')
                
                try:
                    value_decimal = Decimal(str(value))
                    
                    if min_val is not None and value_decimal < Decimal(str(min_val)):
                        invalid_fields.append(field)
                    elif max_val is not None and value_decimal > Decimal(str(max_val)):
                        invalid_fields.append(field)
                except (ValueError, TypeError):
                    invalid_fields.append(field)
        
        return len(invalid_fields) == 0, invalid_fields
    except Exception as e:
        logger.error(f"Error validating field ranges: {e}")
        return False, list(range_mapping.keys())


def validate_order_data(order_data: Dict[str, Any]) -> tuple:
    """验证订单数据
    
    Args:
        order_data: 订单数据
        
    Returns:
        tuple: (验证结果, 错误信息列表)
    """
    try:
        errors = []
        
        # 验证必填字段
        required_fields = ['symbol', 'side', 'type', 'quantity']
        is_valid, missing_fields = validate_required_fields(order_data, required_fields)
        if not is_valid:
            errors.extend([f"Missing required field: {field}" for field in missing_fields])
        
        # 验证字段格式
        if 'symbol' in order_data and not validate_symbol(order_data['symbol']):
            errors.append("Invalid symbol format")
        
        if 'side' in order_data and not validate_order_side(order_data['side']):
            errors.append("Invalid order side")
        
        if 'type' in order_data and not validate_order_type(order_data['type']):
            errors.append("Invalid order type")
        
        if 'quantity' in order_data and not validate_quantity(order_data['quantity']):
            errors.append("Invalid quantity")
        
        if 'price' in order_data and not validate_price(order_data['price']):
            errors.append("Invalid price")
        
        if 'timeInForce' in order_data and not validate_time_in_force(order_data['timeInForce']):
            errors.append("Invalid time in force")
        
        return len(errors) == 0, errors
    except Exception as e:
        logger.error(f"Error validating order data: {e}")
        return False, [str(e)]


def validate_strategy_config(config: Dict[str, Any]) -> tuple:
    """验证策略配置
    
    Args:
        config: 策略配置
        
    Returns:
        tuple: (验证结果, 错误信息列表)
    """
    try:
        errors = []
        
        # 验证必填字段
        required_fields = ['name', 'type', 'parameters']
        is_valid, missing_fields = validate_required_fields(config, required_fields)
        if not is_valid:
            errors.extend([f"Missing required field: {field}" for field in missing_fields])
        
        # 验证策略名称
        if 'name' in config:
            name = config['name']
            if not isinstance(name, str) or len(name.strip()) == 0:
                errors.append("Strategy name must be a non-empty string")
        
        # 验证策略类型
        if 'type' in config:
            strategy_type = config['type']
            valid_types = ['trend_following', 'mean_reversion', 'arbitrage', 'market_making']
            if strategy_type not in valid_types:
                errors.append(f"Invalid strategy type. Must be one of: {valid_types}")
        
        # 验证参数
        if 'parameters' in config:
            parameters = config['parameters']
            if not isinstance(parameters, dict):
                errors.append("Strategy parameters must be a dictionary")
        
        return len(errors) == 0, errors
    except Exception as e:
        logger.error(f"Error validating strategy config: {e}")
        return False, [str(e)]


def sanitize_input(input_str: str, max_length: int = 1000) -> str:
    """清理输入字符串
    
    Args:
        input_str: 输入字符串
        max_length: 最大长度
        
    Returns:
        str: 清理后的字符串
    """
    try:
        if not isinstance(input_str, str):
            input_str = str(input_str)
        
        # 移除危险字符
        dangerous_chars = ['<', '>', '"', "'", '&', ';', '|', '`']
        for char in dangerous_chars:
            input_str = input_str.replace(char, '')
        
        # 限制长度
        if len(input_str) > max_length:
            input_str = input_str[:max_length]
        
        # 移除首尾空白字符
        return input_str.strip()
    except Exception as e:
        logger.error(f"Error sanitizing input: {e}")
        return ""


def validate_config_file(config_data: Dict[str, Any]) -> tuple:
    """验证配置文件
    
    Args:
        config_data: 配置数据
        
    Returns:
        tuple: (验证结果, 错误信息列表)
    """
    try:
        errors = []
        
        # 验证顶级必填字段
        required_sections = ['system', 'database', 'binance', 'trading']
        is_valid, missing_sections = validate_required_fields(config_data, required_sections)
        if not is_valid:
            errors.extend([f"Missing required section: {section}" for section in missing_sections])
        
        # 验证系统配置
        if 'system' in config_data:
            system_config = config_data['system']
            system_required = ['name', 'version', 'environment']
            is_valid, missing_fields = validate_required_fields(system_config, system_required)
            if not is_valid:
                errors.extend([f"Missing system field: {field}" for field in missing_fields])
        
        # 验证数据库配置
        if 'database' in config_data:
            db_config = config_data['database']
            db_required = ['host', 'port', 'name', 'user']
            is_valid, missing_fields = validate_required_fields(db_config, db_required)
            if not is_valid:
                errors.extend([f"Missing database field: {field}" for field in missing_fields])
        
        return len(errors) == 0, errors
    except Exception as e:
        logger.error(f"Error validating config file: {e}")
        return False, [str(e)]