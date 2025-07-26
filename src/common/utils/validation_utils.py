"""Validation Utility Functions

数据验证相关的工具函数。
"""

import re
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
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


# =============================================================================
# 数据质量验证功能 (整合重复的验证逻辑)
# =============================================================================

def validate_dataframe_structure(df: pd.DataFrame, required_columns: List[str] = None,
                                column_types: Dict[str, type] = None) -> Tuple[bool, List[str]]:
    """验证DataFrame结构
    
    Args:
        df: 要验证的DataFrame
        required_columns: 必需的列名列表
        column_types: 列类型映射
        
    Returns:
        Tuple[bool, List[str]]: (验证结果, 错误信息列表)
    """
    try:
        errors = []
        
        if not isinstance(df, pd.DataFrame):
            return False, ["Input is not a pandas DataFrame"]
        
        if df.empty:
            return False, ["DataFrame is empty"]
        
        # 检查必需列
        if required_columns:
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                errors.append(f"Missing required columns: {missing_columns}")
        
        # 检查列类型
        if column_types:
            for col, expected_type in column_types.items():
                if col in df.columns:
                    if expected_type == 'numeric':
                        if not pd.api.types.is_numeric_dtype(df[col]):
                            errors.append(f"Column '{col}' should be numeric")
                    elif expected_type == 'datetime':
                        if not pd.api.types.is_datetime64_any_dtype(df[col]):
                            errors.append(f"Column '{col}' should be datetime")
                    elif expected_type == 'string':
                        if not pd.api.types.is_string_dtype(df[col]) and not pd.api.types.is_object_dtype(df[col]):
                            errors.append(f"Column '{col}' should be string")
        
        return len(errors) == 0, errors
    except Exception as e:
        logger.error(f"Error validating DataFrame structure: {e}")
        return False, [str(e)]


def check_data_completeness(df: pd.DataFrame, missing_threshold: float = 0.1) -> Tuple[bool, Dict[str, Any]]:
    """检查数据完整性
    
    Args:
        df: 要检查的DataFrame
        missing_threshold: 缺失值阈值 (0-1)
        
    Returns:
        Tuple[bool, Dict]: (是否通过检查, 检查详情)
    """
    try:
        total_cells = len(df) * len(df.columns)
        missing_cells = df.isnull().sum().sum()
        missing_rate = missing_cells / total_cells if total_cells > 0 else 0
        
        # 按列统计缺失率
        column_missing = df.isnull().sum() / len(df)
        high_missing_columns = column_missing[column_missing > missing_threshold].to_dict()
        
        details = {
            'total_cells': total_cells,
            'missing_cells': missing_cells,
            'missing_rate': missing_rate,
            'missing_threshold': missing_threshold,
            'high_missing_columns': high_missing_columns,
            'column_missing_rates': column_missing.to_dict()
        }
        
        passed = missing_rate <= missing_threshold
        return passed, details
    except Exception as e:
        logger.error(f"Error checking data completeness: {e}")
        return False, {'error': str(e)}


def check_data_duplicates(df: pd.DataFrame, subset: List[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """检查重复数据
    
    Args:
        df: 要检查的DataFrame
        subset: 检查重复的列子集
        
    Returns:
        Tuple[bool, Dict]: (是否无重复, 检查详情)
    """
    try:
        if subset:
            duplicates = df.duplicated(subset=subset)
        else:
            duplicates = df.duplicated()
        
        duplicate_count = duplicates.sum()
        duplicate_indices = df[duplicates].index.tolist()
        duplicate_rate = duplicate_count / len(df) if len(df) > 0 else 0
        
        details = {
            'total_records': len(df),
            'duplicate_count': duplicate_count,
            'duplicate_rate': duplicate_rate,
            'duplicate_indices': duplicate_indices[:100],  # 限制返回数量
            'subset_columns': subset
        }
        
        passed = duplicate_count == 0
        return passed, details
    except Exception as e:
        logger.error(f"Error checking data duplicates: {e}")
        return False, {'error': str(e)}


def check_data_consistency(df: pd.DataFrame, consistency_rules: Dict[str, Any] = None) -> Tuple[bool, Dict[str, Any]]:
    """检查数据一致性
    
    Args:
        df: 要检查的DataFrame
        consistency_rules: 一致性规则字典
        
    Returns:
        Tuple[bool, Dict]: (是否一致, 检查详情)
    """
    try:
        issues = []
        details = {'consistency_issues': []}
        
        # 检查数值列的异常值
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        for col in numeric_columns:
            if col in df.columns:
                # 使用IQR方法检测异常值
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                if len(outliers) > 0:
                    issues.append(f"Column '{col}' has {len(outliers)} outliers")
                    details['consistency_issues'].append({
                        'column': col,
                        'issue_type': 'outliers',
                        'count': len(outliers),
                        'bounds': {'lower': float(lower_bound), 'upper': float(upper_bound)}
                    })
        
        # 检查自定义一致性规则
        if consistency_rules:
            for rule_name, rule_config in consistency_rules.items():
                try:
                    if rule_config['type'] == 'range_check':
                        col = rule_config['column']
                        min_val = rule_config.get('min')
                        max_val = rule_config.get('max')
                        
                        if col in df.columns:
                            violations = 0
                            if min_val is not None:
                                violations += (df[col] < min_val).sum()
                            if max_val is not None:
                                violations += (df[col] > max_val).sum()
                            
                            if violations > 0:
                                issues.append(f"Rule '{rule_name}': {violations} violations")
                                details['consistency_issues'].append({
                                    'rule': rule_name,
                                    'column': col,
                                    'violations': violations
                                })
                except Exception as rule_error:
                    logger.warning(f"Error applying consistency rule '{rule_name}': {rule_error}")
        
        details['total_issues'] = len(issues)
        details['issue_summary'] = issues
        
        passed = len(issues) == 0
        return passed, details
    except Exception as e:
        logger.error(f"Error checking data consistency: {e}")
        return False, {'error': str(e)}


def validate_market_data(df: pd.DataFrame) -> Tuple[bool, Dict[str, Any]]:
    """验证市场数据
    
    Args:
        df: 市场数据DataFrame
        
    Returns:
        Tuple[bool, Dict]: (验证结果, 验证详情)
    """
    try:
        errors = []
        warnings = []
        
        # 检查必需列
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # 检查价格逻辑
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            # 检查 high >= max(open, close) 和 low <= min(open, close)
            price_logic_errors = 0
            
            # High应该是最高价
            high_errors = ((df['high'] < df['open']) | 
                          (df['high'] < df['close']) | 
                          (df['high'] < df['low'])).sum()
            
            # Low应该是最低价
            low_errors = ((df['low'] > df['open']) | 
                         (df['low'] > df['close']) | 
                         (df['low'] > df['high'])).sum()
            
            price_logic_errors = high_errors + low_errors
            
            if price_logic_errors > 0:
                errors.append(f"Price logic violations: {price_logic_errors} records")
        
        # 检查负值
        price_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in price_columns:
            if col in df.columns:
                negative_count = (df[col] < 0).sum()
                if negative_count > 0:
                    errors.append(f"Negative values in '{col}': {negative_count} records")
        
        # 检查时间序列连续性
        if 'timestamp' in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                time_gaps = df['timestamp'].diff().dropna()
                if len(time_gaps) > 0:
                    median_interval = time_gaps.median()
                    large_gaps = (time_gaps > median_interval * 2).sum()
                    if large_gaps > 0:
                        warnings.append(f"Time series has {large_gaps} large gaps")
        
        details = {
            'total_records': len(df),
            'errors': errors,
            'warnings': warnings,
            'error_count': len(errors),
            'warning_count': len(warnings)
        }
        
        passed = len(errors) == 0
        return passed, details
    except Exception as e:
        logger.error(f"Error validating market data: {e}")
        return False, {'error': str(e)}


def comprehensive_data_validation(df: pd.DataFrame, 
                                validation_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """综合数据验证
    
    Args:
        df: 要验证的DataFrame
        validation_config: 验证配置
        
    Returns:
        Dict: 综合验证结果
    """
    try:
        config = validation_config or {}
        results = {
            'overall_passed': True,
            'validation_timestamp': datetime.now().isoformat(),
            'data_shape': df.shape,
            'checks': {}
        }
        
        # 结构验证
        required_columns = config.get('required_columns', [])
        column_types = config.get('column_types', {})
        structure_passed, structure_errors = validate_dataframe_structure(
            df, required_columns, column_types
        )
        results['checks']['structure'] = {
            'passed': structure_passed,
            'errors': structure_errors
        }
        if not structure_passed:
            results['overall_passed'] = False
        
        # 完整性检查
        missing_threshold = config.get('missing_threshold', 0.1)
        completeness_passed, completeness_details = check_data_completeness(
            df, missing_threshold
        )
        results['checks']['completeness'] = {
            'passed': completeness_passed,
            'details': completeness_details
        }
        if not completeness_passed:
            results['overall_passed'] = False
        
        # 重复检查
        duplicate_subset = config.get('duplicate_subset')
        duplicates_passed, duplicates_details = check_data_duplicates(
            df, duplicate_subset
        )
        results['checks']['duplicates'] = {
            'passed': duplicates_passed,
            'details': duplicates_details
        }
        if not duplicates_passed:
            results['overall_passed'] = False
        
        # 一致性检查
        consistency_rules = config.get('consistency_rules')
        consistency_passed, consistency_details = check_data_consistency(
            df, consistency_rules
        )
        results['checks']['consistency'] = {
            'passed': consistency_passed,
            'details': consistency_details
        }
        if not consistency_passed:
            results['overall_passed'] = False
        
        # 市场数据特定验证
        if config.get('market_data_validation', False):
            market_passed, market_details = validate_market_data(df)
            results['checks']['market_data'] = {
                'passed': market_passed,
                'details': market_details
            }
            if not market_passed:
                results['overall_passed'] = False
        
        return results
    except Exception as e:
        logger.error(f"Error in comprehensive data validation: {e}")
        return {
            'overall_passed': False,
            'error': str(e),
            'validation_timestamp': datetime.now().isoformat()
        }


def validate_order(order: Dict[str, Any]) -> bool:
    """验证订单（兼容性函数）
    
    Args:
        order: 订单数据
        
    Returns:
        bool: 验证结果
    """
    try:
        is_valid, errors = validate_order_data(order)
        if not is_valid:
            logger.warning(f"Order validation failed: {errors}")
        return is_valid
    except Exception as e:
        logger.error(f"Error validating order: {e}")
        return False


def validate_config(config: Dict[str, Any]) -> bool:
    """验证配置（兼容性函数）
    
    Args:
        config: 配置数据
        
    Returns:
        bool: 验证结果
    """
    try:
        is_valid, errors = validate_config_file(config)
        if not is_valid:
            logger.warning(f"Config validation failed: {errors}")
        return is_valid
    except Exception as e:
        logger.error(f"Error validating config: {e}")
        return False