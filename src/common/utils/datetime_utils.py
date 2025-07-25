"""DateTime Utility Functions

时间处理相关的工具函数。
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Union
import pytz
from loguru import logger


def get_current_timestamp() -> int:
    """获取当前时间戳（毫秒）
    
    Returns:
        int: 当前时间戳
    """
    return int(datetime.now(timezone.utc).timestamp() * 1000)


def format_timestamp(timestamp: Union[int, float, datetime], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """格式化时间戳
    
    Args:
        timestamp: 时间戳（秒或毫秒）或datetime对象
        format_str: 格式字符串
        
    Returns:
        str: 格式化后的时间字符串
    """
    try:
        if isinstance(timestamp, datetime):
            dt = timestamp
        else:
            # 判断是秒还是毫秒时间戳
            if timestamp > 1e10:  # 毫秒时间戳
                dt = datetime.fromtimestamp(timestamp / 1000, tz=timezone.utc)
            else:  # 秒时间戳
                dt = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        
        return dt.strftime(format_str)
    except Exception as e:
        logger.error(f"Error formatting timestamp {timestamp}: {e}")
        return str(timestamp)


def parse_timestamp(time_str: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> datetime:
    """解析时间字符串
    
    Args:
        time_str: 时间字符串
        format_str: 格式字符串
        
    Returns:
        datetime: datetime对象
        
    Raises:
        ValueError: 解析失败
    """
    try:
        dt = datetime.strptime(time_str, format_str)
        # 如果没有时区信息，默认为UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError as e:
        logger.error(f"Error parsing timestamp {time_str}: {e}")
        raise


def get_trading_day(dt: Optional[datetime] = None, timezone_str: str = "UTC") -> datetime:
    """获取交易日
    
    Args:
        dt: 指定日期，默认为当前时间
        timezone_str: 时区字符串
        
    Returns:
        datetime: 交易日的开始时间
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    # 转换到指定时区
    tz = pytz.timezone(timezone_str)
    local_dt = dt.astimezone(tz)
    
    # 获取当天的开始时间
    trading_day = local_dt.replace(hour=0, minute=0, second=0, microsecond=0)
    
    return trading_day


def is_trading_hours(dt: Optional[datetime] = None, 
                    start_hour: int = 0, end_hour: int = 24,
                    timezone_str: str = "UTC") -> bool:
    """检查是否在交易时间内
    
    Args:
        dt: 指定时间，默认为当前时间
        start_hour: 交易开始小时
        end_hour: 交易结束小时
        timezone_str: 时区字符串
        
    Returns:
        bool: 是否在交易时间内
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    # 转换到指定时区
    tz = pytz.timezone(timezone_str)
    local_dt = dt.astimezone(tz)
    
    current_hour = local_dt.hour
    
    # 处理跨天的情况
    if start_hour <= end_hour:
        return start_hour <= current_hour < end_hour
    else:
        return current_hour >= start_hour or current_hour < end_hour


def get_next_trading_day(dt: Optional[datetime] = None, 
                        timezone_str: str = "UTC",
                        skip_weekends: bool = True) -> datetime:
    """获取下一个交易日
    
    Args:
        dt: 指定日期，默认为当前时间
        timezone_str: 时区字符串
        skip_weekends: 是否跳过周末
        
    Returns:
        datetime: 下一个交易日
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    # 转换到指定时区
    tz = pytz.timezone(timezone_str)
    local_dt = dt.astimezone(tz)
    
    # 获取下一天
    next_day = local_dt + timedelta(days=1)
    next_day = next_day.replace(hour=0, minute=0, second=0, microsecond=0)
    
    # 如果需要跳过周末
    if skip_weekends:
        while next_day.weekday() >= 5:  # 5=Saturday, 6=Sunday
            next_day += timedelta(days=1)
    
    return next_day


def convert_timezone(dt: datetime, from_tz: str, to_tz: str) -> datetime:
    """转换时区
    
    Args:
        dt: datetime对象
        from_tz: 源时区
        to_tz: 目标时区
        
    Returns:
        datetime: 转换后的datetime对象
    """
    try:
        from_timezone = pytz.timezone(from_tz)
        to_timezone = pytz.timezone(to_tz)
        
        # 如果datetime没有时区信息，先设置源时区
        if dt.tzinfo is None:
            dt = from_timezone.localize(dt)
        else:
            dt = dt.astimezone(from_timezone)
        
        # 转换到目标时区
        return dt.astimezone(to_timezone)
    except Exception as e:
        logger.error(f"Error converting timezone from {from_tz} to {to_tz}: {e}")
        return dt


def get_timeframe_delta(timeframe: str) -> timedelta:
    """获取时间周期对应的时间间隔
    
    Args:
        timeframe: 时间周期（如 '1m', '5m', '1h', '1d'）
        
    Returns:
        timedelta: 时间间隔
        
    Raises:
        ValueError: 不支持的时间周期
    """
    timeframe = timeframe.lower()
    
    # 解析数字和单位
    import re
    match = re.match(r'(\d+)([smhd])', timeframe)
    if not match:
        raise ValueError(f"Invalid timeframe format: {timeframe}")
    
    value = int(match.group(1))
    unit = match.group(2)
    
    if unit == 's':
        return timedelta(seconds=value)
    elif unit == 'm':
        return timedelta(minutes=value)
    elif unit == 'h':
        return timedelta(hours=value)
    elif unit == 'd':
        return timedelta(days=value)
    else:
        raise ValueError(f"Unsupported time unit: {unit}")


def round_to_timeframe(dt: datetime, timeframe: str) -> datetime:
    """将时间舍入到指定的时间周期
    
    Args:
        dt: datetime对象
        timeframe: 时间周期
        
    Returns:
        datetime: 舍入后的时间
    """
    try:
        delta = get_timeframe_delta(timeframe)
        
        # 获取时间周期的秒数
        total_seconds = int(delta.total_seconds())
        
        # 转换为时间戳
        timestamp = int(dt.timestamp())
        
        # 舍入到最近的时间周期
        rounded_timestamp = (timestamp // total_seconds) * total_seconds
        
        # 转换回datetime
        return datetime.fromtimestamp(rounded_timestamp, tz=dt.tzinfo)
    except Exception as e:
        logger.error(f"Error rounding time to timeframe {timeframe}: {e}")
        return dt


def get_market_hours(market: str = "crypto") -> dict:
    """获取市场交易时间
    
    Args:
        market: 市场类型（crypto, forex, stock）
        
    Returns:
        dict: 交易时间信息
    """
    market_hours = {
        "crypto": {
            "is_24_7": True,
            "timezone": "UTC",
            "start_hour": 0,
            "end_hour": 24,
            "trading_days": list(range(7))  # 0-6 (Monday-Sunday)
        },
        "forex": {
            "is_24_7": False,
            "timezone": "UTC",
            "start_hour": 22,  # Sunday 22:00 UTC
            "end_hour": 22,   # Friday 22:00 UTC
            "trading_days": [0, 1, 2, 3, 4]  # Monday-Friday
        },
        "stock": {
            "is_24_7": False,
            "timezone": "US/Eastern",
            "start_hour": 9,   # 9:30 AM
            "end_hour": 16,    # 4:00 PM
            "trading_days": [0, 1, 2, 3, 4]  # Monday-Friday
        }
    }
    
    return market_hours.get(market.lower(), market_hours["crypto"])


def is_market_open(market: str = "crypto", dt: Optional[datetime] = None) -> bool:
    """检查市场是否开放
    
    Args:
        market: 市场类型
        dt: 指定时间，默认为当前时间
        
    Returns:
        bool: 市场是否开放
    """
    if dt is None:
        dt = datetime.now(timezone.utc)
    
    market_info = get_market_hours(market)
    
    # 加密货币市场24/7开放
    if market_info["is_24_7"]:
        return True
    
    # 转换到市场时区
    market_tz = pytz.timezone(market_info["timezone"])
    market_dt = dt.astimezone(market_tz)
    
    # 检查是否为交易日
    if market_dt.weekday() not in market_info["trading_days"]:
        return False
    
    # 检查是否在交易时间内
    return is_trading_hours(
        market_dt,
        market_info["start_hour"],
        market_info["end_hour"],
        market_info["timezone"]
    )