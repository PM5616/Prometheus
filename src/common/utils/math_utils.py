"""Math Utility Functions

数学计算相关的工具函数。
"""

import math
from decimal import Decimal, ROUND_HALF_UP
from typing import List, Union, Optional
import numpy as np
from loguru import logger


def calculate_percentage_change(old_value: Union[float, Decimal], 
                              new_value: Union[float, Decimal]) -> Decimal:
    """计算百分比变化
    
    Args:
        old_value: 原值
        new_value: 新值
        
    Returns:
        Decimal: 百分比变化
    """
    try:
        old_val = Decimal(str(old_value))
        new_val = Decimal(str(new_value))
        
        if old_val == 0:
            return Decimal('0') if new_val == 0 else Decimal('inf')
        
        return ((new_val - old_val) / old_val) * 100
    except Exception as e:
        logger.error(f"Error calculating percentage change: {e}")
        return Decimal('0')


def calculate_volatility(prices: List[Union[float, Decimal]], 
                        periods: int = 20) -> Decimal:
    """计算价格波动率
    
    Args:
        prices: 价格列表
        periods: 计算周期
        
    Returns:
        Decimal: 波动率
    """
    try:
        if len(prices) < 2:
            return Decimal('0')
        
        # 转换为numpy数组进行计算
        price_array = np.array([float(p) for p in prices])
        
        # 计算收益率
        returns = np.diff(price_array) / price_array[:-1]
        
        # 计算标准差
        if len(returns) < periods:
            volatility = np.std(returns)
        else:
            volatility = np.std(returns[-periods:])
        
        # 年化波动率（假设252个交易日）
        annualized_volatility = volatility * math.sqrt(252)
        
        return Decimal(str(annualized_volatility))
    except Exception as e:
        logger.error(f"Error calculating volatility: {e}")
        return Decimal('0')


def calculate_sharpe_ratio(returns: List[Union[float, Decimal]], 
                          risk_free_rate: Union[float, Decimal] = 0.02) -> Decimal:
    """计算夏普比率
    
    Args:
        returns: 收益率列表
        risk_free_rate: 无风险利率
        
    Returns:
        Decimal: 夏普比率
    """
    try:
        if len(returns) < 2:
            return Decimal('0')
        
        returns_array = np.array([float(r) for r in returns])
        
        # 计算平均收益率
        avg_return = np.mean(returns_array)
        
        # 计算收益率标准差
        std_return = np.std(returns_array)
        
        if std_return == 0:
            return Decimal('0')
        
        # 计算夏普比率
        sharpe = (avg_return - float(risk_free_rate)) / std_return
        
        return Decimal(str(sharpe))
    except Exception as e:
        logger.error(f"Error calculating Sharpe ratio: {e}")
        return Decimal('0')


def calculate_max_drawdown(equity_curve: List[Union[float, Decimal]]) -> Decimal:
    """计算最大回撤
    
    Args:
        equity_curve: 权益曲线
        
    Returns:
        Decimal: 最大回撤
    """
    try:
        if len(equity_curve) < 2:
            return Decimal('0')
        
        equity_array = np.array([float(e) for e in equity_curve])
        
        # 计算累计最高点
        peak = np.maximum.accumulate(equity_array)
        
        # 计算回撤
        drawdown = (peak - equity_array) / peak
        
        # 返回最大回撤
        max_dd = np.max(drawdown)
        
        return Decimal(str(max_dd))
    except Exception as e:
        logger.error(f"Error calculating max drawdown: {e}")
        return Decimal('0')


def round_to_precision(value: Union[float, Decimal], precision: int) -> Decimal:
    """按精度舍入
    
    Args:
        value: 要舍入的值
        precision: 小数位数
        
    Returns:
        Decimal: 舍入后的值
    """
    try:
        decimal_value = Decimal(str(value))
        
        if precision < 0:
            # 整数位舍入
            factor = Decimal('10') ** abs(precision)
            return (decimal_value / factor).quantize(Decimal('1'), rounding=ROUND_HALF_UP) * factor
        else:
            # 小数位舍入
            quantizer = Decimal('0.1') ** precision
            return decimal_value.quantize(quantizer, rounding=ROUND_HALF_UP)
    except Exception as e:
        logger.error(f"Error rounding to precision: {e}")
        return Decimal(str(value))


def safe_divide(numerator: Union[float, Decimal], 
               denominator: Union[float, Decimal], 
               default: Union[float, Decimal] = 0) -> Decimal:
    """安全除法，避免除零错误
    
    Args:
        numerator: 分子
        denominator: 分母
        default: 默认值（当分母为0时）
        
    Returns:
        Decimal: 除法结果
    """
    try:
        num = Decimal(str(numerator))
        den = Decimal(str(denominator))
        
        if den == 0:
            return Decimal(str(default))
        
        return num / den
    except Exception as e:
        logger.error(f"Error in safe division: {e}")
        return Decimal(str(default))


def calculate_rsi(prices: List[Union[float, Decimal]], period: int = 14) -> Decimal:
    """计算相对强弱指数(RSI)
    
    Args:
        prices: 价格列表
        period: 计算周期
        
    Returns:
        Decimal: RSI值
    """
    try:
        if len(prices) < period + 1:
            return Decimal('50')  # 默认中性值
        
        price_array = np.array([float(p) for p in prices])
        
        # 计算价格变化
        deltas = np.diff(price_array)
        
        # 分离上涨和下跌
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        # 计算平均收益和损失
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return Decimal('100')
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return Decimal(str(rsi))
    except Exception as e:
        logger.error(f"Error calculating RSI: {e}")
        return Decimal('50')


def calculate_moving_average(prices: List[Union[float, Decimal]], 
                           period: int, 
                           ma_type: str = 'simple') -> Decimal:
    """计算移动平均线
    
    Args:
        prices: 价格列表
        period: 计算周期
        ma_type: 移动平均类型（'simple', 'exponential'）
        
    Returns:
        Decimal: 移动平均值
    """
    try:
        if len(prices) < period:
            return Decimal(str(prices[-1])) if prices else Decimal('0')
        
        price_array = np.array([float(p) for p in prices[-period:]])
        
        if ma_type.lower() == 'simple':
            ma = np.mean(price_array)
        elif ma_type.lower() == 'exponential':
            # 指数移动平均
            alpha = 2 / (period + 1)
            ema = price_array[0]
            for price in price_array[1:]:
                ema = alpha * price + (1 - alpha) * ema
            ma = ema
        else:
            ma = np.mean(price_array)
        
        return Decimal(str(ma))
    except Exception as e:
        logger.error(f"Error calculating moving average: {e}")
        return Decimal('0')


def calculate_bollinger_bands(prices: List[Union[float, Decimal]], 
                             period: int = 20, 
                             std_dev: float = 2) -> dict:
    """计算布林带
    
    Args:
        prices: 价格列表
        period: 计算周期
        std_dev: 标准差倍数
        
    Returns:
        dict: 包含上轨、中轨、下轨的字典
    """
    try:
        if len(prices) < period:
            current_price = Decimal(str(prices[-1])) if prices else Decimal('0')
            return {
                'upper': current_price,
                'middle': current_price,
                'lower': current_price
            }
        
        price_array = np.array([float(p) for p in prices[-period:]])
        
        # 计算中轨（简单移动平均）
        middle = np.mean(price_array)
        
        # 计算标准差
        std = np.std(price_array)
        
        # 计算上下轨
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        return {
            'upper': Decimal(str(upper)),
            'middle': Decimal(str(middle)),
            'lower': Decimal(str(lower))
        }
    except Exception as e:
        logger.error(f"Error calculating Bollinger Bands: {e}")
        current_price = Decimal(str(prices[-1])) if prices else Decimal('0')
        return {
            'upper': current_price,
            'middle': current_price,
            'lower': current_price
        }


def calculate_macd(prices: List[Union[float, Decimal]], 
                  fast_period: int = 12, 
                  slow_period: int = 26, 
                  signal_period: int = 9) -> dict:
    """计算MACD指标
    
    Args:
        prices: 价格列表
        fast_period: 快线周期
        slow_period: 慢线周期
        signal_period: 信号线周期
        
    Returns:
        dict: 包含MACD线、信号线、柱状图的字典
    """
    try:
        if len(prices) < slow_period:
            return {
                'macd': Decimal('0'),
                'signal': Decimal('0'),
                'histogram': Decimal('0')
            }
        
        price_array = np.array([float(p) for p in prices])
        
        # 计算快速和慢速EMA
        fast_ema = _calculate_ema(price_array, fast_period)
        slow_ema = _calculate_ema(price_array, slow_period)
        
        # 计算MACD线
        macd_line = fast_ema - slow_ema
        
        # 计算信号线（MACD的EMA）
        signal_line = _calculate_ema(macd_line, signal_period)
        
        # 计算柱状图
        histogram = macd_line - signal_line
        
        return {
            'macd': Decimal(str(macd_line[-1])),
            'signal': Decimal(str(signal_line[-1])),
            'histogram': Decimal(str(histogram[-1]))
        }
    except Exception as e:
        logger.error(f"Error calculating MACD: {e}")
        return {
            'macd': Decimal('0'),
            'signal': Decimal('0'),
            'histogram': Decimal('0')
        }


def _calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """计算指数移动平均
    
    Args:
        prices: 价格数组
        period: 周期
        
    Returns:
        np.ndarray: EMA数组
    """
    alpha = 2 / (period + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema


def calculate_correlation(x: List[Union[float, Decimal]], 
                         y: List[Union[float, Decimal]]) -> Decimal:
    """计算相关系数
    
    Args:
        x: 第一个数据序列
        y: 第二个数据序列
        
    Returns:
        Decimal: 相关系数
    """
    try:
        if len(x) != len(y) or len(x) < 2:
            return Decimal('0')
        
        x_array = np.array([float(val) for val in x])
        y_array = np.array([float(val) for val in y])
        
        correlation = np.corrcoef(x_array, y_array)[0, 1]
        
        # 处理NaN值
        if np.isnan(correlation):
            return Decimal('0')
        
        return Decimal(str(correlation))
    except Exception as e:
        logger.error(f"Error calculating correlation: {e}")
        return Decimal('0')


def calculate_var(returns: List[Union[float, Decimal]], 
                 confidence_level: float = 0.95) -> Decimal:
    """计算风险价值(VaR)
    
    Args:
        returns: 收益率列表
        confidence_level: 置信水平
        
    Returns:
        Decimal: VaR值
    """
    try:
        if len(returns) < 2:
            return Decimal('0')
        
        returns_array = np.array([float(r) for r in returns])
        
        # 计算分位数
        percentile = (1 - confidence_level) * 100
        var = np.percentile(returns_array, percentile)
        
        return Decimal(str(abs(var)))
    except Exception as e:
        logger.error(f"Error calculating VaR: {e}")
        return Decimal('0')