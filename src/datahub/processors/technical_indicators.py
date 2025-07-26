"""Technical Indicators Module

技术指标计算器，提供各种技术分析指标的计算功能。

主要功能：
- 趋势指标：移动平均线、MACD、ADX等
- 震荡指标：RSI、KDJ、威廉指标等
- 成交量指标：OBV、成交量移动平均等
- 波动率指标：布林带、ATR等
- 支撑阻力：枢轴点、斐波那契回调等
"""

import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union, Tuple
from datetime import datetime
import time

from .base import BaseProcessor, ProcessorType, ProcessingResult
from ...common.exceptions.data import DataProcessingError


class TechnicalIndicators(BaseProcessor):
    """技术指标计算器
    
    提供各种技术分析指标的计算功能，支持批量计算和实时更新。
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """初始化技术指标计算器
        
        Args:
            config: 指标配置
                - default_periods: 默认周期参数
                - price_columns: 价格列名映射
                - volume_column: 成交量列名
                - indicators: 要计算的指标列表
        """
        super().__init__("TechnicalIndicators", ProcessorType.INDICATOR, config)
        
        # 默认周期参数
        self.default_periods = self.config.get('default_periods', {
            'sma': [5, 10, 20, 50],
            'ema': [12, 26],
            'rsi': 14,
            'macd': [12, 26, 9],
            'bollinger': [20, 2],
            'kdj': [9, 3, 3],
            'atr': 14
        })
        
        # 价格列名映射
        self.price_columns = self.config.get('price_columns', {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        })
        
        # 要计算的指标
        self.indicators = self.config.get('indicators', [
            'sma', 'ema', 'rsi', 'macd', 'bollinger', 'kdj'
        ])
        
        # 计算统计
        self.indicator_stats = {
            'indicators_calculated': 0,
            'records_processed': 0,
            'calculation_errors': 0
        }
    
    def initialize(self) -> bool:
        """初始化指标计算器
        
        Returns:
            bool: 初始化是否成功
        """
        try:
            self.logger.info(f"初始化技术指标计算器: {self.name}")
            self.logger.info(f"支持的指标: {', '.join(self.indicators)}")
            self.logger.info(f"默认周期: {self.default_periods}")
            
            self._initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"初始化技术指标计算器失败: {e}")
            return False
    
    def validate_input(self, data: Any) -> bool:
        """验证输入数据
        
        Args:
            data: 输入数据
            
        Returns:
            bool: 数据是否有效
        """
        if not isinstance(data, pd.DataFrame) or data.empty:
            return False
        
        # 检查必要的价格列
        required_columns = ['close']
        for col in required_columns:
            mapped_col = self.price_columns.get(col, col)
            if mapped_col not in data.columns:
                self.logger.error(f"缺少必要的价格列: {mapped_col}")
                return False
        
        return True
    
    def process(self, data: Any, **kwargs) -> ProcessingResult:
        """计算技术指标
        
        Args:
            data: 价格数据（DataFrame）
            **kwargs: 额外参数
                - indicators: 要计算的指标列表
                - periods: 自定义周期参数
                - inplace: 是否在原数据上添加指标
                
        Returns:
            ProcessingResult: 计算结果
        """
        start_time = time.time()
        
        try:
            if not self.validate_input(data):
                return ProcessingResult(
                    data=None,
                    success=False,
                    message="输入数据无效或缺少必要的价格列"
                )
            
            df = data.copy()
            indicators_to_calc = kwargs.get('indicators', self.indicators)
            custom_periods = kwargs.get('periods', {})
            inplace = kwargs.get('inplace', False)
            
            # 合并周期参数
            periods = {**self.default_periods, **custom_periods}
            
            calculated_indicators = []
            
            # 计算各种技术指标
            for indicator in indicators_to_calc:
                try:
                    if indicator == 'sma':
                        df = self._calculate_sma(df, periods.get('sma', [20]))
                    elif indicator == 'ema':
                        df = self._calculate_ema(df, periods.get('ema', [12, 26]))
                    elif indicator == 'rsi':
                        df = self._calculate_rsi(df, periods.get('rsi', 14))
                    elif indicator == 'macd':
                        df = self._calculate_macd(df, periods.get('macd', [12, 26, 9]))
                    elif indicator == 'bollinger':
                        df = self._calculate_bollinger_bands(df, periods.get('bollinger', [20, 2]))
                    elif indicator == 'kdj':
                        df = self._calculate_kdj(df, periods.get('kdj', [9, 3, 3]))
                    elif indicator == 'atr':
                        df = self._calculate_atr(df, periods.get('atr', 14))
                    elif indicator == 'obv':
                        df = self._calculate_obv(df)
                    elif indicator == 'williams':
                        df = self._calculate_williams_r(df, periods.get('williams', 14))
                    elif indicator == 'cci':
                        df = self._calculate_cci(df, periods.get('cci', 20))
                    
                    calculated_indicators.append(indicator)
                    
                except Exception as e:
                    self.logger.error(f"计算指标 {indicator} 时发生错误: {e}")
                    self.indicator_stats['calculation_errors'] += 1
            
            # 如果不是原地修改，只返回指标列
            if not inplace:
                # 获取新添加的指标列
                original_columns = set(data.columns)
                new_columns = set(df.columns) - original_columns
                if new_columns:
                    # 保留原始的索引列和新指标列
                    result_df = df[list(original_columns) + list(new_columns)]
                else:
                    result_df = df
            else:
                result_df = df
            
            processing_time = time.time() - start_time
            
            # 更新统计信息
            self.indicator_stats['indicators_calculated'] += len(calculated_indicators)
            self.indicator_stats['records_processed'] += len(df)
            
            result = ProcessingResult(
                data=result_df,
                success=True,
                message=f"成功计算 {len(calculated_indicators)} 个技术指标",
                metadata={
                    'calculated_indicators': calculated_indicators,
                    'total_records': len(df),
                    'new_columns': len(df.columns) - len(data.columns)
                },
                processing_time=processing_time
            )
            
            self._update_stats(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"技术指标计算失败: {str(e)}"
            self.logger.error(error_msg)
            
            result = ProcessingResult(
                data=None,
                success=False,
                message=error_msg,
                processing_time=processing_time
            )
            
            self._update_stats(result)
            return result
    
    def _get_price_series(self, df: pd.DataFrame, price_type: str) -> pd.Series:
        """获取价格序列
        
        Args:
            df: 数据DataFrame
            price_type: 价格类型 ('open', 'high', 'low', 'close', 'volume')
            
        Returns:
            pd.Series: 价格序列
        """
        column_name = self.price_columns.get(price_type, price_type)
        if column_name not in df.columns:
            raise DataProcessingError(f"找不到价格列: {column_name}")
        return df[column_name]
    
    def _calculate_sma(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """计算简单移动平均线
        
        Args:
            df: 数据DataFrame
            periods: 周期列表
            
        Returns:
            pd.DataFrame: 添加SMA指标的DataFrame
        """
        close = self._get_price_series(df, 'close')
        
        for period in periods:
            df[f'SMA_{period}'] = close.rolling(window=period).mean()
        
        return df
    
    def _calculate_ema(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """计算指数移动平均线
        
        Args:
            df: 数据DataFrame
            periods: 周期列表
            
        Returns:
            pd.DataFrame: 添加EMA指标的DataFrame
        """
        close = self._get_price_series(df, 'close')
        
        for period in periods:
            df[f'EMA_{period}'] = close.ewm(span=period).mean()
        
        return df
    
    def _calculate_rsi(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算相对强弱指标
        
        Args:
            df: 数据DataFrame
            period: 计算周期
            
        Returns:
            pd.DataFrame: 添加RSI指标的DataFrame
        """
        close = self._get_price_series(df, 'close')
        
        # 计算价格变化
        delta = close.diff()
        
        # 分离上涨和下跌
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # 计算平均收益和损失
        avg_gain = gain.rolling(window=period).mean()
        avg_loss = loss.rolling(window=period).mean()
        
        # 计算RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        df[f'RSI_{period}'] = rsi
        
        return df
    
    def _calculate_macd(self, df: pd.DataFrame, periods: List[int]) -> pd.DataFrame:
        """计算MACD指标
        
        Args:
            df: 数据DataFrame
            periods: [快线周期, 慢线周期, 信号线周期]
            
        Returns:
            pd.DataFrame: 添加MACD指标的DataFrame
        """
        if len(periods) != 3:
            raise DataProcessingError("MACD需要3个周期参数: [快线, 慢线, 信号线]")
        
        fast_period, slow_period, signal_period = periods
        close = self._get_price_series(df, 'close')
        
        # 计算快慢EMA
        ema_fast = close.ewm(span=fast_period).mean()
        ema_slow = close.ewm(span=slow_period).mean()
        
        # 计算MACD线
        macd_line = ema_fast - ema_slow
        
        # 计算信号线
        signal_line = macd_line.ewm(span=signal_period).mean()
        
        # 计算柱状图
        histogram = macd_line - signal_line
        
        df['MACD'] = macd_line
        df['MACD_Signal'] = signal_line
        df['MACD_Histogram'] = histogram
        
        return df
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, params: List) -> pd.DataFrame:
        """计算布林带
        
        Args:
            df: 数据DataFrame
            params: [周期, 标准差倍数]
            
        Returns:
            pd.DataFrame: 添加布林带指标的DataFrame
        """
        if len(params) != 2:
            raise DataProcessingError("布林带需要2个参数: [周期, 标准差倍数]")
        
        period, std_dev = params
        close = self._get_price_series(df, 'close')
        
        # 计算中轨（移动平均）
        middle_band = close.rolling(window=period).mean()
        
        # 计算标准差
        std = close.rolling(window=period).std()
        
        # 计算上下轨
        upper_band = middle_band + (std * std_dev)
        lower_band = middle_band - (std * std_dev)
        
        df[f'BB_Upper_{period}'] = upper_band
        df[f'BB_Middle_{period}'] = middle_band
        df[f'BB_Lower_{period}'] = lower_band
        
        # 计算布林带宽度和位置
        df[f'BB_Width_{period}'] = (upper_band - lower_band) / middle_band
        df[f'BB_Position_{period}'] = (close - lower_band) / (upper_band - lower_band)
        
        return df
    
    def _calculate_kdj(self, df: pd.DataFrame, params: List[int]) -> pd.DataFrame:
        """计算KDJ指标
        
        Args:
            df: 数据DataFrame
            params: [K周期, D周期, J周期]
            
        Returns:
            pd.DataFrame: 添加KDJ指标的DataFrame
        """
        if len(params) != 3:
            raise DataProcessingError("KDJ需要3个周期参数: [K, D, J]")
        
        k_period, d_period, j_period = params
        
        high = self._get_price_series(df, 'high')
        low = self._get_price_series(df, 'low')
        close = self._get_price_series(df, 'close')
        
        # 计算最高价和最低价
        highest_high = high.rolling(window=k_period).max()
        lowest_low = low.rolling(window=k_period).min()
        
        # 计算RSV
        rsv = (close - lowest_low) / (highest_high - lowest_low) * 100
        
        # 计算K值
        k_values = []
        k_prev = 50  # 初始K值
        
        for rsv_val in rsv:
            if pd.isna(rsv_val):
                k_values.append(np.nan)
            else:
                k_current = (2/3) * k_prev + (1/3) * rsv_val
                k_values.append(k_current)
                k_prev = k_current
        
        df['K'] = k_values
        
        # 计算D值
        d_values = []
        d_prev = 50  # 初始D值
        
        for k_val in k_values:
            if pd.isna(k_val):
                d_values.append(np.nan)
            else:
                d_current = (2/3) * d_prev + (1/3) * k_val
                d_values.append(d_current)
                d_prev = d_current
        
        df['D'] = d_values
        
        # 计算J值
        df['J'] = 3 * df['K'] - 2 * df['D']
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算平均真实波幅
        
        Args:
            df: 数据DataFrame
            period: 计算周期
            
        Returns:
            pd.DataFrame: 添加ATR指标的DataFrame
        """
        high = self._get_price_series(df, 'high')
        low = self._get_price_series(df, 'low')
        close = self._get_price_series(df, 'close')
        
        # 计算真实波幅
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # 计算ATR
        atr = true_range.rolling(window=period).mean()
        
        df[f'ATR_{period}'] = atr
        
        return df
    
    def _calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算能量潮指标
        
        Args:
            df: 数据DataFrame
            
        Returns:
            pd.DataFrame: 添加OBV指标的DataFrame
        """
        close = self._get_price_series(df, 'close')
        volume = self._get_price_series(df, 'volume')
        
        # 计算价格变化方向
        price_change = close.diff()
        
        # 计算OBV
        obv = []
        obv_value = 0
        
        for i, (price_diff, vol) in enumerate(zip(price_change, volume)):
            if i == 0 or pd.isna(price_diff):
                obv.append(obv_value)
            elif price_diff > 0:
                obv_value += vol
                obv.append(obv_value)
            elif price_diff < 0:
                obv_value -= vol
                obv.append(obv_value)
            else:
                obv.append(obv_value)
        
        df['OBV'] = obv
        
        return df
    
    def _calculate_williams_r(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算威廉指标
        
        Args:
            df: 数据DataFrame
            period: 计算周期
            
        Returns:
            pd.DataFrame: 添加Williams %R指标的DataFrame
        """
        high = self._get_price_series(df, 'high')
        low = self._get_price_series(df, 'low')
        close = self._get_price_series(df, 'close')
        
        # 计算最高价和最低价
        highest_high = high.rolling(window=period).max()
        lowest_low = low.rolling(window=period).min()
        
        # 计算Williams %R
        williams_r = (highest_high - close) / (highest_high - lowest_low) * -100
        
        df[f'Williams_R_{period}'] = williams_r
        
        return df
    
    def _calculate_cci(self, df: pd.DataFrame, period: int) -> pd.DataFrame:
        """计算商品通道指标
        
        Args:
            df: 数据DataFrame
            period: 计算周期
            
        Returns:
            pd.DataFrame: 添加CCI指标的DataFrame
        """
        high = self._get_price_series(df, 'high')
        low = self._get_price_series(df, 'low')
        close = self._get_price_series(df, 'close')
        
        # 计算典型价格
        typical_price = (high + low + close) / 3
        
        # 计算移动平均
        sma_tp = typical_price.rolling(window=period).mean()
        
        # 计算平均偏差
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x)))
        )
        
        # 计算CCI
        cci = (typical_price - sma_tp) / (0.015 * mad)
        
        df[f'CCI_{period}'] = cci
        
        return df
    
    def get_indicator_stats(self) -> Dict:
        """获取指标计算统计信息
        
        Returns:
            Dict: 统计信息
        """
        return self.indicator_stats.copy()
    
    def reset_indicator_stats(self) -> None:
        """重置指标计算统计信息"""
        self.indicator_stats = {
            'indicators_calculated': 0,
            'records_processed': 0,
            'calculation_errors': 0
        }
        self.logger.info("指标计算统计信息已重置")