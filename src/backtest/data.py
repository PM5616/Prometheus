"""Backtest Data Module

回测数据管理模块，负责回测数据的加载、处理和管理。

主要功能：
- 历史数据加载
- 数据预处理
- 数据验证
- 数据缓存
- 基准数据管理
- 数据同步
"""

import asyncio
from typing import Dict, List, Optional, Tuple, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import json
import uuid

from ..common.logging import get_logger
from ..common.exceptions.backtest import BacktestDataError
from ..datahub.data_manager import DataManager


class DataFrequency(Enum):
    """数据频率枚举"""
    TICK = "tick"          # 逐笔数据
    SECOND = "1s"          # 秒级数据
    MINUTE = "1m"          # 分钟数据
    MINUTE_5 = "5m"        # 5分钟数据
    MINUTE_15 = "15m"      # 15分钟数据
    MINUTE_30 = "30m"      # 30分钟数据
    HOUR = "1h"            # 小时数据
    HOUR_4 = "4h"          # 4小时数据
    DAILY = "1d"           # 日线数据
    WEEKLY = "1w"          # 周线数据
    MONTHLY = "1M"         # 月线数据


class DataQuality(Enum):
    """数据质量枚举"""
    EXCELLENT = "excellent"    # 优秀
    GOOD = "good"              # 良好
    FAIR = "fair"              # 一般
    POOR = "poor"              # 较差
    INVALID = "invalid"        # 无效


@dataclass
class DataValidationRule:
    """数据验证规则"""
    name: str
    description: str
    check_function: callable
    severity: str = "warning"  # error, warning, info
    enabled: bool = True


@dataclass
class DataValidationResult:
    """数据验证结果"""
    symbol: str
    rule_name: str
    passed: bool
    message: str
    severity: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DataQualityReport:
    """数据质量报告"""
    symbol: str
    start_date: datetime
    end_date: datetime
    frequency: DataFrequency
    
    # 基本统计
    total_records: int = 0
    missing_records: int = 0
    duplicate_records: int = 0
    
    # 数据完整性
    completeness_ratio: float = 0.0
    missing_dates: List[datetime] = field(default_factory=list)
    
    # 数据质量
    quality_score: float = 0.0
    quality_level: DataQuality = DataQuality.FAIR
    
    # 验证结果
    validation_results: List[DataValidationResult] = field(default_factory=list)
    
    # 统计信息
    price_statistics: Dict[str, float] = field(default_factory=dict)
    volume_statistics: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 报告字典
        """
        return {
            'symbol': self.symbol,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'frequency': self.frequency.value,
            'total_records': self.total_records,
            'missing_records': self.missing_records,
            'duplicate_records': self.duplicate_records,
            'completeness_ratio': self.completeness_ratio,
            'missing_dates': [d.isoformat() for d in self.missing_dates],
            'quality_score': self.quality_score,
            'quality_level': self.quality_level.value,
            'validation_results': [
                {
                    'rule_name': r.rule_name,
                    'passed': r.passed,
                    'message': r.message,
                    'severity': r.severity,
                    'timestamp': r.timestamp.isoformat()
                }
                for r in self.validation_results
            ],
            'price_statistics': self.price_statistics,
            'volume_statistics': self.volume_statistics
        }


@dataclass
class BacktestDataConfig:
    """回测数据配置"""
    # 基本配置
    symbols: List[str] = field(default_factory=list)
    start_date: datetime = field(default_factory=lambda: datetime.now() - timedelta(days=365))
    end_date: datetime = field(default_factory=lambda: datetime.now())
    frequency: DataFrequency = DataFrequency.DAILY
    
    # 数据源配置
    data_sources: List[str] = field(default_factory=list)
    primary_source: Optional[str] = None
    
    # 数据处理配置
    fill_missing: bool = True
    remove_duplicates: bool = True
    validate_data: bool = True
    
    # 缓存配置
    enable_cache: bool = True
    cache_directory: Optional[str] = None
    cache_expiry_hours: int = 24
    
    # 基准配置
    benchmark_symbol: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 配置字典
        """
        return {
            'symbols': self.symbols,
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'frequency': self.frequency.value,
            'data_sources': self.data_sources,
            'primary_source': self.primary_source,
            'fill_missing': self.fill_missing,
            'remove_duplicates': self.remove_duplicates,
            'validate_data': self.validate_data,
            'enable_cache': self.enable_cache,
            'cache_directory': self.cache_directory,
            'cache_expiry_hours': self.cache_expiry_hours,
            'benchmark_symbol': self.benchmark_symbol
        }


class DataValidator:
    """数据验证器
    
    负责验证回测数据的质量和完整性。
    """
    
    def __init__(self):
        """初始化数据验证器"""
        self.validation_rules: Dict[str, DataValidationRule] = {}
        self.logger = get_logger("DataValidator")
        
        # 添加默认验证规则
        self._add_default_rules()
    
    def _add_default_rules(self):
        """添加默认验证规则"""
        # 价格合理性检查
        self.add_rule(DataValidationRule(
            name="price_positive",
            description="检查价格是否为正数",
            check_function=lambda df: (df[['open', 'high', 'low', 'close']] > 0).all().all(),
            severity="error"
        ))
        
        # 高低价关系检查
        self.add_rule(DataValidationRule(
            name="high_low_relationship",
            description="检查最高价是否大于等于最低价",
            check_function=lambda df: (df['high'] >= df['low']).all(),
            severity="error"
        ))
        
        # OHLC关系检查
        self.add_rule(DataValidationRule(
            name="ohlc_relationship",
            description="检查OHLC价格关系",
            check_function=self._check_ohlc_relationship,
            severity="error"
        ))
        
        # 成交量检查
        self.add_rule(DataValidationRule(
            name="volume_non_negative",
            description="检查成交量是否非负",
            check_function=lambda df: (df['volume'] >= 0).all() if 'volume' in df.columns else True,
            severity="warning"
        ))
        
        # 价格跳跃检查
        self.add_rule(DataValidationRule(
            name="price_jump_check",
            description="检查价格异常跳跃",
            check_function=self._check_price_jumps,
            severity="warning"
        ))
        
        # 数据连续性检查
        self.add_rule(DataValidationRule(
            name="data_continuity",
            description="检查数据时间连续性",
            check_function=self._check_data_continuity,
            severity="warning"
        ))
    
    def _check_ohlc_relationship(self, df: pd.DataFrame) -> bool:
        """检查OHLC价格关系
        
        Args:
            df: 数据DataFrame
            
        Returns:
            bool: 是否通过检查
        """
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            return True  # 如果没有完整的OHLC数据，跳过检查
        
        # 检查 high >= max(open, close) 和 low <= min(open, close)
        high_check = (df['high'] >= df[['open', 'close']].max(axis=1)).all()
        low_check = (df['low'] <= df[['open', 'close']].min(axis=1)).all()
        
        return high_check and low_check
    
    def _check_price_jumps(self, df: pd.DataFrame) -> bool:
        """检查价格异常跳跃
        
        Args:
            df: 数据DataFrame
            
        Returns:
            bool: 是否通过检查
        """
        if 'close' not in df.columns or len(df) < 2:
            return True
        
        # 计算价格变化率
        price_changes = df['close'].pct_change().dropna()
        
        # 检查是否有超过50%的单日涨跌幅（可能是数据错误）
        extreme_changes = abs(price_changes) > 0.5
        
        return not extreme_changes.any()
    
    def _check_data_continuity(self, df: pd.DataFrame) -> bool:
        """检查数据时间连续性
        
        Args:
            df: 数据DataFrame
            
        Returns:
            bool: 是否通过检查
        """
        if len(df) < 2:
            return True
        
        # 检查时间索引是否单调递增
        return df.index.is_monotonic_increasing
    
    def add_rule(self, rule: DataValidationRule):
        """添加验证规则
        
        Args:
            rule: 验证规则
        """
        self.validation_rules[rule.name] = rule
        self.logger.info(f"添加验证规则: {rule.name}")
    
    def remove_rule(self, rule_name: str):
        """移除验证规则
        
        Args:
            rule_name: 规则名称
        """
        if rule_name in self.validation_rules:
            del self.validation_rules[rule_name]
            self.logger.info(f"移除验证规则: {rule_name}")
    
    def enable_rule(self, rule_name: str):
        """启用验证规则
        
        Args:
            rule_name: 规则名称
        """
        if rule_name in self.validation_rules:
            self.validation_rules[rule_name].enabled = True
    
    def disable_rule(self, rule_name: str):
        """禁用验证规则
        
        Args:
            rule_name: 规则名称
        """
        if rule_name in self.validation_rules:
            self.validation_rules[rule_name].enabled = False
    
    def validate_data(self, symbol: str, data: pd.DataFrame) -> List[DataValidationResult]:
        """验证数据
        
        Args:
            symbol: 交易品种
            data: 数据DataFrame
            
        Returns:
            List[DataValidationResult]: 验证结果列表
        """
        results = []
        
        for rule_name, rule in self.validation_rules.items():
            if not rule.enabled:
                continue
            
            try:
                passed = rule.check_function(data)
                message = f"规则 {rule.name} 检查通过" if passed else f"规则 {rule.name} 检查失败: {rule.description}"
                
                result = DataValidationResult(
                    symbol=symbol,
                    rule_name=rule_name,
                    passed=passed,
                    message=message,
                    severity=rule.severity
                )
                
                results.append(result)
                
                if not passed:
                    self.logger.warning(f"数据验证失败 {symbol}: {message}")
                    
            except Exception as e:
                error_result = DataValidationResult(
                    symbol=symbol,
                    rule_name=rule_name,
                    passed=False,
                    message=f"验证规则执行失败: {str(e)}",
                    severity="error"
                )
                results.append(error_result)
                self.logger.error(f"验证规则执行失败 {rule_name}: {e}")
        
        return results


class BacktestData:
    """回测数据管理器
    
    负责回测数据的加载、处理和管理。
    """
    
    def __init__(self, data_manager: DataManager, config: BacktestDataConfig):
        """初始化回测数据管理器
        
        Args:
            data_manager: 数据管理器
            config: 数据配置
        """
        self.data_manager = data_manager
        self.config = config
        
        # 数据存储
        self.data: Dict[str, pd.DataFrame] = {}
        self.benchmark_data: Optional[pd.DataFrame] = None
        
        # 数据质量
        self.validator = DataValidator()
        self.quality_reports: Dict[str, DataQualityReport] = {}
        
        # 缓存管理
        self.cache_directory = Path(config.cache_directory) if config.cache_directory else Path.cwd() / "cache"
        self.cache_directory.mkdir(exist_ok=True)
        
        # 状态管理
        self.loaded_symbols: set = set()
        self.loading_errors: Dict[str, str] = {}
        
        # 日志记录
        self.logger = get_logger("BacktestData")
        
        self.logger.info("回测数据管理器初始化完成")
    
    async def load_data(self) -> bool:
        """加载回测数据
        
        Returns:
            bool: 是否加载成功
        """
        self.logger.info("开始加载回测数据...")
        
        success = True
        
        # 加载主要数据
        for symbol in self.config.symbols:
            try:
                data = await self._load_symbol_data(symbol)
                if data is not None and not data.empty:
                    self.data[symbol] = data
                    self.loaded_symbols.add(symbol)
                    self.logger.info(f"加载数据成功: {symbol} - {len(data)}条记录")
                else:
                    self.loading_errors[symbol] = "数据为空"
                    success = False
                    self.logger.warning(f"数据为空: {symbol}")
                    
            except Exception as e:
                self.loading_errors[symbol] = str(e)
                success = False
                self.logger.error(f"加载数据失败 {symbol}: {e}")
        
        # 加载基准数据
        if self.config.benchmark_symbol:
            try:
                benchmark_data = await self._load_symbol_data(self.config.benchmark_symbol)
                if benchmark_data is not None and not benchmark_data.empty:
                    self.benchmark_data = benchmark_data
                    self.logger.info(f"加载基准数据成功: {self.config.benchmark_symbol}")
                else:
                    self.logger.warning(f"基准数据为空: {self.config.benchmark_symbol}")
                    
            except Exception as e:
                self.logger.error(f"加载基准数据失败 {self.config.benchmark_symbol}: {e}")
        
        # 数据后处理
        if self.data:
            await self._post_process_data()
        
        self.logger.info(f"数据加载完成，成功加载 {len(self.loaded_symbols)} 个品种")
        
        return success
    
    async def _load_symbol_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """加载单个品种数据
        
        Args:
            symbol: 交易品种
            
        Returns:
            Optional[pd.DataFrame]: 数据DataFrame
        """
        # 检查缓存
        if self.config.enable_cache:
            cached_data = self._load_from_cache(symbol)
            if cached_data is not None:
                self.logger.debug(f"从缓存加载数据: {symbol}")
                return cached_data
        
        # 从数据源加载
        data = await self.data_manager.get_historical_data(
            symbol=symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            frequency=self.config.frequency.value
        )
        
        if data is not None and not data.empty:
            # 保存到缓存
            if self.config.enable_cache:
                self._save_to_cache(symbol, data)
        
        return data
    
    def _load_from_cache(self, symbol: str) -> Optional[pd.DataFrame]:
        """从缓存加载数据
        
        Args:
            symbol: 交易品种
            
        Returns:
            Optional[pd.DataFrame]: 缓存数据
        """
        cache_file = self._get_cache_file_path(symbol)
        
        if not cache_file.exists():
            return None
        
        try:
            # 检查缓存是否过期
            cache_time = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - cache_time > timedelta(hours=self.config.cache_expiry_hours):
                cache_file.unlink()  # 删除过期缓存
                return None
            
            # 加载缓存数据
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            # 验证缓存数据的时间范围
            if (cache_data['start_date'] <= self.config.start_date and
                cache_data['end_date'] >= self.config.end_date and
                cache_data['frequency'] == self.config.frequency.value):
                
                # 筛选所需时间范围的数据
                data = cache_data['data']
                mask = (data.index >= self.config.start_date) & (data.index <= self.config.end_date)
                return data[mask]
            
        except Exception as e:
            self.logger.warning(f"加载缓存失败 {symbol}: {e}")
            # 删除损坏的缓存文件
            if cache_file.exists():
                cache_file.unlink()
        
        return None
    
    def _save_to_cache(self, symbol: str, data: pd.DataFrame):
        """保存数据到缓存
        
        Args:
            symbol: 交易品种
            data: 数据DataFrame
        """
        try:
            cache_file = self._get_cache_file_path(symbol)
            
            cache_data = {
                'symbol': symbol,
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'frequency': self.config.frequency.value,
                'data': data,
                'cached_at': datetime.now()
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(cache_data, f)
            
            self.logger.debug(f"数据已缓存: {symbol}")
            
        except Exception as e:
            self.logger.warning(f"缓存数据失败 {symbol}: {e}")
    
    def _get_cache_file_path(self, symbol: str) -> Path:
        """获取缓存文件路径
        
        Args:
            symbol: 交易品种
            
        Returns:
            Path: 缓存文件路径
        """
        # 创建安全的文件名
        safe_symbol = symbol.replace('/', '_').replace('\\', '_')
        filename = f"{safe_symbol}_{self.config.frequency.value}_{self.config.start_date.strftime('%Y%m%d')}_{self.config.end_date.strftime('%Y%m%d')}.pkl"
        return self.cache_directory / filename
    
    async def _post_process_data(self):
        """数据后处理"""
        self.logger.info("开始数据后处理...")
        
        for symbol in list(self.data.keys()):
            try:
                data = self.data[symbol]
                
                # 移除重复数据
                if self.config.remove_duplicates:
                    original_len = len(data)
                    data = data[~data.index.duplicated(keep='first')]
                    if len(data) < original_len:
                        self.logger.info(f"移除重复数据 {symbol}: {original_len - len(data)}条")
                
                # 填充缺失数据
                if self.config.fill_missing:
                    data = self._fill_missing_data(data)
                
                # 数据验证
                if self.config.validate_data:
                    await self._validate_and_report(symbol, data)
                
                # 更新数据
                self.data[symbol] = data
                
            except Exception as e:
                self.logger.error(f"数据后处理失败 {symbol}: {e}")
                # 移除有问题的数据
                del self.data[symbol]
                self.loaded_symbols.discard(symbol)
                self.loading_errors[symbol] = f"后处理失败: {e}"
        
        self.logger.info("数据后处理完成")
    
    def _fill_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """填充缺失数据
        
        Args:
            data: 原始数据
            
        Returns:
            pd.DataFrame: 填充后的数据
        """
        # 前向填充价格数据
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill')
        
        # 成交量填充为0
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
        
        return data
    
    async def _validate_and_report(self, symbol: str, data: pd.DataFrame):
        """验证数据并生成报告
        
        Args:
            symbol: 交易品种
            data: 数据DataFrame
        """
        # 数据验证
        validation_results = self.validator.validate_data(symbol, data)
        
        # 生成质量报告
        report = self._generate_quality_report(symbol, data, validation_results)
        self.quality_reports[symbol] = report
        
        # 记录严重问题
        error_results = [r for r in validation_results if r.severity == "error" and not r.passed]
        if error_results:
            error_messages = [r.message for r in error_results]
            self.logger.error(f"数据质量严重问题 {symbol}: {'; '.join(error_messages)}")
    
    def _generate_quality_report(self, symbol: str, data: pd.DataFrame, 
                               validation_results: List[DataValidationResult]) -> DataQualityReport:
        """生成数据质量报告
        
        Args:
            symbol: 交易品种
            data: 数据DataFrame
            validation_results: 验证结果
            
        Returns:
            DataQualityReport: 质量报告
        """
        report = DataQualityReport(
            symbol=symbol,
            start_date=self.config.start_date,
            end_date=self.config.end_date,
            frequency=self.config.frequency
        )
        
        # 基本统计
        report.total_records = len(data)
        report.duplicate_records = data.index.duplicated().sum()
        
        # 计算缺失记录数（基于预期的时间序列）
        expected_periods = self._calculate_expected_periods()
        report.missing_records = max(0, expected_periods - report.total_records)
        report.completeness_ratio = report.total_records / expected_periods if expected_periods > 0 else 0
        
        # 验证结果
        report.validation_results = validation_results
        
        # 计算质量分数
        report.quality_score = self._calculate_quality_score(report, validation_results)
        report.quality_level = self._determine_quality_level(report.quality_score)
        
        # 价格统计
        if 'close' in data.columns:
            report.price_statistics = {
                'mean': float(data['close'].mean()),
                'std': float(data['close'].std()),
                'min': float(data['close'].min()),
                'max': float(data['close'].max()),
                'median': float(data['close'].median())
            }
        
        # 成交量统计
        if 'volume' in data.columns:
            report.volume_statistics = {
                'mean': float(data['volume'].mean()),
                'std': float(data['volume'].std()),
                'min': float(data['volume'].min()),
                'max': float(data['volume'].max()),
                'median': float(data['volume'].median())
            }
        
        return report
    
    def _calculate_expected_periods(self) -> int:
        """计算预期的时间周期数
        
        Returns:
            int: 预期周期数
        """
        # 简化计算，实际应根据具体的频率和交易日历计算
        days = (self.config.end_date - self.config.start_date).days
        
        if self.config.frequency == DataFrequency.DAILY:
            return int(days * 0.7)  # 假设70%的日子是交易日
        elif self.config.frequency == DataFrequency.HOUR:
            return int(days * 0.7 * 24)
        elif self.config.frequency == DataFrequency.MINUTE:
            return int(days * 0.7 * 24 * 60)
        else:
            return days
    
    def _calculate_quality_score(self, report: DataQualityReport, 
                               validation_results: List[DataValidationResult]) -> float:
        """计算数据质量分数
        
        Args:
            report: 质量报告
            validation_results: 验证结果
            
        Returns:
            float: 质量分数 (0-100)
        """
        score = 100.0
        
        # 完整性扣分
        completeness_penalty = (1 - report.completeness_ratio) * 30
        score -= completeness_penalty
        
        # 重复数据扣分
        if report.total_records > 0:
            duplicate_ratio = report.duplicate_records / report.total_records
            duplicate_penalty = duplicate_ratio * 20
            score -= duplicate_penalty
        
        # 验证失败扣分
        error_count = sum(1 for r in validation_results if r.severity == "error" and not r.passed)
        warning_count = sum(1 for r in validation_results if r.severity == "warning" and not r.passed)
        
        error_penalty = error_count * 15
        warning_penalty = warning_count * 5
        
        score -= (error_penalty + warning_penalty)
        
        return max(0, min(100, score))
    
    def _determine_quality_level(self, score: float) -> DataQuality:
        """确定数据质量等级
        
        Args:
            score: 质量分数
            
        Returns:
            DataQuality: 质量等级
        """
        if score >= 90:
            return DataQuality.EXCELLENT
        elif score >= 75:
            return DataQuality.GOOD
        elif score >= 60:
            return DataQuality.FAIR
        elif score >= 40:
            return DataQuality.POOR
        else:
            return DataQuality.INVALID
    
    def get_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """获取数据
        
        Args:
            symbol: 交易品种
            
        Returns:
            Optional[pd.DataFrame]: 数据DataFrame
        """
        return self.data.get(symbol)
    
    def get_benchmark_data(self) -> Optional[pd.DataFrame]:
        """获取基准数据
        
        Returns:
            Optional[pd.DataFrame]: 基准数据
        """
        return self.benchmark_data
    
    def get_all_data(self) -> Dict[str, pd.DataFrame]:
        """获取所有数据
        
        Returns:
            Dict[str, pd.DataFrame]: 所有数据
        """
        return self.data.copy()
    
    def get_quality_report(self, symbol: str) -> Optional[DataQualityReport]:
        """获取质量报告
        
        Args:
            symbol: 交易品种
            
        Returns:
            Optional[DataQualityReport]: 质量报告
        """
        return self.quality_reports.get(symbol)
    
    def get_all_quality_reports(self) -> Dict[str, DataQualityReport]:
        """获取所有质量报告
        
        Returns:
            Dict[str, DataQualityReport]: 所有质量报告
        """
        return self.quality_reports.copy()
    
    def get_loading_errors(self) -> Dict[str, str]:
        """获取加载错误
        
        Returns:
            Dict[str, str]: 加载错误
        """
        return self.loading_errors.copy()
    
    def clear_cache(self, symbol: Optional[str] = None):
        """清理缓存
        
        Args:
            symbol: 交易品种，None表示清理所有缓存
        """
        if symbol:
            cache_file = self._get_cache_file_path(symbol)
            if cache_file.exists():
                cache_file.unlink()
                self.logger.info(f"清理缓存: {symbol}")
        else:
            # 清理所有缓存文件
            for cache_file in self.cache_directory.glob("*.pkl"):
                cache_file.unlink()
            self.logger.info("清理所有缓存")
    
    def export_quality_report(self, output_path: str, format: str = "json"):
        """导出质量报告
        
        Args:
            output_path: 输出路径
            format: 输出格式 (json, csv)
        """
        try:
            if format.lower() == "json":
                report_data = {
                    symbol: report.to_dict() 
                    for symbol, report in self.quality_reports.items()
                }
                
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, ensure_ascii=False)
                    
            elif format.lower() == "csv":
                # 将报告转换为DataFrame并保存为CSV
                rows = []
                for symbol, report in self.quality_reports.items():
                    row = {
                        'symbol': symbol,
                        'quality_score': report.quality_score,
                        'quality_level': report.quality_level.value,
                        'total_records': report.total_records,
                        'missing_records': report.missing_records,
                        'duplicate_records': report.duplicate_records,
                        'completeness_ratio': report.completeness_ratio
                    }
                    rows.append(row)
                
                df = pd.DataFrame(rows)
                df.to_csv(output_path, index=False)
            
            self.logger.info(f"质量报告已导出: {output_path}")
            
        except Exception as e:
            self.logger.error(f"导出质量报告失败: {e}")
            raise BacktestDataError(f"导出质量报告失败: {e}")
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 数据管理器字典
        """
        return {
            'config': self.config.to_dict(),
            'loaded_symbols': list(self.loaded_symbols),
            'loading_errors': self.loading_errors,
            'data_shapes': {symbol: data.shape for symbol, data in self.data.items()},
            'benchmark_shape': self.benchmark_data.shape if self.benchmark_data is not None else None,
            'quality_reports': {symbol: report.to_dict() for symbol, report in self.quality_reports.items()}
        }
    
    def __str__(self) -> str:
        return f"BacktestData(symbols={len(self.loaded_symbols)}, errors={len(self.loading_errors)})"
    
    def __repr__(self) -> str:
        return self.__str__()