"""Signal Processor

信号处理器，负责整合和处理来自不同策略的交易信号
"""

import logging
import asyncio
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import asdict
from enum import Enum
from collections import defaultdict, deque

from ..common.exceptions.base import PrometheusException
from ..common.models.base import BaseModel
from .models import Signal, SignalType, SignalStrength, PositionType


class SignalConflictResolution(Enum):
    """信号冲突解决策略"""
    LATEST = "latest"  # 使用最新信号
    STRONGEST = "strongest"  # 使用最强信号
    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均
    MAJORITY_VOTE = "majority_vote"  # 多数投票
    CONSERVATIVE = "conservative"  # 保守策略（冲突时不交易）


class SignalAggregationMethod(Enum):
    """信号聚合方法"""
    SIMPLE_AVERAGE = "simple_average"  # 简单平均
    WEIGHTED_AVERAGE = "weighted_average"  # 加权平均
    ENSEMBLE = "ensemble"  # 集成学习
    VOTING = "voting"  # 投票机制


class ProcessedSignal:
    """处理后的信号"""
    
    def __init__(
        self,
        asset: str,
        signal_type: SignalType,
        strength: SignalStrength,
        quantity: Decimal,
        price: Decimal,
        position_type: PositionType,
        confidence: float = 0.0,
        source_signals: Optional[List[Signal]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None
    ):
        self.asset = asset
        self.signal_type = signal_type
        self.strength = strength
        self.quantity = quantity
        self.price = price
        self.position_type = position_type
        self.confidence = confidence
        self.source_signals = source_signals or []
        self.metadata = metadata or {}
        self.timestamp = timestamp or datetime.now()
        self.processed_at = datetime.now()
    
    def to_signal(self) -> Signal:
        """转换为标准信号"""
        return Signal(
            signal_id=f"processed_{self.asset}_{int(self.timestamp.timestamp())}",
            asset=self.asset,
            signal_type=self.signal_type,
            strength=self.strength,
            quantity=self.quantity,
            price=self.price,
            position_type=self.position_type,
            confidence=self.confidence,
            metadata=self.metadata,
            timestamp=self.timestamp
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'asset': self.asset,
            'signal_type': self.signal_type.value,
            'strength': self.strength.value,
            'quantity': float(self.quantity),
            'price': float(self.price),
            'position_type': self.position_type.value,
            'confidence': self.confidence,
            'source_count': len(self.source_signals),
            'timestamp': self.timestamp.isoformat(),
            'processed_at': self.processed_at.isoformat()
        }


class SignalProcessorException(PrometheusException):
    """信号处理异常"""
    pass


class SignalProcessor(BaseModel):
    """信号处理器
    
    负责处理和整合来自不同策略的交易信号，包括：
    1. 信号收集和缓存
    2. 信号冲突检测和解决
    3. 信号聚合和融合
    4. 信号过滤和验证
    5. 信号优先级排序
    """
    
    def __init__(
        self,
        conflict_resolution: SignalConflictResolution = SignalConflictResolution.WEIGHTED_AVERAGE,
        aggregation_method: SignalAggregationMethod = SignalAggregationMethod.WEIGHTED_AVERAGE,
        signal_timeout: int = 300,  # 信号超时时间（秒）
        max_signals_per_asset: int = 10,  # 每个资产最大信号数
        min_confidence_threshold: float = 0.3  # 最小置信度阈值
    ):
        super().__init__()
        self.conflict_resolution = conflict_resolution
        self.aggregation_method = aggregation_method
        self.signal_timeout = signal_timeout
        self.max_signals_per_asset = max_signals_per_asset
        self.min_confidence_threshold = min_confidence_threshold
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 信号缓存
        self.signal_buffer: Dict[str, deque] = defaultdict(lambda: deque(maxlen=self.max_signals_per_asset))
        self.processed_signals: Dict[str, ProcessedSignal] = {}
        
        # 策略权重
        self.strategy_weights: Dict[str, float] = {}
        
        # 统计信息
        self.signal_stats = {
            'total_received': 0,
            'total_processed': 0,
            'conflicts_resolved': 0,
            'signals_filtered': 0
        }
        
        self.logger.info("Signal Processor initialized")
    
    async def add_signal(self, signal: Signal, strategy_id: str) -> None:
        """添加新信号
        
        Args:
            signal: 交易信号
            strategy_id: 策略ID
        """
        try:
            # 验证信号
            if not self._validate_signal(signal):
                self.logger.warning(f"Invalid signal rejected: {signal.signal_id}")
                self.signal_stats['signals_filtered'] += 1
                return
            
            # 检查信号是否过期
            if self._is_signal_expired(signal):
                self.logger.warning(f"Expired signal rejected: {signal.signal_id}")
                self.signal_stats['signals_filtered'] += 1
                return
            
            # 添加策略信息到信号元数据
            signal.metadata = signal.metadata or {}
            signal.metadata['strategy_id'] = strategy_id
            signal.metadata['received_at'] = datetime.now().isoformat()
            
            # 添加到缓存
            self.signal_buffer[signal.asset].append(signal)
            self.signal_stats['total_received'] += 1
            
            self.logger.debug(f"Signal added: {signal.signal_id} from {strategy_id}")
            
            # 触发信号处理
            await self._process_signals_for_asset(signal.asset)
            
        except Exception as e:
            self.logger.error(f"Error adding signal: {e}")
            raise SignalProcessorException(f"Failed to add signal: {e}")
    
    async def process_all_signals(self) -> Dict[str, ProcessedSignal]:
        """处理所有资产的信号
        
        Returns:
            处理后的信号字典
        """
        try:
            # 清理过期信号
            await self._cleanup_expired_signals()
            
            # 处理每个资产的信号
            for asset in list(self.signal_buffer.keys()):
                await self._process_signals_for_asset(asset)
            
            return self.processed_signals.copy()
            
        except Exception as e:
            self.logger.error(f"Error processing all signals: {e}")
            return {}
    
    async def get_processed_signal(self, asset: str) -> Optional[ProcessedSignal]:
        """获取指定资产的处理后信号"""
        return self.processed_signals.get(asset)
    
    async def get_all_processed_signals(self) -> Dict[str, ProcessedSignal]:
        """获取所有处理后的信号"""
        return self.processed_signals.copy()
    
    def set_strategy_weight(self, strategy_id: str, weight: float) -> None:
        """设置策略权重"""
        if 0.0 <= weight <= 1.0:
            self.strategy_weights[strategy_id] = weight
            self.logger.info(f"Strategy weight set: {strategy_id} = {weight}")
        else:
            self.logger.warning(f"Invalid weight for strategy {strategy_id}: {weight}")
    
    def get_strategy_weights(self) -> Dict[str, float]:
        """获取策略权重"""
        return self.strategy_weights.copy()
    
    async def _process_signals_for_asset(self, asset: str) -> None:
        """处理指定资产的信号"""
        try:
            signals = list(self.signal_buffer[asset])
            if not signals:
                return
            
            # 过滤有效信号
            valid_signals = [s for s in signals if not self._is_signal_expired(s)]
            
            if not valid_signals:
                # 清空该资产的处理结果
                if asset in self.processed_signals:
                    del self.processed_signals[asset]
                return
            
            # 检测信号冲突
            conflicts = self._detect_conflicts(valid_signals)
            
            if conflicts:
                self.signal_stats['conflicts_resolved'] += 1
                self.logger.debug(f"Signal conflicts detected for {asset}: {len(conflicts)} groups")
            
            # 解决冲突并聚合信号
            processed_signal = await self._aggregate_signals(valid_signals)
            
            if processed_signal:
                self.processed_signals[asset] = processed_signal
                self.signal_stats['total_processed'] += 1
                self.logger.debug(f"Signal processed for {asset}: {processed_signal.signal_type.value}")
            
        except Exception as e:
            self.logger.error(f"Error processing signals for {asset}: {e}")
    
    def _validate_signal(self, signal: Signal) -> bool:
        """验证信号有效性"""
        try:
            # 基本字段检查
            if not signal.asset or not signal.signal_id:
                return False
            
            # 数量检查
            if signal.quantity <= 0:
                return False
            
            # 价格检查
            if signal.price <= 0:
                return False
            
            # 置信度检查
            if signal.confidence < self.min_confidence_threshold:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _is_signal_expired(self, signal: Signal) -> bool:
        """检查信号是否过期"""
        if not signal.timestamp:
            return True
        
        age = (datetime.now() - signal.timestamp).total_seconds()
        return age > self.signal_timeout
    
    def _detect_conflicts(self, signals: List[Signal]) -> List[List[Signal]]:
        """检测信号冲突
        
        Returns:
            冲突信号组列表
        """
        if len(signals) <= 1:
            return []
        
        conflicts = []
        
        # 按信号类型分组
        signal_groups = defaultdict(list)
        for signal in signals:
            signal_groups[signal.signal_type].append(signal)
        
        # 检查买卖信号冲突
        buy_signals = signal_groups.get(SignalType.BUY, [])
        sell_signals = signal_groups.get(SignalType.SELL, [])
        
        if buy_signals and sell_signals:
            conflicts.append(buy_signals + sell_signals)
        
        # 检查同类型信号的参数冲突
        for signal_type, group_signals in signal_groups.items():
            if len(group_signals) > 1:
                # 检查价格差异
                prices = [float(s.price) for s in group_signals]
                if max(prices) - min(prices) > min(prices) * 0.05:  # 5%价格差异
                    conflicts.append(group_signals)
        
        return conflicts
    
    async def _aggregate_signals(self, signals: List[Signal]) -> Optional[ProcessedSignal]:
        """聚合信号"""
        if not signals:
            return None
        
        if len(signals) == 1:
            # 单个信号直接转换
            signal = signals[0]
            return ProcessedSignal(
                asset=signal.asset,
                signal_type=signal.signal_type,
                strength=signal.strength,
                quantity=signal.quantity,
                price=signal.price,
                position_type=signal.position_type,
                confidence=signal.confidence,
                source_signals=signals,
                metadata=signal.metadata,
                timestamp=signal.timestamp
            )
        
        # 多个信号需要聚合
        if self.aggregation_method == SignalAggregationMethod.SIMPLE_AVERAGE:
            return await self._simple_average_aggregation(signals)
        elif self.aggregation_method == SignalAggregationMethod.WEIGHTED_AVERAGE:
            return await self._weighted_average_aggregation(signals)
        elif self.aggregation_method == SignalAggregationMethod.VOTING:
            return await self._voting_aggregation(signals)
        else:
            # 默认使用加权平均
            return await self._weighted_average_aggregation(signals)
    
    async def _simple_average_aggregation(self, signals: List[Signal]) -> Optional[ProcessedSignal]:
        """简单平均聚合"""
        if not signals:
            return None
        
        # 按信号类型分组
        signal_groups = defaultdict(list)
        for signal in signals:
            signal_groups[signal.signal_type].append(signal)
        
        # 选择最多的信号类型
        dominant_type = max(signal_groups.keys(), key=lambda k: len(signal_groups[k]))
        dominant_signals = signal_groups[dominant_type]
        
        # 计算平均值
        avg_quantity = sum(s.quantity for s in dominant_signals) / len(dominant_signals)
        avg_price = sum(s.price for s in dominant_signals) / len(dominant_signals)
        avg_confidence = sum(s.confidence for s in dominant_signals) / len(dominant_signals)
        
        # 确定强度
        strengths = [s.strength for s in dominant_signals]
        avg_strength = self._average_strength(strengths)
        
        # 确定仓位类型
        position_type = self._determine_position_type(dominant_type)
        
        return ProcessedSignal(
            asset=signals[0].asset,
            signal_type=dominant_type,
            strength=avg_strength,
            quantity=avg_quantity,
            price=avg_price,
            position_type=position_type,
            confidence=avg_confidence,
            source_signals=signals,
            timestamp=max(s.timestamp for s in signals)
        )
    
    async def _weighted_average_aggregation(self, signals: List[Signal]) -> Optional[ProcessedSignal]:
        """加权平均聚合"""
        if not signals:
            return None
        
        # 计算权重
        weights = []
        for signal in signals:
            strategy_id = signal.metadata.get('strategy_id', 'unknown')
            strategy_weight = self.strategy_weights.get(strategy_id, 1.0)
            confidence_weight = signal.confidence
            strength_weight = self._strength_to_weight(signal.strength)
            
            total_weight = strategy_weight * confidence_weight * strength_weight
            weights.append(total_weight)
        
        # 归一化权重
        total_weight = sum(weights)
        if total_weight == 0:
            return await self._simple_average_aggregation(signals)
        
        normalized_weights = [w / total_weight for w in weights]
        
        # 按信号类型分组并计算加权平均
        signal_groups = defaultdict(list)
        weight_groups = defaultdict(list)
        
        for signal, weight in zip(signals, normalized_weights):
            signal_groups[signal.signal_type].append(signal)
            weight_groups[signal.signal_type].append(weight)
        
        # 选择权重最大的信号类型
        type_weights = {}
        for signal_type in signal_groups:
            type_weights[signal_type] = sum(weight_groups[signal_type])
        
        dominant_type = max(type_weights.keys(), key=lambda k: type_weights[k])
        dominant_signals = signal_groups[dominant_type]
        dominant_weights = weight_groups[dominant_type]
        
        # 重新归一化权重
        total_dominant_weight = sum(dominant_weights)
        if total_dominant_weight == 0:
            return await self._simple_average_aggregation(dominant_signals)
        
        normalized_dominant_weights = [w / total_dominant_weight for w in dominant_weights]
        
        # 计算加权平均
        weighted_quantity = sum(
            s.quantity * w for s, w in zip(dominant_signals, normalized_dominant_weights)
        )
        weighted_price = sum(
            s.price * w for s, w in zip(dominant_signals, normalized_dominant_weights)
        )
        weighted_confidence = sum(
            s.confidence * w for s, w in zip(dominant_signals, normalized_dominant_weights)
        )
        
        # 确定强度（基于权重）
        weighted_strength_values = [
            self._strength_to_value(s.strength) * w 
            for s, w in zip(dominant_signals, normalized_dominant_weights)
        ]
        avg_strength_value = sum(weighted_strength_values)
        avg_strength = self._value_to_strength(avg_strength_value)
        
        # 确定仓位类型
        position_type = self._determine_position_type(dominant_type)
        
        return ProcessedSignal(
            asset=signals[0].asset,
            signal_type=dominant_type,
            strength=avg_strength,
            quantity=weighted_quantity,
            price=weighted_price,
            position_type=position_type,
            confidence=weighted_confidence,
            source_signals=signals,
            timestamp=max(s.timestamp for s in signals)
        )
    
    async def _voting_aggregation(self, signals: List[Signal]) -> Optional[ProcessedSignal]:
        """投票聚合"""
        if not signals:
            return None
        
        # 统计投票
        votes = defaultdict(int)
        signal_groups = defaultdict(list)
        
        for signal in signals:
            strategy_id = signal.metadata.get('strategy_id', 'unknown')
            weight = self.strategy_weights.get(strategy_id, 1.0)
            
            votes[signal.signal_type] += weight
            signal_groups[signal.signal_type].append(signal)
        
        # 选择得票最多的信号类型
        winning_type = max(votes.keys(), key=lambda k: votes[k])
        winning_signals = signal_groups[winning_type]
        
        # 对获胜信号进行简单平均
        return await self._simple_average_aggregation(winning_signals)
    
    def _strength_to_weight(self, strength: SignalStrength) -> float:
        """将信号强度转换为权重"""
        strength_weights = {
            SignalStrength.WEAK: 0.5,
            SignalStrength.MEDIUM: 1.0,
            SignalStrength.STRONG: 1.5,
            SignalStrength.VERY_STRONG: 2.0
        }
        return strength_weights.get(strength, 1.0)
    
    def _strength_to_value(self, strength: SignalStrength) -> float:
        """将信号强度转换为数值"""
        strength_values = {
            SignalStrength.WEAK: 1.0,
            SignalStrength.MEDIUM: 2.0,
            SignalStrength.STRONG: 3.0,
            SignalStrength.VERY_STRONG: 4.0
        }
        return strength_values.get(strength, 2.0)
    
    def _value_to_strength(self, value: float) -> SignalStrength:
        """将数值转换为信号强度"""
        if value < 1.5:
            return SignalStrength.WEAK
        elif value < 2.5:
            return SignalStrength.MEDIUM
        elif value < 3.5:
            return SignalStrength.STRONG
        else:
            return SignalStrength.VERY_STRONG
    
    def _average_strength(self, strengths: List[SignalStrength]) -> SignalStrength:
        """计算平均强度"""
        if not strengths:
            return SignalStrength.MEDIUM
        
        values = [self._strength_to_value(s) for s in strengths]
        avg_value = sum(values) / len(values)
        return self._value_to_strength(avg_value)
    
    def _determine_position_type(self, signal_type: SignalType) -> PositionType:
        """根据信号类型确定仓位类型"""
        if signal_type == SignalType.BUY:
            return PositionType.LONG
        elif signal_type == SignalType.SELL:
            return PositionType.SHORT
        else:
            return PositionType.CLOSE
    
    async def _cleanup_expired_signals(self) -> None:
        """清理过期信号"""
        try:
            for asset in list(self.signal_buffer.keys()):
                signals = self.signal_buffer[asset]
                
                # 过滤掉过期信号
                valid_signals = deque(
                    [s for s in signals if not self._is_signal_expired(s)],
                    maxlen=self.max_signals_per_asset
                )
                
                if len(valid_signals) != len(signals):
                    self.signal_buffer[asset] = valid_signals
                    expired_count = len(signals) - len(valid_signals)
                    self.signal_stats['signals_filtered'] += expired_count
                    self.logger.debug(f"Cleaned {expired_count} expired signals for {asset}")
                
                # 如果没有有效信号，清空处理结果
                if not valid_signals and asset in self.processed_signals:
                    del self.processed_signals[asset]
        
        except Exception as e:
            self.logger.error(f"Error cleaning expired signals: {e}")
    
    def get_signal_stats(self) -> Dict[str, Any]:
        """获取信号统计信息"""
        stats = self.signal_stats.copy()
        stats.update({
            'buffered_signals': sum(len(signals) for signals in self.signal_buffer.values()),
            'processed_assets': len(self.processed_signals),
            'active_strategies': len(self.strategy_weights)
        })
        return stats
    
    def clear_signals(self, asset: Optional[str] = None) -> None:
        """清空信号
        
        Args:
            asset: 指定资产，如果为None则清空所有
        """
        if asset:
            if asset in self.signal_buffer:
                self.signal_buffer[asset].clear()
            if asset in self.processed_signals:
                del self.processed_signals[asset]
            self.logger.info(f"Signals cleared for asset: {asset}")
        else:
            self.signal_buffer.clear()
            self.processed_signals.clear()
            self.logger.info("All signals cleared")
    
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.signal_stats = {
            'total_received': 0,
            'total_processed': 0,
            'conflicts_resolved': 0,
            'signals_filtered': 0
        }
        self.logger.info("Signal statistics reset")