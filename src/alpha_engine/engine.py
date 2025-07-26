"""Strategy Engine Module

策略引擎核心模块，负责策略的加载、运行、管理和信号生成。

主要功能：
- 策略生命周期管理
- 多策略并行运行
- 信号生成和分发
- 性能监控和统计
- 风险控制和限制
- 策略热更新
"""

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from decimal import Decimal
import time

from .base_strategy import BaseStrategy, StrategyConfig
from .signal import Signal, SignalType, SignalStrength, SignalSource
from src.common.models.market import MarketData, KlineData
from src.common.exceptions.strategy_exceptions import StrategyError, StrategyEngineError
from src.common.utils.performance import PerformanceTimer
from src.common.logging.logger import get_logger


@dataclass
class EngineConfig:
    """引擎配置"""
    max_strategies: int = 50                    # 最大策略数量
    max_workers: int = 10                       # 最大工作线程数
    signal_queue_size: int = 1000              # 信号队列大小
    update_interval: float = 1.0               # 更新间隔（秒）
    performance_log_interval: int = 300        # 性能日志间隔（秒）
    
    # 风险控制
    max_signals_per_second: int = 100          # 每秒最大信号数
    max_signals_per_strategy: int = 10         # 每个策略每次最大信号数
    
    # 资源限制
    max_memory_usage_mb: int = 1024            # 最大内存使用（MB）
    max_cpu_usage_pct: float = 80.0            # 最大CPU使用率
    
    # 监控配置
    enable_performance_monitoring: bool = True  # 启用性能监控
    enable_health_check: bool = True           # 启用健康检查
    health_check_interval: int = 60            # 健康检查间隔（秒）


@dataclass
class EngineStats:
    """引擎统计信息"""
    start_time: datetime = field(default_factory=datetime.now)
    total_strategies: int = 0
    active_strategies: int = 0
    total_signals_generated: int = 0
    total_signals_processed: int = 0
    total_errors: int = 0
    
    # 性能统计
    avg_signal_generation_time: float = 0.0
    avg_signal_processing_time: float = 0.0
    peak_memory_usage_mb: float = 0.0
    peak_cpu_usage_pct: float = 0.0
    
    # 最近统计
    signals_last_minute: int = 0
    signals_last_hour: int = 0
    errors_last_hour: int = 0
    
    def get_uptime(self) -> timedelta:
        """获取运行时间"""
        return datetime.now() - self.start_time
    
    def get_signal_rate(self) -> float:
        """获取信号生成率（每秒）"""
        uptime_seconds = self.get_uptime().total_seconds()
        if uptime_seconds == 0:
            return 0.0
        return self.total_signals_generated / uptime_seconds
    
    def get_error_rate(self) -> float:
        """获取错误率"""
        if self.total_signals_generated == 0:
            return 0.0
        return self.total_errors / self.total_signals_generated


class StrategyEngine:
    """策略引擎
    
    负责管理多个策略的运行，生成和分发交易信号。
    """
    
    def __init__(self, config: EngineConfig):
        """初始化策略引擎
        
        Args:
            config: 引擎配置
        """
        self.config = config
        self.logger = get_logger("strategy_engine")
        
        # 策略管理
        self._strategies: Dict[str, BaseStrategy] = {}
        self._strategy_configs: Dict[str, StrategyConfig] = {}
        
        # 运行状态
        self._is_running = False
        self._is_stopping = False
        
        # 线程管理
        self._executor = ThreadPoolExecutor(max_workers=config.max_workers)
        self._update_thread: Optional[threading.Thread] = None
        self._monitor_thread: Optional[threading.Thread] = None
        
        # 信号管理
        self._signal_queue: asyncio.Queue = asyncio.Queue(maxsize=config.signal_queue_size)
        self._signal_callbacks: List[Callable[[Signal], None]] = []
        
        # 数据缓存
        self._market_data_cache: Dict[str, MarketData] = {}
        self._kline_data_cache: Dict[str, Dict[str, List[KlineData]]] = {}
        
        # 统计信息
        self.stats = EngineStats()
        
        # 性能监控
        self._performance_timers: Dict[str, PerformanceTimer] = {}
        
        # 锁
        self._lock = threading.RLock()
        
        self.logger.info(f"策略引擎初始化完成，最大策略数: {config.max_strategies}")
    
    def add_strategy(self, strategy: BaseStrategy) -> bool:
        """添加策略
        
        Args:
            strategy: 策略实例
            
        Returns:
            是否添加成功
        """
        with self._lock:
            if len(self._strategies) >= self.config.max_strategies:
                self.logger.error(f"策略数量已达上限: {self.config.max_strategies}")
                return False
            
            strategy_name = strategy.config.name
            
            if strategy_name in self._strategies:
                self.logger.warning(f"策略已存在: {strategy_name}")
                return False
            
            try:
                self._strategies[strategy_name] = strategy
                self._strategy_configs[strategy_name] = strategy.config
                self.stats.total_strategies += 1
                
                self.logger.info(f"添加策略: {strategy_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"添加策略失败: {strategy_name}, 错误: {e}")
                return False
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """移除策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            是否移除成功
        """
        with self._lock:
            if strategy_name not in self._strategies:
                self.logger.warning(f"策略不存在: {strategy_name}")
                return False
            
            try:
                strategy = self._strategies[strategy_name]
                
                # 停止策略
                if strategy.state.is_running:
                    strategy.stop()
                
                # 移除策略
                del self._strategies[strategy_name]
                del self._strategy_configs[strategy_name]
                
                self.logger.info(f"移除策略: {strategy_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"移除策略失败: {strategy_name}, 错误: {e}")
                return False
    
    def start_strategy(self, strategy_name: str) -> bool:
        """启动策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            是否启动成功
        """
        with self._lock:
            if strategy_name not in self._strategies:
                self.logger.error(f"策略不存在: {strategy_name}")
                return False
            
            try:
                strategy = self._strategies[strategy_name]
                strategy.start()
                
                if strategy.state.is_running:
                    self.stats.active_strategies += 1
                
                self.logger.info(f"启动策略: {strategy_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"启动策略失败: {strategy_name}, 错误: {e}")
                return False
    
    def stop_strategy(self, strategy_name: str) -> bool:
        """停止策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            是否停止成功
        """
        with self._lock:
            if strategy_name not in self._strategies:
                self.logger.error(f"策略不存在: {strategy_name}")
                return False
            
            try:
                strategy = self._strategies[strategy_name]
                
                if strategy.state.is_running:
                    strategy.stop()
                    self.stats.active_strategies -= 1
                
                self.logger.info(f"停止策略: {strategy_name}")
                return True
                
            except Exception as e:
                self.logger.error(f"停止策略失败: {strategy_name}, 错误: {e}")
                return False
    
    def start(self) -> None:
        """启动引擎"""
        if self._is_running:
            self.logger.warning("策略引擎已在运行中")
            return
        
        self.logger.info("启动策略引擎")
        self._is_running = True
        self._is_stopping = False
        
        # 启动更新线程
        self._update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self._update_thread.start()
        
        # 启动监控线程
        if self.config.enable_performance_monitoring:
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
        
        # 启动所有已配置的策略
        for strategy_name in list(self._strategies.keys()):
            if self._strategy_configs[strategy_name].enabled:
                self.start_strategy(strategy_name)
    
    def stop(self) -> None:
        """停止引擎"""
        if not self._is_running:
            self.logger.warning("策略引擎未在运行")
            return
        
        self.logger.info("停止策略引擎")
        self._is_stopping = True
        
        # 停止所有策略
        for strategy_name in list(self._strategies.keys()):
            self.stop_strategy(strategy_name)
        
        # 等待线程结束
        if self._update_thread and self._update_thread.is_alive():
            self._update_thread.join(timeout=5.0)
        
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=5.0)
        
        # 关闭线程池
        self._executor.shutdown(wait=True)
        
        self._is_running = False
        self.logger.info("策略引擎已停止")
    
    def update_market_data(self, symbol: str, data: MarketData) -> None:
        """更新市场数据
        
        Args:
            symbol: 交易对
            data: 市场数据
        """
        self._market_data_cache[symbol] = data
        
        # 更新所有相关策略的数据
        for strategy in self._strategies.values():
            if symbol in strategy.config.symbols and strategy.state.is_running:
                try:
                    strategy.update_market_data(symbol, data)
                except Exception as e:
                    self.logger.error(f"更新策略市场数据失败: {strategy.config.name}, 错误: {e}")
    
    def update_kline_data(self, symbol: str, timeframe: str, klines: List[KlineData]) -> None:
        """更新K线数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            klines: K线数据列表
        """
        if symbol not in self._kline_data_cache:
            self._kline_data_cache[symbol] = {}
        
        self._kline_data_cache[symbol][timeframe] = klines
        
        # 更新所有相关策略的数据
        for strategy in self._strategies.values():
            if (symbol in strategy.config.symbols and 
                timeframe in strategy.config.timeframes and 
                strategy.state.is_running):
                try:
                    strategy.update_kline_data(symbol, timeframe, klines)
                except Exception as e:
                    self.logger.error(f"更新策略K线数据失败: {strategy.config.name}, 错误: {e}")
    
    def add_signal_callback(self, callback: Callable[[Signal], None]) -> None:
        """添加信号回调
        
        Args:
            callback: 信号回调函数
        """
        self._signal_callbacks.append(callback)
        self.logger.info(f"添加信号回调，当前回调数量: {len(self._signal_callbacks)}")
    
    def remove_signal_callback(self, callback: Callable[[Signal], None]) -> None:
        """移除信号回调
        
        Args:
            callback: 信号回调函数
        """
        if callback in self._signal_callbacks:
            self._signal_callbacks.remove(callback)
            self.logger.info(f"移除信号回调，当前回调数量: {len(self._signal_callbacks)}")
    
    def get_strategy_list(self) -> List[str]:
        """获取策略列表"""
        return list(self._strategies.keys())
    
    def get_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """获取策略实例
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略实例或None
        """
        return self._strategies.get(strategy_name)
    
    def get_engine_stats(self) -> Dict[str, Any]:
        """获取引擎统计信息"""
        return {
            'is_running': self._is_running,
            'uptime_seconds': self.stats.get_uptime().total_seconds(),
            'total_strategies': self.stats.total_strategies,
            'active_strategies': self.stats.active_strategies,
            'total_signals_generated': self.stats.total_signals_generated,
            'total_signals_processed': self.stats.total_signals_processed,
            'total_errors': self.stats.total_errors,
            'signal_rate': self.stats.get_signal_rate(),
            'error_rate': self.stats.get_error_rate(),
            'avg_signal_generation_time': self.stats.avg_signal_generation_time,
            'avg_signal_processing_time': self.stats.avg_signal_processing_time,
            'peak_memory_usage_mb': self.stats.peak_memory_usage_mb,
            'peak_cpu_usage_pct': self.stats.peak_cpu_usage_pct,
            'signals_last_minute': self.stats.signals_last_minute,
            'signals_last_hour': self.stats.signals_last_hour,
            'errors_last_hour': self.stats.errors_last_hour
        }
    
    def get_all_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有策略统计信息"""
        stats = {}
        for name, strategy in self._strategies.items():
            stats[name] = strategy.get_performance_metrics()
        return stats
    
    def _update_loop(self) -> None:
        """更新循环"""
        self.logger.info("策略引擎更新循环启动")
        
        while self._is_running and not self._is_stopping:
            try:
                start_time = time.time()
                
                # 生成信号
                self._generate_signals()
                
                # 处理信号队列
                self._process_signal_queue()
                
                # 更新统计信息
                self._update_stats()
                
                # 计算处理时间
                processing_time = time.time() - start_time
                
                # 等待下次更新
                sleep_time = max(0, self.config.update_interval - processing_time)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
            except Exception as e:
                self.logger.error(f"更新循环发生错误: {e}")
                self.stats.total_errors += 1
                time.sleep(1.0)  # 错误时短暂休眠
        
        self.logger.info("策略引擎更新循环结束")
    
    def _generate_signals(self) -> None:
        """生成信号"""
        if not self._market_data_cache:
            return
        
        # 并行生成信号
        futures = []
        
        for strategy in self._strategies.values():
            if not strategy.state.is_running:
                continue
            
            # 准备策略相关的市场数据
            strategy_data = {}
            for symbol in strategy.config.symbols:
                if symbol in self._market_data_cache:
                    strategy_data[symbol] = self._market_data_cache[symbol]
            
            if not strategy_data:
                continue
            
            # 提交信号生成任务
            future = self._executor.submit(self._generate_strategy_signals, strategy, strategy_data)
            futures.append((strategy.config.name, future))
        
        # 收集结果
        for strategy_name, future in futures:
            try:
                signals = future.result(timeout=5.0)  # 5秒超时
                
                if signals:
                    self.stats.total_signals_generated += len(signals)
                    
                    # 限制每个策略的信号数量
                    if len(signals) > self.config.max_signals_per_strategy:
                        signals = signals[:self.config.max_signals_per_strategy]
                        self.logger.warning(f"策略 {strategy_name} 信号数量超限，已截断")
                    
                    # 添加到信号队列
                    for signal in signals:
                        try:
                            self._signal_queue.put_nowait(signal)
                        except asyncio.QueueFull:
                            self.logger.warning("信号队列已满，丢弃信号")
                            break
                
            except Exception as e:
                self.logger.error(f"策略 {strategy_name} 信号生成失败: {e}")
                self.stats.total_errors += 1
    
    def _generate_strategy_signals(self, strategy: BaseStrategy, market_data: Dict[str, MarketData]) -> List[Signal]:
        """生成策略信号
        
        Args:
            strategy: 策略实例
            market_data: 市场数据
            
        Returns:
            信号列表
        """
        try:
            with PerformanceTimer() as timer:
                signals = strategy.generate_signals(market_data)
            
            # 更新性能统计
            self.stats.avg_signal_generation_time = (
                (self.stats.avg_signal_generation_time * 0.9) + 
                (timer.elapsed_time * 0.1)
            )
            
            return signals or []
            
        except Exception as e:
            self.logger.error(f"策略 {strategy.config.name} 信号生成异常: {e}")
            return []
    
    def _process_signal_queue(self) -> None:
        """处理信号队列"""
        processed_count = 0
        max_process_per_cycle = 100  # 每次循环最多处理的信号数
        
        while processed_count < max_process_per_cycle:
            try:
                signal = self._signal_queue.get_nowait()
                
                # 处理信号
                self._process_signal(signal)
                processed_count += 1
                self.stats.total_signals_processed += 1
                
            except asyncio.QueueEmpty:
                break
            except Exception as e:
                self.logger.error(f"处理信号时发生错误: {e}")
                self.stats.total_errors += 1
    
    def _process_signal(self, signal: Signal) -> None:
        """处理单个信号
        
        Args:
            signal: 交易信号
        """
        try:
            with PerformanceTimer() as timer:
                # 调用所有信号回调
                for callback in self._signal_callbacks:
                    try:
                        callback(signal)
                    except Exception as e:
                        self.logger.error(f"信号回调执行失败: {e}")
            
            # 更新性能统计
            self.stats.avg_signal_processing_time = (
                (self.stats.avg_signal_processing_time * 0.9) + 
                (timer.elapsed_time * 0.1)
            )
            
        except Exception as e:
            self.logger.error(f"处理信号失败: {signal}, 错误: {e}")
    
    def _update_stats(self) -> None:
        """更新统计信息"""
        # 更新活跃策略数量
        active_count = sum(1 for s in self._strategies.values() if s.state.is_running)
        self.stats.active_strategies = active_count
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        self.logger.info("策略引擎监控循环启动")
        
        while self._is_running and not self._is_stopping:
            try:
                # 健康检查
                if self.config.enable_health_check:
                    self._health_check()
                
                # 性能日志
                if self.config.enable_performance_monitoring:
                    self._log_performance()
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环发生错误: {e}")
                time.sleep(10.0)
        
        self.logger.info("策略引擎监控循环结束")
    
    def _health_check(self) -> None:
        """健康检查"""
        # 检查策略状态
        for name, strategy in self._strategies.items():
            if strategy.config.enabled and not strategy.state.is_running:
                self.logger.warning(f"策略 {name} 应该运行但未运行")
        
        # 检查信号队列
        queue_size = self._signal_queue.qsize()
        if queue_size > self.config.signal_queue_size * 0.8:
            self.logger.warning(f"信号队列接近满载: {queue_size}/{self.config.signal_queue_size}")
    
    def _log_performance(self) -> None:
        """记录性能日志"""
        stats = self.get_engine_stats()
        self.logger.info(
            f"引擎性能 - 运行时间: {stats['uptime_seconds']:.1f}s, "
            f"活跃策略: {stats['active_strategies']}, "
            f"信号生成率: {stats['signal_rate']:.2f}/s, "
            f"错误率: {stats['error_rate']:.4f}"
        )
    
    def __del__(self):
        """析构函数"""
        if self._is_running:
            self.stop()