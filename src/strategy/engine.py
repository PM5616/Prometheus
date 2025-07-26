"""Strategy Engine Module

策略引擎模块，负责管理多个策略的运行、调度和协调。

主要功能：
- 策略生命周期管理
- 策略调度和执行
- 数据分发和路由
- 性能监控和统计
- 风险控制和限制
"""

import asyncio
import threading
import time
from typing import Dict, List, Optional, Callable, Any, Union
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
import json

from .base import BaseStrategy, StrategySignal, StrategyState
from ..common.logging import get_logger
from ..common.models.market_data import Kline, Ticker, OrderBook
from ..common.exceptions.strategy import StrategyEngineError, StrategyNotFoundError
from ..datahub.data_manager import DataManager


class StrategyEngine:
    """策略引擎
    
    管理多个策略的运行、调度和协调。
    """
    
    def __init__(self, 
                 data_manager: DataManager = None,
                 config: Dict = None):
        """初始化策略引擎
        
        Args:
            data_manager: 数据管理器
            config: 引擎配置
        """
        self.data_manager = data_manager
        self.config = config or {}
        
        # 策略管理
        self.strategies: Dict[str, BaseStrategy] = {}
        self.strategy_threads: Dict[str, threading.Thread] = {}
        
        # 数据队列
        self.data_queue = Queue(maxsize=self.config.get('max_queue_size', 10000))
        
        # 引擎状态
        self.is_running = False
        self.is_paused = False
        self.started_at = None
        self.stopped_at = None
        
        # 线程池
        max_workers = self.config.get('max_workers', 10)
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # 事件循环
        self.loop = None
        self.loop_thread = None
        
        # 日志记录
        self.logger = get_logger("StrategyEngine")
        
        # 性能统计
        self.stats = {
            'total_strategies': 0,
            'running_strategies': 0,
            'total_signals': 0,
            'processed_data_count': 0,
            'processing_time': 0.0,
            'error_count': 0,
            'last_update_time': None
        }
        
        # 回调函数
        self.on_signal_callbacks = []
        self.on_strategy_state_change_callbacks = []
        self.on_error_callbacks = []
        
        # 风险控制
        self.risk_manager = None
        self.global_risk_limits = self.config.get('global_risk_limits', {})
        
        # 数据订阅
        self.subscribed_symbols = set()
        
        self.logger.info("策略引擎初始化完成")
    
    def add_strategy(self, strategy: BaseStrategy) -> bool:
        """添加策略
        
        Args:
            strategy: 策略实例
            
        Returns:
            bool: 添加是否成功
        """
        try:
            if strategy.name in self.strategies:
                self.logger.error(f"策略 {strategy.name} 已存在")
                return False
            
            # 初始化策略
            if not strategy.initialize():
                self.logger.error(f"策略 {strategy.name} 初始化失败")
                return False
            
            # 添加回调函数
            strategy.add_signal_callback(self._on_strategy_signal)
            strategy.add_error_callback(self._on_strategy_error)
            
            # 注册策略
            self.strategies[strategy.name] = strategy
            self.stats['total_strategies'] += 1
            
            # 订阅数据
            self._subscribe_strategy_data(strategy)
            
            self.logger.info(f"策略 {strategy.name} 添加成功")
            return True
            
        except Exception as e:
            self.logger.error(f"添加策略失败: {e}")
            return False
    
    def remove_strategy(self, strategy_name: str) -> bool:
        """移除策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            bool: 移除是否成功
        """
        try:
            if strategy_name not in self.strategies:
                self.logger.error(f"策略 {strategy_name} 不存在")
                return False
            
            strategy = self.strategies[strategy_name]
            
            # 停止策略
            if strategy.state == StrategyState.RUNNING:
                strategy.stop()
            
            # 取消数据订阅
            self._unsubscribe_strategy_data(strategy)
            
            # 移除策略
            del self.strategies[strategy_name]
            self.stats['total_strategies'] -= 1
            
            # 停止策略线程
            if strategy_name in self.strategy_threads:
                thread = self.strategy_threads[strategy_name]
                if thread.is_alive():
                    thread.join(timeout=5.0)
                del self.strategy_threads[strategy_name]
            
            self.logger.info(f"策略 {strategy_name} 移除成功")
            return True
            
        except Exception as e:
            self.logger.error(f"移除策略失败: {e}")
            return False
    
    def get_strategy(self, strategy_name: str) -> Optional[BaseStrategy]:
        """获取策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            Optional[BaseStrategy]: 策略实例
        """
        return self.strategies.get(strategy_name)
    
    def list_strategies(self) -> List[str]:
        """列出所有策略名称
        
        Returns:
            List[str]: 策略名称列表
        """
        return list(self.strategies.keys())
    
    def start_strategy(self, strategy_name: str) -> bool:
        """启动策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            bool: 启动是否成功
        """
        try:
            if strategy_name not in self.strategies:
                raise StrategyNotFoundError(f"策略 {strategy_name} 不存在")
            
            strategy = self.strategies[strategy_name]
            
            if not strategy.start():
                return False
            
            # 更新统计
            self.stats['running_strategies'] += 1
            
            # 触发状态变化回调
            self._on_strategy_state_change(strategy_name, StrategyState.RUNNING)
            
            self.logger.info(f"策略 {strategy_name} 启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"启动策略失败: {e}")
            return False
    
    def stop_strategy(self, strategy_name: str) -> bool:
        """停止策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            bool: 停止是否成功
        """
        try:
            if strategy_name not in self.strategies:
                raise StrategyNotFoundError(f"策略 {strategy_name} 不存在")
            
            strategy = self.strategies[strategy_name]
            
            if strategy.state == StrategyState.RUNNING:
                self.stats['running_strategies'] -= 1
            
            if not strategy.stop():
                return False
            
            # 触发状态变化回调
            self._on_strategy_state_change(strategy_name, StrategyState.STOPPED)
            
            self.logger.info(f"策略 {strategy_name} 停止成功")
            return True
            
        except Exception as e:
            self.logger.error(f"停止策略失败: {e}")
            return False
    
    def pause_strategy(self, strategy_name: str) -> bool:
        """暂停策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            bool: 暂停是否成功
        """
        try:
            if strategy_name not in self.strategies:
                raise StrategyNotFoundError(f"策略 {strategy_name} 不存在")
            
            strategy = self.strategies[strategy_name]
            
            if not strategy.pause():
                return False
            
            # 触发状态变化回调
            self._on_strategy_state_change(strategy_name, StrategyState.PAUSED)
            
            self.logger.info(f"策略 {strategy_name} 暂停成功")
            return True
            
        except Exception as e:
            self.logger.error(f"暂停策略失败: {e}")
            return False
    
    def resume_strategy(self, strategy_name: str) -> bool:
        """恢复策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            bool: 恢复是否成功
        """
        try:
            if strategy_name not in self.strategies:
                raise StrategyNotFoundError(f"策略 {strategy_name} 不存在")
            
            strategy = self.strategies[strategy_name]
            
            if not strategy.resume():
                return False
            
            # 触发状态变化回调
            self._on_strategy_state_change(strategy_name, StrategyState.RUNNING)
            
            self.logger.info(f"策略 {strategy_name} 恢复成功")
            return True
            
        except Exception as e:
            self.logger.error(f"恢复策略失败: {e}")
            return False
    
    def start_all_strategies(self) -> bool:
        """启动所有策略
        
        Returns:
            bool: 启动是否成功
        """
        success = True
        for strategy_name in self.strategies:
            if not self.start_strategy(strategy_name):
                success = False
        return success
    
    def stop_all_strategies(self) -> bool:
        """停止所有策略
        
        Returns:
            bool: 停止是否成功
        """
        success = True
        for strategy_name in self.strategies:
            if not self.stop_strategy(strategy_name):
                success = False
        return success
    
    def start_engine(self) -> bool:
        """启动引擎
        
        Returns:
            bool: 启动是否成功
        """
        try:
            if self.is_running:
                self.logger.warning("引擎已在运行中")
                return True
            
            self.is_running = True
            self.is_paused = False
            self.started_at = datetime.now()
            
            # 启动事件循环
            self._start_event_loop()
            
            # 启动数据处理线程
            self._start_data_processing()
            
            # 启动所有策略
            self.start_all_strategies()
            
            self.logger.info("策略引擎启动成功")
            return True
            
        except Exception as e:
            self.logger.error(f"启动引擎失败: {e}")
            self.is_running = False
            return False
    
    def stop_engine(self) -> bool:
        """停止引擎
        
        Returns:
            bool: 停止是否成功
        """
        try:
            if not self.is_running:
                self.logger.warning("引擎未在运行中")
                return True
            
            self.is_running = False
            self.stopped_at = datetime.now()
            
            # 停止所有策略
            self.stop_all_strategies()
            
            # 停止事件循环
            self._stop_event_loop()
            
            # 关闭线程池
            self.executor.shutdown(wait=True)
            
            self.logger.info("策略引擎停止成功")
            return True
            
        except Exception as e:
            self.logger.error(f"停止引擎失败: {e}")
            return False
    
    def pause_engine(self) -> bool:
        """暂停引擎
        
        Returns:
            bool: 暂停是否成功
        """
        try:
            if not self.is_running or self.is_paused:
                self.logger.warning("引擎状态错误")
                return False
            
            self.is_paused = True
            
            # 暂停所有运行中的策略
            for strategy in self.strategies.values():
                if strategy.state == StrategyState.RUNNING:
                    strategy.pause()
            
            self.logger.info("策略引擎暂停成功")
            return True
            
        except Exception as e:
            self.logger.error(f"暂停引擎失败: {e}")
            return False
    
    def resume_engine(self) -> bool:
        """恢复引擎
        
        Returns:
            bool: 恢复是否成功
        """
        try:
            if not self.is_running or not self.is_paused:
                self.logger.warning("引擎状态错误")
                return False
            
            self.is_paused = False
            
            # 恢复所有暂停的策略
            for strategy in self.strategies.values():
                if strategy.state == StrategyState.PAUSED:
                    strategy.resume()
            
            self.logger.info("策略引擎恢复成功")
            return True
            
        except Exception as e:
            self.logger.error(f"恢复引擎失败: {e}")
            return False
    
    def feed_data(self, symbol: str, data: Union[Kline, Ticker, OrderBook]):
        """向引擎输入市场数据
        
        Args:
            symbol: 交易对符号
            data: 市场数据
        """
        try:
            if not self.is_running or self.is_paused:
                return
            
            # 将数据放入队列
            try:
                self.data_queue.put_nowait((symbol, data))
            except:
                # 队列满时丢弃最旧的数据
                try:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait((symbol, data))
                except Empty:
                    pass
            
        except Exception as e:
            self.logger.error(f"输入数据失败: {e}")
    
    def _start_event_loop(self):
        """启动事件循环"""
        def run_loop():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
            self.loop.run_forever()
        
        self.loop_thread = threading.Thread(target=run_loop, daemon=True)
        self.loop_thread.start()
    
    def _stop_event_loop(self):
        """停止事件循环"""
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        
        if self.loop_thread and self.loop_thread.is_alive():
            self.loop_thread.join(timeout=5.0)
    
    def _start_data_processing(self):
        """启动数据处理线程"""
        def process_data():
            while self.is_running:
                try:
                    # 从队列获取数据
                    symbol, data = self.data_queue.get(timeout=1.0)
                    
                    if self.is_paused:
                        continue
                    
                    # 分发数据到相关策略
                    self._distribute_data(symbol, data)
                    
                    # 更新统计
                    self.stats['processed_data_count'] += 1
                    self.stats['last_update_time'] = datetime.now()
                    
                except Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"处理数据时发生错误: {e}")
                    self.stats['error_count'] += 1
        
        processing_thread = threading.Thread(target=process_data, daemon=True)
        processing_thread.start()
    
    def _distribute_data(self, symbol: str, data: Union[Kline, Ticker, OrderBook]):
        """分发数据到相关策略
        
        Args:
            symbol: 交易对符号
            data: 市场数据
        """
        start_time = time.time()
        
        # 找到订阅此交易对的策略
        relevant_strategies = [
            strategy for strategy in self.strategies.values()
            if symbol in strategy.symbols and strategy.state == StrategyState.RUNNING
        ]
        
        # 并行处理策略
        futures = []
        for strategy in relevant_strategies:
            future = self.executor.submit(strategy.process_market_data, symbol, data)
            futures.append(future)
        
        # 等待所有策略处理完成
        for future in futures:
            try:
                future.result(timeout=self.config.get('strategy_timeout', 5.0))
            except Exception as e:
                self.logger.error(f"策略处理数据超时或出错: {e}")
        
        # 更新处理时间统计
        processing_time = time.time() - start_time
        self.stats['processing_time'] += processing_time
    
    def _subscribe_strategy_data(self, strategy: BaseStrategy):
        """订阅策略所需的数据
        
        Args:
            strategy: 策略实例
        """
        if not self.data_manager:
            return
        
        for symbol in strategy.symbols:
            if symbol not in self.subscribed_symbols:
                try:
                    # 订阅实时数据
                    self.data_manager.subscribe_ticker(symbol, self._on_ticker_data)
                    self.data_manager.subscribe_kline(symbol, '1m', self._on_kline_data)
                    
                    self.subscribed_symbols.add(symbol)
                    self.logger.info(f"订阅数据: {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"订阅数据失败 {symbol}: {e}")
    
    def _unsubscribe_strategy_data(self, strategy: BaseStrategy):
        """取消订阅策略数据
        
        Args:
            strategy: 策略实例
        """
        if not self.data_manager:
            return
        
        for symbol in strategy.symbols:
            # 检查是否还有其他策略需要此数据
            other_strategies_need = any(
                symbol in s.symbols for s in self.strategies.values() 
                if s != strategy
            )
            
            if not other_strategies_need and symbol in self.subscribed_symbols:
                try:
                    # 取消订阅
                    self.data_manager.unsubscribe_ticker(symbol)
                    self.data_manager.unsubscribe_kline(symbol, '1m')
                    
                    self.subscribed_symbols.remove(symbol)
                    self.logger.info(f"取消订阅数据: {symbol}")
                    
                except Exception as e:
                    self.logger.error(f"取消订阅数据失败 {symbol}: {e}")
    
    def _on_ticker_data(self, symbol: str, ticker: Ticker):
        """处理行情数据
        
        Args:
            symbol: 交易对符号
            ticker: 行情数据
        """
        self.feed_data(symbol, ticker)
    
    def _on_kline_data(self, symbol: str, kline: Kline):
        """处理K线数据
        
        Args:
            symbol: 交易对符号
            kline: K线数据
        """
        self.feed_data(symbol, kline)
    
    def _on_strategy_signal(self, signal: StrategySignal):
        """处理策略信号
        
        Args:
            signal: 策略信号
        """
        try:
            # 全局风险检查
            if not self._check_global_risk_limits(signal):
                self.logger.warning(f"信号 {signal} 被全局风险控制拒绝")
                return
            
            # 更新统计
            self.stats['total_signals'] += 1
            
            # 调用回调函数
            for callback in self.on_signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    self.logger.error(f"信号回调函数执行失败: {e}")
            
            self.logger.info(f"引擎处理信号: {signal}")
            
        except Exception as e:
            self.logger.error(f"处理策略信号失败: {e}")
    
    def _on_strategy_error(self, error: Exception):
        """处理策略错误
        
        Args:
            error: 异常对象
        """
        self.stats['error_count'] += 1
        
        # 调用错误回调函数
        for callback in self.on_error_callbacks:
            try:
                callback(error)
            except Exception as e:
                self.logger.error(f"错误回调函数执行失败: {e}")
    
    def _on_strategy_state_change(self, strategy_name: str, new_state: StrategyState):
        """处理策略状态变化
        
        Args:
            strategy_name: 策略名称
            new_state: 新状态
        """
        # 调用状态变化回调函数
        for callback in self.on_strategy_state_change_callbacks:
            try:
                callback(strategy_name, new_state)
            except Exception as e:
                self.logger.error(f"状态变化回调函数执行失败: {e}")
    
    def _check_global_risk_limits(self, signal: StrategySignal) -> bool:
        """检查全局风险限制
        
        Args:
            signal: 策略信号
            
        Returns:
            bool: 是否通过风险检查
        """
        try:
            # 检查全局信号频率
            max_signals_per_minute = self.global_risk_limits.get('max_signals_per_minute')
            if max_signals_per_minute:
                recent_time = datetime.now() - timedelta(minutes=1)
                recent_signals = sum(
                    1 for strategy in self.strategies.values()
                    for s in strategy.signals
                    if s.timestamp > recent_time
                )
                if recent_signals >= max_signals_per_minute:
                    return False
            
            # 检查全局持仓限制
            max_total_position = self.global_risk_limits.get('max_total_position')
            if max_total_position and signal.quantity:
                total_position = sum(
                    abs(strategy.get_position(signal.symbol).get('quantity', 0))
                    for strategy in self.strategies.values()
                )
                if total_position + abs(signal.quantity) > max_total_position:
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"全局风险检查失败: {e}")
            return False
    
    def add_signal_callback(self, callback: Callable[[StrategySignal], None]):
        """添加信号回调函数
        
        Args:
            callback: 回调函数
        """
        self.on_signal_callbacks.append(callback)
    
    def add_strategy_state_change_callback(self, callback: Callable[[str, StrategyState], None]):
        """添加策略状态变化回调函数
        
        Args:
            callback: 回调函数
        """
        self.on_strategy_state_change_callbacks.append(callback)
    
    def add_error_callback(self, callback: Callable[[Exception], None]):
        """添加错误回调函数
        
        Args:
            callback: 回调函数
        """
        self.on_error_callbacks.append(callback)
    
    def get_engine_stats(self) -> Dict:
        """获取引擎统计信息
        
        Returns:
            Dict: 统计信息
        """
        stats = self.stats.copy()
        
        # 添加运行时间
        if self.started_at:
            if self.stopped_at:
                runtime = (self.stopped_at - self.started_at).total_seconds()
            else:
                runtime = (datetime.now() - self.started_at).total_seconds()
            stats['runtime_seconds'] = runtime
        
        # 添加状态信息
        stats.update({
            'is_running': self.is_running,
            'is_paused': self.is_paused,
            'subscribed_symbols': list(self.subscribed_symbols),
            'queue_size': self.data_queue.qsize(),
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'stopped_at': self.stopped_at.isoformat() if self.stopped_at else None
        })
        
        return stats
    
    def get_all_strategy_stats(self) -> Dict[str, Dict]:
        """获取所有策略的统计信息
        
        Returns:
            Dict[str, Dict]: 策略统计信息字典
        """
        return {
            name: strategy.get_stats()
            for name, strategy in self.strategies.items()
        }
    
    def get_all_signals(self, limit: int = None) -> List[StrategySignal]:
        """获取所有策略的信号
        
        Args:
            limit: 限制数量
            
        Returns:
            List[StrategySignal]: 信号列表
        """
        all_signals = []
        for strategy in self.strategies.values():
            all_signals.extend(strategy.get_signals())
        
        # 按时间排序
        all_signals.sort(key=lambda x: x.timestamp)
        
        if limit:
            all_signals = all_signals[-limit:]
        
        return all_signals
    
    def reset_all_stats(self):
        """重置所有统计信息"""
        # 重置引擎统计
        self.stats = {
            'total_strategies': len(self.strategies),
            'running_strategies': sum(1 for s in self.strategies.values() if s.state == StrategyState.RUNNING),
            'total_signals': 0,
            'processed_data_count': 0,
            'processing_time': 0.0,
            'error_count': 0,
            'last_update_time': None
        }
        
        # 重置所有策略统计
        for strategy in self.strategies.values():
            strategy.reset_stats()
        
        self.logger.info("所有统计信息已重置")
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 引擎字典
        """
        return {
            'config': self.config,
            'stats': self.get_engine_stats(),
            'strategies': {
                name: strategy.to_dict()
                for name, strategy in self.strategies.items()
            },
            'subscribed_symbols': list(self.subscribed_symbols),
            'global_risk_limits': self.global_risk_limits
        }
    
    def __str__(self) -> str:
        return f"StrategyEngine({len(self.strategies)} strategies, running={self.is_running})"
    
    def __repr__(self) -> str:
        return self.__str__()
    
    def __enter__(self):
        """上下文管理器入口"""
        self.start_engine()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.stop_engine()