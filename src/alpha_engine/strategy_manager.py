"""Strategy Manager Module

策略管理器模块，负责策略的生命周期管理、协调和监控。

主要功能：
- 策略生命周期管理
- 策略配置管理
- 策略性能监控
- 策略风险控制
- 策略热更新
- 策略依赖管理
"""

import asyncio
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from decimal import Decimal
import time
import json
import os

from .base_strategy import BaseStrategy, StrategyConfig, StrategyState
from .strategy_loader import StrategyLoader, StrategyMetadata
from .signal import Signal, SignalType, SignalStrength
from ..common.models.market_data import MarketData, KlineData
from ..common.exceptions.strategy_exceptions import StrategyError, StrategyManagerError
from ..common.utils.performance import PerformanceTimer
from ..common.logging.logger import get_logger


@dataclass
class StrategyInstance:
    """策略实例信息"""
    strategy: BaseStrategy
    config: StrategyConfig
    metadata: StrategyMetadata
    created_time: datetime = field(default_factory=datetime.now)
    last_update_time: datetime = field(default_factory=datetime.now)
    
    # 运行统计
    total_signals: int = 0
    successful_signals: int = 0
    failed_signals: int = 0
    total_runtime: float = 0.0
    avg_execution_time: float = 0.0
    
    # 性能指标
    total_return: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    
    def get_success_rate(self) -> float:
        """获取成功率"""
        if self.total_signals == 0:
            return 0.0
        return self.successful_signals / self.total_signals
    
    def get_uptime(self) -> timedelta:
        """获取运行时间"""
        return datetime.now() - self.created_time
    
    def update_performance(self, signal_success: bool, execution_time: float) -> None:
        """更新性能指标
        
        Args:
            signal_success: 信号是否成功
            execution_time: 执行时间
        """
        self.total_signals += 1
        if signal_success:
            self.successful_signals += 1
        else:
            self.failed_signals += 1
        
        self.total_runtime += execution_time
        self.avg_execution_time = self.total_runtime / self.total_signals
        self.last_update_time = datetime.now()


@dataclass
class ManagerConfig:
    """管理器配置"""
    max_strategies: int = 100                   # 最大策略数量
    auto_reload: bool = True                    # 自动重载
    reload_check_interval: int = 60             # 重载检查间隔（秒）
    performance_log_interval: int = 300         # 性能日志间隔（秒）
    
    # 风险控制
    max_signals_per_strategy: int = 10          # 每个策略最大信号数
    max_total_signals: int = 1000               # 总最大信号数
    signal_rate_limit: int = 100                # 信号速率限制（每秒）
    
    # 资源限制
    max_memory_per_strategy_mb: int = 100       # 每个策略最大内存（MB）
    max_cpu_per_strategy_pct: float = 10.0      # 每个策略最大CPU使用率
    
    # 监控配置
    enable_performance_monitoring: bool = True  # 启用性能监控
    enable_health_check: bool = True           # 启用健康检查
    health_check_interval: int = 30            # 健康检查间隔（秒）
    
    # 持久化配置
    save_state_interval: int = 600             # 状态保存间隔（秒）
    state_file_path: str = "strategy_states.json"  # 状态文件路径


class StrategyManager:
    """策略管理器
    
    负责策略的生命周期管理、协调和监控。
    """
    
    def __init__(self, config: ManagerConfig, strategy_loader: StrategyLoader):
        """初始化策略管理器
        
        Args:
            config: 管理器配置
            strategy_loader: 策略加载器
        """
        self.config = config
        self.strategy_loader = strategy_loader
        self.logger = get_logger("strategy_manager")
        
        # 策略实例管理
        self._strategy_instances: Dict[str, StrategyInstance] = {}
        self._strategy_groups: Dict[str, Set[str]] = {}  # 策略分组
        
        # 运行状态
        self._is_running = False
        self._is_stopping = False
        
        # 线程管理
        self._monitor_thread: Optional[threading.Thread] = None
        self._reload_thread: Optional[threading.Thread] = None
        self._save_state_thread: Optional[threading.Thread] = None
        
        # 信号管理
        self._signal_callbacks: List[Callable[[Signal], None]] = []
        self._signal_filters: List[Callable[[Signal], bool]] = []
        
        # 性能统计
        self._total_signals_generated = 0
        self._total_signals_processed = 0
        self._total_errors = 0
        self._start_time = datetime.now()
        
        # 锁
        self._lock = threading.RLock()
        
        self.logger.info(f"策略管理器初始化完成，最大策略数: {config.max_strategies}")
    
    def create_strategy(self, strategy_name: str, config: StrategyConfig) -> bool:
        """创建策略实例
        
        Args:
            strategy_name: 策略名称
            config: 策略配置
            
        Returns:
            是否创建成功
        """
        with self._lock:
            if len(self._strategy_instances) >= self.config.max_strategies:
                self.logger.error(f"策略数量已达上限: {self.config.max_strategies}")
                return False
            
            instance_id = f"{strategy_name}_{config.name}"
            
            if instance_id in self._strategy_instances:
                self.logger.warning(f"策略实例已存在: {instance_id}")
                return False
            
            try:
                # 创建策略实例
                strategy_instance = self.strategy_loader.create_strategy_instance(strategy_name, config)
                if strategy_instance is None:
                    return False
                
                # 获取元数据
                metadata = self.strategy_loader.get_strategy_metadata(strategy_name)
                if metadata is None:
                    self.logger.error(f"获取策略元数据失败: {strategy_name}")
                    return False
                
                # 创建实例信息
                instance_info = StrategyInstance(
                    strategy=strategy_instance,
                    config=config,
                    metadata=metadata
                )
                
                self._strategy_instances[instance_id] = instance_info
                
                # 添加到默认分组
                if 'default' not in self._strategy_groups:
                    self._strategy_groups['default'] = set()
                self._strategy_groups['default'].add(instance_id)
                
                self.logger.info(f"创建策略实例: {instance_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"创建策略实例失败: {instance_id}, 错误: {e}")
                return False
    
    def remove_strategy(self, instance_id: str) -> bool:
        """移除策略实例
        
        Args:
            instance_id: 实例ID
            
        Returns:
            是否移除成功
        """
        with self._lock:
            if instance_id not in self._strategy_instances:
                self.logger.warning(f"策略实例不存在: {instance_id}")
                return False
            
            try:
                instance = self._strategy_instances[instance_id]
                
                # 停止策略
                if instance.strategy.state.is_running:
                    instance.strategy.stop()
                
                # 从分组中移除
                for group_strategies in self._strategy_groups.values():
                    group_strategies.discard(instance_id)
                
                # 移除实例
                del self._strategy_instances[instance_id]
                
                self.logger.info(f"移除策略实例: {instance_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"移除策略实例失败: {instance_id}, 错误: {e}")
                return False
    
    def start_strategy(self, instance_id: str) -> bool:
        """启动策略
        
        Args:
            instance_id: 实例ID
            
        Returns:
            是否启动成功
        """
        with self._lock:
            if instance_id not in self._strategy_instances:
                self.logger.error(f"策略实例不存在: {instance_id}")
                return False
            
            try:
                instance = self._strategy_instances[instance_id]
                instance.strategy.start()
                
                self.logger.info(f"启动策略: {instance_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"启动策略失败: {instance_id}, 错误: {e}")
                return False
    
    def stop_strategy(self, instance_id: str) -> bool:
        """停止策略
        
        Args:
            instance_id: 实例ID
            
        Returns:
            是否停止成功
        """
        with self._lock:
            if instance_id not in self._strategy_instances:
                self.logger.error(f"策略实例不存在: {instance_id}")
                return False
            
            try:
                instance = self._strategy_instances[instance_id]
                instance.strategy.stop()
                
                self.logger.info(f"停止策略: {instance_id}")
                return True
                
            except Exception as e:
                self.logger.error(f"停止策略失败: {instance_id}, 错误: {e}")
                return False
    
    def restart_strategy(self, instance_id: str) -> bool:
        """重启策略
        
        Args:
            instance_id: 实例ID
            
        Returns:
            是否重启成功
        """
        return self.stop_strategy(instance_id) and self.start_strategy(instance_id)
    
    def start_group(self, group_name: str) -> bool:
        """启动策略组
        
        Args:
            group_name: 组名
            
        Returns:
            是否启动成功
        """
        if group_name not in self._strategy_groups:
            self.logger.error(f"策略组不存在: {group_name}")
            return False
        
        success_count = 0
        total_count = len(self._strategy_groups[group_name])
        
        for instance_id in self._strategy_groups[group_name]:
            if self.start_strategy(instance_id):
                success_count += 1
        
        self.logger.info(f"启动策略组 {group_name}: {success_count}/{total_count} 成功")
        return success_count == total_count
    
    def stop_group(self, group_name: str) -> bool:
        """停止策略组
        
        Args:
            group_name: 组名
            
        Returns:
            是否停止成功
        """
        if group_name not in self._strategy_groups:
            self.logger.error(f"策略组不存在: {group_name}")
            return False
        
        success_count = 0
        total_count = len(self._strategy_groups[group_name])
        
        for instance_id in self._strategy_groups[group_name]:
            if self.stop_strategy(instance_id):
                success_count += 1
        
        self.logger.info(f"停止策略组 {group_name}: {success_count}/{total_count} 成功")
        return success_count == total_count
    
    def create_group(self, group_name: str, instance_ids: List[str]) -> bool:
        """创建策略组
        
        Args:
            group_name: 组名
            instance_ids: 实例ID列表
            
        Returns:
            是否创建成功
        """
        with self._lock:
            # 验证所有实例都存在
            for instance_id in instance_ids:
                if instance_id not in self._strategy_instances:
                    self.logger.error(f"策略实例不存在: {instance_id}")
                    return False
            
            self._strategy_groups[group_name] = set(instance_ids)
            self.logger.info(f"创建策略组: {group_name}, 包含 {len(instance_ids)} 个策略")
            return True
    
    def remove_group(self, group_name: str) -> bool:
        """移除策略组
        
        Args:
            group_name: 组名
            
        Returns:
            是否移除成功
        """
        with self._lock:
            if group_name not in self._strategy_groups:
                self.logger.warning(f"策略组不存在: {group_name}")
                return False
            
            del self._strategy_groups[group_name]
            self.logger.info(f"移除策略组: {group_name}")
            return True
    
    def update_market_data(self, symbol: str, data: MarketData) -> None:
        """更新市场数据
        
        Args:
            symbol: 交易对
            data: 市场数据
        """
        for instance in self._strategy_instances.values():
            if (symbol in instance.config.symbols and 
                instance.strategy.state.is_running):
                try:
                    instance.strategy.update_market_data(symbol, data)
                except Exception as e:
                    self.logger.error(f"更新策略市场数据失败: {instance.config.name}, 错误: {e}")
                    self._total_errors += 1
    
    def update_kline_data(self, symbol: str, timeframe: str, klines: List[KlineData]) -> None:
        """更新K线数据
        
        Args:
            symbol: 交易对
            timeframe: 时间周期
            klines: K线数据列表
        """
        for instance in self._strategy_instances.values():
            if (symbol in instance.config.symbols and 
                timeframe in instance.config.timeframes and 
                instance.strategy.state.is_running):
                try:
                    instance.strategy.update_kline_data(symbol, timeframe, klines)
                except Exception as e:
                    self.logger.error(f"更新策略K线数据失败: {instance.config.name}, 错误: {e}")
                    self._total_errors += 1
    
    def generate_signals(self) -> List[Signal]:
        """生成所有策略信号
        
        Returns:
            信号列表
        """
        all_signals = []
        
        for instance_id, instance in self._strategy_instances.items():
            if not instance.strategy.state.is_running:
                continue
            
            try:
                with PerformanceTimer() as timer:
                    signals = instance.strategy.generate_signals({})
                
                if signals:
                    # 应用信号过滤器
                    filtered_signals = []
                    for signal in signals:
                        if all(filter_func(signal) for filter_func in self._signal_filters):
                            filtered_signals.append(signal)
                    
                    # 限制信号数量
                    if len(filtered_signals) > self.config.max_signals_per_strategy:
                        filtered_signals = filtered_signals[:self.config.max_signals_per_strategy]
                        self.logger.warning(f"策略 {instance_id} 信号数量超限，已截断")
                    
                    all_signals.extend(filtered_signals)
                    
                    # 更新统计
                    instance.update_performance(True, timer.elapsed_time)
                    self._total_signals_generated += len(filtered_signals)
                
            except Exception as e:
                self.logger.error(f"策略 {instance_id} 信号生成失败: {e}")
                instance.update_performance(False, 0.0)
                self._total_errors += 1
        
        # 限制总信号数量
        if len(all_signals) > self.config.max_total_signals:
            all_signals = all_signals[:self.config.max_total_signals]
            self.logger.warning(f"总信号数量超限，已截断到 {self.config.max_total_signals}")
        
        # 调用信号回调
        for signal in all_signals:
            for callback in self._signal_callbacks:
                try:
                    callback(signal)
                except Exception as e:
                    self.logger.error(f"信号回调执行失败: {e}")
        
        self._total_signals_processed += len(all_signals)
        return all_signals
    
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
    
    def add_signal_filter(self, filter_func: Callable[[Signal], bool]) -> None:
        """添加信号过滤器
        
        Args:
            filter_func: 过滤函数，返回True表示通过
        """
        self._signal_filters.append(filter_func)
        self.logger.info(f"添加信号过滤器，当前过滤器数量: {len(self._signal_filters)}")
    
    def remove_signal_filter(self, filter_func: Callable[[Signal], bool]) -> None:
        """移除信号过滤器
        
        Args:
            filter_func: 过滤函数
        """
        if filter_func in self._signal_filters:
            self._signal_filters.remove(filter_func)
            self.logger.info(f"移除信号过滤器，当前过滤器数量: {len(self._signal_filters)}")
    
    def get_strategy_list(self) -> List[str]:
        """获取策略实例列表"""
        return list(self._strategy_instances.keys())
    
    def get_group_list(self) -> List[str]:
        """获取策略组列表"""
        return list(self._strategy_groups.keys())
    
    def get_strategy_instance(self, instance_id: str) -> Optional[StrategyInstance]:
        """获取策略实例
        
        Args:
            instance_id: 实例ID
            
        Returns:
            策略实例或None
        """
        return self._strategy_instances.get(instance_id)
    
    def get_strategy_stats(self, instance_id: str) -> Optional[Dict[str, Any]]:
        """获取策略统计信息
        
        Args:
            instance_id: 实例ID
            
        Returns:
            统计信息字典或None
        """
        instance = self._strategy_instances.get(instance_id)
        if instance is None:
            return None
        
        return {
            'instance_id': instance_id,
            'strategy_name': instance.metadata.name,
            'config_name': instance.config.name,
            'is_running': instance.strategy.state.is_running,
            'created_time': instance.created_time.isoformat(),
            'last_update_time': instance.last_update_time.isoformat(),
            'uptime_seconds': instance.get_uptime().total_seconds(),
            'total_signals': instance.total_signals,
            'successful_signals': instance.successful_signals,
            'failed_signals': instance.failed_signals,
            'success_rate': instance.get_success_rate(),
            'avg_execution_time': instance.avg_execution_time,
            'total_return': instance.total_return,
            'sharpe_ratio': instance.sharpe_ratio,
            'max_drawdown': instance.max_drawdown,
            'win_rate': instance.win_rate
        }
    
    def get_all_strategy_stats(self) -> Dict[str, Dict[str, Any]]:
        """获取所有策略统计信息"""
        stats = {}
        for instance_id in self._strategy_instances:
            stats[instance_id] = self.get_strategy_stats(instance_id)
        return stats
    
    def get_manager_stats(self) -> Dict[str, Any]:
        """获取管理器统计信息"""
        uptime = datetime.now() - self._start_time
        active_strategies = sum(1 for i in self._strategy_instances.values() if i.strategy.state.is_running)
        
        return {
            'is_running': self._is_running,
            'uptime_seconds': uptime.total_seconds(),
            'total_strategies': len(self._strategy_instances),
            'active_strategies': active_strategies,
            'total_groups': len(self._strategy_groups),
            'total_signals_generated': self._total_signals_generated,
            'total_signals_processed': self._total_signals_processed,
            'total_errors': self._total_errors,
            'signal_rate': self._total_signals_generated / max(uptime.total_seconds(), 1),
            'error_rate': self._total_errors / max(self._total_signals_generated, 1)
        }
    
    def start(self) -> None:
        """启动管理器"""
        if self._is_running:
            self.logger.warning("策略管理器已在运行中")
            return
        
        self.logger.info("启动策略管理器")
        self._is_running = True
        self._is_stopping = False
        
        # 启动监控线程
        if self.config.enable_performance_monitoring:
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
        
        # 启动自动重载线程
        if self.config.auto_reload:
            self._reload_thread = threading.Thread(target=self._reload_loop, daemon=True)
            self._reload_thread.start()
        
        # 启动状态保存线程
        self._save_state_thread = threading.Thread(target=self._save_state_loop, daemon=True)
        self._save_state_thread.start()
    
    def stop(self) -> None:
        """停止管理器"""
        if not self._is_running:
            self.logger.warning("策略管理器未在运行")
            return
        
        self.logger.info("停止策略管理器")
        self._is_stopping = True
        
        # 停止所有策略
        for instance_id in list(self._strategy_instances.keys()):
            self.stop_strategy(instance_id)
        
        # 等待线程结束
        for thread in [self._monitor_thread, self._reload_thread, self._save_state_thread]:
            if thread and thread.is_alive():
                thread.join(timeout=5.0)
        
        # 保存最终状态
        self._save_state()
        
        self._is_running = False
        self.logger.info("策略管理器已停止")
    
    def _monitor_loop(self) -> None:
        """监控循环"""
        self.logger.info("策略管理器监控循环启动")
        
        while self._is_running and not self._is_stopping:
            try:
                # 健康检查
                if self.config.enable_health_check:
                    self._health_check()
                
                # 性能日志
                self._log_performance()
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                self.logger.error(f"监控循环发生错误: {e}")
                time.sleep(10.0)
        
        self.logger.info("策略管理器监控循环结束")
    
    def _reload_loop(self) -> None:
        """重载循环"""
        self.logger.info("策略自动重载循环启动")
        
        while self._is_running and not self._is_stopping:
            try:
                # 检查文件变更
                changed_strategies = self.strategy_loader.check_file_changes()
                
                for strategy_name in changed_strategies:
                    self.logger.info(f"检测到策略文件变更，重载策略: {strategy_name}")
                    
                    # 找到使用该策略的实例
                    affected_instances = []
                    for instance_id, instance in self._strategy_instances.items():
                        if instance.metadata.name == strategy_name:
                            affected_instances.append(instance_id)
                    
                    # 重载策略类
                    new_strategy_class = self.strategy_loader.reload_strategy(strategy_name)
                    if new_strategy_class:
                        # 重启受影响的实例
                        for instance_id in affected_instances:
                            self.restart_strategy(instance_id)
                
                time.sleep(self.config.reload_check_interval)
                
            except Exception as e:
                self.logger.error(f"重载循环发生错误: {e}")
                time.sleep(30.0)
        
        self.logger.info("策略自动重载循环结束")
    
    def _save_state_loop(self) -> None:
        """状态保存循环"""
        self.logger.info("状态保存循环启动")
        
        while self._is_running and not self._is_stopping:
            try:
                time.sleep(self.config.save_state_interval)
                self._save_state()
                
            except Exception as e:
                self.logger.error(f"状态保存循环发生错误: {e}")
                time.sleep(60.0)
        
        self.logger.info("状态保存循环结束")
    
    def _health_check(self) -> None:
        """健康检查"""
        for instance_id, instance in self._strategy_instances.items():
            # 检查策略状态
            if instance.config.enabled and not instance.strategy.state.is_running:
                self.logger.warning(f"策略 {instance_id} 应该运行但未运行")
            
            # 检查性能指标
            if instance.get_success_rate() < 0.5 and instance.total_signals > 10:
                self.logger.warning(f"策略 {instance_id} 成功率过低: {instance.get_success_rate():.2%}")
    
    def _log_performance(self) -> None:
        """记录性能日志"""
        stats = self.get_manager_stats()
        self.logger.info(
            f"管理器性能 - 运行时间: {stats['uptime_seconds']:.1f}s, "
            f"策略总数: {stats['total_strategies']}, "
            f"活跃策略: {stats['active_strategies']}, "
            f"信号生成率: {stats['signal_rate']:.2f}/s, "
            f"错误率: {stats['error_rate']:.4f}"
        )
    
    def _save_state(self) -> None:
        """保存状态"""
        try:
            state_data = {
                'manager_stats': self.get_manager_stats(),
                'strategy_stats': self.get_all_strategy_stats(),
                'groups': {name: list(strategies) for name, strategies in self._strategy_groups.items()},
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.config.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"状态已保存到: {self.config.state_file_path}")
            
        except Exception as e:
            self.logger.error(f"保存状态失败: {e}")
    
    def load_state(self) -> bool:
        """加载状态
        
        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(self.config.state_file_path):
                self.logger.info(f"状态文件不存在: {self.config.state_file_path}")
                return False
            
            with open(self.config.state_file_path, 'r', encoding='utf-8') as f:
                state_data = json.load(f)
            
            # 恢复策略组
            if 'groups' in state_data:
                for group_name, instance_ids in state_data['groups'].items():
                    self._strategy_groups[group_name] = set(instance_ids)
            
            self.logger.info(f"状态已从文件加载: {self.config.state_file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载状态失败: {e}")
            return False
    
    def __del__(self):
        """析构函数"""
        if self._is_running:
            self.stop()