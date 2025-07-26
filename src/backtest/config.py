"""Backtest Configuration Module

回测配置模块，负责管理回测的各种配置参数。

主要功能：
- 回测参数配置
- 数据源配置
- 交易成本配置
- 风险管理配置
- 输出配置
- 配置验证
"""

import os
import json
import yaml
from typing import Dict, List, Optional, Union, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

from ..common.logging import get_logger
from ..common.exceptions.backtest import BacktestConfigError


class BacktestMode(Enum):
    """回测模式枚举"""
    FULL = "full"                    # 完整回测
    FAST = "fast"                    # 快速回测
    PAPER = "paper"                  # 模拟交易
    LIVE = "live"                    # 实盘交易


class DataSource(Enum):
    """数据源枚举"""
    CSV = "csv"                      # CSV文件
    DATABASE = "database"            # 数据库
    API = "api"                      # API接口
    MEMORY = "memory"                # 内存数据
    MIXED = "mixed"                  # 混合数据源


class OrderType(Enum):
    """订单类型枚举"""
    MARKET = "market"                # 市价单
    LIMIT = "limit"                  # 限价单
    STOP = "stop"                    # 止损单
    STOP_LIMIT = "stop_limit"        # 止损限价单


class CommissionType(Enum):
    """佣金类型枚举"""
    FIXED = "fixed"                  # 固定佣金
    PERCENTAGE = "percentage"        # 百分比佣金
    TIERED = "tiered"                # 阶梯佣金
    CUSTOM = "custom"                # 自定义佣金


@dataclass
class TradingCosts:
    """交易成本配置"""
    # 佣金配置
    commission_type: CommissionType = CommissionType.PERCENTAGE
    commission_rate: float = 0.001  # 0.1%
    min_commission: float = 1.0     # 最小佣金
    max_commission: Optional[float] = None  # 最大佣金
    
    # 滑点配置
    slippage_type: str = "percentage"  # "fixed", "percentage", "bps"
    slippage_rate: float = 0.0005      # 0.05%
    
    # 印花税配置
    stamp_duty_rate: float = 0.001     # 0.1%
    stamp_duty_on_sell_only: bool = True
    
    # 过户费配置
    transfer_fee_rate: float = 0.00002  # 0.002%
    
    # 其他费用
    other_fees: Dict[str, float] = field(default_factory=dict)
    
    def calculate_commission(self, value: float, quantity: int = 1) -> float:
        """计算佣金
        
        Args:
            value: 交易金额
            quantity: 交易数量
            
        Returns:
            float: 佣金金额
        """
        if self.commission_type == CommissionType.FIXED:
            commission = self.commission_rate * quantity
        elif self.commission_type == CommissionType.PERCENTAGE:
            commission = value * self.commission_rate
        else:
            commission = value * self.commission_rate  # 默认按百分比
        
        # 应用最小和最大佣金限制
        if self.min_commission is not None:
            commission = max(commission, self.min_commission)
        if self.max_commission is not None:
            commission = min(commission, self.max_commission)
        
        return commission
    
    def calculate_slippage(self, value: float, side: str = "buy") -> float:
        """计算滑点
        
        Args:
            value: 交易金额
            side: 交易方向 ("buy", "sell")
            
        Returns:
            float: 滑点金额
        """
        if self.slippage_type == "fixed":
            return self.slippage_rate
        elif self.slippage_type == "percentage":
            return value * self.slippage_rate
        elif self.slippage_type == "bps":
            return value * self.slippage_rate / 10000
        else:
            return value * self.slippage_rate
    
    def calculate_total_costs(self, value: float, quantity: int = 1, 
                             side: str = "buy") -> Dict[str, float]:
        """计算总交易成本
        
        Args:
            value: 交易金额
            quantity: 交易数量
            side: 交易方向
            
        Returns:
            Dict[str, float]: 各项成本明细
        """
        costs = {
            'commission': self.calculate_commission(value, quantity),
            'slippage': self.calculate_slippage(value, side),
            'stamp_duty': 0.0,
            'transfer_fee': 0.0,
            'other_fees': 0.0
        }
        
        # 印花税（通常只在卖出时收取）
        if side == "sell" or not self.stamp_duty_on_sell_only:
            costs['stamp_duty'] = value * self.stamp_duty_rate
        
        # 过户费
        costs['transfer_fee'] = value * self.transfer_fee_rate
        
        # 其他费用
        for fee_name, fee_rate in self.other_fees.items():
            costs['other_fees'] += value * fee_rate
        
        costs['total'] = sum(costs.values())
        
        return costs


@dataclass
class DataConfig:
    """数据配置"""
    # 数据源配置
    source: DataSource = DataSource.CSV
    data_path: Optional[str] = None
    database_url: Optional[str] = None
    api_config: Optional[Dict[str, Any]] = None
    
    # 时间配置
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    timezone: str = "Asia/Shanghai"
    
    # 数据频率
    frequency: str = "1d"  # "1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w", "1M"
    
    # 数据字段映射
    field_mapping: Dict[str, str] = field(default_factory=lambda: {
        'open': 'open',
        'high': 'high', 
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'amount': 'amount'
    })
    
    # 数据预处理
    fill_missing: bool = True
    fill_method: str = "forward"  # "forward", "backward", "interpolate", "zero"
    remove_outliers: bool = False
    outlier_threshold: float = 3.0  # 标准差倍数
    
    # 数据缓存
    use_cache: bool = True
    cache_path: Optional[str] = None
    cache_expire_hours: int = 24
    
    def validate(self) -> bool:
        """验证数据配置
        
        Returns:
            bool: 配置是否有效
        """
        if self.source == DataSource.CSV and not self.data_path:
            raise BacktestConfigError("CSV数据源需要指定data_path")
        
        if self.source == DataSource.DATABASE and not self.database_url:
            raise BacktestConfigError("数据库数据源需要指定database_url")
        
        if self.source == DataSource.API and not self.api_config:
            raise BacktestConfigError("API数据源需要指定api_config")
        
        if self.start_date and self.end_date and self.start_date >= self.end_date:
            raise BacktestConfigError("开始时间必须早于结束时间")
        
        return True


@dataclass
class RiskConfig:
    """风险管理配置"""
    # 仓位管理
    max_position_size: float = 1.0      # 最大仓位比例
    max_single_position: float = 0.1    # 单个标的最大仓位
    max_sector_exposure: float = 0.3    # 单个行业最大敞口
    
    # 风险限制
    max_drawdown: float = 0.2           # 最大回撤限制
    max_daily_loss: float = 0.05        # 最大日损失
    max_leverage: float = 1.0           # 最大杠杆
    
    # 止损止盈
    use_stop_loss: bool = False
    stop_loss_pct: float = 0.05         # 止损百分比
    use_take_profit: bool = False
    take_profit_pct: float = 0.1        # 止盈百分比
    
    # 风险监控
    risk_check_frequency: str = "daily"  # "tick", "minute", "hourly", "daily"
    enable_risk_alerts: bool = True
    
    # 压力测试
    stress_test_scenarios: List[Dict[str, float]] = field(default_factory=list)
    
    def validate(self) -> bool:
        """验证风险配置
        
        Returns:
            bool: 配置是否有效
        """
        if not 0 < self.max_position_size <= 1:
            raise BacktestConfigError("最大仓位比例必须在(0,1]范围内")
        
        if not 0 < self.max_single_position <= self.max_position_size:
            raise BacktestConfigError("单个标的最大仓位不能超过总仓位限制")
        
        if self.max_drawdown <= 0 or self.max_drawdown > 1:
            raise BacktestConfigError("最大回撤限制必须在(0,1]范围内")
        
        return True


@dataclass
class ExecutionConfig:
    """执行配置"""
    # 订单执行
    default_order_type: OrderType = OrderType.MARKET
    fill_on_next_bar: bool = True       # 下一根K线成交
    partial_fill_allowed: bool = True   # 允许部分成交
    
    # 流动性限制
    max_volume_pct: float = 0.1         # 最大成交量占比
    min_trade_value: float = 1000.0     # 最小交易金额
    
    # 交易时间
    trading_hours: Dict[str, str] = field(default_factory=lambda: {
        'start': '09:30',
        'end': '15:00',
        'lunch_start': '11:30',
        'lunch_end': '13:00'
    })
    
    # 交易日历
    trading_calendar: Optional[str] = None  # 交易日历文件路径
    exclude_weekends: bool = True
    exclude_holidays: bool = True
    
    # 延迟模拟
    execution_delay: float = 0.0        # 执行延迟（秒）
    order_latency: float = 0.0          # 订单延迟（秒）
    
    def validate(self) -> bool:
        """验证执行配置
        
        Returns:
            bool: 配置是否有效
        """
        if not 0 < self.max_volume_pct <= 1:
            raise BacktestConfigError("最大成交量占比必须在(0,1]范围内")
        
        if self.min_trade_value <= 0:
            raise BacktestConfigError("最小交易金额必须大于0")
        
        return True


@dataclass
class OutputConfig:
    """输出配置"""
    # 输出路径
    output_dir: str = "./backtest_results"
    
    # 报告配置
    generate_report: bool = True
    report_format: List[str] = field(default_factory=lambda: ["html", "json"])
    include_charts: bool = True
    chart_format: str = "png"           # "png", "svg", "pdf"
    
    # 详细输出
    save_trades: bool = True
    save_positions: bool = True
    save_metrics: bool = True
    save_signals: bool = False
    
    # 实时输出
    real_time_update: bool = False
    update_frequency: int = 100         # 更新频率（条数）
    
    # 日志配置
    log_level: str = "INFO"
    log_to_file: bool = True
    log_file_path: Optional[str] = None
    
    def validate(self) -> bool:
        """验证输出配置
        
        Returns:
            bool: 配置是否有效
        """
        # 创建输出目录
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        if self.update_frequency <= 0:
            raise BacktestConfigError("更新频率必须大于0")
        
        return True


@dataclass
class BacktestConfig:
    """回测配置主类"""
    # 基本配置
    name: str = "Backtest"
    description: str = ""
    mode: BacktestMode = BacktestMode.FULL
    
    # 资金配置
    initial_capital: float = 1000000.0  # 初始资金
    currency: str = "CNY"               # 货币单位
    
    # 标的配置
    symbols: List[str] = field(default_factory=list)
    benchmark: Optional[str] = None     # 基准标的
    
    # 子配置
    data_config: DataConfig = field(default_factory=DataConfig)
    trading_costs: TradingCosts = field(default_factory=TradingCosts)
    risk_config: RiskConfig = field(default_factory=RiskConfig)
    execution_config: ExecutionConfig = field(default_factory=ExecutionConfig)
    output_config: OutputConfig = field(default_factory=OutputConfig)
    
    # 策略配置
    strategy_params: Dict[str, Any] = field(default_factory=dict)
    
    # 其他配置
    random_seed: Optional[int] = None
    debug_mode: bool = False
    
    def __post_init__(self):
        """后处理初始化"""
        # 验证配置
        self.validate()
        
        # 设置默认值
        if not self.symbols:
            raise BacktestConfigError("必须指定至少一个交易标的")
        
        if self.initial_capital <= 0:
            raise BacktestConfigError("初始资金必须大于0")
    
    def validate(self) -> bool:
        """验证配置
        
        Returns:
            bool: 配置是否有效
        """
        # 验证子配置
        self.data_config.validate()
        self.risk_config.validate()
        self.execution_config.validate()
        self.output_config.validate()
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式
        
        Returns:
            Dict[str, Any]: 配置字典
        """
        config_dict = asdict(self)
        
        # 转换枚举值
        config_dict['mode'] = self.mode.value
        config_dict['data_config']['source'] = self.data_config.source.value
        config_dict['trading_costs']['commission_type'] = self.trading_costs.commission_type.value
        config_dict['execution_config']['default_order_type'] = self.execution_config.default_order_type.value
        
        # 转换日期时间
        if self.data_config.start_date:
            config_dict['data_config']['start_date'] = self.data_config.start_date.isoformat()
        if self.data_config.end_date:
            config_dict['data_config']['end_date'] = self.data_config.end_date.isoformat()
        
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfig':
        """从字典创建配置
        
        Args:
            config_dict: 配置字典
            
        Returns:
            BacktestConfig: 配置实例
        """
        # 深拷贝字典
        config_dict = config_dict.copy()
        
        # 转换枚举值
        if 'mode' in config_dict:
            config_dict['mode'] = BacktestMode(config_dict['mode'])
        
        if 'data_config' in config_dict:
            data_config = config_dict['data_config']
            if 'source' in data_config:
                data_config['source'] = DataSource(data_config['source'])
            
            # 转换日期时间
            if 'start_date' in data_config and data_config['start_date']:
                data_config['start_date'] = datetime.fromisoformat(data_config['start_date'])
            if 'end_date' in data_config and data_config['end_date']:
                data_config['end_date'] = datetime.fromisoformat(data_config['end_date'])
            
            config_dict['data_config'] = DataConfig(**data_config)
        
        if 'trading_costs' in config_dict:
            trading_costs = config_dict['trading_costs']
            if 'commission_type' in trading_costs:
                trading_costs['commission_type'] = CommissionType(trading_costs['commission_type'])
            config_dict['trading_costs'] = TradingCosts(**trading_costs)
        
        if 'risk_config' in config_dict:
            config_dict['risk_config'] = RiskConfig(**config_dict['risk_config'])
        
        if 'execution_config' in config_dict:
            execution_config = config_dict['execution_config']
            if 'default_order_type' in execution_config:
                execution_config['default_order_type'] = OrderType(execution_config['default_order_type'])
            config_dict['execution_config'] = ExecutionConfig(**execution_config)
        
        if 'output_config' in config_dict:
            config_dict['output_config'] = OutputConfig(**config_dict['output_config'])
        
        return cls(**config_dict)
    
    def save_to_file(self, file_path: str) -> None:
        """保存配置到文件
        
        Args:
            file_path: 文件路径
        """
        file_path = Path(file_path)
        config_dict = self.to_dict()
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)
        else:
            raise BacktestConfigError(f"不支持的文件格式: {file_path.suffix}")
    
    @classmethod
    def load_from_file(cls, file_path: str) -> 'BacktestConfig':
        """从文件加载配置
        
        Args:
            file_path: 文件路径
            
        Returns:
            BacktestConfig: 配置实例
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise BacktestConfigError(f"配置文件不存在: {file_path}")
        
        if file_path.suffix.lower() == '.json':
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
        elif file_path.suffix.lower() in ['.yml', '.yaml']:
            with open(file_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f)
        else:
            raise BacktestConfigError(f"不支持的文件格式: {file_path.suffix}")
        
        return cls.from_dict(config_dict)
    
    def copy(self) -> 'BacktestConfig':
        """创建配置副本
        
        Returns:
            BacktestConfig: 配置副本
        """
        return self.from_dict(self.to_dict())
    
    def update(self, **kwargs) -> 'BacktestConfig':
        """更新配置参数
        
        Args:
            **kwargs: 要更新的参数
            
        Returns:
            BacktestConfig: 更新后的配置
        """
        config_dict = self.to_dict()
        
        # 递归更新字典
        def update_dict(d: Dict, u: Dict) -> Dict:
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d
        
        update_dict(config_dict, kwargs)
        return self.from_dict(config_dict)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取配置摘要
        
        Returns:
            Dict[str, Any]: 配置摘要
        """
        return {
            'name': self.name,
            'mode': self.mode.value,
            'initial_capital': self.initial_capital,
            'currency': self.currency,
            'symbols_count': len(self.symbols),
            'start_date': self.data_config.start_date.isoformat() if self.data_config.start_date else None,
            'end_date': self.data_config.end_date.isoformat() if self.data_config.end_date else None,
            'frequency': self.data_config.frequency,
            'commission_rate': self.trading_costs.commission_rate,
            'slippage_rate': self.trading_costs.slippage_rate,
            'max_position_size': self.risk_config.max_position_size,
            'max_drawdown': self.risk_config.max_drawdown
        }
    
    def __str__(self) -> str:
        return f"BacktestConfig(name='{self.name}', mode={self.mode.value}, symbols={len(self.symbols)})"
    
    def __repr__(self) -> str:
        return self.__str__()


class ConfigManager:
    """配置管理器
    
    提供配置的创建、验证、保存和加载功能。
    """
    
    def __init__(self, config_dir: str = "./configs"):
        """初始化配置管理器
        
        Args:
            config_dir: 配置文件目录
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger("ConfigManager")
    
    def create_default_config(self, name: str = "default") -> BacktestConfig:
        """创建默认配置
        
        Args:
            name: 配置名称
            
        Returns:
            BacktestConfig: 默认配置
        """
        config = BacktestConfig(
            name=name,
            description="默认回测配置",
            symbols=["000001.SZ"],  # 平安银行作为示例
            initial_capital=1000000.0,
            data_config=DataConfig(
                source=DataSource.CSV,
                start_date=datetime(2020, 1, 1),
                end_date=datetime(2023, 12, 31),
                frequency="1d"
            )
        )
        
        return config
    
    def save_config(self, config: BacktestConfig, filename: Optional[str] = None) -> str:
        """保存配置
        
        Args:
            config: 回测配置
            filename: 文件名（可选）
            
        Returns:
            str: 保存的文件路径
        """
        if filename is None:
            filename = f"{config.name}.json"
        
        file_path = self.config_dir / filename
        config.save_to_file(str(file_path))
        
        self.logger.info(f"配置已保存到: {file_path}")
        return str(file_path)
    
    def load_config(self, filename: str) -> BacktestConfig:
        """加载配置
        
        Args:
            filename: 文件名
            
        Returns:
            BacktestConfig: 回测配置
        """
        file_path = self.config_dir / filename
        config = BacktestConfig.load_from_file(str(file_path))
        
        self.logger.info(f"配置已从 {file_path} 加载")
        return config
    
    def list_configs(self) -> List[str]:
        """列出所有配置文件
        
        Returns:
            List[str]: 配置文件列表
        """
        config_files = []
        for file_path in self.config_dir.glob("*.json"):
            config_files.append(file_path.name)
        for file_path in self.config_dir.glob("*.yml"):
            config_files.append(file_path.name)
        for file_path in self.config_dir.glob("*.yaml"):
            config_files.append(file_path.name)
        
        return sorted(config_files)
    
    def delete_config(self, filename: str) -> bool:
        """删除配置文件
        
        Args:
            filename: 文件名
            
        Returns:
            bool: 是否删除成功
        """
        file_path = self.config_dir / filename
        
        if file_path.exists():
            file_path.unlink()
            self.logger.info(f"配置文件已删除: {file_path}")
            return True
        else:
            self.logger.warning(f"配置文件不存在: {file_path}")
            return False
    
    def validate_config(self, config: BacktestConfig) -> List[str]:
        """验证配置
        
        Args:
            config: 回测配置
            
        Returns:
            List[str]: 验证错误列表
        """
        errors = []
        
        try:
            config.validate()
        except BacktestConfigError as e:
            errors.append(str(e))
        
        return errors
    
    def compare_configs(self, config1: BacktestConfig, 
                       config2: BacktestConfig) -> Dict[str, Any]:
        """比较两个配置
        
        Args:
            config1: 配置1
            config2: 配置2
            
        Returns:
            Dict[str, Any]: 比较结果
        """
        dict1 = config1.to_dict()
        dict2 = config2.to_dict()
        
        differences = {}
        
        def compare_dicts(d1: Dict, d2: Dict, path: str = "") -> None:
            for key in set(d1.keys()) | set(d2.keys()):
                current_path = f"{path}.{key}" if path else key
                
                if key not in d1:
                    differences[current_path] = {'config1': None, 'config2': d2[key]}
                elif key not in d2:
                    differences[current_path] = {'config1': d1[key], 'config2': None}
                elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
                    compare_dicts(d1[key], d2[key], current_path)
                elif d1[key] != d2[key]:
                    differences[current_path] = {'config1': d1[key], 'config2': d2[key]}
        
        compare_dicts(dict1, dict2)
        
        return differences
    
    def __str__(self) -> str:
        return f"ConfigManager(config_dir='{self.config_dir}')"
    
    def __repr__(self) -> str:
        return self.__str__()