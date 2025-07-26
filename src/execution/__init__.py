"""Execution Engine Module

执行引擎模块，负责订单执行、交易管理和市场接入。

主要功能：
- 订单管理和执行
- 交易路由和分发
- 市场接入和连接
- 执行算法和策略
- 成交回报处理

支持的执行类型：
- 市价单执行
- 限价单执行
- 算法交易执行
- 批量订单执行
- 条件订单执行

核心组件：
- ExecutionEngine: 执行引擎
- OrderManager: 订单管理器
- TradeRouter: 交易路由器
- MarketConnector: 市场连接器
- ExecutionAlgorithm: 执行算法
"""

from .engine import ExecutionEngine, ExecutionOrder, OrderStatus, OrderType, OrderSide, Trade
from .order_manager import OrderManager
from .trade_router import TradeRouter
from .market_connector import BaseMarketConnector, MockMarketConnector, MarketConnectorManager

__all__ = [
    'ExecutionEngine',
    'ExecutionOrder',
    'OrderManager',
    'OrderStatus',
    'OrderType',
    'OrderSide',
    'Trade',
    'TradeRouter',
    'BaseMarketConnector',
    'MockMarketConnector',
    'MarketConnectorManager'
]

__version__ = '1.0.0'
__author__ = 'Prometheus Team'
__description__ = 'Execution Engine Module for Prometheus Trading System'