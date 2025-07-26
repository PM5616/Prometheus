"""Trading Data Models

交易相关的数据模型，包括订单、交易、持仓等。
"""

from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Optional, Dict, Any
from pydantic import Field, validator

from .base import BaseModel, TimestampMixin, IdentifiableMixin


class Symbol(BaseModel):
    """交易对模型"""
    
    symbol: str = Field(..., description="交易对符号")
    base_asset: str = Field(..., description="基础资产")
    quote_asset: str = Field(..., description="计价资产")
    
    # 精度设置
    price_precision: int = Field(8, description="价格精度")
    quantity_precision: int = Field(8, description="数量精度")
    
    # 最小交易限制
    min_price: Decimal = Field(Decimal('0'), description="最小价格")
    max_price: Decimal = Field(Decimal('0'), description="最大价格")
    min_qty: Decimal = Field(Decimal('0'), description="最小数量")
    max_qty: Decimal = Field(Decimal('0'), description="最大数量")
    min_notional: Decimal = Field(Decimal('0'), description="最小名义价值")
    
    # 状态
    status: str = Field("TRADING", description="交易状态")
    
    @validator('symbol')
    def validate_symbol(cls, v):
        """验证交易对符号"""
        if not v or len(v) < 3:
            raise ValueError('Symbol must be at least 3 characters long')
        return v.upper()
    
    def is_trading(self) -> bool:
        """是否可交易
        
        Returns:
            bool: 是否可交易
        """
        return self.status == "TRADING"


from .enums import OrderSide, OrderType, OrderStatus


class Order(TimestampMixin, IdentifiableMixin, BaseModel):
    """订单模型"""
    
    # 基本信息
    symbol: str = Field(..., description="交易对")
    side: OrderSide = Field(..., description="买卖方向")
    type: OrderType = Field(..., description="订单类型")
    
    # 价格和数量
    quantity: Decimal = Field(..., description="订单数量")
    price: Optional[Decimal] = Field(None, description="订单价格")
    stop_price: Optional[Decimal] = Field(None, description="止损价格")
    
    # 执行信息
    executed_qty: Decimal = Field(Decimal('0'), description="已执行数量")
    executed_quote_qty: Decimal = Field(Decimal('0'), description="已执行金额")
    avg_price: Optional[Decimal] = Field(None, description="平均成交价")
    
    # 状态
    status: OrderStatus = Field(OrderStatus.PENDING, description="订单状态")
    
    # 时间信息
    order_time: datetime = Field(default_factory=datetime.utcnow, description="下单时间")
    update_time: Optional[datetime] = Field(None, description="更新时间")
    
    # 外部ID
    client_order_id: Optional[str] = Field(None, description="客户端订单ID")
    exchange_order_id: Optional[str] = Field(None, description="交易所订单ID")
    
    # 策略信息
    strategy_id: Optional[str] = Field(None, description="策略ID")
    
    @validator('quantity', 'executed_qty', 'executed_quote_qty')
    def validate_positive_decimal(cls, v):
        """验证正数"""
        if v < 0:
            raise ValueError('Value must be non-negative')
        return v
    
    @property
    def remaining_qty(self) -> Decimal:
        """剩余数量
        
        Returns:
            Decimal: 剩余数量
        """
        return self.quantity - self.executed_qty
    
    @property
    def fill_rate(self) -> float:
        """成交率
        
        Returns:
            float: 成交率（0-1）
        """
        if self.quantity == 0:
            return 0.0
        return float(self.executed_qty / self.quantity)
    
    def is_filled(self) -> bool:
        """是否完全成交
        
        Returns:
            bool: 是否完全成交
        """
        return self.status == OrderStatus.FILLED
    
    def is_active(self) -> bool:
        """是否为活跃订单
        
        Returns:
            bool: 是否活跃
        """
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIALLY_FILLED]


class Trade(TimestampMixin, IdentifiableMixin, BaseModel):
    """交易记录模型"""
    
    # 基本信息
    symbol: str = Field(..., description="交易对")
    side: OrderSide = Field(..., description="买卖方向")
    
    # 交易信息
    quantity: Decimal = Field(..., description="交易数量")
    price: Decimal = Field(..., description="交易价格")
    quote_qty: Decimal = Field(..., description="交易金额")
    
    # 手续费
    commission: Decimal = Field(Decimal('0'), description="手续费")
    commission_asset: Optional[str] = Field(None, description="手续费资产")
    
    # 关联信息
    order_id: str = Field(..., description="订单ID")
    trade_id: Optional[str] = Field(None, description="交易ID")
    
    # 时间
    trade_time: datetime = Field(default_factory=datetime.utcnow, description="交易时间")
    
    # 策略信息
    strategy_id: Optional[str] = Field(None, description="策略ID")
    
    @property
    def notional(self) -> Decimal:
        """名义价值
        
        Returns:
            Decimal: 名义价值
        """
        return self.quantity * self.price


class Position(TimestampMixin, BaseModel):
    """持仓模型"""
    
    # 基本信息
    symbol: str = Field(..., description="交易对")
    side: OrderSide = Field(..., description="持仓方向")
    
    # 持仓信息
    quantity: Decimal = Field(..., description="持仓数量")
    avg_price: Decimal = Field(..., description="平均成本价")
    
    # 盈亏信息
    unrealized_pnl: Decimal = Field(Decimal('0'), description="未实现盈亏")
    realized_pnl: Decimal = Field(Decimal('0'), description="已实现盈亏")
    
    # 风险信息
    margin: Decimal = Field(Decimal('0'), description="保证金")
    
    # 策略信息
    strategy_id: Optional[str] = Field(None, description="策略ID")
    
    @property
    def notional(self) -> Decimal:
        """名义价值
        
        Returns:
            Decimal: 名义价值
        """
        return self.quantity * self.avg_price
    
    @property
    def total_pnl(self) -> Decimal:
        """总盈亏
        
        Returns:
            Decimal: 总盈亏
        """
        return self.unrealized_pnl + self.realized_pnl


class Balance(TimestampMixin, BaseModel):
    """余额模型"""
    
    # 资产信息
    asset: str = Field(..., description="资产符号")
    
    # 余额信息
    free: Decimal = Field(..., description="可用余额")
    locked: Decimal = Field(Decimal('0'), description="冻结余额")
    
    @property
    def total(self) -> Decimal:
        """总余额
        
        Returns:
            Decimal: 总余额
        """
        return self.free + self.locked
    
    def can_trade(self, amount: Decimal) -> bool:
        """是否可以交易指定数量
        
        Args:
            amount: 交易数量
            
        Returns:
            bool: 是否可以交易
        """
        return self.free >= amount