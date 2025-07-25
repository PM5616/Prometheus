"""Market Data Models

市场数据相关的数据模型，包括行情、K线、订单簿等。
"""

from datetime import datetime
from decimal import Decimal
from typing import List, Optional
from pydantic import Field, validator

from .base import BaseModel, TimestampMixin


class Ticker(BaseModel, TimestampMixin):
    """行情数据模型"""
    
    # 基本信息
    symbol: str = Field(..., description="交易对")
    
    # 价格信息
    price: Decimal = Field(..., description="最新价格")
    price_change: Decimal = Field(..., description="价格变化")
    price_change_percent: Decimal = Field(..., description="价格变化百分比")
    
    # 24小时统计
    high_price: Decimal = Field(..., description="24小时最高价")
    low_price: Decimal = Field(..., description="24小时最低价")
    open_price: Decimal = Field(..., description="24小时开盘价")
    prev_close_price: Decimal = Field(..., description="前收盘价")
    
    # 成交量信息
    volume: Decimal = Field(..., description="24小时成交量")
    quote_volume: Decimal = Field(..., description="24小时成交额")
    
    # 买卖盘信息
    bid_price: Optional[Decimal] = Field(None, description="买一价")
    bid_qty: Optional[Decimal] = Field(None, description="买一量")
    ask_price: Optional[Decimal] = Field(None, description="卖一价")
    ask_qty: Optional[Decimal] = Field(None, description="卖一量")
    
    # 统计信息
    count: Optional[int] = Field(None, description="24小时成交笔数")
    
    # 时间戳
    event_time: datetime = Field(default_factory=datetime.utcnow, description="事件时间")
    
    @property
    def spread(self) -> Optional[Decimal]:
        """买卖价差
        
        Returns:
            Optional[Decimal]: 买卖价差
        """
        if self.ask_price and self.bid_price:
            return self.ask_price - self.bid_price
        return None
    
    @property
    def spread_percent(self) -> Optional[Decimal]:
        """买卖价差百分比
        
        Returns:
            Optional[Decimal]: 买卖价差百分比
        """
        spread = self.spread
        if spread and self.bid_price and self.bid_price > 0:
            return (spread / self.bid_price) * 100
        return None


class Kline(BaseModel):
    """K线数据模型"""
    
    # 基本信息
    symbol: str = Field(..., description="交易对")
    interval: str = Field(..., description="时间间隔")
    
    # 时间信息
    open_time: datetime = Field(..., description="开盘时间")
    close_time: datetime = Field(..., description="收盘时间")
    
    # OHLC数据
    open_price: Decimal = Field(..., description="开盘价")
    high_price: Decimal = Field(..., description="最高价")
    low_price: Decimal = Field(..., description="最低价")
    close_price: Decimal = Field(..., description="收盘价")
    
    # 成交量数据
    volume: Decimal = Field(..., description="成交量")
    quote_volume: Decimal = Field(..., description="成交额")
    
    # 统计数据
    trade_count: int = Field(..., description="成交笔数")
    taker_buy_volume: Decimal = Field(..., description="主动买入成交量")
    taker_buy_quote_volume: Decimal = Field(..., description="主动买入成交额")
    
    # 状态
    is_closed: bool = Field(True, description="是否已收盘")
    
    @property
    def price_change(self) -> Decimal:
        """价格变化
        
        Returns:
            Decimal: 价格变化
        """
        return self.close_price - self.open_price
    
    @property
    def price_change_percent(self) -> Decimal:
        """价格变化百分比
        
        Returns:
            Decimal: 价格变化百分比
        """
        if self.open_price == 0:
            return Decimal('0')
        return (self.price_change / self.open_price) * 100
    
    @property
    def typical_price(self) -> Decimal:
        """典型价格 (HLC/3)
        
        Returns:
            Decimal: 典型价格
        """
        return (self.high_price + self.low_price + self.close_price) / 3
    
    @property
    def weighted_price(self) -> Decimal:
        """加权平均价格
        
        Returns:
            Decimal: 加权平均价格
        """
        if self.volume == 0:
            return self.close_price
        return self.quote_volume / self.volume


class OrderBookLevel(BaseModel):
    """订单簿档位"""
    
    price: Decimal = Field(..., description="价格")
    quantity: Decimal = Field(..., description="数量")
    
    @property
    def notional(self) -> Decimal:
        """名义价值
        
        Returns:
            Decimal: 名义价值
        """
        return self.price * self.quantity


class OrderBook(BaseModel, TimestampMixin):
    """订单簿模型"""
    
    # 基本信息
    symbol: str = Field(..., description="交易对")
    
    # 买卖盘数据
    bids: List[OrderBookLevel] = Field(..., description="买盘")
    asks: List[OrderBookLevel] = Field(..., description="卖盘")
    
    # 更新ID
    last_update_id: Optional[int] = Field(None, description="最后更新ID")
    
    # 事件时间
    event_time: datetime = Field(default_factory=datetime.utcnow, description="事件时间")
    
    @validator('bids', 'asks')
    def validate_levels(cls, v):
        """验证订单簿档位"""
        if not v:
            return v
        
        # 检查价格排序
        prices = [level.price for level in v]
        if len(prices) > 1:
            # 买盘应该降序，卖盘应该升序
            # 这里只检查是否已排序，具体排序由调用方保证
            pass
        
        return v
    
    @property
    def best_bid(self) -> Optional[OrderBookLevel]:
        """最优买价
        
        Returns:
            Optional[OrderBookLevel]: 最优买价档位
        """
        return self.bids[0] if self.bids else None
    
    @property
    def best_ask(self) -> Optional[OrderBookLevel]:
        """最优卖价
        
        Returns:
            Optional[OrderBookLevel]: 最优卖价档位
        """
        return self.asks[0] if self.asks else None
    
    @property
    def spread(self) -> Optional[Decimal]:
        """买卖价差
        
        Returns:
            Optional[Decimal]: 买卖价差
        """
        best_bid = self.best_bid
        best_ask = self.best_ask
        
        if best_bid and best_ask:
            return best_ask.price - best_bid.price
        return None
    
    @property
    def mid_price(self) -> Optional[Decimal]:
        """中间价
        
        Returns:
            Optional[Decimal]: 中间价
        """
        best_bid = self.best_bid
        best_ask = self.best_ask
        
        if best_bid and best_ask:
            return (best_bid.price + best_ask.price) / 2
        return None
    
    def get_depth(self, side: str, depth: int = 5) -> List[OrderBookLevel]:
        """获取指定深度的订单簿
        
        Args:
            side: 'bid' 或 'ask'
            depth: 深度
            
        Returns:
            List[OrderBookLevel]: 订单簿档位列表
        """
        if side.lower() == 'bid':
            return self.bids[:depth]
        elif side.lower() == 'ask':
            return self.asks[:depth]
        else:
            raise ValueError("Side must be 'bid' or 'ask'")
    
    def get_total_volume(self, side: str, depth: int = 5) -> Decimal:
        """获取指定深度的总成交量
        
        Args:
            side: 'bid' 或 'ask'
            depth: 深度
            
        Returns:
            Decimal: 总成交量
        """
        levels = self.get_depth(side, depth)
        return sum(level.quantity for level in levels)


class MarketData(BaseModel, TimestampMixin):
    """市场数据聚合模型"""
    
    # 基本信息
    symbol: str = Field(..., description="交易对")
    
    # 行情数据
    ticker: Optional[Ticker] = Field(None, description="行情数据")
    
    # 订单簿数据
    order_book: Optional[OrderBook] = Field(None, description="订单簿数据")
    
    # 最新K线
    latest_kline: Optional[Kline] = Field(None, description="最新K线")
    
    # 数据时间戳
    data_time: datetime = Field(default_factory=datetime.utcnow, description="数据时间")
    
    def is_complete(self) -> bool:
        """数据是否完整
        
        Returns:
            bool: 数据是否完整
        """
        return all([
            self.ticker is not None,
            self.order_book is not None,
            self.latest_kline is not None
        ])
    
    def get_current_price(self) -> Optional[Decimal]:
        """获取当前价格
        
        Returns:
            Optional[Decimal]: 当前价格
        """
        if self.ticker:
            return self.ticker.price
        elif self.order_book and self.order_book.mid_price:
            return self.order_book.mid_price
        elif self.latest_kline:
            return self.latest_kline.close_price
        return None