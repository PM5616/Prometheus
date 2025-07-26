"""Position Manager Module

头寸管理模块，负责管理投资组合中的头寸信息。
"""

from typing import Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from loguru import logger

from ..common.exceptions import (
    ValidationError,
    InsufficientBalanceError,
    PositionSizeError
)


@dataclass
class Position:
    """头寸信息"""
    symbol: str
    quantity: Decimal
    avg_price: Decimal
    market_price: Decimal = Decimal('0')
    unrealized_pnl: Decimal = Decimal('0')
    realized_pnl: Decimal = Decimal('0')
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    @property
    def market_value(self) -> Decimal:
        """市场价值"""
        return self.quantity * self.market_price
    
    @property
    def cost_basis(self) -> Decimal:
        """成本基础"""
        return self.quantity * self.avg_price
    
    @property
    def is_long(self) -> bool:
        """是否为多头头寸"""
        return self.quantity > 0
    
    @property
    def is_short(self) -> bool:
        """是否为空头头寸"""
        return self.quantity < 0
    
    def update_market_price(self, price: Decimal) -> None:
        """更新市场价格"""
        self.market_price = price
        self.unrealized_pnl = (price - self.avg_price) * self.quantity
        self.updated_at = datetime.now()


class PositionManager:
    """头寸管理器"""
    
    def __init__(self):
        self._positions: Dict[str, Position] = {}
        self._cash_balance: Decimal = Decimal('0')
        self._initial_capital: Decimal = Decimal('0')
        
    def set_initial_capital(self, capital: Decimal) -> None:
        """设置初始资金"""
        self._initial_capital = capital
        self._cash_balance = capital
        logger.info(f"Set initial capital: {capital}")
    
    def get_cash_balance(self) -> Decimal:
        """获取现金余额"""
        return self._cash_balance
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """获取指定标的的头寸"""
        return self._positions.get(symbol)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """获取所有头寸"""
        return self._positions.copy()
    
    def has_position(self, symbol: str) -> bool:
        """检查是否持有指定标的的头寸"""
        return symbol in self._positions and self._positions[symbol].quantity != 0
    
    def add_position(self, symbol: str, quantity: Decimal, price: Decimal) -> None:
        """添加头寸
        
        Args:
            symbol: 标的代码
            quantity: 数量（正数为买入，负数为卖出）
            price: 价格
        """
        if quantity == 0:
            return
        
        cost = quantity * price
        
        # 检查现金余额（买入时）
        if quantity > 0 and cost > self._cash_balance:
            raise InsufficientBalanceError(f"Insufficient cash balance: {self._cash_balance} < {cost}")
        
        if symbol in self._positions:
            # 更新现有头寸
            existing_pos = self._positions[symbol]
            new_quantity = existing_pos.quantity + quantity
            
            if new_quantity == 0:
                # 头寸平仓
                realized_pnl = -quantity * (price - existing_pos.avg_price)
                existing_pos.realized_pnl += realized_pnl
                existing_pos.quantity = Decimal('0')
                del self._positions[symbol]
                logger.info(f"Closed position for {symbol}, realized PnL: {realized_pnl}")
            else:
                # 计算新的平均价格
                if (existing_pos.quantity > 0 and quantity > 0) or (existing_pos.quantity < 0 and quantity < 0):
                    # 同向加仓
                    total_cost = existing_pos.cost_basis + cost
                    existing_pos.avg_price = total_cost / new_quantity
                else:
                    # 反向减仓
                    realized_pnl = -quantity * (price - existing_pos.avg_price)
                    existing_pos.realized_pnl += realized_pnl
                
                existing_pos.quantity = new_quantity
                existing_pos.updated_at = datetime.now()
        else:
            # 创建新头寸
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_price=price,
                market_price=price
            )
            logger.info(f"Created new position for {symbol}: {quantity} @ {price}")
        
        # 更新现金余额
        self._cash_balance -= cost
    
    def update_market_prices(self, prices: Dict[str, Decimal]) -> None:
        """批量更新市场价格"""
        for symbol, price in prices.items():
            if symbol in self._positions:
                self._positions[symbol].update_market_price(price)
    
    def get_total_market_value(self) -> Decimal:
        """获取总市场价值"""
        total_value = self._cash_balance
        for position in self._positions.values():
            total_value += position.market_value
        return total_value
    
    def get_total_unrealized_pnl(self) -> Decimal:
        """获取总未实现盈亏"""
        return sum(pos.unrealized_pnl for pos in self._positions.values())
    
    def get_total_realized_pnl(self) -> Decimal:
        """获取总已实现盈亏"""
        return sum(pos.realized_pnl for pos in self._positions.values())
    
    def get_portfolio_return(self) -> Decimal:
        """获取投资组合收益率"""
        if self._initial_capital == 0:
            return Decimal('0')
        
        current_value = self.get_total_market_value()
        return (current_value - self._initial_capital) / self._initial_capital
    
    def get_position_weights(self) -> Dict[str, Decimal]:
        """获取头寸权重"""
        total_value = self.get_total_market_value()
        if total_value == 0:
            return {}
        
        weights = {}
        for symbol, position in self._positions.items():
            weights[symbol] = abs(position.market_value) / total_value
        
        return weights
    
    def get_exposure(self) -> Dict[str, Decimal]:
        """获取敞口信息"""
        exposure = {
            'long_exposure': Decimal('0'),
            'short_exposure': Decimal('0'),
            'net_exposure': Decimal('0'),
            'gross_exposure': Decimal('0')
        }
        
        for position in self._positions.values():
            if position.is_long:
                exposure['long_exposure'] += position.market_value
            else:
                exposure['short_exposure'] += abs(position.market_value)
        
        exposure['net_exposure'] = exposure['long_exposure'] - exposure['short_exposure']
        exposure['gross_exposure'] = exposure['long_exposure'] + exposure['short_exposure']
        
        return exposure
    
    def validate_position_size(self, symbol: str, quantity: Decimal, max_position_size: Optional[Decimal] = None) -> bool:
        """验证头寸大小"""
        if max_position_size is None:
            return True
        
        current_quantity = Decimal('0')
        if symbol in self._positions:
            current_quantity = self._positions[symbol].quantity
        
        new_quantity = abs(current_quantity + quantity)
        
        if new_quantity > max_position_size:
            raise PositionSizeError(f"Position size {new_quantity} exceeds maximum {max_position_size}")
        
        return True
    
    def clear_positions(self) -> None:
        """清空所有头寸"""
        self._positions.clear()
        logger.info("Cleared all positions")
    
    def get_summary(self) -> Dict:
        """获取头寸摘要"""
        return {
            'cash_balance': float(self._cash_balance),
            'total_positions': len(self._positions),
            'total_market_value': float(self.get_total_market_value()),
            'total_unrealized_pnl': float(self.get_total_unrealized_pnl()),
            'total_realized_pnl': float(self.get_total_realized_pnl()),
            'portfolio_return': float(self.get_portfolio_return()),
            'exposure': {k: float(v) for k, v in self.get_exposure().items()}
        }