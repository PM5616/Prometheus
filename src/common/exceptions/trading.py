"""Trading Exception Classes

交易相关异常类定义。
"""

from typing import Optional, Dict, Any
from decimal import Decimal
from .base import PrometheusException


class TradingException(PrometheusException):
    """交易异常基类"""
    
    def __init__(self, 
                 message: str = "Trading error occurred",
                 symbol: Optional[str] = None,
                 **kwargs):
        """初始化交易异常
        
        Args:
            message: 错误消息
            symbol: 交易对符号
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if symbol:
            details['symbol'] = symbol
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class OrderException(TradingException):
    """订单异常类"""
    
    def __init__(self, 
                 message: str = "Order error occurred",
                 order_id: Optional[str] = None,
                 client_order_id: Optional[str] = None,
                 **kwargs):
        """初始化订单异常
        
        Args:
            message: 错误消息
            order_id: 订单ID
            client_order_id: 客户端订单ID
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if order_id:
            details['order_id'] = order_id
        if client_order_id:
            details['client_order_id'] = client_order_id
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class InsufficientBalanceError(TradingException):
    """余额不足异常"""
    
    def __init__(self, 
                 message: str = "Insufficient balance",
                 asset: Optional[str] = None,
                 required_amount: Optional[Decimal] = None,
                 available_amount: Optional[Decimal] = None,
                 **kwargs):
        """初始化余额不足异常
        
        Args:
            message: 错误消息
            asset: 资产名称
            required_amount: 需要金额
            available_amount: 可用金额
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if asset:
            details['asset'] = asset
        if required_amount is not None:
            details['required_amount'] = str(required_amount)
        if available_amount is not None:
            details['available_amount'] = str(available_amount)
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class InvalidOrderError(OrderException):
    """无效订单异常"""
    
    def __init__(self, 
                 message: str = "Invalid order",
                 validation_errors: Optional[list] = None,
                 **kwargs):
        """初始化无效订单异常
        
        Args:
            message: 错误消息
            validation_errors: 验证错误列表
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if validation_errors:
            details['validation_errors'] = validation_errors
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class OrderNotFoundError(OrderException):
    """订单未找到异常"""
    
    def __init__(self, 
                 message: str = "Order not found",
                 **kwargs):
        """初始化订单未找到异常
        
        Args:
            message: 错误消息
            **kwargs: 其他参数
        """
        super().__init__(message, **kwargs)


class MarketClosedError(TradingException):
    """市场关闭异常"""
    
    def __init__(self, 
                 message: str = "Market is closed",
                 market_hours: Optional[Dict[str, str]] = None,
                 **kwargs):
        """初始化市场关闭异常
        
        Args:
            message: 错误消息
            market_hours: 市场开放时间
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if market_hours:
            details['market_hours'] = market_hours
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class SymbolNotFoundError(TradingException):
    """交易对未找到异常"""
    
    def __init__(self, 
                 message: str = "Symbol not found",
                 available_symbols: Optional[list] = None,
                 **kwargs):
        """初始化交易对未找到异常
        
        Args:
            message: 错误消息
            available_symbols: 可用交易对列表
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if available_symbols:
            details['available_symbols'] = available_symbols[:10]  # 限制数量
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class PriceOutOfRangeError(TradingException):
    """价格超出范围异常"""
    
    def __init__(self, 
                 message: str = "Price out of range",
                 price: Optional[Decimal] = None,
                 min_price: Optional[Decimal] = None,
                 max_price: Optional[Decimal] = None,
                 **kwargs):
        """初始化价格超出范围异常
        
        Args:
            message: 错误消息
            price: 当前价格
            min_price: 最小价格
            max_price: 最大价格
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if price is not None:
            details['price'] = str(price)
        if min_price is not None:
            details['min_price'] = str(min_price)
        if max_price is not None:
            details['max_price'] = str(max_price)
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class QuantityOutOfRangeError(TradingException):
    """数量超出范围异常"""
    
    def __init__(self, 
                 message: str = "Quantity out of range",
                 quantity: Optional[Decimal] = None,
                 min_quantity: Optional[Decimal] = None,
                 max_quantity: Optional[Decimal] = None,
                 step_size: Optional[Decimal] = None,
                 **kwargs):
        """初始化数量超出范围异常
        
        Args:
            message: 错误消息
            quantity: 当前数量
            min_quantity: 最小数量
            max_quantity: 最大数量
            step_size: 步长
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if quantity is not None:
            details['quantity'] = str(quantity)
        if min_quantity is not None:
            details['min_quantity'] = str(min_quantity)
        if max_quantity is not None:
            details['max_quantity'] = str(max_quantity)
        if step_size is not None:
            details['step_size'] = str(step_size)
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class OrderExecutionError(OrderException):
    """订单执行异常"""
    
    def __init__(self, 
                 message: str = "Order execution failed",
                 execution_stage: Optional[str] = None,
                 **kwargs):
        """初始化订单执行异常
        
        Args:
            message: 错误消息
            execution_stage: 执行阶段
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if execution_stage:
            details['execution_stage'] = execution_stage
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class PositionNotFoundError(TradingException):
    """持仓未找到异常"""
    
    def __init__(self, 
                 message: str = "Position not found",
                 position_id: Optional[str] = None,
                 **kwargs):
        """初始化持仓未找到异常
        
        Args:
            message: 错误消息
            position_id: 持仓ID
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if position_id:
            details['position_id'] = position_id
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class LiquidationError(TradingException):
    """强制平仓异常"""
    
    def __init__(self, 
                 message: str = "Liquidation occurred",
                 liquidation_price: Optional[Decimal] = None,
                 margin_ratio: Optional[Decimal] = None,
                 **kwargs):
        """初始化强制平仓异常
        
        Args:
            message: 错误消息
            liquidation_price: 强平价格
            margin_ratio: 保证金比率
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if liquidation_price is not None:
            details['liquidation_price'] = str(liquidation_price)
        if margin_ratio is not None:
            details['margin_ratio'] = str(margin_ratio)
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class SlippageExceededError(TradingException):
    """滑点超限异常"""
    
    def __init__(self, 
                 message: str = "Slippage exceeded limit",
                 expected_price: Optional[Decimal] = None,
                 actual_price: Optional[Decimal] = None,
                 slippage_limit: Optional[Decimal] = None,
                 **kwargs):
        """初始化滑点超限异常
        
        Args:
            message: 错误消息
            expected_price: 预期价格
            actual_price: 实际价格
            slippage_limit: 滑点限制
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if expected_price is not None:
            details['expected_price'] = str(expected_price)
        if actual_price is not None:
            details['actual_price'] = str(actual_price)
        if slippage_limit is not None:
            details['slippage_limit'] = str(slippage_limit)
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class TradingHaltedError(TradingException):
    """交易暂停异常"""
    
    def __init__(self, 
                 message: str = "Trading halted",
                 halt_reason: Optional[str] = None,
                 resume_time: Optional[str] = None,
                 **kwargs):
        """初始化交易暂停异常
        
        Args:
            message: 错误消息
            halt_reason: 暂停原因
            resume_time: 恢复时间
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if halt_reason:
            details['halt_reason'] = halt_reason
        if resume_time:
            details['resume_time'] = resume_time
        kwargs['details'] = details
        super().__init__(message, **kwargs)