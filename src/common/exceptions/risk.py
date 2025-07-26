"""Risk Management Exception Classes

风险管理相关异常类定义。
"""

from typing import Optional, Dict, Any
from decimal import Decimal
from .base import PrometheusException


class RiskException(PrometheusException):
    """风险管理异常基类"""
    
    def __init__(self, 
                 message: str = "Risk management error occurred",
                 risk_type: Optional[str] = None,
                 **kwargs):
        """初始化风险管理异常
        
        Args:
            message: 错误消息
            risk_type: 风险类型
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if risk_type:
            details['risk_type'] = risk_type
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class RiskLimitExceededError(RiskException):
    """风险限制超限异常"""
    
    def __init__(self, 
                 message: str = "Risk limit exceeded",
                 limit_type: Optional[str] = None,
                 limit_value: Optional[Decimal] = None,
                 current_value: Optional[Decimal] = None,
                 **kwargs):
        """初始化风险限制超限异常
        
        Args:
            message: 错误消息
            limit_type: 限制类型
            limit_value: 限制值
            current_value: 当前值
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if limit_type:
            details['limit_type'] = limit_type
        if limit_value is not None:
            details['limit_value'] = str(limit_value)
        if current_value is not None:
            details['current_value'] = str(current_value)
        kwargs['details'] = details
        super().__init__(message, **kwargs)


class PositionSizeExceededError(RiskLimitExceededError):
    """持仓规模超限异常"""
    
    def __init__(self, 
                 message: str = "Position size exceeded",
                 symbol: Optional[str] = None,
                 **kwargs):
        """初始化持仓规模超限异常
        
        Args:
            message: 错误消息
            symbol: 交易对符号
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if symbol:
            details['symbol'] = symbol
        kwargs['details'] = details
        super().__init__(message, limit_type="position_size", **kwargs)


class DrawdownExceededError(RiskLimitExceededError):
    """回撤超限异常"""
    
    def __init__(self, 
                 message: str = "Drawdown exceeded limit",
                 drawdown_period: Optional[str] = None,
                 **kwargs):
        """初始化回撤超限异常
        
        Args:
            message: 错误消息
            drawdown_period: 回撤周期
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if drawdown_period:
            details['drawdown_period'] = drawdown_period
        kwargs['details'] = details
        super().__init__(message, limit_type="drawdown", **kwargs)


class LeverageExceededError(RiskLimitExceededError):
    """杠杆超限异常"""
    
    def __init__(self, 
                 message: str = "Leverage exceeded limit",
                 **kwargs):
        """初始化杠杆超限异常
        
        Args:
            message: 错误消息
            **kwargs: 其他参数
        """
        super().__init__(message, limit_type="leverage", **kwargs)


class ConcentrationRiskError(RiskException):
    """集中度风险异常"""
    
    def __init__(self, 
                 message: str = "Concentration risk detected",
                 concentration_type: Optional[str] = None,
                 concentration_ratio: Optional[Decimal] = None,
                 threshold: Optional[Decimal] = None,
                 **kwargs):
        """初始化集中度风险异常
        
        Args:
            message: 错误消息
            concentration_type: 集中度类型
            concentration_ratio: 集中度比率
            threshold: 阈值
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if concentration_type:
            details['concentration_type'] = concentration_type
        if concentration_ratio is not None:
            details['concentration_ratio'] = str(concentration_ratio)
        if threshold is not None:
            details['threshold'] = str(threshold)
        kwargs['details'] = details
        super().__init__(message, risk_type="concentration", **kwargs)


class VolatilityRiskError(RiskException):
    """波动率风险异常"""
    
    def __init__(self, 
                 message: str = "Volatility risk detected",
                 volatility_value: Optional[Decimal] = None,
                 volatility_threshold: Optional[Decimal] = None,
                 time_period: Optional[str] = None,
                 **kwargs):
        """初始化波动率风险异常
        
        Args:
            message: 错误消息
            volatility_value: 波动率值
            volatility_threshold: 波动率阈值
            time_period: 时间周期
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if volatility_value is not None:
            details['volatility_value'] = str(volatility_value)
        if volatility_threshold is not None:
            details['volatility_threshold'] = str(volatility_threshold)
        if time_period:
            details['time_period'] = time_period
        kwargs['details'] = details
        super().__init__(message, risk_type="volatility", **kwargs)


class CorrelationRiskError(RiskException):
    """相关性风险异常"""
    
    def __init__(self, 
                 message: str = "Correlation risk detected",
                 correlation_value: Optional[Decimal] = None,
                 correlation_threshold: Optional[Decimal] = None,
                 assets: Optional[list] = None,
                 **kwargs):
        """初始化相关性风险异常
        
        Args:
            message: 错误消息
            correlation_value: 相关性值
            correlation_threshold: 相关性阈值
            assets: 资产列表
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if correlation_value is not None:
            details['correlation_value'] = str(correlation_value)
        if correlation_threshold is not None:
            details['correlation_threshold'] = str(correlation_threshold)
        if assets:
            details['assets'] = assets
        kwargs['details'] = details
        super().__init__(message, risk_type="correlation", **kwargs)


# 为了向后兼容，创建别名
PositionSizeError = PositionSizeExceededError
DrawdownLimitError = DrawdownExceededError
VolatilityLimitError = VolatilityRiskError
ExposureLimitError = ConcentrationRiskError


class LiquidityRiskError(RiskException):
    """流动性风险异常"""
    
    def __init__(self, 
                 message: str = "Liquidity risk detected",
                 liquidity_metric: Optional[str] = None,
                 liquidity_value: Optional[Decimal] = None,
                 liquidity_threshold: Optional[Decimal] = None,
                 **kwargs):
        """初始化流动性风险异常
        
        Args:
            message: 错误消息
            liquidity_metric: 流动性指标
            liquidity_value: 流动性值
            liquidity_threshold: 流动性阈值
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if liquidity_metric:
            details['liquidity_metric'] = liquidity_metric
        if liquidity_value is not None:
            details['liquidity_value'] = str(liquidity_value)
        if liquidity_threshold is not None:
            details['liquidity_threshold'] = str(liquidity_threshold)
        kwargs['details'] = details
        super().__init__(message, risk_type="liquidity", **kwargs)


class MarginCallError(RiskException):
    """保证金追缴异常"""
    
    def __init__(self, 
                 message: str = "Margin call triggered",
                 margin_ratio: Optional[Decimal] = None,
                 maintenance_margin: Optional[Decimal] = None,
                 required_deposit: Optional[Decimal] = None,
                 **kwargs):
        """初始化保证金追缴异常
        
        Args:
            message: 错误消息
            margin_ratio: 保证金比率
            maintenance_margin: 维持保证金
            required_deposit: 需要追加保证金
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if margin_ratio is not None:
            details['margin_ratio'] = str(margin_ratio)
        if maintenance_margin is not None:
            details['maintenance_margin'] = str(maintenance_margin)
        if required_deposit is not None:
            details['required_deposit'] = str(required_deposit)
        kwargs['details'] = details
        super().__init__(message, risk_type="margin", **kwargs)


class VaRExceededError(RiskLimitExceededError):
    """VaR超限异常"""
    
    def __init__(self, 
                 message: str = "VaR limit exceeded",
                 confidence_level: Optional[Decimal] = None,
                 time_horizon: Optional[str] = None,
                 **kwargs):
        """初始化VaR超限异常
        
        Args:
            message: 错误消息
            confidence_level: 置信水平
            time_horizon: 时间范围
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if confidence_level is not None:
            details['confidence_level'] = str(confidence_level)
        if time_horizon:
            details['time_horizon'] = time_horizon
        kwargs['details'] = details
        super().__init__(message, limit_type="var", **kwargs)


class StressTestFailureError(RiskException):
    """压力测试失败异常"""
    
    def __init__(self, 
                 message: str = "Stress test failed",
                 test_scenario: Optional[str] = None,
                 failure_metrics: Optional[Dict[str, Any]] = None,
                 **kwargs):
        """初始化压力测试失败异常
        
        Args:
            message: 错误消息
            test_scenario: 测试场景
            failure_metrics: 失败指标
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if test_scenario:
            details['test_scenario'] = test_scenario
        if failure_metrics:
            details['failure_metrics'] = failure_metrics
        kwargs['details'] = details
        super().__init__(message, risk_type="stress_test", **kwargs)


class RiskModelError(RiskException):
    """风险模型异常"""
    
    def __init__(self, 
                 message: str = "Risk model error",
                 model_name: Optional[str] = None,
                 model_version: Optional[str] = None,
                 **kwargs):
        """初始化风险模型异常
        
        Args:
            message: 错误消息
            model_name: 模型名称
            model_version: 模型版本
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if model_name:
            details['model_name'] = model_name
        if model_version:
            details['model_version'] = model_version
        kwargs['details'] = details
        super().__init__(message, risk_type="model", **kwargs)


class RiskCalculationError(RiskException):
    """风险计算异常"""
    
    def __init__(self, 
                 message: str = "Risk calculation failed",
                 calculation_type: Optional[str] = None,
                 input_data_issues: Optional[list] = None,
                 **kwargs):
        """初始化风险计算异常
        
        Args:
            message: 错误消息
            calculation_type: 计算类型
            input_data_issues: 输入数据问题
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if calculation_type:
            details['calculation_type'] = calculation_type
        if input_data_issues:
            details['input_data_issues'] = input_data_issues
        kwargs['details'] = details
        super().__init__(message, risk_type="calculation", **kwargs)


class ComplianceViolationError(RiskException):
    """合规违规异常"""
    
    def __init__(self, 
                 message: str = "Compliance violation detected",
                 regulation: Optional[str] = None,
                 violation_type: Optional[str] = None,
                 **kwargs):
        """初始化合规违规异常
        
        Args:
            message: 错误消息
            regulation: 法规名称
            violation_type: 违规类型
            **kwargs: 其他参数
        """
        details = kwargs.get('details', {})
        if regulation:
            details['regulation'] = regulation
        if violation_type:
            details['violation_type'] = violation_type
        kwargs['details'] = details
        super().__init__(message, risk_type="compliance", **kwargs)