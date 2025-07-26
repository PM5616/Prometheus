"""Risk Manager

风险管理器，负责投资组合的风险控制和监控
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import asdict
from enum import Enum

from ..common.exceptions.base import PrometheusException
from ..common.models.base import BaseModel
from .models import (
    Portfolio, Position, RiskMetrics, RiskConfig, 
    RiskLevel, PositionType, Signal
)


class RiskViolationType(Enum):
    """风险违规类型"""
    POSITION_SIZE = "position_size"  # 仓位大小超限
    CONCENTRATION = "concentration"  # 集中度超限
    LEVERAGE = "leverage"  # 杠杆超限
    DRAWDOWN = "drawdown"  # 回撤超限
    VAR = "var"  # VaR超限
    CORRELATION = "correlation"  # 相关性超限
    VOLATILITY = "volatility"  # 波动率超限
    SECTOR_EXPOSURE = "sector_exposure"  # 行业暴露超限


class RiskViolation:
    """风险违规记录"""
    
    def __init__(
        self,
        violation_type: RiskViolationType,
        asset: Optional[str] = None,
        current_value: float = 0.0,
        limit_value: float = 0.0,
        severity: RiskLevel = RiskLevel.MEDIUM,
        message: str = "",
        timestamp: Optional[datetime] = None
    ):
        self.violation_type = violation_type
        self.asset = asset
        self.current_value = current_value
        self.limit_value = limit_value
        self.severity = severity
        self.message = message
        self.timestamp = timestamp or datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'violation_type': self.violation_type.value,
            'asset': self.asset,
            'current_value': self.current_value,
            'limit_value': self.limit_value,
            'severity': self.severity.value,
            'message': self.message,
            'timestamp': self.timestamp.isoformat()
        }


class RiskException(PrometheusException):
    """风险管理异常"""
    pass


class RiskManager(BaseModel):
    """风险管理器
    
    负责投资组合的风险控制和监控，包括：
    1. 仓位风险控制
    2. 集中度风险监控
    3. 市场风险度量（VaR, CVaR）
    4. 流动性风险评估
    5. 相关性风险分析
    6. 压力测试
    """
    
    def __init__(self, config: RiskConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 风险违规记录
        self.violations: List[RiskViolation] = []
        self.violation_history: List[RiskViolation] = []
        
        # 风险指标缓存
        self.risk_metrics_cache: Optional[RiskMetrics] = None
        self.cache_timestamp: Optional[datetime] = None
        
        # 历史数据
        self.returns_history: Dict[str, List[float]] = {}
        self.portfolio_value_history: List[float] = []
        
        self.logger.info("Risk Manager initialized")
    
    async def check_pre_trade_risk(
        self, 
        signal: Signal, 
        portfolio: Portfolio,
        current_positions: Dict[str, Position]
    ) -> Tuple[bool, List[RiskViolation]]:
        """交易前风险检查
        
        Args:
            signal: 交易信号
            portfolio: 当前投资组合
            current_positions: 当前持仓
            
        Returns:
            (is_allowed, violations)
        """
        violations = []
        
        try:
            # 模拟交易后的仓位
            simulated_positions = self._simulate_trade(signal, current_positions)
            simulated_portfolio = self._create_simulated_portfolio(portfolio, simulated_positions)
            
            # 检查各种风险限制
            violations.extend(await self._check_position_limits(signal, simulated_positions))
            violations.extend(await self._check_concentration_limits(simulated_portfolio))
            violations.extend(await self._check_leverage_limits(simulated_portfolio))
            violations.extend(await self._check_sector_limits(simulated_positions))
            
            # 判断是否允许交易
            high_severity_violations = [
                v for v in violations 
                if v.severity in [RiskLevel.HIGH, RiskLevel.CRITICAL]
            ]
            
            is_allowed = len(high_severity_violations) == 0
            
            if not is_allowed:
                self.logger.warning(
                    f"Trade blocked due to risk violations: {[v.message for v in high_severity_violations]}"
                )
            
            return is_allowed, violations
            
        except Exception as e:
            self.logger.error(f"Pre-trade risk check failed: {e}")
            # 保守策略：出错时拒绝交易
            violation = RiskViolation(
                violation_type=RiskViolationType.POSITION_SIZE,
                message=f"Risk check error: {e}",
                severity=RiskLevel.CRITICAL
            )
            return False, [violation]
    
    async def calculate_risk_metrics(
        self, 
        portfolio: Portfolio,
        positions: Dict[str, Position],
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> RiskMetrics:
        """计算风险指标"""
        try:
            # 检查缓存
            if (self.risk_metrics_cache and 
                self.cache_timestamp and 
                datetime.now() - self.cache_timestamp < timedelta(minutes=5)):
                return self.risk_metrics_cache
            
            # 计算基础指标
            total_value = portfolio.total_value
            if total_value <= 0:
                return self._create_empty_risk_metrics()
            
            # 计算VaR和CVaR
            var_95, var_99 = await self._calculate_var(positions, market_data)
            cvar_95, cvar_99 = await self._calculate_cvar(positions, market_data)
            
            # 计算最大回撤
            max_drawdown = self._calculate_max_drawdown()
            
            # 计算波动率
            volatility = await self._calculate_portfolio_volatility(positions, market_data)
            
            # 计算贝塔值
            beta = await self._calculate_portfolio_beta(positions, market_data)
            
            # 计算集中度指标
            concentration = self._calculate_concentration(positions, total_value)
            
            # 计算杠杆率
            leverage = self._calculate_leverage(positions, total_value)
            
            # 计算流动性指标
            liquidity_score = await self._calculate_liquidity_score(positions)
            
            # 创建风险指标对象
            risk_metrics = RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                max_drawdown=max_drawdown,
                volatility=volatility,
                beta=beta,
                sharpe_ratio=0.0,  # 需要更多数据计算
                concentration=concentration,
                leverage=leverage,
                liquidity_score=liquidity_score,
                risk_score=self._calculate_overall_risk_score(
                    var_95, max_drawdown, volatility, concentration, leverage
                )
            )
            
            # 缓存结果
            self.risk_metrics_cache = risk_metrics
            self.cache_timestamp = datetime.now()
            
            return risk_metrics
            
        except Exception as e:
            self.logger.error(f"Risk metrics calculation failed: {e}")
            return self._create_empty_risk_metrics()
    
    async def monitor_ongoing_risk(
        self, 
        portfolio: Portfolio,
        positions: Dict[str, Position]
    ) -> List[RiskViolation]:
        """持续风险监控"""
        violations = []
        
        try:
            # 清空当前违规记录
            self.violations.clear()
            
            # 检查各种风险限制
            violations.extend(await self._check_concentration_limits(portfolio))
            violations.extend(await self._check_leverage_limits(portfolio))
            violations.extend(await self._check_drawdown_limits(portfolio))
            violations.extend(await self._check_var_limits(portfolio, positions))
            violations.extend(await self._check_volatility_limits(positions))
            
            # 记录违规
            self.violations = violations
            self.violation_history.extend(violations)
            
            # 限制历史记录长度
            if len(self.violation_history) > 1000:
                self.violation_history = self.violation_history[-1000:]
            
            if violations:
                self.logger.warning(f"Risk violations detected: {len(violations)}")
            
            return violations
            
        except Exception as e:
            self.logger.error(f"Ongoing risk monitoring failed: {e}")
            return []
    
    def _simulate_trade(
        self, 
        signal: Signal, 
        current_positions: Dict[str, Position]
    ) -> Dict[str, Position]:
        """模拟交易后的仓位"""
        simulated_positions = current_positions.copy()
        
        if signal.asset in simulated_positions:
            position = simulated_positions[signal.asset]
            
            # 计算新的数量
            if signal.position_type == PositionType.LONG:
                new_quantity = position.quantity + signal.quantity
            elif signal.position_type == PositionType.SHORT:
                new_quantity = position.quantity - signal.quantity
            else:  # CLOSE
                new_quantity = Decimal('0')
            
            # 更新仓位
            if new_quantity != 0:
                position.quantity = new_quantity
                position.market_value = new_quantity * signal.price
            else:
                # 平仓
                del simulated_positions[signal.asset]
        else:
            # 新建仓位
            if signal.position_type != PositionType.CLOSE:
                quantity = signal.quantity if signal.position_type == PositionType.LONG else -signal.quantity
                simulated_positions[signal.asset] = Position(
                    asset=signal.asset,
                    quantity=quantity,
                    avg_price=signal.price,
                    market_value=quantity * signal.price,
                    unrealized_pnl=Decimal('0'),
                    position_type=signal.position_type
                )
        
        return simulated_positions
    
    def _create_simulated_portfolio(
        self, 
        portfolio: Portfolio, 
        simulated_positions: Dict[str, Position]
    ) -> Portfolio:
        """创建模拟投资组合"""
        total_value = sum(pos.market_value for pos in simulated_positions.values())
        total_value += portfolio.cash  # 假设现金不变
        
        return Portfolio(
            portfolio_id=portfolio.portfolio_id,
            total_value=total_value,
            cash=portfolio.cash,
            positions=simulated_positions,
            unrealized_pnl=sum(pos.unrealized_pnl for pos in simulated_positions.values()),
            realized_pnl=portfolio.realized_pnl
        )
    
    async def _check_position_limits(
        self, 
        signal: Signal, 
        positions: Dict[str, Position]
    ) -> List[RiskViolation]:
        """检查仓位限制"""
        violations = []
        
        if signal.asset in positions:
            position = positions[signal.asset]
            position_value = abs(float(position.market_value))
            
            # 检查单个仓位限制
            if position_value > self.config.max_position_size:
                violations.append(RiskViolation(
                    violation_type=RiskViolationType.POSITION_SIZE,
                    asset=signal.asset,
                    current_value=position_value,
                    limit_value=self.config.max_position_size,
                    severity=RiskLevel.HIGH,
                    message=f"Position size {position_value:.2f} exceeds limit {self.config.max_position_size:.2f}"
                ))
        
        return violations
    
    async def _check_concentration_limits(self, portfolio: Portfolio) -> List[RiskViolation]:
        """检查集中度限制"""
        violations = []
        
        if portfolio.total_value <= 0:
            return violations
        
        # 计算最大仓位占比
        max_weight = 0.0
        max_asset = ""
        
        for asset, position in portfolio.positions.items():
            weight = abs(float(position.market_value)) / float(portfolio.total_value)
            if weight > max_weight:
                max_weight = weight
                max_asset = asset
        
        # 检查集中度限制
        if max_weight > self.config.max_concentration:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.CONCENTRATION,
                asset=max_asset,
                current_value=max_weight,
                limit_value=self.config.max_concentration,
                severity=RiskLevel.MEDIUM,
                message=f"Concentration {max_weight:.2%} exceeds limit {self.config.max_concentration:.2%}"
            ))
        
        return violations
    
    async def _check_leverage_limits(self, portfolio: Portfolio) -> List[RiskViolation]:
        """检查杠杆限制"""
        violations = []
        
        if portfolio.total_value <= 0:
            return violations
        
        # 计算杠杆率
        total_exposure = sum(abs(float(pos.market_value)) for pos in portfolio.positions.values())
        leverage = total_exposure / float(portfolio.total_value)
        
        if leverage > self.config.max_leverage:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.LEVERAGE,
                current_value=leverage,
                limit_value=self.config.max_leverage,
                severity=RiskLevel.HIGH,
                message=f"Leverage {leverage:.2f} exceeds limit {self.config.max_leverage:.2f}"
            ))
        
        return violations
    
    async def _check_sector_limits(self, positions: Dict[str, Position]) -> List[RiskViolation]:
        """检查行业暴露限制"""
        violations = []
        
        # 简化实现：假设资产名称包含行业信息
        # 实际应用中需要从资产数据库获取行业分类
        sector_exposure = {}
        total_value = sum(abs(float(pos.market_value)) for pos in positions.values())
        
        if total_value <= 0:
            return violations
        
        for asset, position in positions.items():
            # 简单的行业分类逻辑
            sector = self._get_asset_sector(asset)
            if sector not in sector_exposure:
                sector_exposure[sector] = 0.0
            sector_exposure[sector] += abs(float(position.market_value))
        
        # 检查行业暴露限制
        for sector, exposure in sector_exposure.items():
            exposure_ratio = exposure / total_value
            if exposure_ratio > self.config.max_sector_exposure:
                violations.append(RiskViolation(
                    violation_type=RiskViolationType.SECTOR_EXPOSURE,
                    asset=sector,
                    current_value=exposure_ratio,
                    limit_value=self.config.max_sector_exposure,
                    severity=RiskLevel.MEDIUM,
                    message=f"Sector {sector} exposure {exposure_ratio:.2%} exceeds limit {self.config.max_sector_exposure:.2%}"
                ))
        
        return violations
    
    async def _check_drawdown_limits(self, portfolio: Portfolio) -> List[RiskViolation]:
        """检查回撤限制"""
        violations = []
        
        max_drawdown = self._calculate_max_drawdown()
        
        if abs(max_drawdown) > self.config.max_drawdown:
            violations.append(RiskViolation(
                violation_type=RiskViolationType.DRAWDOWN,
                current_value=abs(max_drawdown),
                limit_value=self.config.max_drawdown,
                severity=RiskLevel.HIGH,
                message=f"Max drawdown {abs(max_drawdown):.2%} exceeds limit {self.config.max_drawdown:.2%}"
            ))
        
        return violations
    
    async def _check_var_limits(
        self, 
        portfolio: Portfolio, 
        positions: Dict[str, Position]
    ) -> List[RiskViolation]:
        """检查VaR限制"""
        violations = []
        
        try:
            var_95, _ = await self._calculate_var(positions)
            
            if var_95 > self.config.max_var:
                violations.append(RiskViolation(
                    violation_type=RiskViolationType.VAR,
                    current_value=var_95,
                    limit_value=self.config.max_var,
                    severity=RiskLevel.MEDIUM,
                    message=f"VaR 95% {var_95:.2f} exceeds limit {self.config.max_var:.2f}"
                ))
        except Exception as e:
            self.logger.warning(f"VaR calculation failed: {e}")
        
        return violations
    
    async def _check_volatility_limits(self, positions: Dict[str, Position]) -> List[RiskViolation]:
        """检查波动率限制"""
        violations = []
        
        try:
            volatility = await self._calculate_portfolio_volatility(positions)
            
            if volatility > self.config.max_volatility:
                violations.append(RiskViolation(
                    violation_type=RiskViolationType.VOLATILITY,
                    current_value=volatility,
                    limit_value=self.config.max_volatility,
                    severity=RiskLevel.MEDIUM,
                    message=f"Volatility {volatility:.2%} exceeds limit {self.config.max_volatility:.2%}"
                ))
        except Exception as e:
            self.logger.warning(f"Volatility calculation failed: {e}")
        
        return violations
    
    async def _calculate_var(
        self, 
        positions: Dict[str, Position],
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Tuple[float, float]:
        """计算VaR (Value at Risk)"""
        try:
            # 简化的VaR计算
            # 实际应用中需要更复杂的模型
            
            if not positions:
                return 0.0, 0.0
            
            # 使用历史模拟法
            portfolio_returns = self._get_portfolio_returns(positions, market_data)
            
            if len(portfolio_returns) < 30:  # 需要足够的历史数据
                # 使用参数法估算
                total_value = sum(abs(float(pos.market_value)) for pos in positions.values())
                estimated_volatility = 0.02  # 2%日波动率
                var_95 = total_value * estimated_volatility * 1.645  # 95%置信度
                var_99 = total_value * estimated_volatility * 2.326  # 99%置信度
                return var_95, var_99
            
            # 计算分位数
            var_95 = float(np.percentile(portfolio_returns, (1 - 0.95) * 100))
            var_99 = float(np.percentile(portfolio_returns, (1 - 0.99) * 100))
            
            return abs(var_95), abs(var_99)
            
        except Exception as e:
            self.logger.error(f"VaR calculation failed: {e}")
            return 0.0, 0.0
    
    async def _calculate_cvar(
        self, 
        positions: Dict[str, Position],
        market_data: Optional[Dict[str, pd.DataFrame]] = None,
        confidence_levels: List[float] = [0.95, 0.99]
    ) -> Tuple[float, float]:
        """计算CVaR (Conditional Value at Risk)"""
        try:
            portfolio_returns = self._get_portfolio_returns(positions, market_data)
            
            if len(portfolio_returns) < 30:
                # 估算CVaR
                var_95, var_99 = await self._calculate_var(positions, market_data)
                return var_95 * 1.2, var_99 * 1.2  # CVaR通常比VaR高20%左右
            
            # 计算CVaR
            var_95_threshold = np.percentile(portfolio_returns, 5)
            var_99_threshold = np.percentile(portfolio_returns, 1)
            
            cvar_95 = float(np.mean(portfolio_returns[portfolio_returns <= var_95_threshold]))
            cvar_99 = float(np.mean(portfolio_returns[portfolio_returns <= var_99_threshold]))
            
            return abs(cvar_95), abs(cvar_99)
            
        except Exception as e:
            self.logger.error(f"CVaR calculation failed: {e}")
            return 0.0, 0.0
    
    def _calculate_max_drawdown(self) -> float:
        """计算最大回撤"""
        if len(self.portfolio_value_history) < 2:
            return 0.0
        
        values = np.array(self.portfolio_value_history)
        peak = np.maximum.accumulate(values)
        drawdown = (values - peak) / peak
        
        return float(np.min(drawdown))
    
    async def _calculate_portfolio_volatility(
        self, 
        positions: Dict[str, Position],
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> float:
        """计算投资组合波动率"""
        try:
            portfolio_returns = self._get_portfolio_returns(positions, market_data)
            
            if len(portfolio_returns) < 10:
                return 0.02  # 默认2%日波动率
            
            return float(np.std(portfolio_returns))
            
        except Exception as e:
            self.logger.error(f"Volatility calculation failed: {e}")
            return 0.0
    
    async def _calculate_portfolio_beta(
        self, 
        positions: Dict[str, Position],
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> float:
        """计算投资组合贝塔值"""
        try:
            # 简化实现：假设贝塔值为1.0
            # 实际应用中需要计算与市场指数的相关性
            return 1.0
            
        except Exception as e:
            self.logger.error(f"Beta calculation failed: {e}")
            return 1.0
    
    def _calculate_concentration(self, positions: Dict[str, Position], total_value: float) -> float:
        """计算集中度指标（Herfindahl指数）"""
        if total_value <= 0 or not positions:
            return 0.0
        
        weights = [abs(float(pos.market_value)) / total_value for pos in positions.values()]
        return float(sum(w ** 2 for w in weights))
    
    def _calculate_leverage(self, positions: Dict[str, Position], total_value: float) -> float:
        """计算杠杆率"""
        if total_value <= 0:
            return 0.0
        
        total_exposure = sum(abs(float(pos.market_value)) for pos in positions.values())
        return total_exposure / total_value
    
    async def _calculate_liquidity_score(self, positions: Dict[str, Position]) -> float:
        """计算流动性评分"""
        # 简化实现：基于仓位数量
        # 实际应用中需要考虑交易量、买卖价差等因素
        if not positions:
            return 1.0
        
        # 假设所有资产流动性良好
        return 0.8
    
    def _calculate_overall_risk_score(
        self, 
        var_95: float, 
        max_drawdown: float, 
        volatility: float, 
        concentration: float, 
        leverage: float
    ) -> float:
        """计算综合风险评分"""
        # 简化的风险评分模型
        # 实际应用中需要更复杂的权重和标准化
        
        risk_score = 0.0
        
        # VaR贡献（0-30分）
        risk_score += min(30, var_95 / 1000 * 30)
        
        # 回撤贡献（0-25分）
        risk_score += min(25, abs(max_drawdown) * 100 * 25)
        
        # 波动率贡献（0-20分）
        risk_score += min(20, volatility * 100 * 20)
        
        # 集中度贡献（0-15分）
        risk_score += min(15, concentration * 15)
        
        # 杠杆贡献（0-10分）
        risk_score += min(10, max(0, leverage - 1) * 10)
        
        return min(100, risk_score)
    
    def _get_portfolio_returns(
        self, 
        positions: Dict[str, Position],
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> np.ndarray:
        """获取投资组合收益率序列"""
        # 简化实现：使用缓存的收益率数据
        if not self.portfolio_value_history:
            return np.array([])
        
        values = np.array(self.portfolio_value_history)
        if len(values) < 2:
            return np.array([])
        
        returns = np.diff(values) / values[:-1]
        return returns
    
    def _get_asset_sector(self, asset: str) -> str:
        """获取资产行业分类"""
        # 简化实现：基于资产名称推断
        # 实际应用中需要从数据库获取
        if 'BTC' in asset or 'ETH' in asset:
            return 'Crypto'
        elif 'USD' in asset or 'EUR' in asset:
            return 'Forex'
        else:
            return 'Other'
    
    def _create_empty_risk_metrics(self) -> RiskMetrics:
        """创建空的风险指标"""
        return RiskMetrics(
            var_95=0.0,
            var_99=0.0,
            cvar_95=0.0,
            cvar_99=0.0,
            max_drawdown=0.0,
            volatility=0.0,
            beta=1.0,
            sharpe_ratio=0.0,
            concentration=0.0,
            leverage=0.0,
            liquidity_score=1.0,
            risk_score=0.0
        )
    
    def update_portfolio_value(self, value: float) -> None:
        """更新投资组合价值历史"""
        self.portfolio_value_history.append(value)
        
        # 限制历史数据长度
        if len(self.portfolio_value_history) > 1000:
            self.portfolio_value_history = self.portfolio_value_history[-1000:]
    
    def get_current_violations(self) -> List[Dict[str, Any]]:
        """获取当前风险违规"""
        return [v.to_dict() for v in self.violations]
    
    def get_violation_history(
        self, 
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """获取历史风险违规"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_violations = [
            v for v in self.violation_history 
            if v.timestamp >= cutoff_time
        ]
        return [v.to_dict() for v in recent_violations]
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.risk_metrics_cache = None
        self.cache_timestamp = None
        self.logger.info("Risk manager cache cleared")