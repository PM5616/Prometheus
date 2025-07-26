"""Portfolio Optimizer

投资组合优化器，实现现代投资组合理论的优化算法
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal
from typing import Dict, List, Optional, Tuple, Any
from scipy.optimize import minimize
from scipy import linalg

from src.common.exceptions.base import PrometheusException
from src.common.models.base import BaseModel
from .models import Portfolio, OptimizationConfig


class OptimizationException(PrometheusException):
    """优化异常"""
    pass


class PortfolioOptimizer(BaseModel):
    """投资组合优化器
    
    实现多种投资组合优化方法：
    1. 均值方差优化 (Mean-Variance Optimization)
    2. Black-Litterman模型
    3. 风险平价 (Risk Parity)
    4. 最小方差优化
    5. 最大夏普比率优化
    """
    
    def __init__(self, config: OptimizationConfig):
        super().__init__()
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 历史数据缓存
        self.returns_cache: Dict[str, pd.Series] = {}
        self.covariance_cache: Optional[pd.DataFrame] = None
        self.expected_returns_cache: Optional[pd.Series] = None
        self.cache_timestamp: Optional[datetime] = None
        
        self.logger.info(f"Portfolio Optimizer initialized with method: {config.method}")
    
    async def optimize_portfolio(
        self, 
        portfolio: Portfolio, 
        current_weights: Dict[str, float],
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, float]:
        """优化投资组合权重
        
        Args:
            portfolio: 当前投资组合
            current_weights: 当前权重
            market_data: 市场数据
            
        Returns:
            优化后的权重字典
        """
        try:
            if not current_weights:
                self.logger.warning("No current weights provided, returning empty weights")
                return {}
            
            # 获取资产列表
            assets = list(current_weights.keys())
            
            # 准备数据
            if market_data:
                await self._prepare_data(assets, market_data)
            else:
                # 使用缓存数据或默认数据
                await self._prepare_default_data(assets)
            
            # 根据配置选择优化方法
            if self.config.method == "mean_variance":
                optimal_weights = await self._mean_variance_optimization(assets)
            elif self.config.method == "black_litterman":
                optimal_weights = await self._black_litterman_optimization(assets, current_weights)
            elif self.config.method == "risk_parity":
                optimal_weights = await self._risk_parity_optimization(assets)
            else:
                raise OptimizationException(f"Unknown optimization method: {self.config.method}")
            
            # 验证权重
            validated_weights = self._validate_weights(optimal_weights, assets)
            
            self.logger.info(f"Portfolio optimization completed: {validated_weights}")
            return validated_weights
            
        except Exception as e:
            self.logger.error(f"Portfolio optimization failed: {e}")
            # 返回当前权重作为fallback
            return current_weights
    
    async def _prepare_data(
        self, 
        assets: List[str], 
        market_data: Dict[str, pd.DataFrame]
    ) -> None:
        """准备优化所需的数据"""
        try:
            # 计算收益率
            returns_data = {}
            for asset in assets:
                if asset in market_data:
                    df = market_data[asset]
                    if 'close' in df.columns:
                        returns = df['close'].pct_change().dropna()
                        returns_data[asset] = returns
            
            if not returns_data:
                raise OptimizationException("No valid return data available")
            
            # 创建收益率DataFrame
            returns_df = pd.DataFrame(returns_data)
            returns_df = returns_df.dropna()
            
            if len(returns_df) < 30:  # 至少需要30个观测值
                raise OptimizationException("Insufficient historical data")
            
            # 计算期望收益率
            self.expected_returns_cache = returns_df.mean() * 252  # 年化
            
            # 计算协方差矩阵
            self.covariance_cache = returns_df.cov() * 252  # 年化
            
            # 缓存收益率数据
            for asset in assets:
                if asset in returns_df.columns:
                    self.returns_cache[asset] = returns_df[asset]
            
            self.cache_timestamp = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error preparing data: {e}")
            raise OptimizationException(f"Data preparation failed: {e}")
    
    async def _prepare_default_data(self, assets: List[str]) -> None:
        """准备默认数据（当没有市场数据时）"""
        # 使用默认的期望收益率和协方差矩阵
        default_return = 0.08  # 8%年化收益率
        default_volatility = 0.15  # 15%年化波动率
        default_correlation = 0.3  # 30%相关性
        
        # 创建期望收益率
        expected_returns = {asset: default_return for asset in assets}
        self.expected_returns_cache = pd.Series(expected_returns)
        
        # 创建协方差矩阵
        n_assets = len(assets)
        correlation_matrix = np.full((n_assets, n_assets), default_correlation)
        np.fill_diagonal(correlation_matrix, 1.0)
        
        volatilities = np.full(n_assets, default_volatility)
        covariance_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        self.covariance_cache = pd.DataFrame(
            covariance_matrix, 
            index=assets, 
            columns=assets
        )
        
        self.cache_timestamp = datetime.now()
        self.logger.warning("Using default data for optimization")
    
    async def _mean_variance_optimization(self, assets: List[str]) -> Dict[str, float]:
        """均值方差优化"""
        if self.expected_returns_cache is None or self.covariance_cache is None:
            raise OptimizationException("Missing expected returns or covariance data")
        
        n_assets = len(assets)
        
        # 目标函数
        def objective(weights):
            if self.config.objective == "max_sharpe":
                return -self._calculate_sharpe_ratio(weights)
            elif self.config.objective == "min_variance":
                return self._calculate_portfolio_variance(weights)
            elif self.config.objective == "max_return":
                return -self._calculate_portfolio_return(weights)
            else:
                return self._calculate_portfolio_variance(weights)
        
        # 约束条件
        constraints = []
        
        # 权重和为1
        constraints.append({
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        })
        
        # 目标收益率约束
        if self.config.target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda weights: self._calculate_portfolio_return(weights) - self.config.target_return
            })
        
        # 目标波动率约束
        if self.config.target_volatility is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda weights: np.sqrt(self._calculate_portfolio_variance(weights)) - self.config.target_volatility
            })
        
        # 权重边界
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # 初始权重（等权重）
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 优化
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            self.logger.warning(f"Optimization failed: {result.message}")
            # 返回等权重
            return {asset: 1.0 / n_assets for asset in assets}
        
        # 返回优化结果
        optimal_weights = {asset: float(weight) for asset, weight in zip(assets, result.x)}
        return optimal_weights
    
    async def _black_litterman_optimization(
        self, 
        assets: List[str], 
        current_weights: Dict[str, float]
    ) -> Dict[str, float]:
        """Black-Litterman优化"""
        if self.expected_returns_cache is None or self.covariance_cache is None:
            raise OptimizationException("Missing data for Black-Litterman optimization")
        
        # 简化的Black-Litterman实现
        # 在实际应用中需要更复杂的观点矩阵和置信度设置
        
        # 市场隐含收益率（使用当前权重反推）
        current_weights_array = np.array([current_weights.get(asset, 0.0) for asset in assets])
        risk_aversion = self.config.risk_aversion
        
        # 市场隐含收益率 = 风险厌恶系数 * 协方差矩阵 * 市场权重
        implied_returns = risk_aversion * self.covariance_cache.values @ current_weights_array
        
        # 使用隐含收益率作为新的期望收益率
        bl_expected_returns = pd.Series(implied_returns, index=assets)
        
        # 使用均值方差优化
        original_expected_returns = self.expected_returns_cache.copy()
        self.expected_returns_cache = bl_expected_returns
        
        try:
            optimal_weights = await self._mean_variance_optimization(assets)
        finally:
            # 恢复原始期望收益率
            self.expected_returns_cache = original_expected_returns
        
        return optimal_weights
    
    async def _risk_parity_optimization(self, assets: List[str]) -> Dict[str, float]:
        """风险平价优化"""
        if self.covariance_cache is None:
            raise OptimizationException("Missing covariance data for risk parity optimization")
        
        n_assets = len(assets)
        
        # 目标函数：最小化风险贡献的差异
        def objective(weights):
            portfolio_variance = self._calculate_portfolio_variance(weights)
            if portfolio_variance == 0:
                return 1e6
            
            # 计算每个资产的风险贡献
            marginal_contrib = self.covariance_cache.values @ weights
            risk_contrib = weights * marginal_contrib / portfolio_variance
            
            # 目标风险贡献（等权重）
            target_contrib = 1.0 / n_assets
            
            # 最小化风险贡献与目标的差异
            return np.sum((risk_contrib - target_contrib) ** 2)
        
        # 约束条件
        constraints = [{
            'type': 'eq',
            'fun': lambda weights: np.sum(weights) - 1.0
        }]
        
        # 权重边界
        bounds = [(self.config.min_weight, self.config.max_weight) for _ in range(n_assets)]
        
        # 初始权重
        initial_weights = np.array([1.0 / n_assets] * n_assets)
        
        # 优化
        result = minimize(
            objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if not result.success:
            self.logger.warning(f"Risk parity optimization failed: {result.message}")
            return {asset: 1.0 / n_assets for asset in assets}
        
        optimal_weights = {asset: float(weight) for asset, weight in zip(assets, result.x)}
        return optimal_weights
    
    def _calculate_portfolio_return(self, weights: np.ndarray) -> float:
        """计算投资组合期望收益率"""
        if self.expected_returns_cache is None:
            return 0.0
        return float(np.dot(weights, self.expected_returns_cache.values))
    
    def _calculate_portfolio_variance(self, weights: np.ndarray) -> float:
        """计算投资组合方差"""
        if self.covariance_cache is None:
            return 0.0
        return float(weights.T @ self.covariance_cache.values @ weights)
    
    def _calculate_sharpe_ratio(self, weights: np.ndarray, risk_free_rate: float = 0.02) -> float:
        """计算夏普比率"""
        portfolio_return = self._calculate_portfolio_return(weights)
        portfolio_variance = self._calculate_portfolio_variance(weights)
        
        if portfolio_variance == 0:
            return 0.0
        
        portfolio_volatility = np.sqrt(portfolio_variance)
        return (portfolio_return - risk_free_rate) / portfolio_volatility
    
    def _validate_weights(self, weights: Dict[str, float], assets: List[str]) -> Dict[str, float]:
        """验证和调整权重"""
        validated_weights = {}
        
        # 确保所有资产都有权重
        for asset in assets:
            weight = weights.get(asset, 0.0)
            
            # 应用权重限制
            weight = max(self.config.min_weight, min(self.config.max_weight, weight))
            validated_weights[asset] = weight
        
        # 归一化权重
        total_weight = sum(validated_weights.values())
        if total_weight > 0:
            validated_weights = {
                asset: weight / total_weight 
                for asset, weight in validated_weights.items()
            }
        else:
            # 如果总权重为0，使用等权重
            n_assets = len(assets)
            validated_weights = {asset: 1.0 / n_assets for asset in assets}
        
        return validated_weights
    
    async def calculate_efficient_frontier(
        self, 
        assets: List[str], 
        n_points: int = 100
    ) -> Tuple[List[float], List[float], List[Dict[str, float]]]:
        """计算有效前沿
        
        Returns:
            (returns, volatilities, weights_list)
        """
        if self.expected_returns_cache is None or self.covariance_cache is None:
            raise OptimizationException("Missing data for efficient frontier calculation")
        
        min_return = float(self.expected_returns_cache.min())
        max_return = float(self.expected_returns_cache.max())
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        returns = []
        volatilities = []
        weights_list = []
        
        for target_return in target_returns:
            try:
                # 临时设置目标收益率
                original_target = self.config.target_return
                original_objective = self.config.objective
                
                self.config.target_return = target_return
                self.config.objective = "min_variance"
                
                optimal_weights = await self._mean_variance_optimization(assets)
                
                # 计算组合指标
                weights_array = np.array([optimal_weights[asset] for asset in assets])
                portfolio_return = self._calculate_portfolio_return(weights_array)
                portfolio_variance = self._calculate_portfolio_variance(weights_array)
                portfolio_volatility = np.sqrt(portfolio_variance)
                
                returns.append(portfolio_return)
                volatilities.append(portfolio_volatility)
                weights_list.append(optimal_weights)
                
                # 恢复原始配置
                self.config.target_return = original_target
                self.config.objective = original_objective
                
            except Exception as e:
                self.logger.warning(f"Failed to calculate point on efficient frontier: {e}")
                continue
        
        return returns, volatilities, weights_list
    
    async def get_optimization_metrics(
        self, 
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """获取优化指标"""
        if not weights:
            return {}
        
        assets = list(weights.keys())
        weights_array = np.array([weights[asset] for asset in assets])
        
        metrics = {
            'expected_return': self._calculate_portfolio_return(weights_array),
            'volatility': np.sqrt(self._calculate_portfolio_variance(weights_array)),
            'sharpe_ratio': self._calculate_sharpe_ratio(weights_array)
        }
        
        # 计算集中度指标
        metrics['concentration'] = np.sum(weights_array ** 2)  # Herfindahl指数
        metrics['max_weight'] = float(np.max(weights_array))
        metrics['min_weight'] = float(np.min(weights_array))
        
        return metrics
    
    def clear_cache(self) -> None:
        """清空缓存"""
        self.returns_cache.clear()
        self.covariance_cache = None
        self.expected_returns_cache = None
        self.cache_timestamp = None
        self.logger.info("Optimizer cache cleared")