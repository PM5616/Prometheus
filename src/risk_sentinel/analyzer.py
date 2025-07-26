"""Risk Analyzer

风险分析器 - 负责深度风险分析和报告
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from scipy import stats
from sklearn.covariance import EmpiricalCovariance
import warnings
warnings.filterwarnings('ignore')

from .models import (
    RiskType, RiskLevel, RiskMetric, RiskEvent, StressTestScenario, RiskConfig
)
from ..common.logger import get_logger
from ..portfolio_manager.models import Portfolio, Position
from ..datahub.data_manager import DataManager


@dataclass
class VaRResult:
    """VaR计算结果"""
    var_1d: float  # 1日VaR
    var_5d: float  # 5日VaR
    var_10d: float  # 10日VaR
    cvar_1d: float  # 1日CVaR
    cvar_5d: float  # 5日CVaR
    cvar_10d: float  # 10日CVaR
    confidence_level: float
    method: str
    calculation_date: datetime


@dataclass
class StressTestResult:
    """压力测试结果"""
    scenario_id: str
    scenario_name: str
    portfolio_pnl: float
    portfolio_pnl_pct: float
    position_pnl: Dict[str, float]
    risk_metrics: Dict[str, float]
    breach_limits: List[str]
    calculation_date: datetime


@dataclass
class RiskAttribution:
    """风险归因"""
    total_risk: float
    systematic_risk: float
    idiosyncratic_risk: float
    factor_contributions: Dict[str, float]
    position_contributions: Dict[str, float]
    sector_contributions: Dict[str, float]


class RiskAnalyzer:
    """风险分析器
    
    负责:
    1. VaR/CVaR计算
    2. 压力测试
    3. 风险归因分析
    4. 相关性分析
    5. 风险报告生成
    """
    
    def __init__(self, config: RiskConfig, data_manager: DataManager):
        self.config = config
        self.data_manager = data_manager
        self.logger = get_logger(self.__class__.__name__)
        
        # 分析缓存
        self.var_cache: Dict[str, VaRResult] = {}
        self.stress_test_cache: Dict[str, StressTestResult] = {}
        self.correlation_cache: Dict[str, pd.DataFrame] = {}
        
        # 风险因子
        self.risk_factors = [
            'market_factor', 'size_factor', 'value_factor', 
            'momentum_factor', 'quality_factor', 'volatility_factor'
        ]
        
        # 压力测试场景
        self.stress_scenarios = self._setup_stress_scenarios()
    
    def calculate_var(self, portfolio: Portfolio, 
                     confidence_levels: List[float] = None,
                     holding_periods: List[int] = None,
                     method: str = None) -> Dict[float, VaRResult]:
        """计算VaR"""
        if confidence_levels is None:
            confidence_levels = [0.95, 0.99]
        if holding_periods is None:
            holding_periods = [1, 5, 10]
        if method is None:
            method = self.config.var_method
        
        results = {}
        
        try:
            # 获取历史收益率数据
            returns_data = self._get_portfolio_returns(portfolio)
            if returns_data is None or len(returns_data) < 30:
                self.logger.warning("Insufficient data for VaR calculation")
                return results
            
            for confidence_level in confidence_levels:
                if method == 'historical':
                    var_result = self._calculate_historical_var(
                        returns_data, confidence_level, holding_periods
                    )
                elif method == 'parametric':
                    var_result = self._calculate_parametric_var(
                        returns_data, confidence_level, holding_periods
                    )
                elif method == 'monte_carlo':
                    var_result = self._calculate_monte_carlo_var(
                        returns_data, confidence_level, holding_periods
                    )
                else:
                    raise ValueError(f"Unknown VaR method: {method}")
                
                var_result.method = method
                var_result.confidence_level = confidence_level
                var_result.calculation_date = datetime.now()
                
                results[confidence_level] = var_result
                
                # 缓存结果
                cache_key = f"{portfolio.portfolio_id}_{confidence_level}_{method}"
                self.var_cache[cache_key] = var_result
        
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
        
        return results
    
    def run_stress_test(self, portfolio: Portfolio, 
                       scenarios: List[StressTestScenario] = None) -> Dict[str, StressTestResult]:
        """运行压力测试"""
        if scenarios is None:
            scenarios = list(self.stress_scenarios.values())
        
        results = {}
        
        try:
            for scenario in scenarios:
                result = self._run_single_stress_test(portfolio, scenario)
                results[scenario.scenario_id] = result
                
                # 缓存结果
                cache_key = f"{portfolio.portfolio_id}_{scenario.scenario_id}"
                self.stress_test_cache[cache_key] = result
        
        except Exception as e:
            self.logger.error(f"Error running stress test: {e}")
        
        return results
    
    def calculate_risk_attribution(self, portfolio: Portfolio) -> RiskAttribution:
        """计算风险归因"""
        try:
            # 获取收益率数据
            returns_data = self._get_portfolio_returns(portfolio)
            if returns_data is None:
                raise ValueError("No returns data available")
            
            # 获取因子数据
            factor_data = self._get_factor_returns()
            if factor_data is None:
                raise ValueError("No factor data available")
            
            # 计算总风险
            total_risk = np.std(returns_data) * np.sqrt(252)  # 年化波动率
            
            # 因子回归
            factor_loadings, residuals = self._perform_factor_regression(
                returns_data, factor_data
            )
            
            # 计算系统性风险和特异性风险
            systematic_risk = self._calculate_systematic_risk(
                factor_loadings, factor_data
            )
            idiosyncratic_risk = np.std(residuals) * np.sqrt(252)
            
            # 因子贡献
            factor_contributions = self._calculate_factor_contributions(
                factor_loadings, factor_data
            )
            
            # 持仓贡献
            position_contributions = self._calculate_position_contributions(
                portfolio, returns_data
            )
            
            # 行业贡献
            sector_contributions = self._calculate_sector_contributions(
                portfolio, returns_data
            )
            
            return RiskAttribution(
                total_risk=total_risk,
                systematic_risk=systematic_risk,
                idiosyncratic_risk=idiosyncratic_risk,
                factor_contributions=factor_contributions,
                position_contributions=position_contributions,
                sector_contributions=sector_contributions
            )
        
        except Exception as e:
            self.logger.error(f"Error calculating risk attribution: {e}")
            return None
    
    def calculate_correlation_matrix(self, symbols: List[str], 
                                   lookback_days: int = 252) -> pd.DataFrame:
        """计算相关性矩阵"""
        try:
            # 获取价格数据
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback_days + 30)
            
            price_data = {}
            for symbol in symbols:
                prices = self.data_manager.get_price_data(
                    symbol, start_date, end_date
                )
                if prices is not None and len(prices) > 0:
                    price_data[symbol] = prices['close']
            
            if not price_data:
                return pd.DataFrame()
            
            # 创建价格DataFrame
            price_df = pd.DataFrame(price_data)
            price_df = price_df.dropna()
            
            # 计算收益率
            returns_df = price_df.pct_change().dropna()
            
            # 计算相关性矩阵
            correlation_matrix = returns_df.corr()
            
            # 缓存结果
            cache_key = f"corr_{len(symbols)}_{lookback_days}"
            self.correlation_cache[cache_key] = correlation_matrix
            
            return correlation_matrix
        
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def generate_risk_report(self, portfolio: Portfolio) -> Dict[str, Any]:
        """生成风险报告"""
        report = {
            'portfolio_id': portfolio.portfolio_id,
            'report_date': datetime.now(),
            'portfolio_summary': self._get_portfolio_summary(portfolio),
            'var_analysis': {},
            'stress_test_results': {},
            'risk_attribution': None,
            'correlation_analysis': {},
            'risk_metrics': {},
            'recommendations': []
        }
        
        try:
            # VaR分析
            var_results = self.calculate_var(portfolio)
            report['var_analysis'] = {
                conf_level: {
                    'var_1d': result.var_1d,
                    'var_5d': result.var_5d,
                    'var_10d': result.var_10d,
                    'cvar_1d': result.cvar_1d,
                    'cvar_5d': result.cvar_5d,
                    'cvar_10d': result.cvar_10d,
                    'method': result.method
                }
                for conf_level, result in var_results.items()
            }
            
            # 压力测试
            stress_results = self.run_stress_test(portfolio)
            report['stress_test_results'] = {
                scenario_id: {
                    'scenario_name': result.scenario_name,
                    'portfolio_pnl': result.portfolio_pnl,
                    'portfolio_pnl_pct': result.portfolio_pnl_pct,
                    'breach_limits': result.breach_limits
                }
                for scenario_id, result in stress_results.items()
            }
            
            # 风险归因
            risk_attribution = self.calculate_risk_attribution(portfolio)
            if risk_attribution:
                report['risk_attribution'] = {
                    'total_risk': risk_attribution.total_risk,
                    'systematic_risk': risk_attribution.systematic_risk,
                    'idiosyncratic_risk': risk_attribution.idiosyncratic_risk,
                    'factor_contributions': risk_attribution.factor_contributions,
                    'position_contributions': risk_attribution.position_contributions,
                    'sector_contributions': risk_attribution.sector_contributions
                }
            
            # 相关性分析
            symbols = list(portfolio.positions.keys())
            if symbols:
                correlation_matrix = self.calculate_correlation_matrix(symbols)
                if not correlation_matrix.empty:
                    report['correlation_analysis'] = {
                        'avg_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].mean(),
                        'max_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].max(),
                        'min_correlation': correlation_matrix.values[np.triu_indices_from(correlation_matrix.values, k=1)].min(),
                        'high_correlation_pairs': self._find_high_correlation_pairs(correlation_matrix)
                    }
            
            # 风险指标
            report['risk_metrics'] = self._calculate_comprehensive_risk_metrics(portfolio)
            
            # 生成建议
            report['recommendations'] = self._generate_risk_recommendations(report)
        
        except Exception as e:
            self.logger.error(f"Error generating risk report: {e}")
        
        return report
    
    def _get_portfolio_returns(self, portfolio: Portfolio) -> Optional[pd.Series]:
        """获取投资组合收益率"""
        try:
            # 这里需要从数据管理器获取投资组合的历史收益率
            # 简化实现，实际应该从历史数据计算
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # 获取投资组合历史净值数据
            nav_data = self.data_manager.get_portfolio_nav(
                portfolio.portfolio_id, start_date, end_date
            )
            
            if nav_data is not None and len(nav_data) > 1:
                returns = nav_data.pct_change().dropna()
                return returns
            
            return None
        
        except Exception as e:
            self.logger.error(f"Error getting portfolio returns: {e}")
            return None
    
    def _calculate_historical_var(self, returns: pd.Series, 
                                 confidence_level: float,
                                 holding_periods: List[int]) -> VaRResult:
        """计算历史VaR"""
        # 排序收益率
        sorted_returns = np.sort(returns.values)
        
        # 计算分位数
        alpha = 1 - confidence_level
        var_index = int(alpha * len(sorted_returns))
        
        # 1日VaR
        var_1d = -sorted_returns[var_index]
        
        # 多日VaR（假设独立同分布）
        var_5d = var_1d * np.sqrt(5) if 5 in holding_periods else 0
        var_10d = var_1d * np.sqrt(10) if 10 in holding_periods else 0
        
        # CVaR（条件VaR）
        cvar_1d = -np.mean(sorted_returns[:var_index]) if var_index > 0 else var_1d
        cvar_5d = cvar_1d * np.sqrt(5) if 5 in holding_periods else 0
        cvar_10d = cvar_1d * np.sqrt(10) if 10 in holding_periods else 0
        
        return VaRResult(
            var_1d=var_1d,
            var_5d=var_5d,
            var_10d=var_10d,
            cvar_1d=cvar_1d,
            cvar_5d=cvar_5d,
            cvar_10d=cvar_10d,
            confidence_level=confidence_level,
            method='historical',
            calculation_date=datetime.now()
        )
    
    def _calculate_parametric_var(self, returns: pd.Series, 
                                 confidence_level: float,
                                 holding_periods: List[int]) -> VaRResult:
        """计算参数VaR"""
        # 计算均值和标准差
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # 计算Z分数
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)
        
        # 1日VaR
        var_1d = -(mean_return + z_score * std_return)
        
        # 多日VaR
        var_5d = var_1d * np.sqrt(5) if 5 in holding_periods else 0
        var_10d = var_1d * np.sqrt(10) if 10 in holding_periods else 0
        
        # CVaR（正态分布假设下）
        cvar_1d = -(mean_return - std_return * stats.norm.pdf(z_score) / alpha)
        cvar_5d = cvar_1d * np.sqrt(5) if 5 in holding_periods else 0
        cvar_10d = cvar_1d * np.sqrt(10) if 10 in holding_periods else 0
        
        return VaRResult(
            var_1d=var_1d,
            var_5d=var_5d,
            var_10d=var_10d,
            cvar_1d=cvar_1d,
            cvar_5d=cvar_5d,
            cvar_10d=cvar_10d,
            confidence_level=confidence_level,
            method='parametric',
            calculation_date=datetime.now()
        )
    
    def _calculate_monte_carlo_var(self, returns: pd.Series, 
                                  confidence_level: float,
                                  holding_periods: List[int],
                                  num_simulations: int = 10000) -> VaRResult:
        """计算蒙特卡洛VaR"""
        # 拟合分布参数
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # 生成随机收益率
        np.random.seed(42)  # 为了结果可重复
        simulated_returns = np.random.normal(mean_return, std_return, num_simulations)
        
        # 排序
        sorted_returns = np.sort(simulated_returns)
        
        # 计算VaR
        alpha = 1 - confidence_level
        var_index = int(alpha * num_simulations)
        
        var_1d = -sorted_returns[var_index]
        var_5d = var_1d * np.sqrt(5) if 5 in holding_periods else 0
        var_10d = var_1d * np.sqrt(10) if 10 in holding_periods else 0
        
        # CVaR
        cvar_1d = -np.mean(sorted_returns[:var_index]) if var_index > 0 else var_1d
        cvar_5d = cvar_1d * np.sqrt(5) if 5 in holding_periods else 0
        cvar_10d = cvar_1d * np.sqrt(10) if 10 in holding_periods else 0
        
        return VaRResult(
            var_1d=var_1d,
            var_5d=var_5d,
            var_10d=var_10d,
            cvar_1d=cvar_1d,
            cvar_5d=cvar_5d,
            cvar_10d=cvar_10d,
            confidence_level=confidence_level,
            method='monte_carlo',
            calculation_date=datetime.now()
        )
    
    def _run_single_stress_test(self, portfolio: Portfolio, 
                               scenario: StressTestScenario) -> StressTestResult:
        """运行单个压力测试"""
        try:
            total_pnl = 0
            position_pnl = {}
            
            # 计算每个持仓的损益
            for symbol, position in portfolio.positions.items():
                if position.quantity == 0:
                    continue
                
                # 获取冲击因子
                shock_factor = scenario.shock_factors.get(symbol, 0)
                
                # 计算损益
                pnl = position.market_value * shock_factor
                position_pnl[symbol] = pnl
                total_pnl += pnl
            
            # 计算百分比损益
            portfolio_value = portfolio.total_value
            pnl_pct = total_pnl / portfolio_value if portfolio_value > 0 else 0
            
            # 计算压力测试后的风险指标
            risk_metrics = self._calculate_stressed_risk_metrics(
                portfolio, scenario, total_pnl
            )
            
            # 检查限制突破
            breach_limits = self._check_stress_test_breaches(
                risk_metrics, pnl_pct
            )
            
            return StressTestResult(
                scenario_id=scenario.scenario_id,
                scenario_name=scenario.name,
                portfolio_pnl=total_pnl,
                portfolio_pnl_pct=pnl_pct,
                position_pnl=position_pnl,
                risk_metrics=risk_metrics,
                breach_limits=breach_limits,
                calculation_date=datetime.now()
            )
        
        except Exception as e:
            self.logger.error(f"Error running stress test {scenario.scenario_id}: {e}")
            return None
    
    def _get_factor_returns(self) -> Optional[pd.DataFrame]:
        """获取因子收益率数据"""
        # 这里应该从数据源获取因子数据
        # 简化实现，返回模拟数据
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            # 模拟因子数据
            dates = pd.date_range(start_date, end_date, freq='D')
            factor_data = pd.DataFrame(index=dates)
            
            np.random.seed(42)
            for factor in self.risk_factors:
                factor_data[factor] = np.random.normal(0, 0.01, len(dates))
            
            return factor_data.dropna()
        
        except Exception as e:
            self.logger.error(f"Error getting factor returns: {e}")
            return None
    
    def _perform_factor_regression(self, returns: pd.Series, 
                                  factor_data: pd.DataFrame) -> Tuple[Dict[str, float], pd.Series]:
        """执行因子回归"""
        try:
            # 对齐数据
            aligned_data = pd.concat([returns, factor_data], axis=1, join='inner')
            aligned_data = aligned_data.dropna()
            
            y = aligned_data.iloc[:, 0]  # 收益率
            X = aligned_data.iloc[:, 1:]  # 因子
            
            # 添加常数项
            X = X.copy()
            X['const'] = 1
            
            # 回归
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X.drop('const', axis=1), y)
            
            # 因子载荷
            factor_loadings = dict(zip(X.drop('const', axis=1).columns, model.coef_))
            
            # 残差
            y_pred = model.predict(X.drop('const', axis=1))
            residuals = y - y_pred
            
            return factor_loadings, residuals
        
        except Exception as e:
            self.logger.error(f"Error in factor regression: {e}")
            return {}, pd.Series()
    
    def _setup_stress_scenarios(self) -> Dict[str, StressTestScenario]:
        """设置压力测试场景"""
        scenarios = {}
        
        # 市场崩盘场景
        scenarios['market_crash'] = StressTestScenario(
            scenario_id='market_crash',
            name='Market Crash',
            description='Severe market downturn with 20% decline',
            scenario_type='market_crash',
            shock_factors={'default': -0.20}  # 默认20%下跌
        )
        
        # 波动率飙升场景
        scenarios['volatility_spike'] = StressTestScenario(
            scenario_id='volatility_spike',
            name='Volatility Spike',
            description='Sudden increase in market volatility',
            scenario_type='volatility_spike',
            shock_factors={'default': -0.10}  # 默认10%下跌
        )
        
        # 流动性危机场景
        scenarios['liquidity_crisis'] = StressTestScenario(
            scenario_id='liquidity_crisis',
            name='Liquidity Crisis',
            description='Severe liquidity shortage',
            scenario_type='liquidity_crisis',
            shock_factors={'default': -0.15}  # 默认15%下跌
        )
        
        return scenarios
    
    def _calculate_stressed_risk_metrics(self, portfolio: Portfolio, 
                                       scenario: StressTestScenario,
                                       total_pnl: float) -> Dict[str, float]:
        """计算压力测试后的风险指标"""
        metrics = {}
        
        try:
            # 压力测试后的投资组合价值
            stressed_value = portfolio.total_value + total_pnl
            
            # 损失比例
            loss_ratio = abs(total_pnl) / portfolio.total_value if portfolio.total_value > 0 else 0
            
            metrics['stressed_portfolio_value'] = stressed_value
            metrics['loss_ratio'] = loss_ratio
            metrics['absolute_loss'] = abs(total_pnl)
            
            # 其他风险指标可以在这里添加
            
        except Exception as e:
            self.logger.error(f"Error calculating stressed risk metrics: {e}")
        
        return metrics
    
    def _check_stress_test_breaches(self, risk_metrics: Dict[str, float], 
                                   pnl_pct: float) -> List[str]:
        """检查压力测试限制突破"""
        breaches = []
        
        # 检查损失是否超过限制
        if abs(pnl_pct) > self.config.max_portfolio_var:
            breaches.append(f"Portfolio loss {abs(pnl_pct):.2%} exceeds VaR limit {self.config.max_portfolio_var:.2%}")
        
        if abs(pnl_pct) > self.config.max_drawdown:
            breaches.append(f"Portfolio loss {abs(pnl_pct):.2%} exceeds drawdown limit {self.config.max_drawdown:.2%}")
        
        return breaches
    
    def _calculate_systematic_risk(self, factor_loadings: Dict[str, float], 
                                  factor_data: pd.DataFrame) -> float:
        """计算系统性风险"""
        try:
            # 计算因子协方差矩阵
            factor_cov = factor_data.cov() * 252  # 年化
            
            # 因子载荷向量
            loadings = np.array([factor_loadings.get(factor, 0) for factor in factor_data.columns])
            
            # 系统性风险 = sqrt(loadings' * Cov * loadings)
            systematic_variance = np.dot(loadings, np.dot(factor_cov.values, loadings))
            systematic_risk = np.sqrt(systematic_variance)
            
            return systematic_risk
        
        except Exception as e:
            self.logger.error(f"Error calculating systematic risk: {e}")
            return 0.0
    
    def _calculate_factor_contributions(self, factor_loadings: Dict[str, float], 
                                      factor_data: pd.DataFrame) -> Dict[str, float]:
        """计算因子贡献"""
        contributions = {}
        
        try:
            factor_volatilities = factor_data.std() * np.sqrt(252)  # 年化波动率
            
            for factor, loading in factor_loadings.items():
                if factor in factor_volatilities:
                    contribution = abs(loading) * factor_volatilities[factor]
                    contributions[factor] = contribution
        
        except Exception as e:
            self.logger.error(f"Error calculating factor contributions: {e}")
        
        return contributions
    
    def _calculate_position_contributions(self, portfolio: Portfolio, 
                                        returns: pd.Series) -> Dict[str, float]:
        """计算持仓贡献"""
        contributions = {}
        
        try:
            total_value = portfolio.total_value
            if total_value <= 0:
                return contributions
            
            for symbol, position in portfolio.positions.items():
                if position.quantity != 0:
                    weight = abs(position.market_value) / total_value
                    # 简化计算，实际应该使用持仓的历史波动率
                    contribution = weight * np.std(returns) * np.sqrt(252)
                    contributions[symbol] = contribution
        
        except Exception as e:
            self.logger.error(f"Error calculating position contributions: {e}")
        
        return contributions
    
    def _calculate_sector_contributions(self, portfolio: Portfolio, 
                                      returns: pd.Series) -> Dict[str, float]:
        """计算行业贡献"""
        contributions = {}
        
        try:
            # 按行业分组
            sector_weights = {}
            total_value = portfolio.total_value
            
            if total_value <= 0:
                return contributions
            
            for symbol, position in portfolio.positions.items():
                if position.quantity != 0:
                    sector = getattr(position, 'sector', 'Unknown')
                    weight = abs(position.market_value) / total_value
                    
                    if sector not in sector_weights:
                        sector_weights[sector] = 0
                    sector_weights[sector] += weight
            
            # 计算行业贡献（简化）
            portfolio_volatility = np.std(returns) * np.sqrt(252)
            for sector, weight in sector_weights.items():
                contributions[sector] = weight * portfolio_volatility
        
        except Exception as e:
            self.logger.error(f"Error calculating sector contributions: {e}")
        
        return contributions
    
    def _get_portfolio_summary(self, portfolio: Portfolio) -> Dict[str, Any]:
        """获取投资组合摘要"""
        return {
            'total_value': portfolio.total_value,
            'total_positions': len([p for p in portfolio.positions.values() if p.quantity != 0]),
            'cash': portfolio.cash,
            'leverage': getattr(portfolio, 'leverage', 1.0)
        }
    
    def _calculate_comprehensive_risk_metrics(self, portfolio: Portfolio) -> Dict[str, float]:
        """计算综合风险指标"""
        metrics = {}
        
        try:
            total_value = portfolio.total_value
            if total_value <= 0:
                return metrics
            
            # 集中度指标
            position_weights = []
            sector_weights = {}
            
            for position in portfolio.positions.values():
                if position.quantity != 0:
                    weight = abs(position.market_value) / total_value
                    position_weights.append(weight)
                    
                    sector = getattr(position, 'sector', 'Unknown')
                    if sector not in sector_weights:
                        sector_weights[sector] = 0
                    sector_weights[sector] += weight
            
            if position_weights:
                metrics['max_position_weight'] = max(position_weights)
                metrics['avg_position_weight'] = np.mean(position_weights)
                metrics['position_concentration_hhi'] = sum(w**2 for w in position_weights)
            
            if sector_weights:
                metrics['max_sector_weight'] = max(sector_weights.values())
                metrics['sector_concentration_hhi'] = sum(w**2 for w in sector_weights.values())
                metrics['num_sectors'] = len(sector_weights)
            
            # 其他指标
            metrics['leverage'] = getattr(portfolio, 'leverage', 1.0)
            metrics['cash_ratio'] = portfolio.cash / total_value
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive risk metrics: {e}")
        
        return metrics
    
    def _find_high_correlation_pairs(self, correlation_matrix: pd.DataFrame, 
                                   threshold: float = 0.8) -> List[Tuple[str, str, float]]:
        """找出高相关性配对"""
        high_corr_pairs = []
        
        try:
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr = correlation_matrix.iloc[i, j]
                    if abs(corr) >= threshold:
                        symbol1 = correlation_matrix.columns[i]
                        symbol2 = correlation_matrix.columns[j]
                        high_corr_pairs.append((symbol1, symbol2, corr))
        
        except Exception as e:
            self.logger.error(f"Error finding high correlation pairs: {e}")
        
        return high_corr_pairs
    
    def _generate_risk_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """生成风险建议"""
        recommendations = []
        
        try:
            # 基于VaR分析的建议
            var_analysis = report.get('var_analysis', {})
            for conf_level, var_data in var_analysis.items():
                if var_data.get('var_1d', 0) > self.config.max_portfolio_var:
                    recommendations.append(
                        f"Portfolio VaR ({var_data['var_1d']:.2%}) exceeds limit ({self.config.max_portfolio_var:.2%}). Consider reducing position sizes."
                    )
            
            # 基于压力测试的建议
            stress_results = report.get('stress_test_results', {})
            for scenario_id, stress_data in stress_results.items():
                if stress_data.get('breach_limits'):
                    recommendations.append(
                        f"Stress test '{stress_data['scenario_name']}' shows potential breaches. Consider hedging strategies."
                    )
            
            # 基于相关性分析的建议
            corr_analysis = report.get('correlation_analysis', {})
            if corr_analysis.get('avg_correlation', 0) > 0.7:
                recommendations.append(
                    "High average correlation detected. Consider diversifying across uncorrelated assets."
                )
            
            # 基于集中度的建议
            risk_metrics = report.get('risk_metrics', {})
            if risk_metrics.get('max_position_weight', 0) > self.config.max_position_concentration:
                recommendations.append(
                    "Position concentration exceeds limit. Consider reducing largest positions."
                )
            
            if risk_metrics.get('max_sector_weight', 0) > self.config.max_sector_concentration:
                recommendations.append(
                    "Sector concentration exceeds limit. Consider diversifying across sectors."
                )
        
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
        
        return recommendations