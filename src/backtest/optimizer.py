"""Parameter Optimizer Module

参数优化模块，负责策略参数的优化和调参。

主要功能：
- 参数空间定义
- 优化算法实现
- 并行计算支持
- 结果分析
- 过拟合检测
- 稳健性测试
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import itertools
import random
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from abc import ABC, abstractmethod
import warnings

try:
    from scipy.optimize import minimize, differential_evolution, basinhopping
    from scipy.stats import pearsonr
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.model_selection import ParameterGrid, ParameterSampler
    from sklearn.metrics import mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..common.logging import get_logger
from ..common.exceptions.backtest import BacktestOptimizationError
from .metrics import BacktestMetrics, MetricsCalculator


class OptimizationMethod(Enum):
    """优化方法枚举"""
    GRID_SEARCH = "grid_search"              # 网格搜索
    RANDOM_SEARCH = "random_search"          # 随机搜索
    GENETIC_ALGORITHM = "genetic_algorithm"  # 遗传算法
    PARTICLE_SWARM = "particle_swarm"        # 粒子群优化
    BAYESIAN = "bayesian"                    # 贝叶斯优化
    DIFFERENTIAL_EVOLUTION = "diff_evolution" # 差分进化
    SIMULATED_ANNEALING = "simulated_annealing" # 模拟退火


class ObjectiveFunction(Enum):
    """目标函数枚举"""
    SHARPE_RATIO = "sharpe_ratio"            # 夏普比率
    CALMAR_RATIO = "calmar_ratio"            # 卡玛比率
    SORTINO_RATIO = "sortino_ratio"          # 索提诺比率
    TOTAL_RETURN = "total_return"            # 总收益率
    ANNUAL_RETURN = "annual_return"          # 年化收益率
    MAX_DRAWDOWN = "max_drawdown"            # 最大回撤（最小化）
    PROFIT_FACTOR = "profit_factor"          # 盈利因子
    WIN_RATE = "win_rate"                    # 胜率
    CUSTOM = "custom"                        # 自定义


@dataclass
class ParameterRange:
    """参数范围定义"""
    name: str
    min_value: Union[int, float]
    max_value: Union[int, float]
    step: Optional[Union[int, float]] = None
    values: Optional[List[Union[int, float]]] = None
    param_type: str = "float"  # "int", "float", "choice"
    
    def __post_init__(self):
        """后处理初始化"""
        if self.param_type == "choice" and self.values is None:
            raise ValueError(f"参数 {self.name} 类型为 choice 但未提供 values")
        
        if self.param_type in ["int", "float"] and self.min_value >= self.max_value:
            raise ValueError(f"参数 {self.name} 的最小值必须小于最大值")
    
    def generate_values(self, num_samples: Optional[int] = None) -> List[Union[int, float]]:
        """生成参数值
        
        Args:
            num_samples: 采样数量（用于随机搜索）
            
        Returns:
            List[Union[int, float]]: 参数值列表
        """
        if self.param_type == "choice":
            return self.values
        
        elif self.param_type == "int":
            if self.step is not None:
                return list(range(int(self.min_value), int(self.max_value) + 1, int(self.step)))
            else:
                if num_samples:
                    return [random.randint(int(self.min_value), int(self.max_value)) 
                           for _ in range(num_samples)]
                else:
                    return list(range(int(self.min_value), int(self.max_value) + 1))
        
        elif self.param_type == "float":
            if self.step is not None:
                values = []
                current = self.min_value
                while current <= self.max_value:
                    values.append(current)
                    current += self.step
                return values
            else:
                if num_samples:
                    return [random.uniform(self.min_value, self.max_value) 
                           for _ in range(num_samples)]
                else:
                    # 默认生成10个等间距的值
                    return list(np.linspace(self.min_value, self.max_value, 10))
        
        return []
    
    def clip_value(self, value: Union[int, float]) -> Union[int, float]:
        """裁剪参数值到有效范围
        
        Args:
            value: 参数值
            
        Returns:
            Union[int, float]: 裁剪后的值
        """
        if self.param_type == "choice":
            # 找到最接近的值
            return min(self.values, key=lambda x: abs(x - value))
        
        elif self.param_type == "int":
            return int(max(self.min_value, min(self.max_value, value)))
        
        elif self.param_type == "float":
            return max(self.min_value, min(self.max_value, value))
        
        return value


@dataclass
class OptimizationConfig:
    """优化配置"""
    method: OptimizationMethod = OptimizationMethod.GRID_SEARCH
    objective: ObjectiveFunction = ObjectiveFunction.SHARPE_RATIO
    maximize: bool = True  # 是否最大化目标函数
    
    # 搜索配置
    max_iterations: int = 1000
    max_evaluations: int = 10000
    random_samples: int = 100
    
    # 并行配置
    n_jobs: int = 1
    use_multiprocessing: bool = False
    
    # 早停配置
    early_stopping: bool = False
    patience: int = 50
    min_improvement: float = 1e-6
    
    # 交叉验证配置
    cross_validation: bool = False
    cv_folds: int = 5
    test_size: float = 0.2
    
    # 稳健性测试
    robustness_test: bool = False
    noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.05])
    
    # 过拟合检测
    overfitting_detection: bool = True
    validation_split: float = 0.3
    
    # 其他配置
    random_seed: Optional[int] = None
    verbose: bool = True
    save_all_results: bool = False
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 配置字典
        """
        return {
            'method': self.method.value,
            'objective': self.objective.value,
            'maximize': self.maximize,
            'max_iterations': self.max_iterations,
            'max_evaluations': self.max_evaluations,
            'random_samples': self.random_samples,
            'n_jobs': self.n_jobs,
            'use_multiprocessing': self.use_multiprocessing,
            'early_stopping': self.early_stopping,
            'patience': self.patience,
            'min_improvement': self.min_improvement,
            'cross_validation': self.cross_validation,
            'cv_folds': self.cv_folds,
            'test_size': self.test_size,
            'robustness_test': self.robustness_test,
            'noise_levels': self.noise_levels,
            'overfitting_detection': self.overfitting_detection,
            'validation_split': self.validation_split,
            'random_seed': self.random_seed,
            'verbose': self.verbose,
            'save_all_results': self.save_all_results
        }


@dataclass
class OptimizationResult:
    """优化结果"""
    best_params: Dict[str, Union[int, float]]
    best_score: float
    best_metrics: BacktestMetrics
    
    # 优化过程信息
    total_evaluations: int
    optimization_time: float
    convergence_iteration: Optional[int] = None
    
    # 所有结果（如果保存）
    all_results: Optional[List[Dict]] = None
    
    # 交叉验证结果
    cv_scores: Optional[List[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None
    
    # 稳健性测试结果
    robustness_scores: Optional[Dict[float, float]] = None
    
    # 过拟合检测结果
    train_score: Optional[float] = None
    validation_score: Optional[float] = None
    overfitting_ratio: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 结果字典
        """
        return {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_metrics': self.best_metrics.to_dict() if self.best_metrics else None,
            'total_evaluations': self.total_evaluations,
            'optimization_time': self.optimization_time,
            'convergence_iteration': self.convergence_iteration,
            'cv_scores': self.cv_scores,
            'cv_mean': self.cv_mean,
            'cv_std': self.cv_std,
            'robustness_scores': self.robustness_scores,
            'train_score': self.train_score,
            'validation_score': self.validation_score,
            'overfitting_ratio': self.overfitting_ratio
        }


class BaseOptimizer(ABC):
    """优化器基类"""
    
    def __init__(self, config: OptimizationConfig):
        """初始化优化器
        
        Args:
            config: 优化配置
        """
        self.config = config
        self.logger = get_logger(f"{self.__class__.__name__}")
        
        # 设置随机种子
        if config.random_seed is not None:
            random.seed(config.random_seed)
            np.random.seed(config.random_seed)
    
    @abstractmethod
    def optimize(self, parameter_ranges: List[ParameterRange],
                objective_function: Callable) -> OptimizationResult:
        """执行优化
        
        Args:
            parameter_ranges: 参数范围列表
            objective_function: 目标函数
            
        Returns:
            OptimizationResult: 优化结果
        """
        pass
    
    def _evaluate_parameters(self, params: Dict[str, Union[int, float]],
                           objective_function: Callable) -> float:
        """评估参数组合
        
        Args:
            params: 参数字典
            objective_function: 目标函数
            
        Returns:
            float: 目标函数值
        """
        try:
            score = objective_function(params)
            return score if not np.isnan(score) else float('-inf')
        except Exception as e:
            self.logger.warning(f"参数评估失败 {params}: {e}")
            return float('-inf')


class GridSearchOptimizer(BaseOptimizer):
    """网格搜索优化器"""
    
    def optimize(self, parameter_ranges: List[ParameterRange],
                objective_function: Callable) -> OptimizationResult:
        """执行网格搜索优化
        
        Args:
            parameter_ranges: 参数范围列表
            objective_function: 目标函数
            
        Returns:
            OptimizationResult: 优化结果
        """
        start_time = datetime.now()
        
        # 生成参数网格
        param_grid = {}
        for param_range in parameter_ranges:
            param_grid[param_range.name] = param_range.generate_values()
        
        # 生成所有参数组合
        if SKLEARN_AVAILABLE:
            param_combinations = list(ParameterGrid(param_grid))
        else:
            # 手动生成参数组合
            keys = list(param_grid.keys())
            values = list(param_grid.values())
            param_combinations = [dict(zip(keys, combo)) for combo in itertools.product(*values)]
        
        total_combinations = len(param_combinations)
        self.logger.info(f"开始网格搜索，总共 {total_combinations} 个参数组合")
        
        # 限制评估次数
        if total_combinations > self.config.max_evaluations:
            param_combinations = random.sample(param_combinations, self.config.max_evaluations)
            self.logger.warning(f"参数组合数量超过限制，随机采样 {self.config.max_evaluations} 个")
        
        best_score = float('-inf') if self.config.maximize else float('inf')
        best_params = None
        best_metrics = None
        all_results = [] if self.config.save_all_results else None
        
        # 并行或串行评估
        if self.config.n_jobs > 1:
            results = self._parallel_evaluate(param_combinations, objective_function)
        else:
            results = self._serial_evaluate(param_combinations, objective_function)
        
        # 处理结果
        for i, (params, score, metrics) in enumerate(results):
            if self.config.save_all_results:
                all_results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics.to_dict() if metrics else None
                })
            
            # 更新最佳结果
            is_better = (self.config.maximize and score > best_score) or \
                       (not self.config.maximize and score < best_score)
            
            if is_better:
                best_score = score
                best_params = params
                best_metrics = metrics
            
            if self.config.verbose and (i + 1) % max(1, len(param_combinations) // 10) == 0:
                self.logger.info(f"已评估 {i + 1}/{len(param_combinations)} 个参数组合")
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            total_evaluations=len(param_combinations),
            optimization_time=optimization_time,
            all_results=all_results
        )
    
    def _serial_evaluate(self, param_combinations: List[Dict],
                        objective_function: Callable) -> List[Tuple]:
        """串行评估参数组合"""
        results = []
        for params in param_combinations:
            score = self._evaluate_parameters(params, objective_function)
            # 这里假设objective_function返回(score, metrics)
            try:
                score, metrics = objective_function(params)
            except:
                score = objective_function(params)
                metrics = None
            results.append((params, score, metrics))
        return results
    
    def _parallel_evaluate(self, param_combinations: List[Dict],
                          objective_function: Callable) -> List[Tuple]:
        """并行评估参数组合"""
        results = []
        
        executor_class = ProcessPoolExecutor if self.config.use_multiprocessing else ThreadPoolExecutor
        
        with executor_class(max_workers=self.config.n_jobs) as executor:
            # 提交所有任务
            future_to_params = {
                executor.submit(self._evaluate_single, params, objective_function): params
                for params in param_combinations
            }
            
            # 收集结果
            for future in as_completed(future_to_params):
                params = future_to_params[future]
                try:
                    score, metrics = future.result()
                    results.append((params, score, metrics))
                except Exception as e:
                    self.logger.warning(f"参数评估失败 {params}: {e}")
                    results.append((params, float('-inf'), None))
        
        return results
    
    def _evaluate_single(self, params: Dict, objective_function: Callable) -> Tuple:
        """评估单个参数组合"""
        try:
            result = objective_function(params)
            if isinstance(result, tuple):
                return result
            else:
                return result, None
        except Exception as e:
            return float('-inf'), None


class RandomSearchOptimizer(BaseOptimizer):
    """随机搜索优化器"""
    
    def optimize(self, parameter_ranges: List[ParameterRange],
                objective_function: Callable) -> OptimizationResult:
        """执行随机搜索优化
        
        Args:
            parameter_ranges: 参数范围列表
            objective_function: 目标函数
            
        Returns:
            OptimizationResult: 优化结果
        """
        start_time = datetime.now()
        
        best_score = float('-inf') if self.config.maximize else float('inf')
        best_params = None
        best_metrics = None
        all_results = [] if self.config.save_all_results else None
        
        # 早停相关变量
        no_improvement_count = 0
        last_best_score = best_score
        
        self.logger.info(f"开始随机搜索，最大评估次数: {self.config.random_samples}")
        
        for i in range(self.config.random_samples):
            # 随机生成参数
            params = {}
            for param_range in parameter_ranges:
                if param_range.param_type == "choice":
                    params[param_range.name] = random.choice(param_range.values)
                elif param_range.param_type == "int":
                    params[param_range.name] = random.randint(
                        int(param_range.min_value), int(param_range.max_value)
                    )
                elif param_range.param_type == "float":
                    params[param_range.name] = random.uniform(
                        param_range.min_value, param_range.max_value
                    )
            
            # 评估参数
            try:
                result = objective_function(params)
                if isinstance(result, tuple):
                    score, metrics = result
                else:
                    score, metrics = result, None
            except Exception as e:
                self.logger.warning(f"参数评估失败 {params}: {e}")
                score, metrics = float('-inf'), None
            
            if self.config.save_all_results:
                all_results.append({
                    'params': params,
                    'score': score,
                    'metrics': metrics.to_dict() if metrics else None
                })
            
            # 更新最佳结果
            is_better = (self.config.maximize and score > best_score) or \
                       (not self.config.maximize and score < best_score)
            
            if is_better:
                improvement = abs(score - best_score)
                best_score = score
                best_params = params
                best_metrics = metrics
                no_improvement_count = 0
                
                if self.config.verbose:
                    self.logger.info(f"找到更好的参数组合 (评估 {i+1}): score={score:.6f}")
            else:
                no_improvement_count += 1
            
            # 早停检查
            if self.config.early_stopping and no_improvement_count >= self.config.patience:
                self.logger.info(f"早停触发，连续 {self.config.patience} 次无改善")
                break
            
            if self.config.verbose and (i + 1) % max(1, self.config.random_samples // 10) == 0:
                self.logger.info(f"已评估 {i + 1}/{self.config.random_samples} 个参数组合")
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            total_evaluations=i + 1,
            optimization_time=optimization_time,
            convergence_iteration=i + 1 - no_improvement_count if best_params else None,
            all_results=all_results
        )


class GeneticAlgorithmOptimizer(BaseOptimizer):
    """遗传算法优化器"""
    
    def __init__(self, config: OptimizationConfig, 
                 population_size: int = 50,
                 mutation_rate: float = 0.1,
                 crossover_rate: float = 0.8,
                 elite_size: int = 5):
        """初始化遗传算法优化器
        
        Args:
            config: 优化配置
            population_size: 种群大小
            mutation_rate: 变异率
            crossover_rate: 交叉率
            elite_size: 精英个体数量
        """
        super().__init__(config)
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
    
    def optimize(self, parameter_ranges: List[ParameterRange],
                objective_function: Callable) -> OptimizationResult:
        """执行遗传算法优化
        
        Args:
            parameter_ranges: 参数范围列表
            objective_function: 目标函数
            
        Returns:
            OptimizationResult: 优化结果
        """
        start_time = datetime.now()
        
        # 初始化种群
        population = self._initialize_population(parameter_ranges)
        
        best_score = float('-inf') if self.config.maximize else float('inf')
        best_params = None
        best_metrics = None
        all_results = [] if self.config.save_all_results else None
        
        # 早停相关变量
        no_improvement_count = 0
        
        self.logger.info(f"开始遗传算法优化，种群大小: {self.population_size}，最大代数: {self.config.max_iterations}")
        
        for generation in range(self.config.max_iterations):
            # 评估种群
            fitness_scores = []
            for individual in population:
                params = self._decode_individual(individual, parameter_ranges)
                try:
                    result = objective_function(params)
                    if isinstance(result, tuple):
                        score, metrics = result
                    else:
                        score, metrics = result, None
                except Exception as e:
                    score, metrics = float('-inf'), None
                
                fitness_scores.append(score)
                
                if self.config.save_all_results:
                    all_results.append({
                        'generation': generation,
                        'params': params,
                        'score': score,
                        'metrics': metrics.to_dict() if metrics else None
                    })
                
                # 更新最佳结果
                is_better = (self.config.maximize and score > best_score) or \
                           (not self.config.maximize and score < best_score)
                
                if is_better:
                    best_score = score
                    best_params = params
                    best_metrics = metrics
                    no_improvement_count = 0
            
            # 选择、交叉、变异
            population = self._evolve_population(population, fitness_scores, parameter_ranges)
            
            # 早停检查
            if no_improvement_count >= self.config.patience:
                self.logger.info(f"早停触发，连续 {self.config.patience} 代无改善")
                break
            
            no_improvement_count += 1
            
            if self.config.verbose and (generation + 1) % max(1, self.config.max_iterations // 10) == 0:
                self.logger.info(f"第 {generation + 1} 代，最佳得分: {best_score:.6f}")
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        return OptimizationResult(
            best_params=best_params,
            best_score=best_score,
            best_metrics=best_metrics,
            total_evaluations=(generation + 1) * self.population_size,
            optimization_time=optimization_time,
            convergence_iteration=generation + 1 - no_improvement_count if best_params else None,
            all_results=all_results
        )
    
    def _initialize_population(self, parameter_ranges: List[ParameterRange]) -> List[List[float]]:
        """初始化种群"""
        population = []
        for _ in range(self.population_size):
            individual = []
            for param_range in parameter_ranges:
                if param_range.param_type == "choice":
                    # 对于选择类型，使用索引
                    individual.append(random.randint(0, len(param_range.values) - 1))
                else:
                    # 对于数值类型，使用0-1之间的值
                    individual.append(random.random())
            population.append(individual)
        return population
    
    def _decode_individual(self, individual: List[float], 
                          parameter_ranges: List[ParameterRange]) -> Dict[str, Union[int, float]]:
        """解码个体为参数字典"""
        params = {}
        for i, param_range in enumerate(parameter_ranges):
            if param_range.param_type == "choice":
                idx = int(individual[i]) % len(param_range.values)
                params[param_range.name] = param_range.values[idx]
            elif param_range.param_type == "int":
                value = param_range.min_value + individual[i] * (param_range.max_value - param_range.min_value)
                params[param_range.name] = int(value)
            elif param_range.param_type == "float":
                value = param_range.min_value + individual[i] * (param_range.max_value - param_range.min_value)
                params[param_range.name] = value
        return params
    
    def _evolve_population(self, population: List[List[float]], 
                          fitness_scores: List[float],
                          parameter_ranges: List[ParameterRange]) -> List[List[float]]:
        """进化种群"""
        # 选择精英个体
        elite_indices = sorted(range(len(fitness_scores)), 
                              key=lambda i: fitness_scores[i], 
                              reverse=self.config.maximize)[:self.elite_size]
        new_population = [population[i][:] for i in elite_indices]
        
        # 生成新个体
        while len(new_population) < self.population_size:
            # 选择父母
            parent1 = self._tournament_selection(population, fitness_scores)
            parent2 = self._tournament_selection(population, fitness_scores)
            
            # 交叉
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = parent1[:], parent2[:]
            
            # 变异
            child1 = self._mutate(child1, parameter_ranges)
            child2 = self._mutate(child2, parameter_ranges)
            
            new_population.extend([child1, child2])
        
        return new_population[:self.population_size]
    
    def _tournament_selection(self, population: List[List[float]], 
                             fitness_scores: List[float], 
                             tournament_size: int = 3) -> List[float]:
        """锦标赛选择"""
        tournament_indices = random.sample(range(len(population)), 
                                         min(tournament_size, len(population)))
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i] 
                      if self.config.maximize else -fitness_scores[i])
        return population[best_idx][:]
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """单点交叉"""
        crossover_point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2
    
    def _mutate(self, individual: List[float], 
               parameter_ranges: List[ParameterRange]) -> List[float]:
        """变异"""
        mutated = individual[:]
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                if parameter_ranges[i].param_type == "choice":
                    mutated[i] = random.randint(0, len(parameter_ranges[i].values) - 1)
                else:
                    # 高斯变异
                    mutated[i] += random.gauss(0, 0.1)
                    mutated[i] = max(0, min(1, mutated[i]))  # 限制在[0,1]范围内
        return mutated


class ParameterOptimizer:
    """参数优化器
    
    统一的参数优化接口，支持多种优化算法。
    """
    
    def __init__(self, config: OptimizationConfig):
        """初始化参数优化器
        
        Args:
            config: 优化配置
        """
        self.config = config
        self.logger = get_logger("ParameterOptimizer")
        self.metrics_calculator = MetricsCalculator()
        
        # 初始化优化器
        self.optimizer = self._create_optimizer()
    
    def _create_optimizer(self) -> BaseOptimizer:
        """创建优化器实例
        
        Returns:
            BaseOptimizer: 优化器实例
        """
        if self.config.method == OptimizationMethod.GRID_SEARCH:
            return GridSearchOptimizer(self.config)
        elif self.config.method == OptimizationMethod.RANDOM_SEARCH:
            return RandomSearchOptimizer(self.config)
        elif self.config.method == OptimizationMethod.GENETIC_ALGORITHM:
            return GeneticAlgorithmOptimizer(self.config)
        else:
            raise BacktestOptimizationError(f"不支持的优化方法: {self.config.method}")
    
    def optimize(self, backtest_function: Callable,
                parameter_ranges: List[ParameterRange],
                custom_objective: Optional[Callable] = None) -> OptimizationResult:
        """执行参数优化
        
        Args:
            backtest_function: 回测函数，接受参数字典，返回BacktestMetrics
            parameter_ranges: 参数范围列表
            custom_objective: 自定义目标函数
            
        Returns:
            OptimizationResult: 优化结果
        """
        self.logger.info(f"开始参数优化，方法: {self.config.method.value}")
        
        # 创建目标函数
        if custom_objective is not None:
            objective_function = custom_objective
        else:
            objective_function = self._create_objective_function(backtest_function)
        
        # 执行优化
        result = self.optimizer.optimize(parameter_ranges, objective_function)
        
        # 后处理结果
        if self.config.cross_validation:
            result = self._add_cross_validation(result, backtest_function, parameter_ranges)
        
        if self.config.robustness_test:
            result = self._add_robustness_test(result, backtest_function)
        
        if self.config.overfitting_detection:
            result = self._add_overfitting_detection(result, backtest_function)
        
        self.logger.info(f"参数优化完成，最佳得分: {result.best_score:.6f}")
        
        return result
    
    def _create_objective_function(self, backtest_function: Callable) -> Callable:
        """创建目标函数
        
        Args:
            backtest_function: 回测函数
            
        Returns:
            Callable: 目标函数
        """
        def objective(params: Dict[str, Union[int, float]]) -> Tuple[float, BacktestMetrics]:
            try:
                # 执行回测
                metrics = backtest_function(params)
                
                if metrics is None:
                    return float('-inf'), None
                
                # 计算目标函数值
                if self.config.objective == ObjectiveFunction.SHARPE_RATIO:
                    score = metrics.sharpe_ratio
                elif self.config.objective == ObjectiveFunction.CALMAR_RATIO:
                    score = metrics.calmar_ratio
                elif self.config.objective == ObjectiveFunction.SORTINO_RATIO:
                    score = metrics.sortino_ratio
                elif self.config.objective == ObjectiveFunction.TOTAL_RETURN:
                    score = metrics.total_return
                elif self.config.objective == ObjectiveFunction.ANNUAL_RETURN:
                    score = metrics.annual_return
                elif self.config.objective == ObjectiveFunction.MAX_DRAWDOWN:
                    score = -metrics.max_drawdown  # 最小化回撤
                elif self.config.objective == ObjectiveFunction.PROFIT_FACTOR:
                    score = metrics.trading_stats.profit_factor
                elif self.config.objective == ObjectiveFunction.WIN_RATE:
                    score = metrics.trading_stats.win_rate
                else:
                    score = metrics.sharpe_ratio  # 默认使用夏普比率
                
                return score, metrics
                
            except Exception as e:
                self.logger.warning(f"回测执行失败 {params}: {e}")
                return float('-inf'), None
        
        return objective
    
    def _add_cross_validation(self, result: OptimizationResult,
                             backtest_function: Callable,
                             parameter_ranges: List[ParameterRange]) -> OptimizationResult:
        """添加交叉验证结果
        
        Args:
            result: 优化结果
            backtest_function: 回测函数
            parameter_ranges: 参数范围
            
        Returns:
            OptimizationResult: 更新后的结果
        """
        self.logger.info("执行交叉验证...")
        
        # 这里简化实现，实际应该根据时间序列特点进行分割
        cv_scores = []
        for fold in range(self.config.cv_folds):
            try:
                # 模拟不同的数据分割
                metrics = backtest_function(result.best_params)
                if metrics:
                    if self.config.objective == ObjectiveFunction.SHARPE_RATIO:
                        score = metrics.sharpe_ratio
                    else:
                        score = metrics.total_return
                    cv_scores.append(score)
            except Exception as e:
                self.logger.warning(f"交叉验证第 {fold+1} 折失败: {e}")
        
        if cv_scores:
            result.cv_scores = cv_scores
            result.cv_mean = np.mean(cv_scores)
            result.cv_std = np.std(cv_scores)
        
        return result
    
    def _add_robustness_test(self, result: OptimizationResult,
                            backtest_function: Callable) -> OptimizationResult:
        """添加稳健性测试
        
        Args:
            result: 优化结果
            backtest_function: 回测函数
            
        Returns:
            OptimizationResult: 更新后的结果
        """
        self.logger.info("执行稳健性测试...")
        
        robustness_scores = {}
        
        for noise_level in self.config.noise_levels:
            try:
                # 添加噪声到最佳参数
                noisy_params = {}
                for key, value in result.best_params.items():
                    if isinstance(value, (int, float)):
                        noise = np.random.normal(0, abs(value) * noise_level)
                        noisy_params[key] = value + noise
                    else:
                        noisy_params[key] = value
                
                # 执行回测
                metrics = backtest_function(noisy_params)
                if metrics:
                    if self.config.objective == ObjectiveFunction.SHARPE_RATIO:
                        score = metrics.sharpe_ratio
                    else:
                        score = metrics.total_return
                    robustness_scores[noise_level] = score
                    
            except Exception as e:
                self.logger.warning(f"稳健性测试失败 (噪声水平 {noise_level}): {e}")
        
        result.robustness_scores = robustness_scores
        return result
    
    def _add_overfitting_detection(self, result: OptimizationResult,
                                  backtest_function: Callable) -> OptimizationResult:
        """添加过拟合检测
        
        Args:
            result: 优化结果
            backtest_function: 回测函数
            
        Returns:
            OptimizationResult: 更新后的结果
        """
        self.logger.info("执行过拟合检测...")
        
        try:
            # 在训练集和验证集上测试最佳参数
            # 这里简化实现，实际应该使用不同的数据集
            train_metrics = backtest_function(result.best_params)
            validation_metrics = backtest_function(result.best_params)  # 应该使用验证集
            
            if train_metrics and validation_metrics:
                if self.config.objective == ObjectiveFunction.SHARPE_RATIO:
                    train_score = train_metrics.sharpe_ratio
                    validation_score = validation_metrics.sharpe_ratio
                else:
                    train_score = train_metrics.total_return
                    validation_score = validation_metrics.total_return
                
                result.train_score = train_score
                result.validation_score = validation_score
                
                # 计算过拟合比率
                if validation_score != 0:
                    result.overfitting_ratio = (train_score - validation_score) / abs(validation_score)
                else:
                    result.overfitting_ratio = 0.0
                
        except Exception as e:
            self.logger.warning(f"过拟合检测失败: {e}")
        
        return result
    
    def analyze_parameter_sensitivity(self, backtest_function: Callable,
                                    base_params: Dict[str, Union[int, float]],
                                    parameter_ranges: List[ParameterRange],
                                    sensitivity_range: float = 0.2) -> Dict[str, List[Tuple]]:
        """分析参数敏感性
        
        Args:
            backtest_function: 回测函数
            base_params: 基准参数
            parameter_ranges: 参数范围
            sensitivity_range: 敏感性测试范围（相对于基准值的比例）
            
        Returns:
            Dict[str, List[Tuple]]: 参数敏感性分析结果
        """
        self.logger.info("开始参数敏感性分析...")
        
        sensitivity_results = {}
        
        for param_range in parameter_ranges:
            param_name = param_range.name
            base_value = base_params.get(param_name)
            
            if base_value is None:
                continue
            
            # 生成测试值
            if param_range.param_type == "choice":
                test_values = param_range.values
            else:
                # 在基准值周围生成测试点
                if param_range.param_type == "int":
                    delta = max(1, int(abs(base_value) * sensitivity_range))
                    test_values = list(range(
                        max(int(param_range.min_value), base_value - delta),
                        min(int(param_range.max_value), base_value + delta) + 1
                    ))
                else:
                    delta = abs(base_value) * sensitivity_range
                    test_values = np.linspace(
                        max(param_range.min_value, base_value - delta),
                        min(param_range.max_value, base_value + delta),
                        11  # 11个测试点
                    )
            
            # 测试每个值
            param_results = []
            for test_value in test_values:
                test_params = base_params.copy()
                test_params[param_name] = test_value
                
                try:
                    metrics = backtest_function(test_params)
                    if metrics:
                        if self.config.objective == ObjectiveFunction.SHARPE_RATIO:
                            score = metrics.sharpe_ratio
                        else:
                            score = metrics.total_return
                        param_results.append((test_value, score))
                except Exception as e:
                    self.logger.warning(f"敏感性测试失败 {param_name}={test_value}: {e}")
            
            sensitivity_results[param_name] = param_results
        
        return sensitivity_results
    
    def __str__(self) -> str:
        return f"ParameterOptimizer(method={self.config.method.value}, objective={self.config.objective.value})"
    
    def __repr__(self) -> str:
        return self.__str__()