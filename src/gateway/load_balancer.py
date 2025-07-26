"""Load Balancer Module

负载均衡器模块，提供多种负载均衡策略。
"""

import random
import threading
from abc import ABC, abstractmethod
from enum import Enum
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from src.common.logging import get_logger

logger = get_logger(__name__)


class LoadBalanceStrategy(Enum):
    """负载均衡策略"""
    ROUND_ROBIN = "round_robin"     # 轮询
    RANDOM = "random"               # 随机
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"  # 加权轮询
    LEAST_CONNECTIONS = "least_connections"        # 最少连接
    IP_HASH = "ip_hash"             # IP哈希


@dataclass
class ServerNode:
    """服务器节点"""
    host: str
    port: int
    weight: int = 1
    active_connections: int = 0
    is_healthy: bool = True
    
    @property
    def address(self) -> str:
        """获取地址"""
        return f"{self.host}:{self.port}"


@dataclass
class LoadBalancerConfig:
    """负载均衡器配置"""
    strategy: LoadBalanceStrategy = LoadBalanceStrategy.ROUND_ROBIN
    health_check_interval: float = 30.0  # 健康检查间隔（秒）
    max_retries: int = 3                 # 最大重试次数
    retry_delay: float = 1.0             # 重试延迟（秒）


class LoadBalanceAlgorithm(ABC):
    """负载均衡算法基类"""
    
    @abstractmethod
    def select_server(self, servers: List[ServerNode], client_info: Dict[str, Any] = None) -> Optional[ServerNode]:
        """选择服务器
        
        Args:
            servers: 可用服务器列表
            client_info: 客户端信息
            
        Returns:
            ServerNode: 选中的服务器节点
        """
        pass


class RoundRobinAlgorithm(LoadBalanceAlgorithm):
    """轮询算法"""
    
    def __init__(self):
        self.current_index = 0
        self.lock = threading.Lock()
    
    def select_server(self, servers: List[ServerNode], client_info: Dict[str, Any] = None) -> Optional[ServerNode]:
        if not servers:
            return None
        
        healthy_servers = [s for s in servers if s.is_healthy]
        if not healthy_servers:
            return None
        
        with self.lock:
            server = healthy_servers[self.current_index % len(healthy_servers)]
            self.current_index += 1
            return server


class RandomAlgorithm(LoadBalanceAlgorithm):
    """随机算法"""
    
    def select_server(self, servers: List[ServerNode], client_info: Dict[str, Any] = None) -> Optional[ServerNode]:
        healthy_servers = [s for s in servers if s.is_healthy]
        if not healthy_servers:
            return None
        
        return random.choice(healthy_servers)


class WeightedRoundRobinAlgorithm(LoadBalanceAlgorithm):
    """加权轮询算法"""
    
    def __init__(self):
        self.current_weights = {}
        self.lock = threading.Lock()
    
    def select_server(self, servers: List[ServerNode], client_info: Dict[str, Any] = None) -> Optional[ServerNode]:
        healthy_servers = [s for s in servers if s.is_healthy]
        if not healthy_servers:
            return None
        
        with self.lock:
            # 初始化权重
            for server in healthy_servers:
                if server.address not in self.current_weights:
                    self.current_weights[server.address] = 0
            
            # 增加当前权重
            for server in healthy_servers:
                self.current_weights[server.address] += server.weight
            
            # 选择权重最大的服务器
            max_weight_server = max(healthy_servers, key=lambda s: self.current_weights[s.address])
            
            # 减少选中服务器的权重
            total_weight = sum(s.weight for s in healthy_servers)
            self.current_weights[max_weight_server.address] -= total_weight
            
            return max_weight_server


class LeastConnectionsAlgorithm(LoadBalanceAlgorithm):
    """最少连接算法"""
    
    def select_server(self, servers: List[ServerNode], client_info: Dict[str, Any] = None) -> Optional[ServerNode]:
        healthy_servers = [s for s in servers if s.is_healthy]
        if not healthy_servers:
            return None
        
        return min(healthy_servers, key=lambda s: s.active_connections)


class IPHashAlgorithm(LoadBalanceAlgorithm):
    """IP哈希算法"""
    
    def select_server(self, servers: List[ServerNode], client_info: Dict[str, Any] = None) -> Optional[ServerNode]:
        healthy_servers = [s for s in servers if s.is_healthy]
        if not healthy_servers:
            return None
        
        if not client_info or 'client_ip' not in client_info:
            return healthy_servers[0]
        
        client_ip = client_info['client_ip']
        hash_value = hash(client_ip)
        index = hash_value % len(healthy_servers)
        return healthy_servers[index]


class LoadBalancer:
    """负载均衡器
    
    提供多种负载均衡策略的实现。
    """
    
    def __init__(self, config: LoadBalancerConfig):
        """初始化负载均衡器
        
        Args:
            config: 负载均衡器配置
        """
        self.config = config
        self.servers: List[ServerNode] = []
        self.algorithm = self._create_algorithm(config.strategy)
        self.lock = threading.RLock()
        
        logger.info(f"Load balancer initialized with strategy: {config.strategy.value}")
    
    def _create_algorithm(self, strategy: LoadBalanceStrategy) -> LoadBalanceAlgorithm:
        """创建负载均衡算法
        
        Args:
            strategy: 负载均衡策略
            
        Returns:
            LoadBalanceAlgorithm: 负载均衡算法实例
        """
        algorithms = {
            LoadBalanceStrategy.ROUND_ROBIN: RoundRobinAlgorithm,
            LoadBalanceStrategy.RANDOM: RandomAlgorithm,
            LoadBalanceStrategy.WEIGHTED_ROUND_ROBIN: WeightedRoundRobinAlgorithm,
            LoadBalanceStrategy.LEAST_CONNECTIONS: LeastConnectionsAlgorithm,
            LoadBalanceStrategy.IP_HASH: IPHashAlgorithm
        }
        
        algorithm_class = algorithms.get(strategy, RoundRobinAlgorithm)
        return algorithm_class()
    
    def add_server(self, server: ServerNode):
        """添加服务器
        
        Args:
            server: 服务器节点
        """
        with self.lock:
            self.servers.append(server)
            logger.info(f"Server {server.address} added to load balancer")
    
    def remove_server(self, address: str):
        """移除服务器
        
        Args:
            address: 服务器地址
        """
        with self.lock:
            self.servers = [s for s in self.servers if s.address != address]
            logger.info(f"Server {address} removed from load balancer")
    
    def select_server(self, client_info: Dict[str, Any] = None) -> Optional[ServerNode]:
        """选择服务器
        
        Args:
            client_info: 客户端信息
            
        Returns:
            ServerNode: 选中的服务器节点
        """
        with self.lock:
            return self.algorithm.select_server(self.servers, client_info)
    
    def mark_server_healthy(self, address: str):
        """标记服务器为健康
        
        Args:
            address: 服务器地址
        """
        with self.lock:
            for server in self.servers:
                if server.address == address:
                    server.is_healthy = True
                    logger.info(f"Server {address} marked as healthy")
                    break
    
    def mark_server_unhealthy(self, address: str):
        """标记服务器为不健康
        
        Args:
            address: 服务器地址
        """
        with self.lock:
            for server in self.servers:
                if server.address == address:
                    server.is_healthy = False
                    logger.warning(f"Server {address} marked as unhealthy")
                    break
    
    def increment_connections(self, address: str):
        """增加连接数
        
        Args:
            address: 服务器地址
        """
        with self.lock:
            for server in self.servers:
                if server.address == address:
                    server.active_connections += 1
                    break
    
    def decrement_connections(self, address: str):
        """减少连接数
        
        Args:
            address: 服务器地址
        """
        with self.lock:
            for server in self.servers:
                if server.address == address:
                    server.active_connections = max(0, server.active_connections - 1)
                    break
    
    def get_servers(self) -> List[ServerNode]:
        """获取所有服务器
        
        Returns:
            List[ServerNode]: 服务器列表
        """
        with self.lock:
            return self.servers.copy()
    
    def get_healthy_servers(self) -> List[ServerNode]:
        """获取健康的服务器
        
        Returns:
            List[ServerNode]: 健康的服务器列表
        """
        with self.lock:
            return [s for s in self.servers if s.is_healthy]
    
    def get_metrics(self) -> Dict[str, Any]:
        """获取指标
        
        Returns:
            Dict[str, Any]: 指标数据
        """
        with self.lock:
            healthy_count = len([s for s in self.servers if s.is_healthy])
            total_connections = sum(s.active_connections for s in self.servers)
            
            return {
                'strategy': self.config.strategy.value,
                'total_servers': len(self.servers),
                'healthy_servers': healthy_count,
                'unhealthy_servers': len(self.servers) - healthy_count,
                'total_connections': total_connections,
                'servers': [
                    {
                        'address': s.address,
                        'weight': s.weight,
                        'active_connections': s.active_connections,
                        'is_healthy': s.is_healthy
                    }
                    for s in self.servers
                ]
            }