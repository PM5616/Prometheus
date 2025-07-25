"""Configuration Loader

配置加载器，支持从YAML文件加载配置。
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class ConfigLoader:
    """配置加载器
    
    支持从YAML文件加载配置，并提供配置合并功能。
    """
    
    def __init__(self, config_dir: Optional[Path] = None):
        """初始化配置加载器
        
        Args:
            config_dir: 配置文件目录，默认为项目根目录下的configs文件夹
        """
        if config_dir is None:
            # 获取项目根目录
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            config_dir = project_root / "configs"
        
        self.config_dir = Path(config_dir)
        self._config_cache: Dict[str, Dict[str, Any]] = {}
    
    def load_config(self, config_name: str, use_cache: bool = True) -> Dict[str, Any]:
        """加载配置文件
        
        Args:
            config_name: 配置文件名（不包含扩展名）
            use_cache: 是否使用缓存
            
        Returns:
            Dict[str, Any]: 配置字典
            
        Raises:
            FileNotFoundError: 配置文件不存在
            yaml.YAMLError: YAML解析错误
        """
        if use_cache and config_name in self._config_cache:
            return self._config_cache[config_name]
        
        config_file = self.config_dir / f"{config_name}.yaml"
        
        if not config_file.exists():
            # 尝试.yml扩展名
            config_file = self.config_dir / f"{config_name}.yml"
            if not config_file.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_name}")
        
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                config = {}
            
            if use_cache:
                self._config_cache[config_name] = config
            
            logger.info(f"Loaded configuration from {config_file}")
            return config
            
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {config_file}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading configuration file {config_file}: {e}")
            raise
    
    def load_system_config(self) -> Dict[str, Any]:
        """加载系统配置
        
        Returns:
            Dict[str, Any]: 系统配置字典
        """
        return self.load_config("system_config")
    
    def load_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """加载策略配置
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            Dict[str, Any]: 策略配置字典
        """
        try:
            return self.load_config(f"strategies/{strategy_name}")
        except FileNotFoundError:
            logger.warning(f"Strategy config not found for {strategy_name}, using defaults")
            return {}
    
    def merge_configs(self, *configs: Dict[str, Any]) -> Dict[str, Any]:
        """合并多个配置字典
        
        Args:
            *configs: 要合并的配置字典
            
        Returns:
            Dict[str, Any]: 合并后的配置字典
        """
        merged = {}
        
        for config in configs:
            if not isinstance(config, dict):
                continue
            
            merged = self._deep_merge(merged, config)
        
        return merged
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """深度合并两个字典
        
        Args:
            dict1: 第一个字典
            dict2: 第二个字典
            
        Returns:
            Dict[str, Any]: 合并后的字典
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def save_config(self, config_name: str, config: Dict[str, Any]) -> None:
        """保存配置到文件
        
        Args:
            config_name: 配置文件名（不包含扩展名）
            config: 要保存的配置字典
        """
        config_file = self.config_dir / f"{config_name}.yaml"
        
        # 确保目录存在
        config_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
            
            # 更新缓存
            self._config_cache[config_name] = config
            
            logger.info(f"Saved configuration to {config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration file {config_file}: {e}")
            raise
    
    def clear_cache(self) -> None:
        """清空配置缓存"""
        self._config_cache.clear()
        logger.info("Configuration cache cleared")
    
    def get_config_path(self, config_name: str) -> Path:
        """获取配置文件路径
        
        Args:
            config_name: 配置文件名
            
        Returns:
            Path: 配置文件路径
        """
        return self.config_dir / f"{config_name}.yaml"