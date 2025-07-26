"""Strategy Loader Module

策略加载器模块，负责动态加载、验证和管理策略插件。

主要功能：
- 动态策略加载
- 策略验证和检查
- 策略热更新
- 策略依赖管理
- 策略版本控制
"""

import os
import sys
import importlib
import importlib.util
import inspect
import logging
from typing import Dict, List, Type, Optional, Any, Set
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

from .base_strategy import BaseStrategy, StrategyConfig
from src.common.exceptions.strategy_exceptions import StrategyLoadError, StrategyValidationError
from src.common.logging.logger import get_logger


@dataclass
class StrategyMetadata:
    """策略元数据"""
    name: str
    version: str
    description: str
    author: str
    created_time: datetime
    file_path: str
    file_hash: str
    class_name: str
    dependencies: List[str]
    requirements: List[str]
    tags: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'author': self.author,
            'created_time': self.created_time.isoformat(),
            'file_path': self.file_path,
            'file_hash': self.file_hash,
            'class_name': self.class_name,
            'dependencies': self.dependencies,
            'requirements': self.requirements,
            'tags': self.tags
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StrategyMetadata':
        """从字典创建"""
        return cls(
            name=data['name'],
            version=data['version'],
            description=data['description'],
            author=data['author'],
            created_time=datetime.fromisoformat(data['created_time']),
            file_path=data['file_path'],
            file_hash=data['file_hash'],
            class_name=data['class_name'],
            dependencies=data['dependencies'],
            requirements=data['requirements'],
            tags=data['tags']
        )


class StrategyValidator:
    """策略验证器"""
    
    def __init__(self):
        self.logger = get_logger("strategy_validator")
    
    def validate_strategy_class(self, strategy_class: Type[BaseStrategy]) -> bool:
        """验证策略类
        
        Args:
            strategy_class: 策略类
            
        Returns:
            是否验证通过
            
        Raises:
            StrategyValidationError: 验证失败
        """
        try:
            # 检查是否继承自BaseStrategy
            if not issubclass(strategy_class, BaseStrategy):
                raise StrategyValidationError(f"策略类 {strategy_class.__name__} 必须继承自 BaseStrategy")
            
            # 检查必要的方法
            required_methods = ['_initialize', 'generate_signals', 'update_parameters']
            for method_name in required_methods:
                if not hasattr(strategy_class, method_name):
                    raise StrategyValidationError(f"策略类 {strategy_class.__name__} 缺少必要方法: {method_name}")
                
                method = getattr(strategy_class, method_name)
                if not callable(method):
                    raise StrategyValidationError(f"策略类 {strategy_class.__name__} 的 {method_name} 不是可调用方法")
            
            # 检查构造函数
            init_signature = inspect.signature(strategy_class.__init__)
            if 'config' not in init_signature.parameters:
                raise StrategyValidationError(f"策略类 {strategy_class.__name__} 构造函数必须包含 config 参数")
            
            # 检查类属性
            if not hasattr(strategy_class, '__doc__') or not strategy_class.__doc__:
                self.logger.warning(f"策略类 {strategy_class.__name__} 缺少文档字符串")
            
            self.logger.info(f"策略类 {strategy_class.__name__} 验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"策略类验证失败: {e}")
            raise StrategyValidationError(f"策略类验证失败: {e}")
    
    def validate_strategy_config(self, config: StrategyConfig) -> bool:
        """验证策略配置
        
        Args:
            config: 策略配置
            
        Returns:
            是否验证通过
            
        Raises:
            StrategyValidationError: 验证失败
        """
        try:
            # 检查必要字段
            if not config.name:
                raise StrategyValidationError("策略名称不能为空")
            
            if not config.version:
                raise StrategyValidationError("策略版本不能为空")
            
            if not config.symbols:
                raise StrategyValidationError("策略交易对不能为空")
            
            if not config.timeframes:
                raise StrategyValidationError("策略时间周期不能为空")
            
            # 检查数值范围
            if config.lookback_period <= 0:
                raise StrategyValidationError("回看周期必须大于0")
            
            if config.risk_params.max_position_size <= 0:
                raise StrategyValidationError("最大仓位必须大于0")
            
            if not (0 <= config.risk_params.stop_loss_pct <= 1):
                raise StrategyValidationError("止损比例必须在0-1之间")
            
            if not (0 <= config.risk_params.take_profit_pct <= 1):
                raise StrategyValidationError("止盈比例必须在0-1之间")
            
            self.logger.info(f"策略配置 {config.name} 验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"策略配置验证失败: {e}")
            raise StrategyValidationError(f"策略配置验证失败: {e}")
    
    def validate_strategy_file(self, file_path: str) -> bool:
        """验证策略文件
        
        Args:
            file_path: 策略文件路径
            
        Returns:
            是否验证通过
            
        Raises:
            StrategyValidationError: 验证失败
        """
        try:
            # 检查文件存在
            if not os.path.exists(file_path):
                raise StrategyValidationError(f"策略文件不存在: {file_path}")
            
            # 检查文件扩展名
            if not file_path.endswith('.py'):
                raise StrategyValidationError(f"策略文件必须是Python文件: {file_path}")
            
            # 检查文件大小
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                raise StrategyValidationError(f"策略文件为空: {file_path}")
            
            if file_size > 10 * 1024 * 1024:  # 10MB
                raise StrategyValidationError(f"策略文件过大: {file_path}")
            
            # 检查文件语法
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            try:
                compile(content, file_path, 'exec')
            except SyntaxError as e:
                raise StrategyValidationError(f"策略文件语法错误: {file_path}, {e}")
            
            self.logger.info(f"策略文件 {file_path} 验证通过")
            return True
            
        except Exception as e:
            self.logger.error(f"策略文件验证失败: {e}")
            raise StrategyValidationError(f"策略文件验证失败: {e}")


class StrategyLoader:
    """策略加载器
    
    负责动态加载、验证和管理策略插件。
    """
    
    def __init__(self, strategy_dirs: List[str]):
        """初始化策略加载器
        
        Args:
            strategy_dirs: 策略目录列表
        """
        self.strategy_dirs = [Path(d) for d in strategy_dirs]
        self.logger = get_logger("strategy_loader")
        self.validator = StrategyValidator()
        
        # 策略注册表
        self._strategy_classes: Dict[str, Type[BaseStrategy]] = {}
        self._strategy_metadata: Dict[str, StrategyMetadata] = {}
        self._loaded_modules: Dict[str, Any] = {}
        
        # 文件监控
        self._file_hashes: Dict[str, str] = {}
        
        self.logger.info(f"策略加载器初始化完成，策略目录: {[str(d) for d in self.strategy_dirs]}")
    
    def scan_strategies(self) -> List[str]:
        """扫描策略文件
        
        Returns:
            策略文件路径列表
        """
        strategy_files = []
        
        for strategy_dir in self.strategy_dirs:
            if not strategy_dir.exists():
                self.logger.warning(f"策略目录不存在: {strategy_dir}")
                continue
            
            # 递归扫描Python文件
            for file_path in strategy_dir.rglob('*.py'):
                if file_path.name.startswith('_'):
                    continue  # 跳过私有文件
                
                if file_path.name in ['__init__.py', 'base_strategy.py']:
                    continue  # 跳过基础文件
                
                strategy_files.append(str(file_path))
        
        self.logger.info(f"扫描到 {len(strategy_files)} 个策略文件")
        return strategy_files
    
    def load_strategy_from_file(self, file_path: str) -> Optional[Type[BaseStrategy]]:
        """从文件加载策略
        
        Args:
            file_path: 策略文件路径
            
        Returns:
            策略类或None
        """
        try:
            # 验证文件
            self.validator.validate_strategy_file(file_path)
            
            # 计算文件哈希
            file_hash = self._calculate_file_hash(file_path)
            
            # 检查是否已加载且未修改
            if file_path in self._file_hashes and self._file_hashes[file_path] == file_hash:
                strategy_name = self._get_strategy_name_by_file(file_path)
                if strategy_name and strategy_name in self._strategy_classes:
                    self.logger.debug(f"策略文件未修改，使用缓存: {file_path}")
                    return self._strategy_classes[strategy_name]
            
            # 动态导入模块
            module_name = self._get_module_name(file_path)
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            
            if spec is None or spec.loader is None:
                raise StrategyLoadError(f"无法创建模块规范: {file_path}")
            
            module = importlib.util.module_from_spec(spec)
            
            # 添加到sys.modules以支持相对导入
            sys.modules[module_name] = module
            
            try:
                spec.loader.exec_module(module)
            except Exception as e:
                # 清理sys.modules
                if module_name in sys.modules:
                    del sys.modules[module_name]
                raise StrategyLoadError(f"执行模块失败: {file_path}, {e}")
            
            # 查找策略类
            strategy_class = self._find_strategy_class(module)
            
            if strategy_class is None:
                raise StrategyLoadError(f"未找到策略类: {file_path}")
            
            # 验证策略类
            self.validator.validate_strategy_class(strategy_class)
            
            # 创建元数据
            metadata = self._create_metadata(strategy_class, file_path, file_hash)
            
            # 注册策略
            strategy_name = strategy_class.__name__
            self._strategy_classes[strategy_name] = strategy_class
            self._strategy_metadata[strategy_name] = metadata
            self._loaded_modules[strategy_name] = module
            self._file_hashes[file_path] = file_hash
            
            self.logger.info(f"成功加载策略: {strategy_name} from {file_path}")
            return strategy_class
            
        except Exception as e:
            self.logger.error(f"加载策略失败: {file_path}, 错误: {e}")
            return None
    
    def load_all_strategies(self) -> Dict[str, Type[BaseStrategy]]:
        """加载所有策略
        
        Returns:
            策略类字典
        """
        strategy_files = self.scan_strategies()
        loaded_strategies = {}
        
        for file_path in strategy_files:
            strategy_class = self.load_strategy_from_file(file_path)
            if strategy_class:
                loaded_strategies[strategy_class.__name__] = strategy_class
        
        self.logger.info(f"成功加载 {len(loaded_strategies)} 个策略")
        return loaded_strategies
    
    def reload_strategy(self, strategy_name: str) -> Optional[Type[BaseStrategy]]:
        """重新加载策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略类或None
        """
        if strategy_name not in self._strategy_metadata:
            self.logger.error(f"策略不存在: {strategy_name}")
            return None
        
        file_path = self._strategy_metadata[strategy_name].file_path
        
        # 清理旧的加载信息
        if strategy_name in self._strategy_classes:
            del self._strategy_classes[strategy_name]
        
        if strategy_name in self._strategy_metadata:
            del self._strategy_metadata[strategy_name]
        
        if strategy_name in self._loaded_modules:
            module_name = self._loaded_modules[strategy_name].__name__
            if module_name in sys.modules:
                del sys.modules[module_name]
            del self._loaded_modules[strategy_name]
        
        if file_path in self._file_hashes:
            del self._file_hashes[file_path]
        
        # 重新加载
        return self.load_strategy_from_file(file_path)
    
    def unload_strategy(self, strategy_name: str) -> bool:
        """卸载策略
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            是否卸载成功
        """
        try:
            if strategy_name not in self._strategy_classes:
                self.logger.warning(f"策略未加载: {strategy_name}")
                return False
            
            # 获取文件路径
            file_path = self._strategy_metadata[strategy_name].file_path
            
            # 清理所有相关信息
            del self._strategy_classes[strategy_name]
            del self._strategy_metadata[strategy_name]
            
            if strategy_name in self._loaded_modules:
                module_name = self._loaded_modules[strategy_name].__name__
                if module_name in sys.modules:
                    del sys.modules[module_name]
                del self._loaded_modules[strategy_name]
            
            if file_path in self._file_hashes:
                del self._file_hashes[file_path]
            
            self.logger.info(f"成功卸载策略: {strategy_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"卸载策略失败: {strategy_name}, 错误: {e}")
            return False
    
    def get_strategy_class(self, strategy_name: str) -> Optional[Type[BaseStrategy]]:
        """获取策略类
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略类或None
        """
        return self._strategy_classes.get(strategy_name)
    
    def get_strategy_metadata(self, strategy_name: str) -> Optional[StrategyMetadata]:
        """获取策略元数据
        
        Args:
            strategy_name: 策略名称
            
        Returns:
            策略元数据或None
        """
        return self._strategy_metadata.get(strategy_name)
    
    def get_all_strategy_names(self) -> List[str]:
        """获取所有策略名称"""
        return list(self._strategy_classes.keys())
    
    def get_all_metadata(self) -> Dict[str, StrategyMetadata]:
        """获取所有策略元数据"""
        return self._strategy_metadata.copy()
    
    def create_strategy_instance(self, strategy_name: str, config: StrategyConfig) -> Optional[BaseStrategy]:
        """创建策略实例
        
        Args:
            strategy_name: 策略名称
            config: 策略配置
            
        Returns:
            策略实例或None
        """
        try:
            # 验证配置
            self.validator.validate_strategy_config(config)
            
            # 获取策略类
            strategy_class = self.get_strategy_class(strategy_name)
            if strategy_class is None:
                self.logger.error(f"策略类不存在: {strategy_name}")
                return None
            
            # 创建实例
            strategy_instance = strategy_class(config)
            
            self.logger.info(f"成功创建策略实例: {strategy_name}")
            return strategy_instance
            
        except Exception as e:
            self.logger.error(f"创建策略实例失败: {strategy_name}, 错误: {e}")
            return None
    
    def check_file_changes(self) -> List[str]:
        """检查文件变更
        
        Returns:
            变更的策略名称列表
        """
        changed_strategies = []
        
        for strategy_name, metadata in self._strategy_metadata.items():
            file_path = metadata.file_path
            
            if not os.path.exists(file_path):
                self.logger.warning(f"策略文件已删除: {file_path}")
                changed_strategies.append(strategy_name)
                continue
            
            current_hash = self._calculate_file_hash(file_path)
            if current_hash != metadata.file_hash:
                self.logger.info(f"策略文件已修改: {file_path}")
                changed_strategies.append(strategy_name)
        
        return changed_strategies
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件哈希
        
        Args:
            file_path: 文件路径
            
        Returns:
            文件哈希值
        """
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def _get_module_name(self, file_path: str) -> str:
        """获取模块名称
        
        Args:
            file_path: 文件路径
            
        Returns:
            模块名称
        """
        # 使用文件路径的哈希作为模块名，避免冲突
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
        file_name = Path(file_path).stem
        return f"strategy_{file_name}_{path_hash}"
    
    def _find_strategy_class(self, module: Any) -> Optional[Type[BaseStrategy]]:
        """在模块中查找策略类
        
        Args:
            module: 模块对象
            
        Returns:
            策略类或None
        """
        for name in dir(module):
            obj = getattr(module, name)
            
            if (inspect.isclass(obj) and 
                issubclass(obj, BaseStrategy) and 
                obj is not BaseStrategy):
                return obj
        
        return None
    
    def _create_metadata(self, strategy_class: Type[BaseStrategy], file_path: str, file_hash: str) -> StrategyMetadata:
        """创建策略元数据
        
        Args:
            strategy_class: 策略类
            file_path: 文件路径
            file_hash: 文件哈希
            
        Returns:
            策略元数据
        """
        # 从类属性或文档字符串中提取信息
        name = getattr(strategy_class, 'STRATEGY_NAME', strategy_class.__name__)
        version = getattr(strategy_class, 'STRATEGY_VERSION', '1.0.0')
        description = getattr(strategy_class, 'STRATEGY_DESCRIPTION', strategy_class.__doc__ or '')
        author = getattr(strategy_class, 'STRATEGY_AUTHOR', 'Unknown')
        dependencies = getattr(strategy_class, 'STRATEGY_DEPENDENCIES', [])
        requirements = getattr(strategy_class, 'STRATEGY_REQUIREMENTS', [])
        tags = getattr(strategy_class, 'STRATEGY_TAGS', [])
        
        return StrategyMetadata(
            name=name,
            version=version,
            description=description,
            author=author,
            created_time=datetime.now(),
            file_path=file_path,
            file_hash=file_hash,
            class_name=strategy_class.__name__,
            dependencies=dependencies,
            requirements=requirements,
            tags=tags
        )
    
    def _get_strategy_name_by_file(self, file_path: str) -> Optional[str]:
        """根据文件路径获取策略名称
        
        Args:
            file_path: 文件路径
            
        Returns:
            策略名称或None
        """
        for name, metadata in self._strategy_metadata.items():
            if metadata.file_path == file_path:
                return name
        return None
    
    def save_metadata_to_file(self, file_path: str) -> bool:
        """保存元数据到文件
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否保存成功
        """
        try:
            metadata_dict = {}
            for name, metadata in self._strategy_metadata.items():
                metadata_dict[name] = metadata.to_dict()
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(metadata_dict, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"策略元数据已保存到: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"保存策略元数据失败: {e}")
            return False
    
    def load_metadata_from_file(self, file_path: str) -> bool:
        """从文件加载元数据
        
        Args:
            file_path: 文件路径
            
        Returns:
            是否加载成功
        """
        try:
            if not os.path.exists(file_path):
                self.logger.warning(f"元数据文件不存在: {file_path}")
                return False
            
            with open(file_path, 'r', encoding='utf-8') as f:
                metadata_dict = json.load(f)
            
            for name, data in metadata_dict.items():
                metadata = StrategyMetadata.from_dict(data)
                self._strategy_metadata[name] = metadata
            
            self.logger.info(f"策略元数据已从文件加载: {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"加载策略元数据失败: {e}")
            return False