"""Base Data Models

基础数据模型，提供通用的数据结构和功能。
"""

from datetime import datetime
from typing import Any, Dict, Optional
from pydantic import BaseModel as PydanticBaseModel, Field
from decimal import Decimal
import uuid


class BaseModel(PydanticBaseModel):
    """基础数据模型
    
    所有数据模型的基类，提供通用功能。
    """
    
    model_config = {
        # 允许使用Decimal类型
        "arbitrary_types_allowed": True,
        # 验证赋值
        "validate_assignment": True,
        # 使用枚举值
        "use_enum_values": True
    }
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            Dict[str, Any]: 字典表示
        """
        return self.model_dump()
    
    def to_json(self) -> str:
        """转换为JSON字符串
        
        Returns:
            str: JSON字符串
        """
        return self.model_dump_json()
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """从字典创建实例
        
        Args:
            data: 字典数据
            
        Returns:
            BaseModel: 模型实例
        """
        return cls(**data)
    
    @classmethod
    def from_json(cls, json_str: str):
        """从JSON字符串创建实例
        
        Args:
            json_str: JSON字符串
            
        Returns:
            BaseModel: 模型实例
        """
        return cls.model_validate_json(json_str)


class TimestampMixin(BaseModel):
    """时间戳混入类
    
    为模型添加创建时间和更新时间字段。
    """
    
    created_at: datetime = Field(default_factory=datetime.utcnow, description="创建时间")
    updated_at: Optional[datetime] = Field(None, description="更新时间")
    
    def update_timestamp(self) -> None:
        """更新时间戳"""
        self.updated_at = datetime.utcnow()


class IdentifiableMixin(BaseModel):
    """可识别混入类
    
    为模型添加唯一标识符。
    """
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="唯一标识符")


class MetadataMixin(BaseModel):
    """元数据混入类
    
    为模型添加元数据字段。
    """
    
    metadata: Optional[Dict[str, Any]] = Field(None, description="元数据")
    
    def set_metadata(self, key: str, value: Any) -> None:
        """设置元数据
        
        Args:
            key: 键
            value: 值
        """
        if self.metadata is None:
            self.metadata = {}
        self.metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """获取元数据
        
        Args:
            key: 键
            default: 默认值
            
        Returns:
            Any: 元数据值
        """
        if self.metadata is None:
            return default
        return self.metadata.get(key, default)


class StatusMixin(BaseModel):
    """状态混入类
    
    为模型添加状态字段。
    """
    
    status: str = Field("active", description="状态")
    
    def is_active(self) -> bool:
        """是否为活跃状态
        
        Returns:
            bool: 是否活跃
        """
        return self.status == "active"
    
    def activate(self) -> None:
        """激活"""
        self.status = "active"
    
    def deactivate(self) -> None:
        """停用"""
        self.status = "inactive"