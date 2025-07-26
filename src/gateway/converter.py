"""Protocol Converter Module

协议转换器模块，提供不同协议之间的转换功能。
"""

import json
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass
from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ConversionRequest:
    """转换请求"""
    data: Any
    source_format: str
    target_format: str
    headers: Dict[str, str] = None
    metadata: Dict[str, Any] = None


@dataclass
class ConversionResponse:
    """转换响应"""
    data: Any
    format: str
    headers: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    success: bool = True
    error: str = None


class Converter(ABC):
    """转换器基类"""
    
    @abstractmethod
    def can_convert(self, source_format: str, target_format: str) -> bool:
        """检查是否支持转换
        
        Args:
            source_format: 源格式
            target_format: 目标格式
            
        Returns:
            bool: 是否支持转换
        """
        pass
    
    @abstractmethod
    def convert(self, request: ConversionRequest) -> ConversionResponse:
        """执行转换
        
        Args:
            request: 转换请求
            
        Returns:
            ConversionResponse: 转换响应
        """
        pass


class JSONConverter(Converter):
    """JSON转换器"""
    
    def can_convert(self, source_format: str, target_format: str) -> bool:
        supported_formats = {'json', 'dict', 'object'}
        return source_format.lower() in supported_formats or target_format.lower() in supported_formats
    
    def convert(self, request: ConversionRequest) -> ConversionResponse:
        try:
            source_format = request.source_format.lower()
            target_format = request.target_format.lower()
            
            # 转换为JSON
            if target_format == 'json':
                if source_format in {'dict', 'object'}:
                    data = json.dumps(request.data, ensure_ascii=False, indent=2)
                elif source_format == 'json':
                    # 验证并格式化JSON
                    parsed = json.loads(request.data) if isinstance(request.data, str) else request.data
                    data = json.dumps(parsed, ensure_ascii=False, indent=2)
                else:
                    data = json.dumps(request.data, ensure_ascii=False, indent=2)
                
                return ConversionResponse(
                    data=data,
                    format='json',
                    headers={'Content-Type': 'application/json'}
                )
            
            # 从JSON转换
            elif source_format == 'json':
                if isinstance(request.data, str):
                    data = json.loads(request.data)
                else:
                    data = request.data
                
                return ConversionResponse(
                    data=data,
                    format=target_format,
                    headers={'Content-Type': 'application/json'}
                )
            
            else:
                return ConversionResponse(
                    data=request.data,
                    format=target_format,
                    success=False,
                    error=f"Unsupported conversion: {source_format} -> {target_format}"
                )
                
        except Exception as e:
            logger.error(f"JSON conversion error: {e}")
            return ConversionResponse(
                data=None,
                format=request.target_format,
                success=False,
                error=str(e)
            )


class XMLConverter(Converter):
    """XML转换器"""
    
    def can_convert(self, source_format: str, target_format: str) -> bool:
        supported_formats = {'xml', 'dict', 'object'}
        return source_format.lower() in supported_formats or target_format.lower() in supported_formats
    
    def convert(self, request: ConversionRequest) -> ConversionResponse:
        try:
            source_format = request.source_format.lower()
            target_format = request.target_format.lower()
            
            # 转换为XML
            if target_format == 'xml':
                if source_format in {'dict', 'object'}:
                    data = self._dict_to_xml(request.data)
                elif source_format == 'xml':
                    # 验证XML
                    ET.fromstring(request.data)
                    data = request.data
                else:
                    data = self._dict_to_xml({'data': request.data})
                
                return ConversionResponse(
                    data=data,
                    format='xml',
                    headers={'Content-Type': 'application/xml'}
                )
            
            # 从XML转换
            elif source_format == 'xml':
                data = self._xml_to_dict(request.data)
                
                return ConversionResponse(
                    data=data,
                    format=target_format,
                    headers={'Content-Type': 'application/xml'}
                )
            
            else:
                return ConversionResponse(
                    data=request.data,
                    format=target_format,
                    success=False,
                    error=f"Unsupported conversion: {source_format} -> {target_format}"
                )
                
        except Exception as e:
            logger.error(f"XML conversion error: {e}")
            return ConversionResponse(
                data=None,
                format=request.target_format,
                success=False,
                error=str(e)
            )
    
    def _dict_to_xml(self, data: Dict[str, Any], root_name: str = 'root') -> str:
        """字典转XML"""
        root = ET.Element(root_name)
        self._build_xml_element(root, data)
        return ET.tostring(root, encoding='unicode')
    
    def _build_xml_element(self, parent: ET.Element, data: Any):
        """构建XML元素"""
        if isinstance(data, dict):
            for key, value in data.items():
                child = ET.SubElement(parent, str(key))
                self._build_xml_element(child, value)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                child = ET.SubElement(parent, f'item_{i}')
                self._build_xml_element(child, item)
        else:
            parent.text = str(data)
    
    def _xml_to_dict(self, xml_data: str) -> Dict[str, Any]:
        """XML转字典"""
        root = ET.fromstring(xml_data)
        return {root.tag: self._parse_xml_element(root)}
    
    def _parse_xml_element(self, element: ET.Element) -> Any:
        """解析XML元素"""
        if len(element) == 0:
            return element.text
        
        result = {}
        for child in element:
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(self._parse_xml_element(child))
            else:
                result[child.tag] = self._parse_xml_element(child)
        
        return result


class ProtocolConverter:
    """协议转换器
    
    提供多种协议格式之间的转换功能。
    """
    
    def __init__(self):
        """初始化协议转换器"""
        self.converters = [
            JSONConverter(),
            XMLConverter()
        ]
        
        logger.info("Protocol converter initialized")
    
    def add_converter(self, converter: Converter):
        """添加转换器
        
        Args:
            converter: 转换器实例
        """
        self.converters.append(converter)
        logger.info(f"Converter {converter.__class__.__name__} added")
    
    def convert(self, request: ConversionRequest) -> ConversionResponse:
        """执行转换
        
        Args:
            request: 转换请求
            
        Returns:
            ConversionResponse: 转换响应
        """
        # 查找合适的转换器
        for converter in self.converters:
            if converter.can_convert(request.source_format, request.target_format):
                logger.debug(f"Using {converter.__class__.__name__} for conversion")
                return converter.convert(request)
        
        # 没有找到合适的转换器
        logger.warning(f"No converter found for {request.source_format} -> {request.target_format}")
        return ConversionResponse(
            data=request.data,
            format=request.target_format,
            success=False,
            error=f"No converter available for {request.source_format} -> {request.target_format}"
        )
    
    def get_supported_formats(self) -> Dict[str, list]:
        """获取支持的格式
        
        Returns:
            Dict[str, list]: 支持的格式列表
        """
        formats = set()
        
        # 测试常见格式组合
        test_formats = ['json', 'xml', 'dict', 'object', 'text', 'binary']
        
        for source in test_formats:
            for target in test_formats:
                if source != target:
                    for converter in self.converters:
                        if converter.can_convert(source, target):
                            formats.add((source, target))
        
        # 按源格式分组
        result = {}
        for source, target in formats:
            if source not in result:
                result[source] = []
            result[source].append(target)
        
        return result
    
    def validate_format(self, data: Any, format_type: str) -> bool:
        """验证数据格式
        
        Args:
            data: 数据
            format_type: 格式类型
            
        Returns:
            bool: 是否有效
        """
        try:
            format_type = format_type.lower()
            
            if format_type == 'json':
                if isinstance(data, str):
                    json.loads(data)
                else:
                    json.dumps(data)
                return True
            
            elif format_type == 'xml':
                if isinstance(data, str):
                    ET.fromstring(data)
                    return True
                return False
            
            elif format_type in {'dict', 'object'}:
                return isinstance(data, dict)
            
            elif format_type == 'text':
                return isinstance(data, str)
            
            else:
                return True  # 未知格式，假设有效
                
        except Exception as e:
            logger.debug(f"Format validation failed for {format_type}: {e}")
            return False