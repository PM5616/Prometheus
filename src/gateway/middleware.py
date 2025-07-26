"""Middleware Module

中间件模块，提供请求处理的中间件功能。
"""

import time
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from src.common.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Request:
    """请求对象"""
    method: str
    path: str
    headers: Dict[str, str]
    body: Any = None
    params: Dict[str, str] = None
    query: Dict[str, str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}
        if self.query is None:
            self.query = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class Response:
    """响应对象"""
    status_code: int
    headers: Dict[str, str]
    body: Any = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.headers is None:
            self.headers = {}
        if self.metadata is None:
            self.metadata = {}


class Middleware(ABC):
    """中间件基类"""
    
    @abstractmethod
    def process_request(self, request: Request) -> Optional[Response]:
        """处理请求
        
        Args:
            request: 请求对象
            
        Returns:
            Optional[Response]: 如果返回Response，则中断处理链
        """
        pass
    
    @abstractmethod
    def process_response(self, request: Request, response: Response) -> Response:
        """处理响应
        
        Args:
            request: 请求对象
            response: 响应对象
            
        Returns:
            Response: 处理后的响应
        """
        pass


class LoggingMiddleware(Middleware):
    """日志中间件"""
    
    def __init__(self, log_requests: bool = True, log_responses: bool = True):
        """初始化日志中间件
        
        Args:
            log_requests: 是否记录请求
            log_responses: 是否记录响应
        """
        self.log_requests = log_requests
        self.log_responses = log_responses
    
    def process_request(self, request: Request) -> Optional[Response]:
        if self.log_requests:
            request.metadata['start_time'] = time.time()
            request.metadata['request_id'] = str(uuid.uuid4())
            
            logger.info(
                f"Request [{request.metadata['request_id']}]: "
                f"{request.method} {request.path}"
            )
            logger.debug(
                f"Request details [{request.metadata['request_id']}]: "
                f"headers={request.headers}, params={request.params}, query={request.query}"
            )
        
        return None
    
    def process_response(self, request: Request, response: Response) -> Response:
        if self.log_responses:
            request_id = request.metadata.get('request_id', 'unknown')
            start_time = request.metadata.get('start_time')
            
            duration = None
            if start_time:
                duration = (time.time() - start_time) * 1000  # 毫秒
            
            logger.info(
                f"Response [{request_id}]: {response.status_code} "
                f"({duration:.2f}ms)" if duration else f"Response [{request_id}]: {response.status_code}"
            )
            logger.debug(
                f"Response details [{request_id}]: headers={response.headers}"
            )
        
        return response


class CORSMiddleware(Middleware):
    """CORS中间件"""
    
    def __init__(self, 
                 allow_origins: List[str] = None,
                 allow_methods: List[str] = None,
                 allow_headers: List[str] = None,
                 allow_credentials: bool = False,
                 max_age: int = 86400):
        """初始化CORS中间件
        
        Args:
            allow_origins: 允许的源
            allow_methods: 允许的方法
            allow_headers: 允许的头部
            allow_credentials: 是否允许凭证
            max_age: 预检请求缓存时间
        """
        self.allow_origins = allow_origins or ['*']
        self.allow_methods = allow_methods or ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS']
        self.allow_headers = allow_headers or ['*']
        self.allow_credentials = allow_credentials
        self.max_age = max_age
    
    def process_request(self, request: Request) -> Optional[Response]:
        # 处理预检请求
        if request.method == 'OPTIONS':
            return Response(
                status_code=200,
                headers=self._get_cors_headers(request),
                body=''
            )
        
        return None
    
    def process_response(self, request: Request, response: Response) -> Response:
        # 添加CORS头部
        cors_headers = self._get_cors_headers(request)
        response.headers.update(cors_headers)
        
        return response
    
    def _get_cors_headers(self, request: Request) -> Dict[str, str]:
        """获取CORS头部"""
        headers = {}
        
        # Access-Control-Allow-Origin
        origin = request.headers.get('Origin')
        if '*' in self.allow_origins:
            headers['Access-Control-Allow-Origin'] = '*'
        elif origin and origin in self.allow_origins:
            headers['Access-Control-Allow-Origin'] = origin
        
        # Access-Control-Allow-Methods
        headers['Access-Control-Allow-Methods'] = ', '.join(self.allow_methods)
        
        # Access-Control-Allow-Headers
        if '*' in self.allow_headers:
            requested_headers = request.headers.get('Access-Control-Request-Headers')
            if requested_headers:
                headers['Access-Control-Allow-Headers'] = requested_headers
            else:
                headers['Access-Control-Allow-Headers'] = '*'
        else:
            headers['Access-Control-Allow-Headers'] = ', '.join(self.allow_headers)
        
        # Access-Control-Allow-Credentials
        if self.allow_credentials:
            headers['Access-Control-Allow-Credentials'] = 'true'
        
        # Access-Control-Max-Age
        headers['Access-Control-Max-Age'] = str(self.max_age)
        
        return headers


class SecurityMiddleware(Middleware):
    """安全中间件"""
    
    def __init__(self, 
                 add_security_headers: bool = True,
                 content_type_nosniff: bool = True,
                 frame_options: str = 'DENY',
                 xss_protection: str = '1; mode=block'):
        """初始化安全中间件
        
        Args:
            add_security_headers: 是否添加安全头部
            content_type_nosniff: 是否添加X-Content-Type-Options
            frame_options: X-Frame-Options值
            xss_protection: X-XSS-Protection值
        """
        self.add_security_headers = add_security_headers
        self.content_type_nosniff = content_type_nosniff
        self.frame_options = frame_options
        self.xss_protection = xss_protection
    
    def process_request(self, request: Request) -> Optional[Response]:
        return None
    
    def process_response(self, request: Request, response: Response) -> Response:
        if self.add_security_headers:
            # X-Content-Type-Options
            if self.content_type_nosniff:
                response.headers['X-Content-Type-Options'] = 'nosniff'
            
            # X-Frame-Options
            if self.frame_options:
                response.headers['X-Frame-Options'] = self.frame_options
            
            # X-XSS-Protection
            if self.xss_protection:
                response.headers['X-XSS-Protection'] = self.xss_protection
            
            # Referrer-Policy
            response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
        
        return response


class CompressionMiddleware(Middleware):
    """压缩中间件"""
    
    def __init__(self, min_size: int = 1024, compression_level: int = 6):
        """初始化压缩中间件
        
        Args:
            min_size: 最小压缩大小
            compression_level: 压缩级别
        """
        self.min_size = min_size
        self.compression_level = compression_level
    
    def process_request(self, request: Request) -> Optional[Response]:
        return None
    
    def process_response(self, request: Request, response: Response) -> Response:
        # 检查是否支持压缩
        accept_encoding = request.headers.get('Accept-Encoding', '')
        if 'gzip' not in accept_encoding.lower():
            return response
        
        # 检查内容大小
        if isinstance(response.body, str):
            content_size = len(response.body.encode('utf-8'))
        elif isinstance(response.body, bytes):
            content_size = len(response.body)
        else:
            return response
        
        if content_size < self.min_size:
            return response
        
        # 执行压缩（这里只是示例，实际需要使用gzip库）
        try:
            import gzip
            
            if isinstance(response.body, str):
                content = response.body.encode('utf-8')
            else:
                content = response.body
            
            compressed = gzip.compress(content, compresslevel=self.compression_level)
            
            response.body = compressed
            response.headers['Content-Encoding'] = 'gzip'
            response.headers['Content-Length'] = str(len(compressed))
            
        except Exception as e:
            logger.warning(f"Compression failed: {e}")
        
        return response


class MiddlewareChain:
    """中间件链
    
    管理和执行中间件链。
    """
    
    def __init__(self):
        """初始化中间件链"""
        self.middlewares: List[Middleware] = []
        
        logger.info("Middleware chain initialized")
    
    def add_middleware(self, middleware: Middleware):
        """添加中间件
        
        Args:
            middleware: 中间件实例
        """
        self.middlewares.append(middleware)
        logger.info(f"Middleware {middleware.__class__.__name__} added to chain")
    
    def process_request(self, request: Request) -> Optional[Response]:
        """处理请求
        
        Args:
            request: 请求对象
            
        Returns:
            Optional[Response]: 如果中间件返回响应，则中断处理
        """
        for middleware in self.middlewares:
            try:
                response = middleware.process_request(request)
                if response is not None:
                    logger.debug(f"Request processing stopped by {middleware.__class__.__name__}")
                    return response
            except Exception as e:
                logger.error(f"Error in middleware {middleware.__class__.__name__}: {e}")
                return Response(
                    status_code=500,
                    headers={'Content-Type': 'application/json'},
                    body={'error': 'Internal server error'}
                )
        
        return None
    
    def process_response(self, request: Request, response: Response) -> Response:
        """处理响应
        
        Args:
            request: 请求对象
            response: 响应对象
            
        Returns:
            Response: 处理后的响应
        """
        # 反向执行中间件
        for middleware in reversed(self.middlewares):
            try:
                response = middleware.process_response(request, response)
            except Exception as e:
                logger.error(f"Error in middleware {middleware.__class__.__name__}: {e}")
        
        return response
    
    def get_middlewares(self) -> List[Middleware]:
        """获取中间件列表
        
        Returns:
            List[Middleware]: 中间件列表
        """
        return self.middlewares.copy()
    
    def clear(self):
        """清空中间件链"""
        self.middlewares.clear()
        logger.info("Middleware chain cleared")