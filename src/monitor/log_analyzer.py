"""Log Analyzer Module

日志分析器，负责日志收集、分析和监控。

主要功能：
- 日志收集和解析
- 日志模式识别
- 异常日志检测
- 日志统计分析
- 日志告警
"""

import re
import time
import asyncio
import json
from typing import Dict, List, Optional, Any, Callable, Pattern, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict, Counter
from datetime import datetime, timedelta
import os
import glob
from pathlib import Path

from src.common.logging.logger import get_logger
from src.common.exceptions.monitor_exceptions import MonitorError
from src.common.models import LogLevel


class LogSource(Enum):
    """日志源"""
    FILE = "file"
    SYSLOG = "syslog"
    JOURNAL = "journal"
    NETWORK = "network"
    DATABASE = "database"
    APPLICATION = "application"


class PatternType(Enum):
    """模式类型"""
    ERROR_PATTERN = "error_pattern"
    WARNING_PATTERN = "warning_pattern"
    SECURITY_PATTERN = "security_pattern"
    PERFORMANCE_PATTERN = "performance_pattern"
    CUSTOM_PATTERN = "custom_pattern"


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: float
    level: LogLevel
    source: str
    message: str
    raw_line: str
    fields: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'timestamp': self.timestamp,
            'level': self.level.value,
            'source': self.source,
            'message': self.message,
            'raw_line': self.raw_line,
            'fields': self.fields,
            'tags': self.tags
        }


@dataclass
class LogPattern:
    """日志模式"""
    id: str
    name: str
    pattern: str
    pattern_type: PatternType
    compiled_pattern: Pattern
    description: str = ""
    severity: int = 1  # 1-10，10最严重
    enabled: bool = True
    tags: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """初始化后处理"""
        if isinstance(self.compiled_pattern, str):
            self.compiled_pattern = re.compile(self.pattern)
    
    def match(self, log_entry: LogEntry) -> bool:
        """匹配日志条目
        
        Args:
            log_entry: 日志条目
            
        Returns:
            是否匹配
        """
        if not self.enabled:
            return False
        
        return bool(self.compiled_pattern.search(log_entry.message) or 
                   self.compiled_pattern.search(log_entry.raw_line))
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'pattern': self.pattern,
            'pattern_type': self.pattern_type.value,
            'description': self.description,
            'severity': self.severity,
            'enabled': self.enabled,
            'tags': self.tags
        }


@dataclass
class LogAlert:
    """日志告警"""
    id: str
    pattern_id: str
    pattern_name: str
    log_entry: LogEntry
    severity: int
    count: int = 1
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'id': self.id,
            'pattern_id': self.pattern_id,
            'pattern_name': self.pattern_name,
            'log_entry': self.log_entry.to_dict(),
            'severity': self.severity,
            'count': self.count,
            'first_seen': self.first_seen,
            'last_seen': self.last_seen
        }


@dataclass
class LogStats:
    """日志统计"""
    total_logs: int = 0
    level_counts: Dict[LogLevel, int] = field(default_factory=lambda: defaultdict(int))
    source_counts: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    pattern_matches: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_rate: float = 0.0
    warning_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'total_logs': self.total_logs,
            'level_counts': {level.value: count for level, count in self.level_counts.items()},
            'source_counts': dict(self.source_counts),
            'pattern_matches': dict(self.pattern_matches),
            'error_rate': self.error_rate,
            'warning_rate': self.warning_rate
        }


class LogParser:
    """日志解析器"""
    
    def __init__(self):
        """初始化日志解析器"""
        self.logger = get_logger("log_parser")
        
        # 常见日志格式的正则表达式
        self.log_patterns = {
            'syslog': re.compile(
                r'(?P<timestamp>\w{3}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2})\s+'
                r'(?P<hostname>\S+)\s+'
                r'(?P<program>\S+)(?:\[(?P<pid>\d+)\])?:\s*'
                r'(?P<message>.*)'
            ),
            'apache': re.compile(
                r'(?P<ip>\S+)\s+\S+\s+\S+\s+'
                r'\[(?P<timestamp>[^\]]+)\]\s+'
                r'"(?P<method>\S+)\s+(?P<url>\S+)\s+(?P<protocol>\S+)"\s+'
                r'(?P<status>\d+)\s+(?P<size>\S+)'
            ),
            'nginx': re.compile(
                r'(?P<ip>\S+)\s+-\s+-\s+'
                r'\[(?P<timestamp>[^\]]+)\]\s+'
                r'"(?P<method>\S+)\s+(?P<url>\S+)\s+(?P<protocol>\S+)"\s+'
                r'(?P<status>\d+)\s+(?P<size>\S+)\s+'
                r'"(?P<referer>[^"]*)"\s+"(?P<user_agent>[^"]*)"'
            ),
            'python': re.compile(
                r'(?P<timestamp>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3})\s+'
                r'(?P<level>\w+)\s+'
                r'(?P<logger>\S+)\s+'
                r'(?P<message>.*)'
            ),
            'json': re.compile(r'^\s*{.*}\s*$')
        }
        
        # 日志级别映射
        self.level_mapping = {
            'debug': LogLevel.DEBUG,
            'info': LogLevel.INFO,
            'warn': LogLevel.WARNING,
            'warning': LogLevel.WARNING,
            'error': LogLevel.ERROR,
            'err': LogLevel.ERROR,
            'critical': LogLevel.CRITICAL,
            'crit': LogLevel.CRITICAL,
            'fatal': LogLevel.FATAL,
            'emerg': LogLevel.CRITICAL,
            'alert': LogLevel.CRITICAL
        }
    
    def parse_line(self, line: str, source: str = "unknown") -> Optional[LogEntry]:
        """解析日志行
        
        Args:
            line: 日志行
            source: 日志源
            
        Returns:
            解析后的日志条目
        """
        line = line.strip()
        if not line:
            return None
        
        # 尝试JSON格式
        if self.log_patterns['json'].match(line):
            return self._parse_json_log(line, source)
        
        # 尝试其他格式
        for format_name, pattern in self.log_patterns.items():
            if format_name == 'json':
                continue
                
            match = pattern.match(line)
            if match:
                return self._parse_structured_log(match, line, source, format_name)
        
        # 默认解析
        return self._parse_generic_log(line, source)
    
    def _parse_json_log(self, line: str, source: str) -> Optional[LogEntry]:
        """解析JSON格式日志
        
        Args:
            line: 日志行
            source: 日志源
            
        Returns:
            日志条目
        """
        try:
            data = json.loads(line)
            
            # 提取时间戳
            timestamp = time.time()
            if 'timestamp' in data:
                timestamp = self._parse_timestamp(data['timestamp'])
            elif 'time' in data:
                timestamp = self._parse_timestamp(data['time'])
            elif '@timestamp' in data:
                timestamp = self._parse_timestamp(data['@timestamp'])
            
            # 提取日志级别
            level = LogLevel.INFO
            if 'level' in data:
                level = self._parse_level(data['level'])
            elif 'severity' in data:
                level = self._parse_level(data['severity'])
            
            # 提取消息
            message = data.get('message', data.get('msg', str(data)))
            
            return LogEntry(
                timestamp=timestamp,
                level=level,
                source=source,
                message=message,
                raw_line=line,
                fields=data
            )
            
        except json.JSONDecodeError:
            return None
    
    def _parse_structured_log(self, match: re.Match, line: str, source: str, format_name: str) -> LogEntry:
        """解析结构化日志
        
        Args:
            match: 正则匹配结果
            line: 日志行
            source: 日志源
            format_name: 格式名称
            
        Returns:
            日志条目
        """
        groups = match.groupdict()
        
        # 解析时间戳
        timestamp = time.time()
        if 'timestamp' in groups and groups['timestamp']:
            timestamp = self._parse_timestamp(groups['timestamp'])
        
        # 解析日志级别
        level = LogLevel.INFO
        if 'level' in groups and groups['level']:
            level = self._parse_level(groups['level'])
        elif format_name in ['apache', 'nginx'] and 'status' in groups:
            # HTTP状态码映射到日志级别
            status = int(groups['status'])
            if status >= 500:
                level = LogLevel.ERROR
            elif status >= 400:
                level = LogLevel.WARNING
        
        # 提取消息
        message = groups.get('message', line)
        
        return LogEntry(
            timestamp=timestamp,
            level=level,
            source=source,
            message=message,
            raw_line=line,
            fields=groups
        )
    
    def _parse_generic_log(self, line: str, source: str) -> LogEntry:
        """解析通用日志
        
        Args:
            line: 日志行
            source: 日志源
            
        Returns:
            日志条目
        """
        # 尝试从行中提取日志级别
        level = LogLevel.INFO
        for level_str, log_level in self.level_mapping.items():
            if level_str.upper() in line.upper():
                level = log_level
                break
        
        return LogEntry(
            timestamp=time.time(),
            level=level,
            source=source,
            message=line,
            raw_line=line
        )
    
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """解析时间戳
        
        Args:
            timestamp_str: 时间戳字符串
            
        Returns:
            Unix时间戳
        """
        try:
            # 尝试多种时间格式
            formats = [
                '%Y-%m-%d %H:%M:%S,%f',
                '%Y-%m-%d %H:%M:%S.%f',
                '%Y-%m-%d %H:%M:%S',
                '%Y-%m-%dT%H:%M:%S.%fZ',
                '%Y-%m-%dT%H:%M:%SZ',
                '%d/%b/%Y:%H:%M:%S %z',
                '%b %d %H:%M:%S'
            ]
            
            for fmt in formats:
                try:
                    dt = datetime.strptime(timestamp_str, fmt)
                    return dt.timestamp()
                except ValueError:
                    continue
            
            # 尝试解析Unix时间戳
            return float(timestamp_str)
            
        except (ValueError, TypeError):
            return time.time()
    
    def _parse_level(self, level_str: str) -> LogLevel:
        """解析日志级别
        
        Args:
            level_str: 级别字符串
            
        Returns:
            日志级别
        """
        level_str = level_str.lower().strip()
        return self.level_mapping.get(level_str, LogLevel.INFO)


class LogAnalyzer:
    """日志分析器
    
    负责日志收集、分析和监控。
    """
    
    def __init__(self, max_entries: int = 10000):
        """初始化日志分析器
        
        Args:
            max_entries: 最大日志条目数
        """
        self.max_entries = max_entries
        self.logger = get_logger("log_analyzer")
        
        # 日志存储
        self.log_entries: deque = deque(maxlen=max_entries)
        self.log_patterns: Dict[str, LogPattern] = {}
        self.log_alerts: Dict[str, LogAlert] = {}
        
        # 统计信息
        self.stats = LogStats()
        
        # 解析器
        self.parser = LogParser()
        
        # 监控的文件
        self.monitored_files: Dict[str, Dict[str, Any]] = {}
        
        # 运行状态
        self._running = False
        self._monitor_task: Optional[asyncio.Task] = None
        
        # 回调函数
        self.log_callbacks: List[Callable[[LogEntry], None]] = []
        self.alert_callbacks: List[Callable[[LogAlert], None]] = []
        
        # 初始化默认模式
        self._init_default_patterns()
        
        self.logger.info(f"日志分析器初始化完成，最大条目数: {max_entries}")
    
    def _init_default_patterns(self) -> None:
        """初始化默认模式"""
        default_patterns = [
            LogPattern(
                id="error_pattern",
                name="错误模式",
                pattern=r'(?i)(error|exception|failed|failure|fatal)',
                pattern_type=PatternType.ERROR_PATTERN,
                compiled_pattern=re.compile(r'(?i)(error|exception|failed|failure|fatal)'),
                description="检测错误相关的日志",
                severity=8,
                tags=["error"]
            ),
            LogPattern(
                id="warning_pattern",
                name="警告模式",
                pattern=r'(?i)(warning|warn|deprecated)',
                pattern_type=PatternType.WARNING_PATTERN,
                compiled_pattern=re.compile(r'(?i)(warning|warn|deprecated)'),
                description="检测警告相关的日志",
                severity=5,
                tags=["warning"]
            ),
            LogPattern(
                id="security_pattern",
                name="安全模式",
                pattern=r'(?i)(unauthorized|forbidden|authentication|login.*failed|security|breach)',
                pattern_type=PatternType.SECURITY_PATTERN,
                compiled_pattern=re.compile(r'(?i)(unauthorized|forbidden|authentication|login.*failed|security|breach)'),
                description="检测安全相关的日志",
                severity=9,
                tags=["security"]
            ),
            LogPattern(
                id="performance_pattern",
                name="性能模式",
                pattern=r'(?i)(timeout|slow|performance|latency|response.*time)',
                pattern_type=PatternType.PERFORMANCE_PATTERN,
                compiled_pattern=re.compile(r'(?i)(timeout|slow|performance|latency|response.*time)'),
                description="检测性能相关的日志",
                severity=6,
                tags=["performance"]
            )
        ]
        
        for pattern in default_patterns:
            self.log_patterns[pattern.id] = pattern
    
    async def start(self) -> None:
        """启动日志分析器"""
        if self._running:
            self.logger.warning("日志分析器已在运行")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_files())
        self.logger.info("日志分析器已启动")
    
    async def stop(self) -> None:
        """停止日志分析器"""
        if not self._running:
            return
        
        self._running = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("日志分析器已停止")
    
    async def _monitor_files(self) -> None:
        """监控文件变化"""
        while self._running:
            try:
                for file_path, file_info in self.monitored_files.items():
                    await self._check_file_changes(file_path, file_info)
                
                await asyncio.sleep(1.0)  # 每秒检查一次
                
            except Exception as e:
                self.logger.error(f"文件监控出错: {e}")
                await asyncio.sleep(5.0)
    
    async def _check_file_changes(self, file_path: str, file_info: Dict[str, Any]) -> None:
        """检查文件变化
        
        Args:
            file_path: 文件路径
            file_info: 文件信息
        """
        try:
            if not os.path.exists(file_path):
                return
            
            stat = os.stat(file_path)
            current_size = stat.st_size
            current_mtime = stat.st_mtime
            
            # 检查文件是否有变化
            if (current_size != file_info.get('size', 0) or 
                current_mtime != file_info.get('mtime', 0)):
                
                # 读取新增内容
                await self._read_file_changes(file_path, file_info, current_size)
                
                # 更新文件信息
                file_info['size'] = current_size
                file_info['mtime'] = current_mtime
                
        except Exception as e:
            self.logger.error(f"检查文件变化失败 {file_path}: {e}")
    
    async def _read_file_changes(self, file_path: str, file_info: Dict[str, Any], current_size: int) -> None:
        """读取文件变化
        
        Args:
            file_path: 文件路径
            file_info: 文件信息
            current_size: 当前文件大小
        """
        try:
            last_position = file_info.get('position', 0)
            
            # 如果文件被截断，从头开始读取
            if current_size < last_position:
                last_position = 0
            
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                f.seek(last_position)
                
                for line in f:
                    log_entry = self.parser.parse_line(line.strip(), file_path)
                    if log_entry:
                        self.add_log_entry(log_entry)
                
                # 更新位置
                file_info['position'] = f.tell()
                
        except Exception as e:
            self.logger.error(f"读取文件变化失败 {file_path}: {e}")
    
    def add_log_file(self, file_path: str) -> None:
        """添加监控的日志文件
        
        Args:
            file_path: 文件路径
        """
        try:
            # 支持通配符
            if '*' in file_path or '?' in file_path:
                files = glob.glob(file_path)
                for f in files:
                    self._add_single_file(f)
            else:
                self._add_single_file(file_path)
                
        except Exception as e:
            self.logger.error(f"添加日志文件失败 {file_path}: {e}")
    
    def _add_single_file(self, file_path: str) -> None:
        """添加单个文件
        
        Args:
            file_path: 文件路径
        """
        if not os.path.exists(file_path):
            self.logger.warning(f"文件不存在: {file_path}")
            return
        
        if file_path in self.monitored_files:
            return
        
        stat = os.stat(file_path)
        self.monitored_files[file_path] = {
            'size': stat.st_size,
            'mtime': stat.st_mtime,
            'position': stat.st_size  # 从文件末尾开始监控
        }
        
        self.logger.info(f"已添加监控文件: {file_path}")
    
    def remove_log_file(self, file_path: str) -> None:
        """移除监控的日志文件
        
        Args:
            file_path: 文件路径
        """
        if file_path in self.monitored_files:
            del self.monitored_files[file_path]
            self.logger.info(f"已移除监控文件: {file_path}")
    
    def add_log_entry(self, log_entry: LogEntry) -> None:
        """添加日志条目
        
        Args:
            log_entry: 日志条目
        """
        # 存储日志条目
        self.log_entries.append(log_entry)
        
        # 更新统计信息
        self._update_stats(log_entry)
        
        # 检查模式匹配
        self._check_patterns(log_entry)
        
        # 调用回调
        for callback in self.log_callbacks:
            try:
                callback(log_entry)
            except Exception as e:
                self.logger.error(f"日志回调执行失败: {e}")
    
    def _update_stats(self, log_entry: LogEntry) -> None:
        """更新统计信息
        
        Args:
            log_entry: 日志条目
        """
        self.stats.total_logs += 1
        self.stats.level_counts[log_entry.level] += 1
        self.stats.source_counts[log_entry.source] += 1
        
        # 计算错误率和警告率
        if self.stats.total_logs > 0:
            error_count = (self.stats.level_counts[LogLevel.ERROR] + 
                          self.stats.level_counts[LogLevel.CRITICAL] + 
                          self.stats.level_counts[LogLevel.FATAL])
            warning_count = self.stats.level_counts[LogLevel.WARNING]
            
            self.stats.error_rate = error_count / self.stats.total_logs * 100
            self.stats.warning_rate = warning_count / self.stats.total_logs * 100
    
    def _check_patterns(self, log_entry: LogEntry) -> None:
        """检查模式匹配
        
        Args:
            log_entry: 日志条目
        """
        for pattern in self.log_patterns.values():
            if pattern.match(log_entry):
                self.stats.pattern_matches[pattern.id] += 1
                
                # 创建或更新告警
                self._create_or_update_alert(pattern, log_entry)
    
    def _create_or_update_alert(self, pattern: LogPattern, log_entry: LogEntry) -> None:
        """创建或更新告警
        
        Args:
            pattern: 日志模式
            log_entry: 日志条目
        """
        alert_key = f"{pattern.id}_{log_entry.source}"
        
        if alert_key in self.log_alerts:
            # 更新现有告警
            alert = self.log_alerts[alert_key]
            alert.count += 1
            alert.last_seen = log_entry.timestamp
            alert.log_entry = log_entry  # 更新为最新的日志条目
        else:
            # 创建新告警
            alert = LogAlert(
                id=alert_key,
                pattern_id=pattern.id,
                pattern_name=pattern.name,
                log_entry=log_entry,
                severity=pattern.severity,
                first_seen=log_entry.timestamp,
                last_seen=log_entry.timestamp
            )
            self.log_alerts[alert_key] = alert
            
            # 调用告警回调
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"告警回调执行失败: {e}")
    
    def add_pattern(self, pattern: LogPattern) -> None:
        """添加日志模式
        
        Args:
            pattern: 日志模式
        """
        self.log_patterns[pattern.id] = pattern
        self.logger.info(f"已添加日志模式: {pattern.name}")
    
    def remove_pattern(self, pattern_id: str) -> bool:
        """移除日志模式
        
        Args:
            pattern_id: 模式ID
            
        Returns:
            是否成功移除
        """
        if pattern_id in self.log_patterns:
            del self.log_patterns[pattern_id]
            
            # 移除相关告警
            alerts_to_remove = [
                alert_id for alert_id, alert in self.log_alerts.items()
                if alert.pattern_id == pattern_id
            ]
            
            for alert_id in alerts_to_remove:
                del self.log_alerts[alert_id]
            
            self.logger.info(f"已移除日志模式: {pattern_id}")
            return True
        
        return False
    
    def search_logs(self, query: str, limit: int = 100, 
                   level: Optional[LogLevel] = None,
                   source: Optional[str] = None,
                   start_time: Optional[float] = None,
                   end_time: Optional[float] = None) -> List[LogEntry]:
        """搜索日志
        
        Args:
            query: 搜索查询
            limit: 限制数量
            level: 日志级别过滤
            source: 日志源过滤
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            匹配的日志条目
        """
        try:
            pattern = re.compile(query, re.IGNORECASE)
        except re.error:
            # 如果不是有效的正则表达式，使用字符串匹配
            pattern = None
        
        results = []
        
        for log_entry in reversed(self.log_entries):  # 从最新开始搜索
            # 时间过滤
            if start_time and log_entry.timestamp < start_time:
                continue
            if end_time and log_entry.timestamp > end_time:
                continue
            
            # 级别过滤
            if level and log_entry.level != level:
                continue
            
            # 源过滤
            if source and source not in log_entry.source:
                continue
            
            # 内容匹配
            if pattern:
                if pattern.search(log_entry.message) or pattern.search(log_entry.raw_line):
                    results.append(log_entry)
            else:
                if query.lower() in log_entry.message.lower() or query.lower() in log_entry.raw_line.lower():
                    results.append(log_entry)
            
            if len(results) >= limit:
                break
        
        return results
    
    def get_recent_logs(self, limit: int = 100, level: Optional[LogLevel] = None) -> List[LogEntry]:
        """获取最近的日志
        
        Args:
            limit: 限制数量
            level: 日志级别过滤
            
        Returns:
            最近的日志条目
        """
        logs = list(self.log_entries)
        
        if level:
            logs = [log for log in logs if log.level == level]
        
        # 按时间倒序排列
        logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return logs[:limit] if limit > 0 else logs
    
    def get_alerts(self, severity_threshold: int = 0) -> List[LogAlert]:
        """获取告警
        
        Args:
            severity_threshold: 严重程度阈值
            
        Returns:
            告警列表
        """
        alerts = [
            alert for alert in self.log_alerts.values()
            if alert.severity >= severity_threshold
        ]
        
        # 按严重程度和最后出现时间排序
        alerts.sort(key=lambda x: (x.severity, x.last_seen), reverse=True)
        
        return alerts
    
    def add_log_callback(self, callback: Callable[[LogEntry], None]) -> None:
        """添加日志回调
        
        Args:
            callback: 回调函数
        """
        self.log_callbacks.append(callback)
    
    def add_alert_callback(self, callback: Callable[[LogAlert], None]) -> None:
        """添加告警回调
        
        Args:
            callback: 回调函数
        """
        self.alert_callbacks.append(callback)
    
    def get_stats(self) -> LogStats:
        """获取统计信息
        
        Returns:
            日志统计信息
        """
        return self.stats
    
    def clear_logs(self) -> None:
        """清空日志"""
        self.log_entries.clear()
        self.log_alerts.clear()
        self.stats = LogStats()
        self.logger.info("日志数据已清空")
    
    def export_logs(self, file_path: str, format: str = 'json') -> None:
        """导出日志
        
        Args:
            file_path: 导出文件路径
            format: 导出格式（json或csv）
        """
        try:
            if format.lower() == 'json':
                data = {
                    'logs': [log.to_dict() for log in self.log_entries],
                    'patterns': [pattern.to_dict() for pattern in self.log_patterns.values()],
                    'alerts': [alert.to_dict() for alert in self.log_alerts.values()],
                    'stats': self.stats.to_dict()
                }
                
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
            
            elif format.lower() == 'csv':
                import csv
                
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['timestamp', 'level', 'source', 'message'])
                    
                    for log in self.log_entries:
                        writer.writerow([
                            datetime.fromtimestamp(log.timestamp).isoformat(),
                            log.level.value,
                            log.source,
                            log.message
                        ])
            
            self.logger.info(f"日志已导出到: {file_path}")
            
        except Exception as e:
            self.logger.error(f"导出日志失败: {e}")