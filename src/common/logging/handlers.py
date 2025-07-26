"""Logging Handlers

自定义日志处理器，支持数据库、Elasticsearch、Slack、邮件等多种输出方式。
"""

import logging
import json
import smtplib
import threading
from datetime import datetime
from typing import Dict, Any, Optional, List
from pathlib import Path

try:
    from email.mime.text import MimeText
    from email.mime.multipart import MimeMultipart
except ImportError:
    MimeText = None
    MimeMultipart = None

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor
except ImportError:
    psycopg2 = None

try:
    import elasticsearch
except ImportError:
    elasticsearch = None

try:
    import requests
except ImportError:
    requests = None


class DatabaseHandler(logging.Handler):
    """数据库日志处理器"""
    
    def __init__(self, connection_string: str, table_name: str = 'logs'):
        super().__init__()
        self.connection_string = connection_string
        self.table_name = table_name
        self._connection = None
        self._lock = threading.Lock()
        
        # 初始化数据库连接
        self._init_database()
    
    def _init_database(self):
        """初始化数据库连接和表结构"""
        if not psycopg2:
            raise ImportError("psycopg2 is required for DatabaseHandler")
        
        try:
            self._connection = psycopg2.connect(self.connection_string)
            self._create_table_if_not_exists()
        except Exception as e:
            self.handleError(None)
            raise e
    
    def _create_table_if_not_exists(self):
        """创建日志表（如果不存在）"""
        create_table_sql = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            logger_name VARCHAR(255),
            level VARCHAR(20),
            message TEXT,
            module VARCHAR(255),
            function_name VARCHAR(255),
            line_number INTEGER,
            process_id INTEGER,
            thread_id BIGINT,
            extra_data JSONB,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON {self.table_name} (timestamp);
        CREATE INDEX IF NOT EXISTS idx_logs_level ON {self.table_name} (level);
        CREATE INDEX IF NOT EXISTS idx_logs_logger_name ON {self.table_name} (logger_name);
        """
        
        with self._connection.cursor() as cursor:
            cursor.execute(create_table_sql)
            self._connection.commit()
    
    def emit(self, record: logging.LogRecord):
        """发送日志记录到数据库"""
        if not self._connection:
            return
        
        try:
            with self._lock:
                # 准备日志数据
                log_data = {
                    'timestamp': datetime.fromtimestamp(record.created),
                    'logger_name': record.name,
                    'level': record.levelname,
                    'message': record.getMessage(),
                    'module': record.module,
                    'function_name': record.funcName,
                    'line_number': record.lineno,
                    'process_id': record.process,
                    'thread_id': record.thread,
                    'extra_data': json.dumps(getattr(record, 'extra_data', {}))
                }
                
                # 插入数据库
                insert_sql = f"""
                INSERT INTO {self.table_name} 
                (timestamp, logger_name, level, message, module, function_name, 
                 line_number, process_id, thread_id, extra_data)
                VALUES (%(timestamp)s, %(logger_name)s, %(level)s, %(message)s, 
                        %(module)s, %(function_name)s, %(line_number)s, 
                        %(process_id)s, %(thread_id)s, %(extra_data)s)
                """
                
                with self._connection.cursor() as cursor:
                    cursor.execute(insert_sql, log_data)
                    self._connection.commit()
                    
        except Exception:
            self.handleError(record)
    
    def close(self):
        """关闭数据库连接"""
        if self._connection:
            self._connection.close()
        super().close()


class ElasticsearchHandler(logging.Handler):
    """Elasticsearch日志处理器"""
    
    def __init__(self, hosts: List[str], index_name: str = 'prometheus-logs'):
        super().__init__()
        self.hosts = hosts
        self.index_name = index_name
        self._client = None
        
        # 初始化Elasticsearch客户端
        self._init_elasticsearch()
    
    def _init_elasticsearch(self):
        """初始化Elasticsearch客户端"""
        if not elasticsearch:
            raise ImportError("elasticsearch is required for ElasticsearchHandler")
        
        try:
            self._client = elasticsearch.Elasticsearch(self.hosts)
            # 测试连接
            self._client.ping()
        except Exception as e:
            self.handleError(None)
            raise e
    
    def emit(self, record: logging.LogRecord):
        """发送日志记录到Elasticsearch"""
        if not self._client:
            return
        
        try:
            # 准备日志文档
            doc = {
                '@timestamp': datetime.fromtimestamp(record.created).isoformat(),
                'logger_name': record.name,
                'level': record.levelname,
                'message': record.getMessage(),
                'module': record.module,
                'function_name': record.funcName,
                'line_number': record.lineno,
                'process_id': record.process,
                'thread_id': record.thread,
                'pathname': record.pathname,
                'filename': record.filename
            }
            
            # 添加额外数据
            if hasattr(record, 'extra_data'):
                doc.update(record.extra_data)
            
            # 索引文档
            index_name = f"{self.index_name}-{datetime.now().strftime('%Y.%m.%d')}"
            self._client.index(
                index=index_name,
                body=doc
            )
            
        except Exception:
            self.handleError(record)


class SlackHandler(logging.Handler):
    """Slack日志处理器"""
    
    def __init__(self, webhook_url: str, channel: str = '#alerts', 
                 username: str = 'Prometheus', min_level: int = logging.ERROR):
        super().__init__()
        self.webhook_url = webhook_url
        self.channel = channel
        self.username = username
        self.min_level = min_level
        
        if not requests:
            raise ImportError("requests is required for SlackHandler")
    
    def emit(self, record: logging.LogRecord):
        """发送日志记录到Slack"""
        if record.levelno < self.min_level:
            return
        
        try:
            # 准备Slack消息
            color = self._get_color_for_level(record.levelname)
            
            payload = {
                'channel': self.channel,
                'username': self.username,
                'attachments': [{
                    'color': color,
                    'title': f'{record.levelname}: {record.name}',
                    'text': record.getMessage(),
                    'fields': [
                        {
                            'title': 'Module',
                            'value': record.module,
                            'short': True
                        },
                        {
                            'title': 'Function',
                            'value': record.funcName,
                            'short': True
                        },
                        {
                            'title': 'Line',
                            'value': str(record.lineno),
                            'short': True
                        },
                        {
                            'title': 'Time',
                            'value': datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S'),
                            'short': True
                        }
                    ],
                    'ts': record.created
                }]
            }
            
            # 发送到Slack
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            response.raise_for_status()
            
        except Exception:
            self.handleError(record)
    
    def _get_color_for_level(self, level: str) -> str:
        """根据日志级别获取颜色"""
        colors = {
            'DEBUG': '#36a64f',     # 绿色
            'INFO': '#36a64f',      # 绿色
            'WARNING': '#ff9900',   # 橙色
            'ERROR': '#ff0000',     # 红色
            'CRITICAL': '#990000'   # 深红色
        }
        return colors.get(level, '#36a64f')


class EmailHandler(logging.Handler):
    """邮件日志处理器"""
    
    def __init__(self, smtp_host: str, smtp_port: int, username: str, password: str,
                 from_addr: str, to_addrs: List[str], subject: str = 'Prometheus Alert',
                 min_level: int = logging.ERROR, use_tls: bool = True):
        super().__init__()
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addrs = to_addrs
        self.subject = subject
        self.min_level = min_level
        self.use_tls = use_tls
    
    def emit(self, record: logging.LogRecord):
        """发送日志记录到邮箱"""
        if record.levelno < self.min_level:
            return
        
        try:
            # 创建邮件消息
            msg = MimeMultipart()
            msg['From'] = self.from_addr
            msg['To'] = ', '.join(self.to_addrs)
            msg['Subject'] = f"{self.subject} - {record.levelname}"
            
            # 邮件正文
            body = self._format_email_body(record)
            msg.attach(MimeText(body, 'html'))
            
            # 发送邮件
            with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
                if self.use_tls:
                    server.starttls()
                server.login(self.username, self.password)
                server.send_message(msg)
                
        except Exception:
            self.handleError(record)
    
    def _format_email_body(self, record: logging.LogRecord) -> str:
        """格式化邮件正文"""
        timestamp = datetime.fromtimestamp(record.created).strftime('%Y-%m-%d %H:%M:%S')
        
        html_body = f"""
        <html>
        <body>
            <h2>Prometheus Alert - {record.levelname}</h2>
            <table border="1" cellpadding="5" cellspacing="0">
                <tr><td><b>Time:</b></td><td>{timestamp}</td></tr>
                <tr><td><b>Logger:</b></td><td>{record.name}</td></tr>
                <tr><td><b>Level:</b></td><td>{record.levelname}</td></tr>
                <tr><td><b>Module:</b></td><td>{record.module}</td></tr>
                <tr><td><b>Function:</b></td><td>{record.funcName}</td></tr>
                <tr><td><b>Line:</b></td><td>{record.lineno}</td></tr>
                <tr><td><b>Message:</b></td><td>{record.getMessage()}</td></tr>
            </table>
            
            {self._format_exception_info(record)}
        </body>
        </html>
        """
        
        return html_body
    
    def _format_exception_info(self, record: logging.LogRecord) -> str:
        """格式化异常信息"""
        if record.exc_info:
            import traceback
            exc_text = ''.join(traceback.format_exception(*record.exc_info))
            return f"<h3>Exception Info:</h3><pre>{exc_text}</pre>"
        return ""


class FileRotatingHandler(logging.handlers.RotatingFileHandler):
    """增强的文件轮转处理器"""
    
    def __init__(self, filename: str, mode: str = 'a', maxBytes: int = 0,
                 backupCount: int = 0, encoding: Optional[str] = None,
                 delay: bool = False, compress: bool = False):
        super().__init__(filename, mode, maxBytes, backupCount, encoding, delay)
        self.compress = compress
    
    def doRollover(self):
        """执行日志轮转"""
        super().doRollover()
        
        # 如果启用压缩，压缩旧的日志文件
        if self.compress and self.backupCount > 0:
            self._compress_old_logs()
    
    def _compress_old_logs(self):
        """压缩旧的日志文件"""
        import gzip
        import shutil
        
        for i in range(1, self.backupCount + 1):
            log_file = f"{self.baseFilename}.{i}"
            if Path(log_file).exists():
                gz_file = f"{log_file}.gz"
                
                # 如果压缩文件不存在，则压缩
                if not Path(gz_file).exists():
                    with open(log_file, 'rb') as f_in:
                        with gzip.open(gz_file, 'wb') as f_out:
                            shutil.copyfileobj(f_in, f_out)
                    
                    # 删除原始文件
                    Path(log_file).unlink()


class AsyncHandler(logging.Handler):
    """异步日志处理器"""
    
    def __init__(self, target_handler: logging.Handler, queue_size: int = 1000):
        super().__init__()
        self.target_handler = target_handler
        self.queue_size = queue_size
        self._queue = None
        self._thread = None
        self._stop_event = None
        
        # 启动异步处理
        self._start_async_processing()
    
    def _start_async_processing(self):
        """启动异步处理线程"""
        import queue
        
        self._queue = queue.Queue(maxsize=self.queue_size)
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._process_logs, daemon=True)
        self._thread.start()
    
    def emit(self, record: logging.LogRecord):
        """异步发送日志记录"""
        try:
            self._queue.put_nowait(record)
        except:
            # 队列满时丢弃日志
            pass
    
    def _process_logs(self):
        """处理日志队列"""
        while not self._stop_event.is_set():
            try:
                record = self._queue.get(timeout=1)
                self.target_handler.emit(record)
                self._queue.task_done()
            except:
                continue
    
    def close(self):
        """关闭异步处理器"""
        if self._stop_event:
            self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self.target_handler:
            self.target_handler.close()
        super().close()