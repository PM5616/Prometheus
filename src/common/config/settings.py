"""System Settings Configuration

系统设置配置类，使用Pydantic进行配置验证和管理。
支持从环境变量自动加载配置。
"""

from typing import List, Optional
from pydantic import BaseSettings, Field, validator
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    """数据库配置"""
    
    # PostgreSQL配置
    database_url: str = Field(..., env="DATABASE_URL")
    pool_size: int = Field(10, env="DB_POOL_SIZE")
    max_overflow: int = Field(20, env="DB_MAX_OVERFLOW")
    pool_timeout: int = Field(30, env="DB_POOL_TIMEOUT")
    
    # Redis配置
    redis_url: str = Field(..., env="REDIS_URL")
    redis_max_connections: int = Field(100, env="REDIS_MAX_CONNECTIONS")
    
    # InfluxDB配置
    influxdb_url: str = Field(..., env="INFLUXDB_URL")
    influxdb_token: str = Field(..., env="INFLUXDB_TOKEN")
    influxdb_org: str = Field(..., env="INFLUXDB_ORG")
    influxdb_bucket: str = Field(..., env="INFLUXDB_BUCKET")


class BinanceSettings(BaseSettings):
    """币安API配置"""
    
    api_key: str = Field(..., env="BINANCE_API_KEY")
    secret_key: str = Field(..., env="BINANCE_SECRET_KEY")
    testnet: bool = Field(True, env="BINANCE_TESTNET")
    
    # API端点
    base_url: str = "https://api.binance.com"
    testnet_url: str = "https://testnet.binance.vision"
    websocket_url: str = "wss://stream.binance.com:9443"
    testnet_websocket_url: str = "wss://testnet.binance.vision"
    
    # 连接配置
    timeout: int = 30
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    @property
    def api_base_url(self) -> str:
        """获取API基础URL"""
        return self.testnet_url if self.testnet else self.base_url
    
    @property
    def ws_base_url(self) -> str:
        """获取WebSocket基础URL"""
        return self.testnet_websocket_url if self.testnet else self.websocket_url


class TradingSettings(BaseSettings):
    """交易配置"""
    
    # 支持的交易对
    symbols: List[str] = Field(
        default=["BTCUSDT", "ETHUSDT", "BNBUSDT", "ADAUSDT", "DOTUSDT"],
        env="TRADING_SYMBOLS"
    )
    
    # 交易参数
    min_order_size: float = Field(10.0, env="MIN_ORDER_SIZE")
    max_order_size: float = Field(1000.0, env="MAX_ORDER_SIZE")
    price_precision: int = Field(8, env="PRICE_PRECISION")
    quantity_precision: int = Field(8, env="QUANTITY_PRECISION")
    
    # 滑点控制
    max_slippage: float = Field(0.001, env="MAX_SLIPPAGE")
    
    @validator('symbols', pre=True)
    def parse_symbols(cls, v):
        """解析交易对列表"""
        if isinstance(v, str):
            return [s.strip() for s in v.split(',')]
        return v


class RiskSettings(BaseSettings):
    """风险管理配置"""
    
    # 全局风险限制
    max_total_position: float = Field(0.8, env="MAX_TOTAL_POSITION")
    max_single_position: float = Field(0.2, env="MAX_SINGLE_POSITION")
    max_daily_loss: float = Field(0.05, env="MAX_DAILY_LOSS")
    max_drawdown: float = Field(0.15, env="MAX_DRAWDOWN")
    
    # 止损设置
    stop_loss_threshold: float = Field(0.02, env="STOP_LOSS_THRESHOLD")
    emergency_stop_threshold: float = Field(0.05, env="EMERGENCY_STOP_THRESHOLD")
    
    # 风险检查频率（秒）
    check_interval: int = Field(5, env="RISK_CHECK_INTERVAL")


class MonitoringSettings(BaseSettings):
    """监控配置"""
    
    # Prometheus监控
    prometheus_port: int = Field(8000, env="PROMETHEUS_PORT")
    metrics_interval: int = Field(10, env="METRICS_INTERVAL")
    
    # Grafana配置
    grafana_url: str = Field("http://localhost:3000", env="GRAFANA_URL")
    
    # 告警配置
    alert_email_enabled: bool = Field(False, env="ALERT_EMAIL_ENABLED")
    alert_webhook_enabled: bool = Field(False, env="ALERT_WEBHOOK_ENABLED")
    alert_webhook_url: Optional[str] = Field(None, env="ALERT_WEBHOOK_URL")


class LoggingSettings(BaseSettings):
    """日志配置"""
    
    level: str = Field("INFO", env="LOG_LEVEL")
    format: str = Field(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        env="LOG_FORMAT"
    )
    max_file_size: str = Field("10MB", env="LOG_MAX_FILE_SIZE")
    backup_count: int = Field(5, env="LOG_BACKUP_COUNT")
    
    @validator('level')
    def validate_log_level(cls, v):
        """验证日志级别"""
        valid_levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
        if v.upper() not in valid_levels:
            raise ValueError(f'Invalid log level: {v}. Must be one of {valid_levels}')
        return v.upper()


class SecuritySettings(BaseSettings):
    """安全配置"""
    
    secret_key: str = Field(..., env="SECRET_KEY")
    jwt_secret: str = Field(..., env="JWT_SECRET")
    encryption_key: str = Field(..., env="ENCRYPTION_KEY")
    
    # JWT配置
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30


class Settings(BaseSettings):
    """主配置类"""
    
    # 系统基础配置
    app_name: str = "Prometheus Trading System"
    app_version: str = "0.1.0"
    environment: str = Field("development", env="ENVIRONMENT")
    debug: bool = Field(True, env="DEBUG")
    timezone: str = Field("UTC", env="TIMEZONE")
    
    # 子配置
    database: DatabaseSettings = DatabaseSettings()
    binance: BinanceSettings = BinanceSettings()
    trading: TradingSettings = TradingSettings()
    risk: RiskSettings = RiskSettings()
    monitoring: MonitoringSettings = MonitoringSettings()
    logging: LoggingSettings = LoggingSettings()
    security: SecuritySettings = SecuritySettings()
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """获取系统设置单例
    
    Returns:
        Settings: 系统设置实例
    """
    return Settings()