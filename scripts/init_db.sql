-- Prometheus Trading System Database Initialization
-- 创建数据库表结构

-- 策略配置表
CREATE TABLE IF NOT EXISTS strategies (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    description TEXT,
    strategy_type VARCHAR(50) NOT NULL,
    parameters JSONB NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 交易对配置表
CREATE TABLE IF NOT EXISTS trading_pairs (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    base_asset VARCHAR(10) NOT NULL,
    quote_asset VARCHAR(10) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    min_order_size DECIMAL(20, 8),
    max_order_size DECIMAL(20, 8),
    price_precision INTEGER,
    quantity_precision INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 订单记录表
CREATE TABLE IF NOT EXISTS orders (
    id SERIAL PRIMARY KEY,
    order_id VARCHAR(50) NOT NULL UNIQUE,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- BUY, SELL
    order_type VARCHAR(20) NOT NULL, -- MARKET, LIMIT, STOP_LOSS, etc.
    quantity DECIMAL(20, 8) NOT NULL,
    price DECIMAL(20, 8),
    filled_quantity DECIMAL(20, 8) DEFAULT 0,
    filled_price DECIMAL(20, 8),
    status VARCHAR(20) NOT NULL, -- NEW, PARTIALLY_FILLED, FILLED, CANCELED, REJECTED
    strategy_id INTEGER REFERENCES strategies(id),
    commission DECIMAL(20, 8),
    commission_asset VARCHAR(10),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 持仓记录表
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL, -- LONG, SHORT
    quantity DECIMAL(20, 8) NOT NULL,
    entry_price DECIMAL(20, 8) NOT NULL,
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    realized_pnl DECIMAL(20, 8) DEFAULT 0,
    strategy_id INTEGER REFERENCES strategies(id),
    opened_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    closed_at TIMESTAMP WITH TIME ZONE,
    is_open BOOLEAN DEFAULT true
);

-- 风险限制表
CREATE TABLE IF NOT EXISTS risk_limits (
    id SERIAL PRIMARY KEY,
    limit_type VARCHAR(50) NOT NULL, -- POSITION_SIZE, DAILY_LOSS, DRAWDOWN, etc.
    symbol VARCHAR(20), -- NULL for global limits
    strategy_id INTEGER REFERENCES strategies(id), -- NULL for global limits
    limit_value DECIMAL(20, 8) NOT NULL,
    current_value DECIMAL(20, 8) DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 系统日志表
CREATE TABLE IF NOT EXISTS system_logs (
    id SERIAL PRIMARY KEY,
    level VARCHAR(10) NOT NULL, -- DEBUG, INFO, WARNING, ERROR, CRITICAL
    module VARCHAR(50) NOT NULL,
    message TEXT NOT NULL,
    details JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 交易信号表
CREATE TABLE IF NOT EXISTS trade_signals (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    symbol VARCHAR(20) NOT NULL,
    signal_type VARCHAR(20) NOT NULL, -- BUY, SELL, HOLD
    strength DECIMAL(5, 4), -- 信号强度 0-1
    price DECIMAL(20, 8),
    quantity DECIMAL(20, 8),
    confidence DECIMAL(5, 4), -- 置信度 0-1
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 账户余额表
CREATE TABLE IF NOT EXISTS account_balances (
    id SERIAL PRIMARY KEY,
    asset VARCHAR(10) NOT NULL,
    free_balance DECIMAL(20, 8) NOT NULL,
    locked_balance DECIMAL(20, 8) NOT NULL,
    total_balance DECIMAL(20, 8) GENERATED ALWAYS AS (free_balance + locked_balance) STORED,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 性能指标表
CREATE TABLE IF NOT EXISTS performance_metrics (
    id SERIAL PRIMARY KEY,
    strategy_id INTEGER REFERENCES strategies(id),
    metric_type VARCHAR(50) NOT NULL, -- SHARPE_RATIO, MAX_DRAWDOWN, WIN_RATE, etc.
    metric_value DECIMAL(20, 8) NOT NULL,
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders(symbol);
CREATE INDEX IF NOT EXISTS idx_orders_strategy_id ON orders(strategy_id);
CREATE INDEX IF NOT EXISTS idx_orders_created_at ON orders(created_at);
CREATE INDEX IF NOT EXISTS idx_positions_symbol ON positions(symbol);
CREATE INDEX IF NOT EXISTS idx_positions_strategy_id ON positions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_positions_is_open ON positions(is_open);
CREATE INDEX IF NOT EXISTS idx_trade_signals_strategy_id ON trade_signals(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trade_signals_symbol ON trade_signals(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_signals_created_at ON trade_signals(created_at);
CREATE INDEX IF NOT EXISTS idx_system_logs_level ON system_logs(level);
CREATE INDEX IF NOT EXISTS idx_system_logs_module ON system_logs(module);
CREATE INDEX IF NOT EXISTS idx_system_logs_created_at ON system_logs(created_at);

-- 插入初始数据
INSERT INTO trading_pairs (symbol, base_asset, quote_asset, min_order_size, max_order_size, price_precision, quantity_precision) VALUES
('BTCUSDT', 'BTC', 'USDT', 10.0, 10000.0, 2, 6),
('ETHUSDT', 'ETH', 'USDT', 10.0, 10000.0, 2, 5),
('BNBUSDT', 'BNB', 'USDT', 10.0, 10000.0, 2, 3),
('ADAUSDT', 'ADA', 'USDT', 10.0, 10000.0, 4, 1),
('DOTUSDT', 'DOT', 'USDT', 10.0, 10000.0, 3, 2)
ON CONFLICT (symbol) DO NOTHING;

-- 插入默认风险限制
INSERT INTO risk_limits (limit_type, limit_value) VALUES
('MAX_TOTAL_POSITION', 0.8),
('MAX_SINGLE_POSITION', 0.2),
('MAX_DAILY_LOSS', 0.05),
('MAX_DRAWDOWN', 0.15)
ON CONFLICT DO NOTHING;

-- 创建更新时间触发器函数
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- 为需要的表创建更新时间触发器
CREATE TRIGGER update_strategies_updated_at BEFORE UPDATE ON strategies
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_risk_limits_updated_at BEFORE UPDATE ON risk_limits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_account_balances_updated_at BEFORE UPDATE ON account_balances
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();