# Prometheus Trading System

## 项目概述

Prometheus是一个AI驱动的多策略量化交易系统，专为数字货币市场设计。系统采用微服务架构，基于严格的数学建模和概率优势，能够在不同市场环境中实现风险可控的稳定收益。

## 核心特性

- 🚀 **微服务架构**: 高内聚、低耦合的模块化设计
- 📊 **多策略支持**: 震荡套利、趋势跟踪、市场状态识别
- 🛡️ **严格风控**: 多层次风险管理框架
- 📈 **实时监控**: 全方位系统状态和交易监控
- 🔧 **插件化策略**: 支持策略的快速开发和部署
- 🐳 **容器化部署**: 基于Docker的一键部署

## 系统架构

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gateway       │    │   DataHub       │    │  AlphaEngine    │
│   网关服务      │◄──►│   数据中心      │◄──►│   策略引擎      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Executioner    │    │PortfolioManager │    │ RiskSentinel    │
│   执行服务      │◄──►│ 投资组合管理    │◄──►│   风控哨兵      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────┐
                    │    Monitor      │
                    │    监控面板     │
                    └─────────────────┘
```

## 技术栈

### 核心技术
- **Python 3.13+**: 主要开发语言
- **FastAPI**: Web框架
- **WebSocket**: 实时数据传输
- **Redis**: 缓存和消息队列
- **PostgreSQL**: 关系型数据库
- **InfluxDB**: 时序数据库
- **Docker**: 容器化部署

### 量化分析
- **NumPy/Pandas**: 数据处理
- **SciPy/Statsmodels**: 统计建模
- **Scikit-learn**: 机器学习
- **TA-Lib**: 技术分析
- **PyKalman**: 卡尔曼滤波
- **hmmlearn**: 隐马尔可夫模型

## 快速开始

### 环境要求
- Python 3.13+
- Docker & Docker Compose
- Git

### 安装步骤

1. **克隆项目**
```bash
git clone <repository-url>
cd Prometheus
```

2. **配置环境变量**
```bash
cp .env.example .env
# 编辑.env文件，填入您的币安API密钥等配置
```

3. **安装Python依赖**
```bash
pip install -r requirements.txt
```

4. **启动基础服务**
```bash
docker-compose up -d
```

5. **初始化数据库**
```bash
python scripts/init_database.py
```

6. **运行系统**
```bash
python -m src.main
```

## 项目结构

```
Prometheus/
├── binance-connector-python/          # 币安API库（不可修改）
├── src/                               # 源代码目录
│   ├── gateway/                       # 网关服务
│   ├── datahub/                       # 数据中心
│   ├── alpha_engine/                  # 策略引擎
│   ├── portfolio_manager/             # 投资组合管理
│   ├── executioner/                   # 执行服务
│   ├── risk_sentinel/                 # 风控哨兵
│   ├── monitor/                       # 监控服务
│   ├── common/                        # 公共模块
│   └── strategies/                    # 策略插件目录
├── tests/                             # 测试代码
├── docs/                              # 文档
├── configs/                           # 配置文件
├── scripts/                           # 脚本文件
└── docker/                            # Docker配置
```

## 核心策略

### 1. 震荡套利 (Mean Reversion)
- **协整配对交易**: 基于统计套利的配对交易策略
- **布林带回归**: 基于价格回归的震荡策略
- **Ornstein-Uhlenbeck过程**: 数学建模的均值回归

### 2. 趋势跟踪 (Trend Following)
- **卡尔曼滤波**: 基于状态空间模型的趋势识别
- **动态突破**: 自适应的突破策略
- **海龟交易法则**: 经典趋势跟踪策略的改进版

### 3. 市场状态识别 (Market Regime Detection)
- **隐马尔可夫模型**: 识别市场的不同状态
- **动态权重调整**: 根据市场状态调整策略权重

## 风险管理

- **全局仓位控制**: 严格限制总体风险敞口
- **单策略限制**: 每个策略的独立风险控制
- **实时监控**: 24/7监控系统状态和风险指标
- **紧急止损**: 自动化的风险控制机制

## 监控面板

访问 http://localhost:3000 查看Grafana监控面板，包括：
- 系统性能指标
- 交易执行状况
- 策略表现分析
- 风险控制状态

## 开发指南

### 代码规范
- 遵循PEP 8代码风格
- 使用类型注解
- 编写完整的文档字符串
- 单元测试覆盖率不低于80%

### 添加新策略
1. 在`src/strategies/`目录下创建策略文件
2. 继承`BaseStrategy`类
3. 实现必要的方法
4. 编写单元测试
5. 更新配置文件

### 运行测试
```bash
pytest tests/ -v --cov=src
```

### 代码格式化
```bash
black src/ tests/
mypy src/
flake8 src/ tests/
```

## 部署指南

### 生产环境部署
1. 配置生产环境变量
2. 使用Docker Compose部署
3. 配置反向代理（Nginx）
4. 设置监控告警
5. 配置数据备份

### 安全注意事项
- 妥善保管API密钥
- 使用HTTPS/WSS协议
- 定期更新依赖包
- 监控异常访问

## 许可证

MIT License - 详见 [LICENSE](LICENSE) 文件

## 贡献指南

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 联系我们

- 项目主页: [GitHub Repository]
- 问题反馈: [Issues]
- 邮箱: team@prometheus.ai

---

**免责声明**: 本系统仅供学习和研究使用。数字货币交易存在高风险，请谨慎投资，风险自负。