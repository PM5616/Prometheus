项目计划书 (Project Plan)

1. 项目概述:
本计划书详细阐述了“普罗米修斯计划”的实施路径，包括技术架构、开发阶段、资源需求和风险管理细则。
2. 技术架构 (微服务架构):
我们将采用模块化、高内聚、低耦合的微服务架构，确保系统的可扩展性和鲁棒性。
* Prometheus-Gateway (网关服务):
    * 技术: Python, binance-connector-python, WebSocket, FastAPI
    * 职责: 所有与币安交易所的交互入口。管理WebSocket连接，分发实时市场数据；管理REST API请求，执行交易指令。
* Prometheus-DataHub (数据中心):
    * 技术: Time-series Database (如 InfluxDB, TimescaleDB), Redis, Python
    * 职责: 订阅Gateway的数据流，对原始数据进行清洗、转换（如生成K线），并持久化到时序数据库中。提供历史数据查询服务。Redis用于缓存热数据。
* Prometheus-AlphaEngine (策略引擎):
    * 技术: Python, NumPy, Pandas, SciPy, Statsmodels, Scikit-learn
    * 职责: 系统的“大脑”。以插件化形式加载多个策略模型。每个模型独立运行，接收DataHub的数据，产生交易信号（Signal）。
* Prometheus-PortfolioManager (投资组合管理服务):
    * 技术: Python, a simple rules engine
    * 职责: 接收来自AlphaEngine的多个原始信号。根据全局风险参数（如Kelly Criterion凯利公式）、市场状态和策略表现，对信号进行加权、过滤和整合，生成最终的投资组合调整指令（Target Portfolio）。
* Prometheus-Executioner (执行服务):
    * 技术: Python, binance-connector-python
    * 职责: 接收PortfolioManager的指令，计算需要执行的具体订单（买/卖、数量），并通过Gateway发送至交易所。处理订单成交回报。
* Prometheus-RiskSentinel (风控哨兵):
    * 技术: Python, Redis
    * 职责: 独立于交易流程，实时监控全局仓位、 PnL（盈亏）、保证金率。拥有最高权限，在触及风控阈值时，可直接绕过策略层，强制平仓或禁止开仓。
* Prometheus-Monitor (监控面板):
    * 技术: Grafana, Prometheus (monitoring tool)
    * 职责: 可视化所有服务的运行状态、系统日志、资金曲线、关键绩效指标（KPIs）。
3. 数学与策略核心:
* 震荡套利 (Mean Reversion):
    * 模型1: 协整配对交易 (Cointegration):
        1. 找出市场上具有长期稳定协整关系的交易对 (e.g., BTC/ETH)。
        2. 使用 Engle-Granger 两步法或 Johansen 测试检验协整关系。
        3. 对价差序列（Spread）应用 Ornstein-Uhlenbeck 过程进行建模：dXt =θ(μ−Xt )dt+σdWt 。
        4. 当价差偏离其均值 μ 超过一定阈值（如2倍标准差）时，做空高估资产，做多低估资产。
* 趋势获利 (Trend Following):
    * 模型2: 卡尔曼滤波趋势系统 (Kalman Filter Trend System):
        1. 将价格序列视为一个被噪声污染的动态系统。状态方程描述潜在趋势的运动，观测方程描述我们看到的实际价格。
        2. 应用卡尔曼滤波器，从带噪声的价格中估计出更平滑、更真实的潜在价值和趋势方向（状态）。
        3. 当估计出的趋势斜率（一阶导数）和加速度（二阶导数）满足特定条件时（例如，斜率由负转正且持续增长），产生趋势开始的信号。
* 市场状态识别 (Market Regime Detection):
    * 模型3: 隐马尔可夫模型 (HMM):
        1. 假设市场存在两个不可观测的状态：“震荡”和“趋势”。
        2. 使用收益率序列或波动率序列作为观测值，训练HMM模型，估计出状态转移概率矩阵和每个状态下的观测概率分布。
        3. 系统运行时，实时计算当前处于各个状态的概率，PortfolioManager根据此概率动态调整“震荡套利”和“趋势获利”策略的资金权重。
4. 项目开发计划 (分阶段进行):
* 第一阶段：基础建设与数据回测 (第1 - 4周)
    * W1-W2: 环境搭建（Git, Docker, Python环境），完成Gateway和DataHub的核心开发，实现稳定可靠的数据流入与存储。
    * W3-W4: 开发一个强大的离线回测框架。使用历史数据，实现上述三个核心数学模型的原型，并进行初步回测验证。
* 第二阶段：核心服务开发与集成 (第5 - 12周)
    * W5-W7: 开发AlphaEngine，实现策略的插件化加载。将回测验证后的模型作为首批策略插件集成。
    * W8-W9: 开发PortfolioManager和Executioner，打通从信号产生到订单执行的完整链路。
    * W10: 开发RiskSentinel和Monitor，建立生命保障和眼睛。
    * W11-W12: 系统端到端集成测试，进行模拟盘（Paper Trading）交易，修复Bug，优化性能。
* 第三阶段：小规模实盘测试 (第13 - 16周)
    * W13-W14: 部署至生产服务器（建议使用云服务器如AWS/Google Cloud）。投入小额资金（如总资本的1%）进行实盘测试。
    * W15-W16: 密切监控系统表现，与模拟盘结果进行对比，分析滑点、延迟等现实世界摩擦的影响。
* 第四阶段：扩大规模与持续迭代 (第17周以后)
    * 根据实盘测试结果，逐步增加投入资本。
    * 启动第二轮策略研发，不断为AlphaEngine输送新的“弹药”。
    * 对现有模型进行滚动优化和参数自适应升级。
5. 技术栈与依赖管理:
* 核心技术栈:
    * Python 3.13+ (主要开发语言)
    * binance-connector-python (币安API连接器，严禁修改源码)
    * FastAPI (Web框架，用于API服务)
    * WebSocket (实时数据传输)
    * Redis (缓存和消息队列)
    * InfluxDB/TimescaleDB (时序数据库)
    * PostgreSQL (关系型数据库)
    * Docker & Docker Compose (容器化部署)
    * Grafana (监控面板)
    * Prometheus (系统监控)

* 数据科学与量化库:
    * NumPy (数值计算)
    * Pandas (数据处理)
    * SciPy (科学计算)
    * Statsmodels (统计建模)
    * Scikit-learn (机器学习)
    * TA-Lib (技术分析指标)
    * PyKalman (卡尔曼滤波)
    * hmmlearn (隐马尔可夫模型)
    * arch (GARCH模型)

* 开发与测试工具:
    * pytest (单元测试)
    * black (代码格式化)
    * mypy (类型检查)
    * flake8 (代码质量检查)
    * pre-commit (Git钩子)
    * Jupyter Notebook (策略研发)

6. 项目目录结构:
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
│   │   ├── config/                    # 配置管理
│   │   ├── utils/                     # 工具函数
│   │   ├── models/                    # 数据模型
│   │   └── exceptions/                # 异常定义
│   └── strategies/                    # 策略插件目录
├── tests/                             # 测试代码
├── docs/                              # 文档
├── configs/                           # 配置文件
├── scripts/                           # 脚本文件
├── docker/                            # Docker配置
├── requirements.txt                   # Python依赖
├── pyproject.toml                     # 项目配置
├── docker-compose.yml                 # 容器编排
└── README.md                          # 项目说明
```

7. 数据库设计:
* 时序数据库 (InfluxDB):
    * market_data: 实时行情数据
    * kline_data: K线数据
    * trade_signals: 交易信号
    * portfolio_metrics: 投资组合指标
    * system_metrics: 系统性能指标

* 关系型数据库 (PostgreSQL):
    * strategies: 策略配置表
    * orders: 订单记录表
    * positions: 持仓记录表
    * risk_limits: 风控限制表
    * system_logs: 系统日志表

8. 配置管理:
* 环境变量配置:
    * BINANCE_API_KEY: 币安API密钥
    * BINANCE_SECRET_KEY: 币安密钥
    * DATABASE_URL: 数据库连接字符串
    * REDIS_URL: Redis连接字符串
    * LOG_LEVEL: 日志级别

* 策略配置文件 (YAML格式):
    * 策略参数
    * 风险限制
    * 交易对配置
    * 回测参数

9. 资源与预算 (初步):
* 人力: 您（AI开发）、我（架构与指导）
* 软件: Python及相关开源库 (免费), Docker (免费), InfluxDB (开源版免费), Grafana (开源版免费)
* 硬件/云服务: 一台性能稳定的云服务器（如AWS EC2 t3.medium或更高配置），预计每月成本 $50 - $150。
* 交易资本: 根据您的风险偏好自行决定。

10. 质量保证与测试策略:
* 单元测试: 每个模块必须有对应的单元测试，覆盖率不低于80%
* 集成测试: 测试各服务间的交互
* 回测测试: 使用历史数据验证策略有效性
* 压力测试: 测试系统在高负载下的表现
* 模拟交易: 在真实市场环境下进行无资金风险的测试

11. 部署与运维:
* 容器化部署: 使用Docker容器化所有服务
* 服务编排: 使用Docker Compose管理多服务部署
* 负载均衡: 使用Nginx进行负载均衡
* 自动化部署: 使用CI/CD流水线自动化部署
* 监控告警: 24/7监控系统状态，异常时自动告警
* 数据备份: 定期备份关键数据，确保数据安全