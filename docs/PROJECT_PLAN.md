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
4. 项目开发计划 (重新规划):
* 第一阶段：系统集成与测试 (第1 - 6周)
  - W1-W2: 模块间集成测试，修复接口兼容性问题
  - W3-W4: 端到端测试，验证完整交易流程
  - W5-W6: 性能优化，压力测试，系统调优

* 第二阶段：策略验证与回测 (第7 - 12周)
  - W7-W8: 策略模型验证，使用更多历史数据回测
  - W9-W10: 策略组合优化，风险-收益平衡调整
  - W11-W12: 模拟交易环境搭建，Paper Trading测试

* 第三阶段：小规模实盘部署 (第13 - 18周)
  - W13-W14: 生产环境部署，监控系统完善
  - W15-W16: 小资金实盘测试（总资金1%）
  - W17-W18: 结果分析，系统优化，准备扩大规模

* 第四阶段：规模化运营 (第19周以后)
  - 逐步增加资金投入
  - 新策略研发和部署
  - 系统持续优化和维护
5. 技术栈与依赖管理:
* 核心技术栈:
    * Python 3.13+ (主要开发语言)
    * binance-connector-python (币安API连接器，严禁修改源码)
    * FastAPI 0.110+ (Web框架，用于API服务)
    * WebSocket (实时数据传输)
    * Redis 7.2+ (缓存和消息队列)
    * InfluxDB/TimescaleDB (时序数据库)
    * PostgreSQL 16+ (关系型数据库)
    * Kubernetes (容器编排，替代Docker Compose)
    * Grafana (监控面板)
    * Prometheus (系统监控)

* 新增技术组件:
    * OpenTelemetry (分布式追踪)
    * Jaeger (链路追踪)
    * ELK Stack (日志分析)
    * ArgoCD (GitOps部署)
    * Vault (密钥管理)
    * Terraform (基础设施即代码)
    * Ansible (配置管理)

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
* 单元测试：每个模块必须有>=90%的代码覆盖率
* 集成测试：验证模块间接口的正确性
* 端到端测试：完整交易流程自动化测试
* 回测测试：使用历史数据验证策略的有效性
* 压力测试：模拟高频交易场景下的系统性能
* 安全测试：API密钥管理、数据传输加密等安全性验证
* 混沌工程：故障注入测试，验证系统容错能力
* 性能基准测试：建立性能基线，持续监控性能退化
* 合规性测试：确保符合金融监管要求
* 自动化测试流水线：CI/CD集成，自动触发测试
* 模拟交易: 在真实市场环境下进行无资金风险的测试

11. 部署与运维:
* 容器化部署：使用Docker进行应用打包
* 编排管理：Kubernetes集群管理多服务部署
* CI/CD流水线：GitLab CI/Jenkins + ArgoCD自动化部署
* 监控告警：Grafana + Prometheus + AlertManager全方位监控
* 日志管理：ELK Stack集中化日志收集与分析
* 链路追踪：OpenTelemetry + Jaeger分布式追踪
* 配置管理：Ansible自动化配置管理
* 基础设施即代码：Terraform管理云资源
* 密钥管理：HashiCorp Vault安全密钥存储
* 备份策略：数据库定期备份，配置文件版本控制
* 灾难恢复：多区域部署，自动故障转移
* 蓝绿部署：零停机时间部署策略
* 金丝雀发布：渐进式发布降低风险