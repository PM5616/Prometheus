"""Backtest Reporter Module

回测报告生成模块，负责生成详细的回测报告。

主要功能：
- 报告生成
- 图表绘制
- 数据导出
- 模板管理
- 格式化输出
- 多格式支持
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import base64
from io import BytesIO

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import seaborn as sns
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

try:
    from jinja2 import Template, Environment, FileSystemLoader
    JINJA2_AVAILABLE = True
except ImportError:
    JINJA2_AVAILABLE = False

from ..common.logging import get_logger
from ..common.exceptions.backtest import BacktestReportError
from .metrics import BacktestMetrics, MetricsCalculator


class ReportFormat(Enum):
    """报告格式枚举"""
    HTML = "html"
    PDF = "pdf"
    JSON = "json"
    EXCEL = "excel"
    CSV = "csv"
    MARKDOWN = "markdown"


class ChartType(Enum):
    """图表类型枚举"""
    EQUITY_CURVE = "equity_curve"          # 资产曲线
    DRAWDOWN = "drawdown"                  # 回撤图
    RETURNS_DISTRIBUTION = "returns_dist"   # 收益率分布
    ROLLING_METRICS = "rolling_metrics"    # 滚动指标
    MONTHLY_RETURNS = "monthly_returns"    # 月度收益
    CORRELATION_MATRIX = "correlation"     # 相关性矩阵
    RISK_RETURN_SCATTER = "risk_return"    # 风险收益散点图
    UNDERWATER_CHART = "underwater"        # 水下图


@dataclass
class ChartConfig:
    """图表配置"""
    chart_type: ChartType
    title: str
    width: int = 12
    height: int = 8
    dpi: int = 100
    style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    save_format: str = "png"
    show_grid: bool = True
    show_legend: bool = True
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 配置字典
        """
        return {
            'chart_type': self.chart_type.value,
            'title': self.title,
            'width': self.width,
            'height': self.height,
            'dpi': self.dpi,
            'style': self.style,
            'color_palette': self.color_palette,
            'save_format': self.save_format,
            'show_grid': self.show_grid,
            'show_legend': self.show_legend
        }


@dataclass
class ReportConfig:
    """报告配置"""
    title: str = "回测报告"
    subtitle: str = ""
    author: str = ""
    company: str = ""
    logo_path: Optional[str] = None
    
    # 报告内容配置
    include_summary: bool = True
    include_metrics: bool = True
    include_charts: bool = True
    include_trades: bool = True
    include_positions: bool = True
    include_risk_analysis: bool = True
    
    # 图表配置
    chart_configs: List[ChartConfig] = field(default_factory=list)
    
    # 格式配置
    decimal_places: int = 4
    percentage_places: int = 2
    currency_symbol: str = "¥"
    
    # 输出配置
    output_dir: str = "reports"
    filename_prefix: str = "backtest_report"
    include_timestamp: bool = True
    
    def to_dict(self) -> Dict:
        """转换为字典格式
        
        Returns:
            Dict: 配置字典
        """
        return {
            'title': self.title,
            'subtitle': self.subtitle,
            'author': self.author,
            'company': self.company,
            'logo_path': self.logo_path,
            'include_summary': self.include_summary,
            'include_metrics': self.include_metrics,
            'include_charts': self.include_charts,
            'include_trades': self.include_trades,
            'include_positions': self.include_positions,
            'include_risk_analysis': self.include_risk_analysis,
            'chart_configs': [config.to_dict() for config in self.chart_configs],
            'decimal_places': self.decimal_places,
            'percentage_places': self.percentage_places,
            'currency_symbol': self.currency_symbol,
            'output_dir': self.output_dir,
            'filename_prefix': self.filename_prefix,
            'include_timestamp': self.include_timestamp
        }


@dataclass
class ReportData:
    """报告数据"""
    metrics: BacktestMetrics
    equity_curve: pd.Series
    benchmark_curve: Optional[pd.Series] = None
    trades: Optional[List[Dict]] = None
    positions: Optional[pd.DataFrame] = None
    returns: Optional[pd.Series] = None
    rolling_metrics: Optional[pd.DataFrame] = None
    
    def validate(self) -> bool:
        """验证数据完整性
        
        Returns:
            bool: 数据是否有效
        """
        if self.metrics is None:
            return False
        
        if self.equity_curve is None or self.equity_curve.empty:
            return False
        
        return True


class ChartGenerator:
    """图表生成器
    
    负责生成各种回测图表。
    """
    
    def __init__(self):
        """初始化图表生成器"""
        self.logger = get_logger("ChartGenerator")
        
        if not PLOTTING_AVAILABLE:
            self.logger.warning("matplotlib未安装，图表功能不可用")
        
        # 设置中文字体
        if PLOTTING_AVAILABLE:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
    
    def generate_chart(self, chart_config: ChartConfig, 
                      report_data: ReportData) -> Optional[str]:
        """生成图表
        
        Args:
            chart_config: 图表配置
            report_data: 报告数据
            
        Returns:
            Optional[str]: 图表的base64编码字符串
        """
        if not PLOTTING_AVAILABLE:
            self.logger.warning("matplotlib不可用，跳过图表生成")
            return None
        
        try:
            # 设置图表样式
            if hasattr(plt, 'style') and chart_config.style:
                try:
                    plt.style.use(chart_config.style)
                except:
                    plt.style.use('default')
            
            # 创建图表
            fig, ax = plt.subplots(figsize=(chart_config.width, chart_config.height), 
                                 dpi=chart_config.dpi)
            
            # 根据图表类型生成不同的图表
            if chart_config.chart_type == ChartType.EQUITY_CURVE:
                self._plot_equity_curve(ax, report_data, chart_config)
            elif chart_config.chart_type == ChartType.DRAWDOWN:
                self._plot_drawdown(ax, report_data, chart_config)
            elif chart_config.chart_type == ChartType.RETURNS_DISTRIBUTION:
                self._plot_returns_distribution(ax, report_data, chart_config)
            elif chart_config.chart_type == ChartType.ROLLING_METRICS:
                self._plot_rolling_metrics(ax, report_data, chart_config)
            elif chart_config.chart_type == ChartType.MONTHLY_RETURNS:
                self._plot_monthly_returns(ax, report_data, chart_config)
            elif chart_config.chart_type == ChartType.UNDERWATER_CHART:
                self._plot_underwater_chart(ax, report_data, chart_config)
            else:
                self.logger.warning(f"不支持的图表类型: {chart_config.chart_type}")
                plt.close(fig)
                return None
            
            # 设置标题和网格
            ax.set_title(chart_config.title, fontsize=14, fontweight='bold')
            
            if chart_config.show_grid:
                ax.grid(True, alpha=0.3)
            
            if chart_config.show_legend:
                ax.legend()
            
            plt.tight_layout()
            
            # 转换为base64字符串
            buffer = BytesIO()
            plt.savefig(buffer, format=chart_config.save_format, 
                       bbox_inches='tight', dpi=chart_config.dpi)
            buffer.seek(0)
            
            chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            plt.close(fig)
            buffer.close()
            
            return chart_base64
            
        except Exception as e:
            self.logger.error(f"生成图表失败: {e}")
            if 'fig' in locals():
                plt.close(fig)
            return None
    
    def _plot_equity_curve(self, ax, report_data: ReportData, config: ChartConfig):
        """绘制资产曲线"""
        # 绘制策略曲线
        ax.plot(report_data.equity_curve.index, report_data.equity_curve.values, 
               label='策略', linewidth=2, color='blue')
        
        # 绘制基准曲线
        if report_data.benchmark_curve is not None:
            ax.plot(report_data.benchmark_curve.index, report_data.benchmark_curve.values,
                   label='基准', linewidth=2, color='red', alpha=0.7)
        
        ax.set_xlabel('日期')
        ax.set_ylabel('资产价值')
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_drawdown(self, ax, report_data: ReportData, config: ChartConfig):
        """绘制回撤图"""
        equity_curve = report_data.equity_curve
        
        # 计算回撤
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        # 绘制回撤
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                       alpha=0.3, color='red', label='回撤')
        ax.plot(drawdown.index, drawdown.values, color='red', linewidth=1)
        
        ax.set_xlabel('日期')
        ax.set_ylabel('回撤 (%)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_returns_distribution(self, ax, report_data: ReportData, config: ChartConfig):
        """绘制收益率分布"""
        if report_data.returns is None:
            returns = report_data.equity_curve.pct_change().dropna()
        else:
            returns = report_data.returns
        
        # 绘制直方图
        ax.hist(returns, bins=50, alpha=0.7, density=True, color='skyblue', 
               edgecolor='black', label='收益率分布')
        
        # 绘制正态分布拟合
        mu, sigma = returns.mean(), returns.std()
        x = np.linspace(returns.min(), returns.max(), 100)
        y = ((np.pi * sigma) * np.sqrt(2 * np.pi)) ** -1 * np.exp(-0.5 * (x - mu) ** 2 / sigma ** 2)
        ax.plot(x, y, 'r-', linewidth=2, label=f'正态分布 (μ={mu:.4f}, σ={sigma:.4f})')
        
        ax.set_xlabel('日收益率')
        ax.set_ylabel('密度')
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2%}'))
    
    def _plot_rolling_metrics(self, ax, report_data: ReportData, config: ChartConfig):
        """绘制滚动指标"""
        if report_data.rolling_metrics is None:
            self.logger.warning("没有滚动指标数据")
            return
        
        rolling_metrics = report_data.rolling_metrics
        
        # 绘制滚动夏普比率
        if 'sharpe_ratio' in rolling_metrics.columns:
            ax.plot(rolling_metrics.index, rolling_metrics['sharpe_ratio'], 
                   label='滚动夏普比率', linewidth=2)
        
        ax.set_xlabel('日期')
        ax.set_ylabel('夏普比率')
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_monthly_returns(self, ax, report_data: ReportData, config: ChartConfig):
        """绘制月度收益"""
        if report_data.returns is None:
            returns = report_data.equity_curve.pct_change().dropna()
        else:
            returns = report_data.returns
        
        # 计算月度收益
        monthly_returns = returns.resample('M').apply(lambda x: (1 + x).prod() - 1)
        
        # 绘制柱状图
        colors = ['green' if x > 0 else 'red' for x in monthly_returns]
        ax.bar(monthly_returns.index, monthly_returns.values, color=colors, alpha=0.7)
        
        ax.set_xlabel('月份')
        ax.set_ylabel('月度收益率')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    def _plot_underwater_chart(self, ax, report_data: ReportData, config: ChartConfig):
        """绘制水下图"""
        equity_curve = report_data.equity_curve
        
        # 计算回撤
        peak = equity_curve.expanding().max()
        drawdown = (equity_curve - peak) / peak
        
        # 绘制水下图
        ax.fill_between(drawdown.index, drawdown.values, 0, 
                       where=(drawdown < 0), alpha=0.5, color='red', 
                       interpolate=True, label='水下期间')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        
        ax.set_xlabel('日期')
        ax.set_ylabel('回撤 (%)')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))
        
        # 格式化x轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)


class BacktestReporter:
    """回测报告生成器
    
    负责生成完整的回测报告。
    """
    
    def __init__(self, config: Optional[ReportConfig] = None):
        """初始化报告生成器
        
        Args:
            config: 报告配置
        """
        self.config = config or ReportConfig()
        self.logger = get_logger("BacktestReporter")
        self.chart_generator = ChartGenerator()
        self.metrics_calculator = MetricsCalculator()
        
        # 创建输出目录
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化模板
        self._init_templates()
    
    def _init_templates(self):
        """初始化报告模板"""
        self.html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ config.title }}</title>
    <style>
        body {
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            border-bottom: 2px solid #007acc;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .header h1 {
            color: #007acc;
            margin: 0;
            font-size: 2.5em;
        }
        .header h2 {
            color: #666;
            margin: 10px 0 0 0;
            font-weight: normal;
        }
        .section {
            margin-bottom: 40px;
        }
        .section h3 {
            color: #007acc;
            border-bottom: 1px solid #ddd;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background-color: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #007acc;
        }
        .metric-card h4 {
            margin: 0 0 10px 0;
            color: #333;
            font-size: 0.9em;
            text-transform: uppercase;
        }
        .metric-card .value {
            font-size: 1.8em;
            font-weight: bold;
            color: #007acc;
        }
        .chart {
            text-align: center;
            margin: 30px 0;
        }
        .chart img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        }
        .table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        .table th, .table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        .table th {
            background-color: #007acc;
            color: white;
            font-weight: bold;
        }
        .table tr:hover {
            background-color: #f5f5f5;
        }
        .positive {
            color: #28a745;
        }
        .negative {
            color: #dc3545;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            color: #666;
            font-size: 0.9em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ config.title }}</h1>
            {% if config.subtitle %}
            <h2>{{ config.subtitle }}</h2>
            {% endif %}
            <p>生成时间: {{ generation_time }}</p>
        </div>
        
        {% if config.include_summary %}
        <div class="section">
            <h3>📊 回测概要</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>回测期间</h4>
                    <div class="value">{{ metrics.start_date.strftime('%Y-%m-%d') }} 至 {{ metrics.end_date.strftime('%Y-%m-%d') }}</div>
                </div>
                <div class="metric-card">
                    <h4>交易天数</h4>
                    <div class="value">{{ metrics.trading_days }} 天</div>
                </div>
                <div class="metric-card">
                    <h4>总收益率</h4>
                    <div class="value {{ 'positive' if metrics.total_return > 0 else 'negative' }}">{{ "{:.2%}".format(metrics.total_return) }}</div>
                </div>
                <div class="metric-card">
                    <h4>年化收益率</h4>
                    <div class="value {{ 'positive' if metrics.annual_return > 0 else 'negative' }}">{{ "{:.2%}".format(metrics.annual_return) }}</div>
                </div>
            </div>
        </div>
        {% endif %}
        
        {% if config.include_metrics %}
        <div class="section">
            <h3>📈 性能指标</h3>
            <div class="metrics-grid">
                <div class="metric-card">
                    <h4>波动率</h4>
                    <div class="value">{{ "{:.2%}".format(metrics.volatility) }}</div>
                </div>
                <div class="metric-card">
                    <h4>夏普比率</h4>
                    <div class="value">{{ "{:.4f}".format(metrics.sharpe_ratio) }}</div>
                </div>
                <div class="metric-card">
                    <h4>最大回撤</h4>
                    <div class="value negative">{{ "{:.2%}".format(metrics.max_drawdown) }}</div>
                </div>
                <div class="metric-card">
                    <h4>卡玛比率</h4>
                    <div class="value">{{ "{:.4f}".format(metrics.calmar_ratio) }}</div>
                </div>
                <div class="metric-card">
                    <h4>胜率</h4>
                    <div class="value">{{ "{:.2%}".format(metrics.trading_stats.win_rate) }}</div>
                </div>
                <div class="metric-card">
                    <h4>盈利因子</h4>
                    <div class="value">{{ "{:.4f}".format(metrics.trading_stats.profit_factor) }}</div>
                </div>
            </div>
        </div>
        {% endif %}
        
        {% if config.include_charts and charts %}
        <div class="section">
            <h3>📊 图表分析</h3>
            {% for chart_name, chart_data in charts.items() %}
            <div class="chart">
                <h4>{{ chart_name }}</h4>
                <img src="data:image/png;base64,{{ chart_data }}" alt="{{ chart_name }}">
            </div>
            {% endfor %}
        </div>
        {% endif %}
        
        <div class="footer">
            <p>报告由 Prometheus 量化交易系统生成</p>
            {% if config.author %}
            <p>作者: {{ config.author }}</p>
            {% endif %}
        </div>
    </div>
</body>
</html>
        """
    
    def generate_report(self, report_data: ReportData, 
                       output_format: ReportFormat = ReportFormat.HTML,
                       filename: Optional[str] = None) -> str:
        """生成回测报告
        
        Args:
            report_data: 报告数据
            output_format: 输出格式
            filename: 文件名
            
        Returns:
            str: 报告文件路径
        """
        if not report_data.validate():
            raise BacktestReportError("报告数据无效")
        
        self.logger.info(f"开始生成{output_format.value}格式的回测报告...")
        
        # 生成文件名
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if self.config.include_timestamp else ""
            filename = f"{self.config.filename_prefix}_{timestamp}.{output_format.value}"
        
        output_path = self.output_dir / filename
        
        try:
            if output_format == ReportFormat.HTML:
                self._generate_html_report(report_data, output_path)
            elif output_format == ReportFormat.JSON:
                self._generate_json_report(report_data, output_path)
            elif output_format == ReportFormat.EXCEL:
                self._generate_excel_report(report_data, output_path)
            elif output_format == ReportFormat.CSV:
                self._generate_csv_report(report_data, output_path)
            elif output_format == ReportFormat.MARKDOWN:
                self._generate_markdown_report(report_data, output_path)
            else:
                raise BacktestReportError(f"不支持的报告格式: {output_format}")
            
            self.logger.info(f"报告生成完成: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"生成报告失败: {e}")
            raise BacktestReportError(f"生成报告失败: {e}")
    
    def _generate_html_report(self, report_data: ReportData, output_path: Path):
        """生成HTML报告"""
        # 生成图表
        charts = {}
        if self.config.include_charts:
            charts = self._generate_charts(report_data)
        
        # 准备模板数据
        template_data = {
            'config': self.config,
            'metrics': report_data.metrics,
            'charts': charts,
            'generation_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # 渲染模板
        if JINJA2_AVAILABLE:
            template = Template(self.html_template)
            html_content = template.render(**template_data)
        else:
            # 简单的字符串替换
            html_content = self._render_simple_template(self.html_template, template_data)
        
        # 写入文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
    
    def _generate_json_report(self, report_data: ReportData, output_path: Path):
        """生成JSON报告"""
        report_dict = {
            'config': self.config.to_dict(),
            'metrics': report_data.metrics.to_dict(),
            'generation_time': datetime.now().isoformat()
        }
        
        # 添加数据
        if report_data.equity_curve is not None:
            report_dict['equity_curve'] = report_data.equity_curve.to_dict()
        
        if report_data.trades is not None:
            report_dict['trades'] = report_data.trades
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, ensure_ascii=False, indent=2, default=str)
    
    def _generate_excel_report(self, report_data: ReportData, output_path: Path):
        """生成Excel报告"""
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # 指标摘要
            metrics_df = self._create_metrics_dataframe(report_data.metrics)
            metrics_df.to_excel(writer, sheet_name='指标摘要', index=False)
            
            # 资产曲线
            if report_data.equity_curve is not None:
                equity_df = pd.DataFrame({
                    '日期': report_data.equity_curve.index,
                    '资产价值': report_data.equity_curve.values
                })
                equity_df.to_excel(writer, sheet_name='资产曲线', index=False)
            
            # 交易记录
            if report_data.trades is not None:
                trades_df = pd.DataFrame(report_data.trades)
                trades_df.to_excel(writer, sheet_name='交易记录', index=False)
            
            # 持仓记录
            if report_data.positions is not None:
                report_data.positions.to_excel(writer, sheet_name='持仓记录')
    
    def _generate_csv_report(self, report_data: ReportData, output_path: Path):
        """生成CSV报告"""
        # 创建指标DataFrame
        metrics_df = self._create_metrics_dataframe(report_data.metrics)
        metrics_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    def _generate_markdown_report(self, report_data: ReportData, output_path: Path):
        """生成Markdown报告"""
        md_content = f"""# {self.config.title}

{self.config.subtitle}

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 回测概要

- **回测期间**: {report_data.metrics.start_date.strftime('%Y-%m-%d')} 至 {report_data.metrics.end_date.strftime('%Y-%m-%d')}
- **交易天数**: {report_data.metrics.trading_days} 天
- **总收益率**: {report_data.metrics.total_return:.2%}
- **年化收益率**: {report_data.metrics.annual_return:.2%}

## 性能指标

| 指标 | 数值 |
|------|------|
| 波动率 | {report_data.metrics.volatility:.2%} |
| 夏普比率 | {report_data.metrics.sharpe_ratio:.4f} |
| 最大回撤 | {report_data.metrics.max_drawdown:.2%} |
| 卡玛比率 | {report_data.metrics.calmar_ratio:.4f} |
| 胜率 | {report_data.metrics.trading_stats.win_rate:.2%} |
| 盈利因子 | {report_data.metrics.trading_stats.profit_factor:.4f} |

## 风险指标

| 指标 | 数值 |
|------|------|
| VaR (95%) | {report_data.metrics.var_95:.2%} |
| CVaR (95%) | {report_data.metrics.cvar_95:.2%} |
| 下行波动率 | {report_data.metrics.downside_volatility:.2%} |
| 索提诺比率 | {report_data.metrics.sortino_ratio:.4f} |

---

*报告由 Prometheus 量化交易系统生成*
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
    
    def _generate_charts(self, report_data: ReportData) -> Dict[str, str]:
        """生成所有图表
        
        Args:
            report_data: 报告数据
            
        Returns:
            Dict[str, str]: 图表名称到base64编码的映射
        """
        charts = {}
        
        # 默认图表配置
        default_charts = [
            ChartConfig(ChartType.EQUITY_CURVE, "资产曲线"),
            ChartConfig(ChartType.DRAWDOWN, "回撤分析"),
            ChartConfig(ChartType.RETURNS_DISTRIBUTION, "收益率分布"),
            ChartConfig(ChartType.MONTHLY_RETURNS, "月度收益"),
            ChartConfig(ChartType.UNDERWATER_CHART, "水下图")
        ]
        
        # 使用配置中的图表或默认图表
        chart_configs = self.config.chart_configs if self.config.chart_configs else default_charts
        
        for chart_config in chart_configs:
            try:
                chart_data = self.chart_generator.generate_chart(chart_config, report_data)
                if chart_data:
                    charts[chart_config.title] = chart_data
            except Exception as e:
                self.logger.warning(f"生成图表 {chart_config.title} 失败: {e}")
        
        return charts
    
    def _create_metrics_dataframe(self, metrics: BacktestMetrics) -> pd.DataFrame:
        """创建指标DataFrame
        
        Args:
            metrics: 回测指标
            
        Returns:
            pd.DataFrame: 指标DataFrame
        """
        data = [
            ['回测期间', f"{metrics.start_date.strftime('%Y-%m-%d')} 至 {metrics.end_date.strftime('%Y-%m-%d')}"],
            ['交易天数', f"{metrics.trading_days} 天"],
            ['总收益率', f"{metrics.total_return:.2%}"],
            ['年化收益率', f"{metrics.annual_return:.2%}"],
            ['波动率', f"{metrics.volatility:.2%}"],
            ['夏普比率', f"{metrics.sharpe_ratio:.4f}"],
            ['索提诺比率', f"{metrics.sortino_ratio:.4f}"],
            ['卡玛比率', f"{metrics.calmar_ratio:.4f}"],
            ['最大回撤', f"{metrics.max_drawdown:.2%}"],
            ['VaR (95%)', f"{metrics.var_95:.2%}"],
            ['CVaR (95%)', f"{metrics.cvar_95:.2%}"],
            ['胜率', f"{metrics.trading_stats.win_rate:.2%}"],
            ['盈利因子', f"{metrics.trading_stats.profit_factor:.4f}"]
        ]
        
        return pd.DataFrame(data, columns=['指标', '数值'])
    
    def _render_simple_template(self, template: str, data: Dict) -> str:
        """简单模板渲染（当Jinja2不可用时）
        
        Args:
            template: 模板字符串
            data: 数据字典
            
        Returns:
            str: 渲染后的字符串
        """
        # 这是一个简化的模板渲染，仅用于基本替换
        result = template
        
        # 替换基本变量
        replacements = {
            '{{ config.title }}': data['config'].title,
            '{{ config.subtitle }}': data['config'].subtitle or '',
            '{{ generation_time }}': data['generation_time'],
            # 添加更多替换规则...
        }
        
        for placeholder, value in replacements.items():
            result = result.replace(placeholder, str(value))
        
        return result
    
    def export_data(self, report_data: ReportData, 
                   export_format: str = "csv") -> str:
        """导出数据
        
        Args:
            report_data: 报告数据
            export_format: 导出格式
            
        Returns:
            str: 导出文件路径
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if export_format.lower() == "csv":
            # 导出资产曲线
            if report_data.equity_curve is not None:
                equity_path = self.output_dir / f"equity_curve_{timestamp}.csv"
                report_data.equity_curve.to_csv(equity_path)
                
            # 导出交易记录
            if report_data.trades is not None:
                trades_path = self.output_dir / f"trades_{timestamp}.csv"
                pd.DataFrame(report_data.trades).to_csv(trades_path, index=False)
                
            return str(equity_path)
        
        elif export_format.lower() == "json":
            export_path = self.output_dir / f"backtest_data_{timestamp}.json"
            
            export_data = {
                'metrics': report_data.metrics.to_dict(),
                'equity_curve': report_data.equity_curve.to_dict() if report_data.equity_curve is not None else None,
                'trades': report_data.trades,
                'export_time': datetime.now().isoformat()
            }
            
            with open(export_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
            
            return str(export_path)
        
        else:
            raise BacktestReportError(f"不支持的导出格式: {export_format}")
    
    def __str__(self) -> str:
        return f"BacktestReporter(title='{self.config.title}', output_dir='{self.output_dir}')"
    
    def __repr__(self) -> str:
        return self.__str__()