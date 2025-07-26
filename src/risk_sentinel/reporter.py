"""Risk Reporter

风险报告生成器 - 负责生成各种风险报告
"""

import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 可选依赖导入
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
    plt = None
    sns = None

try:
    from jinja2 import Template
    HAS_JINJA2 = True
except ImportError:
    HAS_JINJA2 = False
    Template = None

from .models import (
    RiskType, RiskLevel, RiskAlert, RiskEvent, RiskConfig
)
from .analyzer import RiskAnalyzer, VaRResult, StressTestResult, RiskAttribution
from src.common.logging import get_logger
from src.portfolio_manager.models import Portfolio


@dataclass
class ReportConfig:
    """报告配置"""
    output_dir: str = "reports"
    template_dir: str = "templates"
    include_charts: bool = True
    chart_format: str = "png"  # png, svg, pdf
    chart_dpi: int = 300
    report_formats: List[str] = None  # html, pdf, json
    
    def __post_init__(self):
        if self.report_formats is None:
            self.report_formats = ['html', 'json']


@dataclass
class ReportMetadata:
    """报告元数据"""
    report_id: str
    report_type: str
    portfolio_id: str
    generation_time: datetime
    report_period_start: datetime
    report_period_end: datetime
    generated_by: str = "Risk Sentinel"
    version: str = "1.0"


class RiskReporter:
    """风险报告生成器
    
    负责:
    1. 日常风险报告
    2. 压力测试报告
    3. VaR报告
    4. 风险事件报告
    5. 合规报告
    6. 图表生成
    7. 报告导出
    """
    
    def __init__(self, config: ReportConfig, risk_analyzer: RiskAnalyzer = None):
        self.config = config
        self.risk_analyzer = risk_analyzer
        self.logger = get_logger(self.__class__.__name__)
        
        # 创建输出目录
        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.charts_path = self.output_path / "charts"
        self.charts_path.mkdir(exist_ok=True)
        
        # 设置图表样式（如果可用）
        if HAS_PLOTTING:
            try:
                plt.style.use('seaborn-v0_8')
                sns.set_palette("husl")
            except Exception:
                # 如果seaborn样式不可用，使用默认样式
                pass
        else:
            self.logger.warning("Plotting libraries not available. Charts will be disabled.")
            self.config.include_charts = False
    
    def generate_daily_risk_report(self, portfolio: Portfolio, 
                                 alerts: List[RiskAlert] = None,
                                 events: List[RiskEvent] = None) -> Dict[str, str]:
        """生成日常风险报告"""
        try:
            # 生成报告元数据
            metadata = ReportMetadata(
                report_id=f"daily_risk_{portfolio.portfolio_id}_{datetime.now().strftime('%Y%m%d')}",
                report_type="daily_risk",
                portfolio_id=portfolio.portfolio_id,
                generation_time=datetime.now(),
                report_period_start=datetime.now() - timedelta(days=1),
                report_period_end=datetime.now()
            )
            
            # 获取风险分析数据
            risk_report = self.risk_analyzer.generate_risk_report(portfolio)
            
            # 准备报告数据
            report_data = {
                'metadata': asdict(metadata),
                'portfolio_summary': risk_report.get('portfolio_summary', {}),
                'risk_metrics': risk_report.get('risk_metrics', {}),
                'var_analysis': risk_report.get('var_analysis', {}),
                'stress_test_results': risk_report.get('stress_test_results', {}),
                'risk_attribution': risk_report.get('risk_attribution', {}),
                'correlation_analysis': risk_report.get('correlation_analysis', {}),
                'alerts': [asdict(alert) for alert in (alerts or [])],
                'events': [asdict(event) for event in (events or [])],
                'recommendations': risk_report.get('recommendations', [])
            }
            
            # 生成图表
            if self.config.include_charts:
                chart_paths = self._generate_daily_risk_charts(portfolio, report_data)
                report_data['charts'] = chart_paths
            
            # 生成报告文件
            report_files = self._export_report(report_data, metadata)
            
            self.logger.info(f"Daily risk report generated: {metadata.report_id}")
            return report_files
        
        except Exception as e:
            self.logger.error(f"Error generating daily risk report: {e}")
            return {}
    
    def generate_var_report(self, portfolio: Portfolio, 
                           confidence_levels: List[float] = None,
                           holding_periods: List[int] = None) -> Dict[str, str]:
        """生成VaR报告"""
        try:
            if confidence_levels is None:
                confidence_levels = [0.95, 0.99]
            if holding_periods is None:
                holding_periods = [1, 5, 10]
            
            # 生成报告元数据
            metadata = ReportMetadata(
                report_id=f"var_report_{portfolio.portfolio_id}_{datetime.now().strftime('%Y%m%d')}",
                report_type="var_report",
                portfolio_id=portfolio.portfolio_id,
                generation_time=datetime.now(),
                report_period_start=datetime.now() - timedelta(days=252),
                report_period_end=datetime.now()
            )
            
            # 计算VaR
            var_results = self.risk_analyzer.calculate_var(
                portfolio, confidence_levels, holding_periods
            )
            
            # 准备报告数据
            report_data = {
                'metadata': asdict(metadata),
                'portfolio_summary': {
                    'portfolio_id': portfolio.portfolio_id,
                    'total_value': portfolio.total_value,
                    'positions_count': len([p for p in portfolio.positions.values() if p.quantity != 0])
                },
                'var_results': {
                    str(conf_level): asdict(result) 
                    for conf_level, result in var_results.items()
                },
                'var_analysis': self._analyze_var_results(var_results),
                'historical_var_trend': self._get_historical_var_trend(portfolio)
            }
            
            # 生成图表
            if self.config.include_charts:
                chart_paths = self._generate_var_charts(var_results, report_data)
                report_data['charts'] = chart_paths
            
            # 生成报告文件
            report_files = self._export_report(report_data, metadata)
            
            self.logger.info(f"VaR report generated: {metadata.report_id}")
            return report_files
        
        except Exception as e:
            self.logger.error(f"Error generating VaR report: {e}")
            return {}
    
    def generate_stress_test_report(self, portfolio: Portfolio, 
                                  stress_results: Dict[str, StressTestResult]) -> Dict[str, str]:
        """生成压力测试报告"""
        try:
            # 生成报告元数据
            metadata = ReportMetadata(
                report_id=f"stress_test_{portfolio.portfolio_id}_{datetime.now().strftime('%Y%m%d')}",
                report_type="stress_test",
                portfolio_id=portfolio.portfolio_id,
                generation_time=datetime.now(),
                report_period_start=datetime.now(),
                report_period_end=datetime.now()
            )
            
            # 准备报告数据
            report_data = {
                'metadata': asdict(metadata),
                'portfolio_summary': {
                    'portfolio_id': portfolio.portfolio_id,
                    'total_value': portfolio.total_value,
                    'positions_count': len([p for p in portfolio.positions.values() if p.quantity != 0])
                },
                'stress_test_results': {
                    scenario_id: asdict(result) 
                    for scenario_id, result in stress_results.items()
                },
                'stress_analysis': self._analyze_stress_results(stress_results),
                'worst_case_scenario': self._find_worst_case_scenario(stress_results),
                'breach_summary': self._summarize_breaches(stress_results)
            }
            
            # 生成图表
            if self.config.include_charts:
                chart_paths = self._generate_stress_test_charts(stress_results, report_data)
                report_data['charts'] = chart_paths
            
            # 生成报告文件
            report_files = self._export_report(report_data, metadata)
            
            self.logger.info(f"Stress test report generated: {metadata.report_id}")
            return report_files
        
        except Exception as e:
            self.logger.error(f"Error generating stress test report: {e}")
            return {}
    
    def generate_risk_attribution_report(self, portfolio: Portfolio, 
                                       attribution: RiskAttribution) -> Dict[str, str]:
        """生成风险归因报告"""
        try:
            # 生成报告元数据
            metadata = ReportMetadata(
                report_id=f"risk_attribution_{portfolio.portfolio_id}_{datetime.now().strftime('%Y%m%d')}",
                report_type="risk_attribution",
                portfolio_id=portfolio.portfolio_id,
                generation_time=datetime.now(),
                report_period_start=datetime.now() - timedelta(days=252),
                report_period_end=datetime.now()
            )
            
            # 准备报告数据
            report_data = {
                'metadata': asdict(metadata),
                'portfolio_summary': {
                    'portfolio_id': portfolio.portfolio_id,
                    'total_value': portfolio.total_value
                },
                'risk_attribution': asdict(attribution),
                'attribution_analysis': self._analyze_risk_attribution(attribution)
            }
            
            # 生成图表
            if self.config.include_charts:
                chart_paths = self._generate_attribution_charts(attribution, report_data)
                report_data['charts'] = chart_paths
            
            # 生成报告文件
            report_files = self._export_report(report_data, metadata)
            
            self.logger.info(f"Risk attribution report generated: {metadata.report_id}")
            return report_files
        
        except Exception as e:
            self.logger.error(f"Error generating risk attribution report: {e}")
            return {}
    
    def generate_compliance_report(self, portfolio: Portfolio, 
                                 violations: List[Dict[str, Any]],
                                 period_start: datetime = None,
                                 period_end: datetime = None) -> Dict[str, str]:
        """生成合规报告"""
        try:
            if period_start is None:
                period_start = datetime.now() - timedelta(days=30)
            if period_end is None:
                period_end = datetime.now()
            
            # 生成报告元数据
            metadata = ReportMetadata(
                report_id=f"compliance_{portfolio.portfolio_id}_{datetime.now().strftime('%Y%m%d')}",
                report_type="compliance",
                portfolio_id=portfolio.portfolio_id,
                generation_time=datetime.now(),
                report_period_start=period_start,
                report_period_end=period_end
            )
            
            # 准备报告数据
            report_data = {
                'metadata': asdict(metadata),
                'portfolio_summary': {
                    'portfolio_id': portfolio.portfolio_id,
                    'total_value': portfolio.total_value
                },
                'compliance_summary': self._generate_compliance_summary(violations),
                'violations': violations,
                'compliance_metrics': self._calculate_compliance_metrics(violations, period_start, period_end)
            }
            
            # 生成图表
            if self.config.include_charts:
                chart_paths = self._generate_compliance_charts(violations, report_data)
                report_data['charts'] = chart_paths
            
            # 生成报告文件
            report_files = self._export_report(report_data, metadata)
            
            self.logger.info(f"Compliance report generated: {metadata.report_id}")
            return report_files
        
        except Exception as e:
            self.logger.error(f"Error generating compliance report: {e}")
            return {}
    
    def _generate_daily_risk_charts(self, portfolio: Portfolio, 
                                  report_data: Dict[str, Any]) -> Dict[str, str]:
        """生成日常风险图表"""
        chart_paths = {}
        
        try:
            # 风险指标雷达图
            chart_paths['risk_radar'] = self._create_risk_radar_chart(
                report_data.get('risk_metrics', {})
            )
            
            # VaR趋势图
            var_analysis = report_data.get('var_analysis', {})
            if var_analysis:
                chart_paths['var_trend'] = self._create_var_trend_chart(var_analysis)
            
            # 持仓集中度图
            chart_paths['concentration'] = self._create_concentration_chart(portfolio)
            
            # 相关性热力图
            correlation_analysis = report_data.get('correlation_analysis', {})
            if correlation_analysis:
                chart_paths['correlation_heatmap'] = self._create_correlation_heatmap(
                    portfolio, correlation_analysis
                )
        
        except Exception as e:
            self.logger.error(f"Error generating daily risk charts: {e}")
        
        return chart_paths
    
    def _generate_var_charts(self, var_results: Dict[float, VaRResult], 
                           report_data: Dict[str, Any]) -> Dict[str, str]:
        """生成VaR图表"""
        chart_paths = {}
        
        try:
            # VaR对比图
            chart_paths['var_comparison'] = self._create_var_comparison_chart(var_results)
            
            # VaR vs CVaR对比
            chart_paths['var_cvar_comparison'] = self._create_var_cvar_chart(var_results)
            
            # 历史VaR趋势
            historical_trend = report_data.get('historical_var_trend', {})
            if historical_trend:
                chart_paths['historical_var_trend'] = self._create_historical_var_chart(historical_trend)
        
        except Exception as e:
            self.logger.error(f"Error generating VaR charts: {e}")
        
        return chart_paths
    
    def _generate_stress_test_charts(self, stress_results: Dict[str, StressTestResult], 
                                   report_data: Dict[str, Any]) -> Dict[str, str]:
        """生成压力测试图表"""
        chart_paths = {}
        
        try:
            # 压力测试结果对比
            chart_paths['stress_comparison'] = self._create_stress_comparison_chart(stress_results)
            
            # 最坏情况分析
            chart_paths['worst_case_analysis'] = self._create_worst_case_chart(
                report_data.get('worst_case_scenario', {})
            )
            
            # 持仓损益分布
            chart_paths['position_pnl_distribution'] = self._create_position_pnl_chart(stress_results)
        
        except Exception as e:
            self.logger.error(f"Error generating stress test charts: {e}")
        
        return chart_paths
    
    def _generate_attribution_charts(self, attribution: RiskAttribution, 
                                   report_data: Dict[str, Any]) -> Dict[str, str]:
        """生成风险归因图表"""
        chart_paths = {}
        
        try:
            # 风险分解饼图
            chart_paths['risk_decomposition'] = self._create_risk_decomposition_chart(attribution)
            
            # 因子贡献条形图
            chart_paths['factor_contributions'] = self._create_factor_contribution_chart(
                attribution.factor_contributions
            )
            
            # 持仓贡献图
            chart_paths['position_contributions'] = self._create_position_contribution_chart(
                attribution.position_contributions
            )
            
            # 行业贡献图
            chart_paths['sector_contributions'] = self._create_sector_contribution_chart(
                attribution.sector_contributions
            )
        
        except Exception as e:
            self.logger.error(f"Error generating attribution charts: {e}")
        
        return chart_paths
    
    def _generate_compliance_charts(self, violations: List[Dict[str, Any]], 
                                  report_data: Dict[str, Any]) -> Dict[str, str]:
        """生成合规图表"""
        chart_paths = {}
        
        try:
            # 违规类型分布
            chart_paths['violation_types'] = self._create_violation_types_chart(violations)
            
            # 违规趋势
            chart_paths['violation_trend'] = self._create_violation_trend_chart(violations)
            
            # 合规评分
            compliance_metrics = report_data.get('compliance_metrics', {})
            if compliance_metrics:
                chart_paths['compliance_score'] = self._create_compliance_score_chart(compliance_metrics)
        
        except Exception as e:
            self.logger.error(f"Error generating compliance charts: {e}")
        
        return chart_paths
    
    def _create_risk_radar_chart(self, risk_metrics: Dict[str, float]) -> str:
        """创建风险雷达图"""
        try:
            fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(projection='polar'))
            
            # 选择关键风险指标
            metrics_to_plot = {
                'Position Concentration': risk_metrics.get('position_concentration_hhi', 0),
                'Sector Concentration': risk_metrics.get('sector_concentration_hhi', 0),
                'Leverage': min(risk_metrics.get('leverage', 1), 2) / 2,  # 标准化到0-1
                'Max Position Weight': risk_metrics.get('max_position_weight', 0),
                'Cash Ratio': 1 - risk_metrics.get('cash_ratio', 0)  # 反向，现金少风险高
            }
            
            # 角度
            angles = [n / len(metrics_to_plot) * 2 * 3.14159 for n in range(len(metrics_to_plot))]
            angles += angles[:1]  # 闭合
            
            # 数值
            values = list(metrics_to_plot.values())
            values += values[:1]  # 闭合
            
            # 绘制
            ax.plot(angles, values, 'o-', linewidth=2, label='Current Risk')
            ax.fill(angles, values, alpha=0.25)
            
            # 设置标签
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metrics_to_plot.keys())
            ax.set_ylim(0, 1)
            
            plt.title('Risk Metrics Radar Chart', size=16, weight='bold')
            plt.tight_layout()
            
            # 保存
            filename = f"risk_radar_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config.chart_format}"
            filepath = self.charts_path / filename
            plt.savefig(filepath, dpi=self.config.chart_dpi, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.error(f"Error creating risk radar chart: {e}")
            return ""
    
    def _create_var_comparison_chart(self, var_results: Dict[float, VaRResult]) -> str:
        """创建VaR对比图"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            confidence_levels = list(var_results.keys())
            var_1d = [result.var_1d for result in var_results.values()]
            var_5d = [result.var_5d for result in var_results.values()]
            var_10d = [result.var_10d for result in var_results.values()]
            
            x = range(len(confidence_levels))
            width = 0.25
            
            ax.bar([i - width for i in x], var_1d, width, label='1-Day VaR', alpha=0.8)
            ax.bar(x, var_5d, width, label='5-Day VaR', alpha=0.8)
            ax.bar([i + width for i in x], var_10d, width, label='10-Day VaR', alpha=0.8)
            
            ax.set_xlabel('Confidence Level')
            ax.set_ylabel('VaR (%)')
            ax.set_title('Value at Risk Comparison')
            ax.set_xticks(x)
            ax.set_xticklabels([f'{cl:.0%}' for cl in confidence_levels])
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # 格式化y轴为百分比
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            plt.tight_layout()
            
            # 保存
            filename = f"var_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config.chart_format}"
            filepath = self.charts_path / filename
            plt.savefig(filepath, dpi=self.config.chart_dpi, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.error(f"Error creating VaR comparison chart: {e}")
            return ""
    
    def _create_stress_comparison_chart(self, stress_results: Dict[str, StressTestResult]) -> str:
        """创建压力测试对比图"""
        try:
            fig, ax = plt.subplots(figsize=(12, 8))
            
            scenarios = list(stress_results.keys())
            pnl_pcts = [result.portfolio_pnl_pct for result in stress_results.values()]
            scenario_names = [result.scenario_name for result in stress_results.values()]
            
            # 颜色编码：损失为红色，盈利为绿色
            colors = ['red' if pnl < 0 else 'green' for pnl in pnl_pcts]
            
            bars = ax.bar(range(len(scenarios)), pnl_pcts, color=colors, alpha=0.7)
            
            ax.set_xlabel('Stress Test Scenarios')
            ax.set_ylabel('Portfolio P&L (%)')
            ax.set_title('Stress Test Results Comparison')
            ax.set_xticks(range(len(scenarios)))
            ax.set_xticklabels(scenario_names, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            
            # 添加数值标签
            for bar, pnl in zip(bars, pnl_pcts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{pnl:.1%}', ha='center', va='bottom' if height >= 0 else 'top')
            
            # 格式化y轴为百分比
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
            
            plt.tight_layout()
            
            # 保存
            filename = f"stress_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{self.config.chart_format}"
            filepath = self.charts_path / filename
            plt.savefig(filepath, dpi=self.config.chart_dpi, bbox_inches='tight')
            plt.close()
            
            return str(filepath)
        
        except Exception as e:
            self.logger.error(f"Error creating stress comparison chart: {e}")
            return ""
    
    def _export_report(self, report_data: Dict[str, Any], 
                      metadata: ReportMetadata) -> Dict[str, str]:
        """导出报告"""
        report_files = {}
        
        try:
            # JSON格式
            if 'json' in self.config.report_formats:
                json_file = self.output_path / f"{metadata.report_id}.json"
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(report_data, f, indent=2, default=str, ensure_ascii=False)
                report_files['json'] = str(json_file)
            
            # HTML格式
            if 'html' in self.config.report_formats:
                html_file = self._generate_html_report(report_data, metadata)
                if html_file:
                    report_files['html'] = html_file
            
            # PDF格式（如果需要）
            if 'pdf' in self.config.report_formats:
                pdf_file = self._generate_pdf_report(report_data, metadata)
                if pdf_file:
                    report_files['pdf'] = pdf_file
        
        except Exception as e:
            self.logger.error(f"Error exporting report: {e}")
        
        return report_files
    
    def _generate_html_report(self, report_data: Dict[str, Any], 
                            metadata: ReportMetadata) -> Optional[str]:
        """生成HTML报告"""
        try:
            # 简化的HTML模板
            html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>{{ metadata.report_type|title }} Report - {{ metadata.portfolio_id }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; border-radius: 5px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .metric { display: inline-block; margin: 10px; padding: 10px; background-color: #f9f9f9; border-radius: 3px; }
        .alert { color: red; font-weight: bold; }
        .chart { text-align: center; margin: 20px 0; }
        table { border-collapse: collapse; width: 100%; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
    </style>
</head>
<body>
    <div class="header">
        <h1>{{ metadata.report_type|title }} Report</h1>
        <p><strong>Portfolio:</strong> {{ metadata.portfolio_id }}</p>
        <p><strong>Generated:</strong> {{ metadata.generation_time }}</p>
        <p><strong>Period:</strong> {{ metadata.report_period_start }} to {{ metadata.report_period_end }}</p>
    </div>
    
    {% if portfolio_summary %}
    <div class="section">
        <h2>Portfolio Summary</h2>
        {% for key, value in portfolio_summary.items() %}
        <div class="metric">
            <strong>{{ key|replace('_', ' ')|title }}:</strong> {{ value }}
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    {% if risk_metrics %}
    <div class="section">
        <h2>Risk Metrics</h2>
        {% for key, value in risk_metrics.items() %}
        <div class="metric">
            <strong>{{ key|replace('_', ' ')|title }}:</strong> 
            {% if value is number %}
                {{ "%.2f"|format(value) }}{% if 'ratio' in key or 'weight' in key %}%{% endif %}
            {% else %}
                {{ value }}
            {% endif %}
        </div>
        {% endfor %}
    </div>
    {% endif %}
    
    {% if recommendations %}
    <div class="section">
        <h2>Recommendations</h2>
        <ul>
        {% for rec in recommendations %}
            <li class="alert">{{ rec }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
    
    {% if charts %}
    <div class="section">
        <h2>Charts</h2>
        {% for chart_name, chart_path in charts.items() %}
        <div class="chart">
            <h3>{{ chart_name|replace('_', ' ')|title }}</h3>
            <img src="{{ chart_path }}" alt="{{ chart_name }}" style="max-width: 100%; height: auto;">
        </div>
        {% endfor %}
    </div>
    {% endif %}
</body>
</html>
            """
            
            template = Template(html_template)
            html_content = template.render(**report_data)
            
            # 保存HTML文件
            html_file = self.output_path / f"{metadata.report_id}.html"
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return str(html_file)
        
        except Exception as e:
            self.logger.error(f"Error generating HTML report: {e}")
            return None
    
    def _analyze_var_results(self, var_results: Dict[float, VaRResult]) -> Dict[str, Any]:
        """分析VaR结果"""
        analysis = {}
        
        try:
            if not var_results:
                return analysis
            
            # 计算平均VaR
            avg_var_1d = sum(result.var_1d for result in var_results.values()) / len(var_results)
            avg_var_5d = sum(result.var_5d for result in var_results.values()) / len(var_results)
            
            analysis['average_var_1d'] = avg_var_1d
            analysis['average_var_5d'] = avg_var_5d
            
            # 找出最高和最低VaR
            max_var_result = max(var_results.values(), key=lambda x: x.var_1d)
            min_var_result = min(var_results.values(), key=lambda x: x.var_1d)
            
            analysis['max_var'] = {
                'confidence_level': max_var_result.confidence_level,
                'var_1d': max_var_result.var_1d
            }
            analysis['min_var'] = {
                'confidence_level': min_var_result.confidence_level,
                'var_1d': min_var_result.var_1d
            }
        
        except Exception as e:
            self.logger.error(f"Error analyzing VaR results: {e}")
        
        return analysis
    
    def _analyze_stress_results(self, stress_results: Dict[str, StressTestResult]) -> Dict[str, Any]:
        """分析压力测试结果"""
        analysis = {}
        
        try:
            if not stress_results:
                return analysis
            
            # 计算平均损失
            avg_loss = sum(result.portfolio_pnl_pct for result in stress_results.values()) / len(stress_results)
            analysis['average_loss_pct'] = avg_loss
            
            # 统计违规数量
            total_breaches = sum(len(result.breach_limits) for result in stress_results.values())
            analysis['total_breaches'] = total_breaches
            
            # 最大损失场景
            worst_result = min(stress_results.values(), key=lambda x: x.portfolio_pnl_pct)
            analysis['worst_scenario'] = {
                'scenario_name': worst_result.scenario_name,
                'loss_pct': worst_result.portfolio_pnl_pct
            }
        
        except Exception as e:
            self.logger.error(f"Error analyzing stress results: {e}")
        
        return analysis
    
    def _find_worst_case_scenario(self, stress_results: Dict[str, StressTestResult]) -> Dict[str, Any]:
        """找出最坏情况场景"""
        try:
            if not stress_results:
                return {}
            
            worst_result = min(stress_results.values(), key=lambda x: x.portfolio_pnl_pct)
            return asdict(worst_result)
        
        except Exception as e:
            self.logger.error(f"Error finding worst case scenario: {e}")
            return {}
    
    def _summarize_breaches(self, stress_results: Dict[str, StressTestResult]) -> Dict[str, Any]:
        """汇总违规情况"""
        summary = {
            'total_scenarios': len(stress_results),
            'scenarios_with_breaches': 0,
            'total_breaches': 0,
            'breach_types': {}
        }
        
        try:
            for result in stress_results.values():
                if result.breach_limits:
                    summary['scenarios_with_breaches'] += 1
                    summary['total_breaches'] += len(result.breach_limits)
                    
                    for breach in result.breach_limits:
                        breach_type = breach.split()[0]  # 简化分类
                        summary['breach_types'][breach_type] = summary['breach_types'].get(breach_type, 0) + 1
        
        except Exception as e:
            self.logger.error(f"Error summarizing breaches: {e}")
        
        return summary
    
    def _get_historical_var_trend(self, portfolio: Portfolio) -> Dict[str, Any]:
        """获取历史VaR趋势（简化实现）"""
        # 这里应该从历史数据获取VaR趋势
        # 简化实现，返回空字典
        return {}
    
    def _analyze_risk_attribution(self, attribution: RiskAttribution) -> Dict[str, Any]:
        """分析风险归因"""
        analysis = {}
        
        try:
            # 系统性风险占比
            total_risk = attribution.total_risk
            if total_risk > 0:
                systematic_ratio = attribution.systematic_risk / total_risk
                idiosyncratic_ratio = attribution.idiosyncratic_risk / total_risk
                
                analysis['systematic_risk_ratio'] = systematic_ratio
                analysis['idiosyncratic_risk_ratio'] = idiosyncratic_ratio
            
            # 主要风险因子
            if attribution.factor_contributions:
                top_factor = max(attribution.factor_contributions.items(), key=lambda x: abs(x[1]))
                analysis['top_risk_factor'] = {
                    'factor': top_factor[0],
                    'contribution': top_factor[1]
                }
            
            # 主要风险持仓
            if attribution.position_contributions:
                top_position = max(attribution.position_contributions.items(), key=lambda x: abs(x[1]))
                analysis['top_risk_position'] = {
                    'position': top_position[0],
                    'contribution': top_position[1]
                }
        
        except Exception as e:
            self.logger.error(f"Error analyzing risk attribution: {e}")
        
        return analysis
    
    def _generate_compliance_summary(self, violations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """生成合规摘要"""
        summary = {
            'total_violations': len(violations),
            'violation_types': {},
            'severity_distribution': {},
            'recent_violations': 0
        }
        
        try:
            recent_threshold = datetime.now() - timedelta(days=7)
            
            for violation in violations:
                # 违规类型统计
                violation_type = violation.get('type', 'Unknown')
                summary['violation_types'][violation_type] = summary['violation_types'].get(violation_type, 0) + 1
                
                # 严重程度统计
                severity = violation.get('severity', 'Unknown')
                summary['severity_distribution'][severity] = summary['severity_distribution'].get(severity, 0) + 1
                
                # 最近违规统计
                violation_date = violation.get('date')
                if violation_date and isinstance(violation_date, datetime) and violation_date >= recent_threshold:
                    summary['recent_violations'] += 1
        
        except Exception as e:
            self.logger.error(f"Error generating compliance summary: {e}")
        
        return summary
    
    def _calculate_compliance_metrics(self, violations: List[Dict[str, Any]], 
                                    period_start: datetime, 
                                    period_end: datetime) -> Dict[str, float]:
        """计算合规指标"""
        metrics = {}
        
        try:
            period_days = (period_end - period_start).days
            if period_days <= 0:
                return metrics
            
            # 违规频率
            metrics['violation_frequency'] = len(violations) / period_days
            
            # 合规评分（简化计算）
            if violations:
                severity_weights = {'low': 1, 'medium': 3, 'high': 5, 'critical': 10}
                total_severity_score = sum(
                    severity_weights.get(v.get('severity', 'medium'), 3) 
                    for v in violations
                )
                # 评分：100分制，违规越多分数越低
                compliance_score = max(0, 100 - total_severity_score)
            else:
                compliance_score = 100
            
            metrics['compliance_score'] = compliance_score
        
        except Exception as e:
            self.logger.error(f"Error calculating compliance metrics: {e}")
        
        return metrics
    
    # 其他图表创建方法的简化实现
    def _create_var_trend_chart(self, var_analysis: Dict[str, Any]) -> str:
        """创建VaR趋势图（简化）"""
        return ""
    
    def _create_concentration_chart(self, portfolio: Portfolio) -> str:
        """创建集中度图（简化）"""
        return ""
    
    def _create_correlation_heatmap(self, portfolio: Portfolio, correlation_analysis: Dict[str, Any]) -> str:
        """创建相关性热力图（简化）"""
        return ""
    
    def _create_var_cvar_chart(self, var_results: Dict[float, VaRResult]) -> str:
        """创建VaR vs CVaR图（简化）"""
        return ""
    
    def _create_historical_var_chart(self, historical_trend: Dict[str, Any]) -> str:
        """创建历史VaR图（简化）"""
        return ""
    
    def _create_worst_case_chart(self, worst_case: Dict[str, Any]) -> str:
        """创建最坏情况图（简化）"""
        return ""
    
    def _create_position_pnl_chart(self, stress_results: Dict[str, StressTestResult]) -> str:
        """创建持仓损益图（简化）"""
        return ""
    
    def _create_risk_decomposition_chart(self, attribution: RiskAttribution) -> str:
        """创建风险分解图（简化）"""
        return ""
    
    def _create_factor_contribution_chart(self, factor_contributions: Dict[str, float]) -> str:
        """创建因子贡献图（简化）"""
        return ""
    
    def _create_position_contribution_chart(self, position_contributions: Dict[str, float]) -> str:
        """创建持仓贡献图（简化）"""
        return ""
    
    def _create_sector_contribution_chart(self, sector_contributions: Dict[str, float]) -> str:
        """创建行业贡献图（简化）"""
        return ""
    
    def _create_violation_types_chart(self, violations: List[Dict[str, Any]]) -> str:
        """创建违规类型图（简化）"""
        return ""
    
    def _create_violation_trend_chart(self, violations: List[Dict[str, Any]]) -> str:
        """创建违规趋势图（简化）"""
        return ""
    
    def _create_compliance_score_chart(self, compliance_metrics: Dict[str, float]) -> str:
        """创建合规评分图（简化）"""
        return ""
    
    def _generate_pdf_report(self, report_data: Dict[str, Any], 
                           metadata: ReportMetadata) -> Optional[str]:
        """生成PDF报告（简化实现）"""
        # PDF生成需要额外的库如reportlab，这里简化实现
        return None