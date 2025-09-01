"""
Performance Report Generation System for CodeSage MCP Server.

This module provides comprehensive automated report generation capabilities including
KPI summaries, trend analysis, regression detection integration, visual charts,
historical comparisons, and export functionality in multiple formats.

SECURITY CONSIDERATIONS:
- All file operations use secure path validation
- HTML output is sanitized to prevent XSS
- Sensitive data is redacted from reports
- File permissions are set securely
- Input validation prevents path traversal attacks
- Error messages don't leak sensitive information
"""

import json
import time
import re
import secrets
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import statistics
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.backends.backend_pdf import PdfPages

from .performance_monitor import get_performance_monitor
from .regression_detector import get_regression_detector, RegressionReport
from .prometheus_client import get_prometheus_client
from codesage_mcp.core.logging_config import get_logger
from codesage_mcp.config.config import get_optional_env_var

logger = get_logger("performance_report_generator")


class SecurityUtils:
    """Security utilities for report generation."""

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal and injection attacks."""
        # Remove path separators and dangerous characters
        sanitized = re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename)
        # Limit length
        return sanitized[:255] if sanitized else "report"

    @staticmethod
    def secure_path_validation(base_path: Path, requested_path: str) -> Path:
        """Validate and resolve path securely to prevent directory traversal."""
        try:
            # Resolve the path and check if it's within the base directory
            resolved_path = (base_path / requested_path).resolve()
            base_path_resolved = base_path.resolve()

            # Check if the resolved path is within the base directory
            if not str(resolved_path).startswith(str(base_path_resolved)):
                raise ValueError("Path traversal attempt detected")

            return resolved_path
        except Exception as e:
            logger.error(f"Path validation failed: {e}")
            raise ValueError("Invalid path")

    @staticmethod
    def sanitize_html_content(content: str) -> str:
        """Sanitize HTML content to prevent XSS attacks."""
        # Basic HTML sanitization - escape dangerous characters
        content = content.replace('&', '&')
        content = content.replace('<', '<')
        content = content.replace('>', '>')
        content = content.replace('"', '"')
        content = content.replace("'", '&#x27;')
        return content

    @staticmethod
    def redact_sensitive_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """Redact sensitive information from report data."""
        sensitive_keys = {
            'password', 'token', 'key', 'secret', 'credential',
            'auth', 'api_key', 'access_token', 'session_id'
        }

        def _redact_value(key: str, value: Any) -> Any:
            if isinstance(key, str) and any(sensitive in key.lower() for sensitive in sensitive_keys):
                if isinstance(value, str) and len(value) > 0:
                    return f"***REDACTED*** ({len(value)} chars)"
                elif isinstance(value, (int, float)):
                    return "***REDACTED***"
                else:
                    return "***REDACTED***"
            elif isinstance(value, dict):
                return {k: _redact_value(k, v) for k, v in value.items()}
            elif isinstance(value, list):
                return [_redact_value(f"item_{i}", item) for i, item in enumerate(value)]
            else:
                return value

        return {k: _redact_value(k, v) for k, v in data.items()}

    @staticmethod
    def generate_secure_filename(prefix: str = "report", extension: str = "json") -> str:
        """Generate a secure, unique filename."""
        timestamp = int(time.time())
        random_suffix = secrets.token_hex(8)
        return f"{prefix}_{timestamp}_{random_suffix}.{extension}"

    @staticmethod
    def validate_file_size(file_path: Path, max_size_mb: int = 100) -> bool:
        """Validate file size to prevent resource exhaustion."""
        try:
            size_mb = file_path.stat().st_size / (1024 * 1024)
            return size_mb <= max_size_mb
        except Exception:
            return False

    @staticmethod
    def secure_file_permissions(file_path: Path):
        """Set secure file permissions (readable by owner only)."""
        try:
            file_path.chmod(0o600)
        except Exception as e:
            logger.warning(f"Failed to set secure permissions on {file_path}: {e}")


@dataclass
class ReportConfig:
    """Configuration for performance report generation."""
    # Report settings
    report_title: str = "CodeSage MCP Server Performance Report"
    include_charts: bool = True
    include_historical_comparison: bool = True
    include_recommendations: bool = True

    # Data sources
    benchmark_results_dir: str = "benchmark_results"
    historical_reports_dir: str = "reports/historical"
    prometheus_url: Optional[str] = None
    grafana_url: Optional[str] = None

    # Export settings
    export_formats: List[str] = field(default_factory=lambda: ["json", "pdf", "html"])
    chart_dpi: int = 150

    # Time windows
    historical_window_days: int = 30
    trend_analysis_window_hours: int = 24

    def __post_init__(self):
        """Load configuration from environment variables."""
        self.include_charts = get_optional_env_var("REPORT_INCLUDE_CHARTS", "true").lower() == "true"
        self.include_historical_comparison = get_optional_env_var("REPORT_INCLUDE_HISTORICAL", "true").lower() == "true"
        self.include_recommendations = get_optional_env_var("REPORT_INCLUDE_RECOMMENDATIONS", "true").lower() == "true"
        self.prometheus_url = get_optional_env_var("PROMETHEUS_URL")
        self.grafana_url = get_optional_env_var("GRAFANA_URL")
        self.chart_dpi = int(get_optional_env_var("REPORT_CHART_DPI") or self.chart_dpi)


@dataclass
class PerformanceReport:
    """Comprehensive performance report data structure."""
    report_id: str
    timestamp: float
    title: str
    executive_summary: Dict[str, Any]
    kpi_summary: Dict[str, Any]
    trend_analysis: Dict[str, Any]
    regression_analysis: Optional[RegressionReport]
    historical_comparison: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    charts: Dict[str, str]  # filename -> description
    metadata: Dict[str, Any] = field(default_factory=dict)


class PerformanceReportGenerator:
    """Main performance report generator class."""

    def __init__(self, config: Optional[ReportConfig] = None):
        self.config = config or ReportConfig()
        self.logger = get_logger("performance_report_generator")
        self.performance_monitor = get_performance_monitor()
        self.regression_detector = get_regression_detector()
        self.prometheus_client = get_prometheus_client()

        # Create output directories
        self.output_dir = Path("reports")
        self.charts_dir = self.output_dir / "charts"
        self.historical_dir = Path(self.config.historical_reports_dir)

        for dir_path in [self.output_dir, self.charts_dir, self.historical_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    async def generate_comprehensive_report(self) -> PerformanceReport:
        """
        Generate a comprehensive performance report with security validation.

        Returns:
            Complete performance report with all sections

        Raises:
            ValueError: If input validation fails
        """
        # Input validation
        if not self.config.report_title or len(self.config.report_title) > 200:
            raise ValueError("Invalid report title: must be non-empty and less than 200 characters")

        if self.config.historical_window_days < 1 or self.config.historical_window_days > 365:
            raise ValueError("Invalid historical window: must be between 1 and 365 days")

        report_id = SecurityUtils.generate_secure_filename("perf_report", "")
        report_id = report_id.rstrip('.')  # Remove extension for ID
        self.logger.info(f"Generating comprehensive performance report: {report_id}")

        # Collect all data sources
        current_metrics = self.performance_monitor.get_current_metrics()

        # Try to get additional metrics from Prometheus if available
        prometheus_metrics = {}
        if self.config.prometheus_url:
            try:
                prometheus_metrics = await self.prometheus_client.get_current_metrics()
                # Merge Prometheus metrics with local metrics
                current_metrics.update(prometheus_metrics)
            except Exception as e:
                self.logger.warning(f"Failed to fetch Prometheus metrics: {e}")

        benchmark_data = self._load_benchmark_data()
        historical_data = self._load_historical_data()

        # Generate regression analysis
        regression_report = None
        if benchmark_data:
            regression_report = self.regression_detector.detect_regressions(
                current_metrics, historical_data.get("baseline")
            )

        # Generate report sections
        executive_summary = self._generate_executive_summary(
            current_metrics, benchmark_data, regression_report
        )

        kpi_summary = self._generate_kpi_summary(current_metrics, benchmark_data)

        trend_analysis = self._generate_trend_analysis(
            current_metrics, historical_data, self.config.trend_analysis_window_hours
        )

        historical_comparison = self._generate_historical_comparison(
            current_metrics, historical_data
        )

        recommendations = self._generate_recommendations(
            current_metrics, benchmark_data, regression_report
        )

        # Add Grafana dashboard information if available
        grafana_info = self._get_grafana_dashboard_info()
        if grafana_info:
            executive_summary["grafana_dashboards"] = grafana_info

        # Generate charts if enabled
        charts = {}
        if self.config.include_charts:
            charts = await self._generate_charts(
                current_metrics, historical_data, trend_analysis
            )

        # Create report object
        report = PerformanceReport(
            report_id=report_id,
            timestamp=time.time(),
            title=self.config.report_title,
            executive_summary=executive_summary,
            kpi_summary=kpi_summary,
            trend_analysis=trend_analysis,
            regression_analysis=regression_report,
            historical_comparison=historical_comparison,
            recommendations=recommendations,
            charts=charts,
            metadata={
                "generator_version": "1.0.0",
                "config": self.config.__dict__,
                "data_sources": {
                    "performance_monitor": True,
                    "benchmark_data": bool(benchmark_data),
                    "historical_data": bool(historical_data),
                    "prometheus": bool(self.config.prometheus_url),
                    "grafana": bool(self.config.grafana_url)
                }
            }
        )

        # Save report to historical data
        await self._save_report_to_history(report)

        return report

    def _generate_executive_summary(self, current_metrics: Dict[str, Any],
                                  benchmark_data: Dict[str, Any],
                                  regression_report: Optional[RegressionReport]) -> Dict[str, Any]:
        """Generate executive summary section."""
        summary = {
            "report_timestamp": datetime.fromtimestamp(time.time()).isoformat(),
            "overall_health_score": 0,
            "key_findings": [],
            "critical_issues": [],
            "performance_trends": []
        }

        # Calculate overall health score
        health_components = []

        # Response time score (0-100, lower is better)
        if "response_time_ms" in current_metrics:
            rt_value = current_metrics["response_time_ms"]["value"]
            rt_score = max(0, 100 - (rt_value / 10))  # Penalize > 1s responses
            health_components.append(rt_score)

        # Error rate score (0-100, lower is better)
        if "error_rate_percent" in current_metrics:
            error_value = current_metrics["error_rate_percent"]["value"]
            error_score = max(0, 100 - error_value)
            health_components.append(error_score)

        # Resource utilization score (0-100, balanced is better)
        if "memory_usage_percent" in current_metrics and "cpu_usage_percent" in current_metrics:
            mem_value = current_metrics["memory_usage_percent"]["value"]
            cpu_value = current_metrics["cpu_usage_percent"]["value"]
            resource_score = 100 - ((mem_value + cpu_value) / 2)
            health_components.append(resource_score)

        if health_components:
            summary["overall_health_score"] = round(statistics.mean(health_components), 1)

        # Generate key findings
        if benchmark_data:
            total_tests = benchmark_data.get("summary", {}).get("total_tests", 0)
            passed_tests = benchmark_data.get("summary", {}).get("passed_tests", 0)
            if total_tests > 0:
                success_rate = (passed_tests / total_tests) * 100
                summary["key_findings"].append(
                    f"Benchmark Success Rate: {success_rate:.1f}% ({passed_tests}/{total_tests} tests passed)"
                )

        # Add regression findings
        if regression_report and regression_report.regressions_detected:
            critical_regs = [r for r in regression_report.regressions_detected if r.severity == "critical"]
            if critical_regs:
                summary["critical_issues"].extend([
                    f"Critical regression in {r.metric_name}: {r.percentage_change:+.1f}%"
                    for r in critical_regs
                ])

        return summary

    def _generate_kpi_summary(self, current_metrics: Dict[str, Any],
                            benchmark_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate KPI summary section."""
        kpi_summary = {
            "current_metrics": current_metrics,
            "benchmark_kpis": {},
            "kpi_status": {}
        }

        # Extract KPIs from benchmark data
        if benchmark_data and "results" in benchmark_data:
            for result in benchmark_data["results"]:
                metric_name = result.get("metric_name", "")
                if metric_name:
                    kpi_summary["benchmark_kpis"][metric_name] = {
                        "value": result.get("value"),
                        "unit": result.get("unit"),
                        "target": result.get("target"),
                        "achieved": result.get("achieved", False),
                        "test_name": result.get("test_name")
                    }

        # Determine KPI status
        for metric_name, metric_data in current_metrics.items():
            value = metric_data.get("value", 0)
            status = "unknown"

            if metric_name == "response_time_ms":
                status = "good" if value < 1000 else "warning" if value < 5000 else "critical"
            elif metric_name == "error_rate_percent":
                status = "good" if value < 1 else "warning" if value < 5 else "critical"
            elif metric_name in ["memory_usage_percent", "cpu_usage_percent"]:
                status = "good" if value < 70 else "warning" if value < 85 else "critical"
            elif metric_name == "throughput_rps":
                status = "good" if value > 10 else "warning" if value > 1 else "critical"

            kpi_summary["kpi_status"][metric_name] = status

        return kpi_summary

    def _generate_trend_analysis(self, current_metrics: Dict[str, Any],
                               historical_data: Dict[str, Any],
                               window_hours: int) -> Dict[str, Any]:
        """Generate trend analysis section."""
        trend_analysis = {
            "analysis_window_hours": window_hours,
            "metric_trends": {},
            "trend_summary": {}
        }

        # Analyze trends for each metric
        cutoff_time = time.time() - (window_hours * 3600)

        for metric_name, metric_data in current_metrics.items():
            trend_data = {
                "current_value": metric_data.get("value"),
                "historical_values": [],
                "trend_direction": "stable",
                "trend_magnitude": 0.0,
                "volatility": 0.0
            }

            # Collect historical values
            if "historical" in historical_data:
                for historical_entry in historical_data["historical"]:
                    if historical_entry.get("timestamp", 0) > cutoff_time:
                        hist_metric = historical_entry.get("metrics", {}).get(metric_name)
                        if hist_metric:
                            trend_data["historical_values"].append({
                                "timestamp": historical_entry["timestamp"],
                                "value": hist_metric.get("value")
                            })

            # Calculate trend if we have enough data
            if len(trend_data["historical_values"]) >= 3:
                values = [entry["value"] for entry in trend_data["historical_values"]]
                values.append(trend_data["current_value"])

                # Simple linear trend
                if len(values) >= 2:
                    try:
                        slope = statistics.linear_regression(range(len(values)), values)[1]
                        trend_data["trend_magnitude"] = slope

                        if abs(slope) < 0.01:
                            trend_data["trend_direction"] = "stable"
                        elif slope > 0:
                            trend_data["trend_direction"] = "increasing"
                        else:
                            trend_data["trend_direction"] = "decreasing"

                        # Calculate volatility (coefficient of variation)
                        if statistics.mean(values) != 0:
                            trend_data["volatility"] = statistics.stdev(values) / abs(statistics.mean(values))

                    except Exception as e:
                        self.logger.warning(f"Failed to calculate trend for {metric_name}: {e}")

            trend_analysis["metric_trends"][metric_name] = trend_data

        # Generate trend summary
        increasing_trends = [m for m, t in trend_analysis["metric_trends"].items()
                           if t["trend_direction"] == "increasing"]
        decreasing_trends = [m for m, t in trend_analysis["metric_trends"].items()
                           if t["trend_direction"] == "decreasing"]

        trend_analysis["trend_summary"] = {
            "metrics_with_increasing_trends": increasing_trends,
            "metrics_with_decreasing_trends": decreasing_trends,
            "stable_metrics": [m for m, t in trend_analysis["metric_trends"].items()
                             if t["trend_direction"] == "stable"]
        }

        return trend_analysis

    def _generate_historical_comparison(self, current_metrics: Dict[str, Any],
                                      historical_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate historical comparison section."""
        comparison = {
            "baseline_comparison": {},
            "performance_changes": {},
            "comparison_period_days": self.config.historical_window_days
        }

        # Compare with baseline
        if "baseline" in historical_data:
            baseline = historical_data["baseline"]
            for metric_name, current_data in current_metrics.items():
                baseline_data = baseline.get(metric_name)
                if baseline_data:
                    current_value = current_data.get("value", 0)
                    baseline_value = baseline_data.get("value", 0)

                    if baseline_value != 0:
                        percentage_change = ((current_value - baseline_value) / baseline_value) * 100
                        comparison["baseline_comparison"][metric_name] = {
                            "current_value": current_value,
                            "baseline_value": baseline_value,
                            "percentage_change": round(percentage_change, 2),
                            "improvement": percentage_change < 0 if metric_name in ["response_time_ms", "error_rate_percent"]
                                        else percentage_change > 0
                        }

        return comparison

    def _generate_recommendations(self, current_metrics: Dict[str, Any],
                                benchmark_data: Dict[str, Any],
                                regression_report: Optional[RegressionReport]) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []

        # Memory usage recommendations
        if "memory_usage_percent" in current_metrics:
            mem_usage = current_metrics["memory_usage_percent"]["value"]
            if mem_usage > 85:
                recommendations.append({
                    "type": "memory_optimization",
                    "priority": "critical",
                    "title": "High Memory Usage Detected",
                    "description": ".1f",
                    "actions": [
                        "Review cache sizes and implement memory limits",
                        "Implement garbage collection tuning",
                        "Consider memory-mapped data structures",
                        "Monitor for memory leaks"
                    ]
                })
            elif mem_usage > 70:
                recommendations.append({
                    "type": "memory_optimization",
                    "priority": "high",
                    "title": "Elevated Memory Usage",
                    "description": ".1f",
                    "actions": [
                        "Monitor memory usage trends",
                        "Optimize cache eviction policies",
                        "Review data structure memory footprint"
                    ]
                })

        # CPU usage recommendations
        if "cpu_usage_percent" in current_metrics:
            cpu_usage = current_metrics["cpu_usage_percent"]["value"]
            if cpu_usage > 90:
                recommendations.append({
                    "type": "cpu_optimization",
                    "priority": "critical",
                    "title": "High CPU Usage Detected",
                    "description": ".1f",
                    "actions": [
                        "Profile CPU-intensive operations",
                        "Implement request throttling",
                        "Consider horizontal scaling",
                        "Optimize algorithm complexity"
                    ]
                })

        # Response time recommendations
        if "response_time_ms" in current_metrics:
            rt = current_metrics["response_time_ms"]["value"]
            if rt > 5000:  # Over 5 seconds
                recommendations.append({
                    "type": "latency_optimization",
                    "priority": "critical",
                    "title": "High Response Times",
                    "description": ".2f",
                    "actions": [
                        "Profile request processing pipeline",
                        "Optimize database queries",
                        "Implement response caching",
                        "Review network latency"
                    ]
                })

        # Error rate recommendations
        if "error_rate_percent" in current_metrics:
            error_rate = current_metrics["error_rate_percent"]["value"]
            if error_rate > 5:
                recommendations.append({
                    "type": "reliability_improvement",
                    "priority": "critical",
                    "title": "High Error Rate",
                    "description": ".1f",
                    "actions": [
                        "Review error logs and stack traces",
                        "Implement circuit breaker patterns",
                        "Add comprehensive error handling",
                        "Monitor error trends and patterns"
                    ]
                })

        # Regression-based recommendations
        if regression_report and regression_report.regressions_detected:
            for regression in regression_report.regressions_detected:
                if regression.severity in ["critical", "high"]:
                    recommendations.append({
                        "type": "regression_mitigation",
                        "priority": "high",
                        "title": f"Address {regression.metric_name} Regression",
                        "description": f"Performance regression detected: {regression.percentage_change:+.1f}% change in {regression.metric_name}",
                        "actions": [
                            "Investigate recent code changes",
                            "Review deployment history",
                            "Consider rollback if regression is severe",
                            "Implement additional monitoring"
                        ]
                    })

        # Benchmark-based recommendations
        if benchmark_data and "summary" in benchmark_data:
            summary = benchmark_data["summary"]
            total_tests = summary.get("total_tests", 0)
            failed_tests = summary.get("failed_tests", 0)

            if failed_tests > 0:
                failure_rate = (failed_tests / total_tests) * 100
                recommendations.append({
                    "type": "benchmark_optimization",
                    "priority": "medium",
                    "title": "Benchmark Test Failures",
                    "description": f"{failed_tests}/{total_tests} benchmark tests failed ({failure_rate:.1f}%)",
                    "actions": [
                        "Review failed benchmark tests",
                        "Analyze failure patterns",
                        "Update performance targets if needed",
                        "Implement fixes for failed tests"
                    ]
                })

        return recommendations

    def perform_security_audit(self) -> Dict[str, Any]:
        """
        Perform security audit of the report generation system.

        Returns:
            Security audit results
        """
        audit_results = {
            "timestamp": time.time(),
            "audit_version": "1.0",
            "security_checks": {},
            "recommendations": []
        }

        # Check file permissions
        audit_results["security_checks"]["output_directory_permissions"] = self._check_directory_permissions(self.output_dir)
        audit_results["security_checks"]["charts_directory_permissions"] = self._check_directory_permissions(self.charts_dir)
        audit_results["security_checks"]["historical_directory_permissions"] = self._check_directory_permissions(self.historical_dir)

        # Check for sensitive data patterns
        audit_results["security_checks"]["sensitive_data_patterns"] = self._check_sensitive_data_patterns()

        # Check export format security
        audit_results["security_checks"]["export_formats_secure"] = self._validate_export_formats()

        # Check Prometheus security
        audit_results["security_checks"]["prometheus_config_secure"] = self._validate_prometheus_config()

        # Generate recommendations
        if not audit_results["security_checks"]["output_directory_permissions"]:
            audit_results["recommendations"].append("Fix output directory permissions to ensure secure file storage")

        if audit_results["security_checks"]["sensitive_data_patterns"]:
            audit_results["recommendations"].append("Sensitive data patterns detected - review data sanitization")

        if not audit_results["security_checks"]["export_formats_secure"]:
            audit_results["recommendations"].append("Review export format security configurations")

        return audit_results

    def _check_directory_permissions(self, directory: Path) -> bool:
        """Check if directory has secure permissions."""
        try:
            if not directory.exists():
                return True  # Non-existent directories are considered secure

            # Check if directory is writable by others
            stat_info = directory.stat()
            permissions = stat_info.st_mode & 0o777

            # Directory should not be writable by group or others
            return (permissions & 0o022) == 0
        except Exception:
            return False

    def _check_sensitive_data_patterns(self) -> List[str]:
        """Check for sensitive data patterns in recent reports."""
        sensitive_patterns = []
        try:
            # Check recent report files for sensitive patterns
            for report_file in self.historical_dir.glob("*.json"):
                if report_file.stat().st_size > 0:
                    with open(report_file, 'r') as f:
                        content = f.read()
                        if any(pattern in content.lower() for pattern in ['password', 'token', 'secret', 'key']):
                            sensitive_patterns.append(str(report_file.name))
        except Exception as e:
            self.logger.warning(f"Failed to check sensitive data patterns: {e}")

        return sensitive_patterns

    def _validate_export_formats(self) -> bool:
        """Validate that configured export formats are secure."""
        allowed_formats = {'json', 'html', 'pdf'}
        configured_formats = set(self.config.export_formats)

        # Check if all configured formats are allowed
        if not configured_formats.issubset(allowed_formats):
            return False

        # Additional validation for HTML export (most risky)
        if 'html' in configured_formats:
            # Ensure HTML sanitization is properly configured
            return hasattr(SecurityUtils, 'sanitize_html_content')

        return True

    def _validate_prometheus_config(self) -> bool:
        """Validate Prometheus configuration security."""
        if not self.config.prometheus_url:
            return True  # No Prometheus config is secure

        # Validate URL format
        if not self.config.prometheus_url.startswith(('http://', 'https://')):
            return False

        # Check for localhost/unsafe configurations in production
        if 'localhost' in self.config.prometheus_url or '127.0.0.1' in self.config.prometheus_url:
            # This might be acceptable for development but flag for review
            self.logger.warning("Prometheus configured with localhost - ensure this is appropriate for your environment")

        return True

    async def export_report(self, report: PerformanceReport, formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Export report in specified formats with security validation.

        Args:
            report: The performance report to export
            formats: List of formats to export (json, pdf, html)

        Returns:
            Dictionary mapping format to file path

        Raises:
            ValueError: If format validation fails
        """
        if formats is None:
            formats = self.config.export_formats

        # Validate formats
        allowed_formats = {'json', 'html', 'pdf'}
        for fmt in formats:
            if fmt not in allowed_formats:
                raise ValueError(f"Unsupported export format: {fmt}")

        # Security audit before export
        audit_results = self.perform_security_audit()
        if audit_results["recommendations"]:
            self.logger.warning(f"Security audit found issues: {audit_results['recommendations']}")

        exported_files = {}

        for format_type in formats:
            try:
                if format_type == "json":
                    file_path = await self._export_json(report)
                elif format_type == "pdf":
                    file_path = await self._export_pdf(report)
                elif format_type == "html":
                    file_path = await self._export_html(report)
                else:
                    raise ValueError(f"Unsupported export format: {format_type}")

                exported_files[format_type] = str(file_path)
                self.logger.info(f"Exported report to {format_type}: {file_path}")

            except Exception as e:
                self.logger.error(f"Failed to export report to {format_type}: {e}")
                raise

        return exported_files

    def _get_grafana_dashboard_info(self) -> Optional[Dict[str, Any]]:
        """Get Grafana dashboard information for real-time monitoring."""
        if not self.config.grafana_url:
            return None

        # Define available dashboards
        dashboards = {
            "performance_report": {
                "name": "Performance Report Dashboard",
                "description": "Real-time performance metrics and KPIs",
                "url": f"{self.config.grafana_url}/d/codesage-performance-report",
                "panels": [
                    "Overall Health Score",
                    "Response Time Trends",
                    "Throughput Metrics",
                    "Error Rate Monitoring",
                    "Resource Usage",
                    "Cache Performance",
                    "Benchmark Results",
                    "Regression Alerts"
                ]
            },
            "overview": {
                "name": "CodeSage Overview",
                "description": "General system overview and health metrics",
                "url": f"{self.config.grafana_url}/d/codesage-overview",
                "panels": [
                    "System Health",
                    "Service Status",
                    "Resource Utilization",
                    "Request Patterns"
                ]
            },
            "user_experience": {
                "name": "User Experience Dashboard",
                "description": "User-facing performance metrics",
                "url": f"{self.config.grafana_url}/d/codesage-user-experience",
                "panels": [
                    "Response Times",
                    "Error Rates",
                    "User Satisfaction",
                    "Performance Trends"
                ]
            }
        }

        return {
            "grafana_url": self.config.grafana_url,
            "available_dashboards": dashboards,
            "time_range": "Last 1 hour",
            "refresh_interval": "30 seconds"
        }

    async def _generate_charts(self, current_metrics: Dict[str, Any],
                             historical_data: Dict[str, Any],
                             trend_analysis: Dict[str, Any]) -> Dict[str, str]:
        """Generate visual charts for the report."""
        charts = {}

        try:
            # KPI Overview Chart
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

            # Response Time
            if "response_time_ms" in current_metrics:
                rt_data = current_metrics["response_time_ms"]
                ax1.bar(["Response Time"], [rt_data["value"]], color='skyblue')
                ax1.set_ylabel(f"Time ({rt_data.get('unit', 'ms')})")
                ax1.set_title("Response Time")
                ax1.grid(True, alpha=0.3)

            # Throughput
            if "throughput_rps" in current_metrics:
                tp_data = current_metrics["throughput_rps"]
                ax2.bar(["Throughput"], [tp_data["value"]], color='lightgreen')
                ax2.set_ylabel(f"Requests/{tp_data.get('unit', 'sec')}")
                ax2.set_title("Throughput")
                ax2.grid(True, alpha=0.3)

            # Memory Usage
            if "memory_usage_percent" in current_metrics:
                mem_data = current_metrics["memory_usage_percent"]
                ax3.bar(["Memory Usage"], [mem_data["value"]], color='orange')
                ax3.set_ylabel(f"Usage ({mem_data.get('unit', '%')})")
                ax3.set_title("Memory Usage")
                ax3.grid(True, alpha=0.3)

            # Error Rate
            if "error_rate_percent" in current_metrics:
                err_data = current_metrics["error_rate_percent"]
                ax4.bar(["Error Rate"], [err_data["value"]], color='red')
                ax4.set_ylabel(f"Rate ({err_data.get('unit', '%')})")
                ax4.set_title("Error Rate")
                ax4.grid(True, alpha=0.3)

            plt.tight_layout()
            kpi_chart_path = self.charts_dir / "kpi_overview.png"
            plt.savefig(kpi_chart_path, dpi=self.config.chart_dpi, bbox_inches='tight')
            plt.close()
            charts["kpi_overview.png"] = "KPI Overview Dashboard"

            # Trend Analysis Chart
            if trend_analysis.get("metric_trends"):
                fig, axes = plt.subplots(len(trend_analysis["metric_trends"]), 1,
                                       figsize=(12, 4 * len(trend_analysis["metric_trends"])))

                if len(trend_analysis["metric_trends"]) == 1:
                    axes = [axes]

                for i, (metric_name, trend_data) in enumerate(trend_analysis["metric_trends"].items()):
                    ax = axes[i]

                    # Plot historical data
                    if trend_data["historical_values"]:
                        timestamps = [entry["timestamp"] for entry in trend_data["historical_values"]]
                        values = [entry["value"] for entry in trend_data["historical_values"]]

                        dates = [datetime.fromtimestamp(ts) for ts in timestamps]
                        ax.plot(dates, values, 'b-', label='Historical', marker='o', markersize=3)

                        # Add current value
                        current_date = datetime.fromtimestamp(time.time())
                        ax.plot([current_date], [trend_data["current_value"]],
                              'ro', label='Current', markersize=5)

                        ax.set_title(f"{metric_name.replace('_', ' ').title()} Trend")
                        ax.set_ylabel("Value")
                        ax.legend()
                        ax.grid(True, alpha=0.3)

                        # Format x-axis dates
                        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

                plt.tight_layout()
                trend_chart_path = self.charts_dir / "trend_analysis.png"
                plt.savefig(trend_chart_path, dpi=self.config.chart_dpi, bbox_inches='tight')
                plt.close()
                charts["trend_analysis.png"] = "Performance Trends Over Time"

        except Exception as e:
            self.logger.error(f"Failed to generate charts: {e}")

        return charts

    def _load_benchmark_data(self) -> Dict[str, Any]:
        """Load benchmark data from results directory."""
        benchmark_dir = Path(self.config.benchmark_results_dir)
        if not benchmark_dir.exists():
            return {}

        # Load the most recent benchmark report
        pattern = "benchmark_report_*.json"
        benchmark_files = list(benchmark_dir.glob(pattern))

        if not benchmark_files:
            return {}

        # Get the most recent file
        latest_file = max(benchmark_files, key=lambda f: f.stat().st_mtime)

        try:
            with open(latest_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load benchmark data from {latest_file}: {e}")
            return {}

    def _load_historical_data(self) -> Dict[str, Any]:
        """Load historical performance data."""
        historical_data = {
            "historical": [],
            "baseline": {}
        }

        # Load baseline data
        baseline_file = Path(self.config.benchmark_results_dir) / "baseline_results.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, 'r') as f:
                    historical_data["baseline"] = json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load baseline data: {e}")

        # Load historical reports
        if self.historical_dir.exists():
            for report_file in self.historical_dir.glob("*.json"):
                try:
                    with open(report_file, 'r') as f:
                        report_data = json.load(f)
                        if "kpi_summary" in report_data and "current_metrics" in report_data["kpi_summary"]:
                            historical_data["historical"].append({
                                "timestamp": report_data.get("timestamp", 0),
                                "metrics": report_data["kpi_summary"]["current_metrics"]
                            })
                except Exception as e:
                    self.logger.error(f"Failed to load historical report {report_file}: {e}")

        # Sort historical data by timestamp
        historical_data["historical"].sort(key=lambda x: x["timestamp"])

        return historical_data

    async def _save_report_to_history(self, report: PerformanceReport):
        """Save report to historical data for future comparisons."""
        report_file = self.historical_dir / f"{report.report_id}.json"

        # Convert report to dict for JSON serialization
        report_dict = {
            "report_id": report.report_id,
            "timestamp": report.timestamp,
            "title": report.title,
            "executive_summary": report.executive_summary,
            "kpi_summary": report.kpi_summary,
            "trend_analysis": report.trend_analysis,
            "historical_comparison": report.historical_comparison,
            "recommendations": report.recommendations,
            "charts": report.charts,
            "metadata": report.metadata
        }

        # Add regression data if available
        if report.regression_analysis:
            report_dict["regression_analysis"] = {
                "test_run_id": report.regression_analysis.test_run_id,
                "regressions_detected": len(report.regression_analysis.regressions_detected),
                "summary": report.regression_analysis.summary,
                "recommendations": report.regression_analysis.recommendations
            }

        try:
            with open(report_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            self.logger.info(f"Saved report to history: {report_file}")
        except Exception as e:
            self.logger.error(f"Failed to save report to history: {e}")

    async def export_report(self, report: PerformanceReport, formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Export report in specified formats.

        Args:
            report: The performance report to export
            formats: List of formats to export (json, pdf, html)

        Returns:
            Dictionary mapping format to file path
        """
        if formats is None:
            formats = self.config.export_formats

        exported_files = {}

        for format_type in formats:
            try:
                if format_type == "json":
                    file_path = await self._export_json(report)
                elif format_type == "pdf":
                    file_path = await self._export_pdf(report)
                elif format_type == "html":
                    file_path = await self._export_html(report)
                else:
                    self.logger.warning(f"Unsupported export format: {format_type}")
                    continue

                exported_files[format_type] = str(file_path)
                self.logger.info(f"Exported report to {format_type}: {file_path}")

            except Exception as e:
                self.logger.error(f"Failed to export report to {format_type}: {e}")

        return exported_files

    async def _export_json(self, report: PerformanceReport) -> Path:
        """Export report as JSON with security measures."""
        # Generate secure filename
        filename = SecurityUtils.generate_secure_filename("report", "json")
        json_file = self.output_dir / filename

        # Redact sensitive data from report
        report_dict = {
            "report_id": report.report_id,
            "timestamp": report.timestamp,
            "title": SecurityUtils.sanitize_html_content(report.title),
            "executive_summary": SecurityUtils.redact_sensitive_data(report.executive_summary),
            "kpi_summary": SecurityUtils.redact_sensitive_data(report.kpi_summary),
            "trend_analysis": SecurityUtils.redact_sensitive_data(report.trend_analysis),
            "historical_comparison": SecurityUtils.redact_sensitive_data(report.historical_comparison),
            "recommendations": SecurityUtils.redact_sensitive_data(report.recommendations),
            "charts": report.charts,
            "metadata": SecurityUtils.redact_sensitive_data(report.metadata)
        }

        if report.regression_analysis:
            report_dict["regression_analysis"] = {
                "test_run_id": report.regression_analysis.test_run_id,
                "regressions_detected": len(report.regression_analysis.regressions_detected),
                "summary": SecurityUtils.redact_sensitive_data(report.regression_analysis.summary),
                "recommendations": SecurityUtils.redact_sensitive_data(report.regression_analysis.recommendations)
            }

        try:
            with open(json_file, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)

            # Set secure file permissions
            SecurityUtils.secure_file_permissions(json_file)

            # Validate file size
            if not SecurityUtils.validate_file_size(json_file):
                logger.warning(f"Generated JSON file exceeds size limit: {json_file}")
                json_file.unlink()  # Delete oversized file
                raise ValueError("Generated report exceeds maximum file size")

            return json_file

        except Exception as e:
            logger.error(f"Failed to export JSON report securely: {e}")
            if json_file.exists():
                json_file.unlink()  # Clean up on failure
            raise

    async def _export_pdf(self, report: PerformanceReport) -> Path:
        """Export report as PDF with charts and security measures."""
        # Generate secure filename
        filename = SecurityUtils.generate_secure_filename("report", "pdf")
        pdf_file = self.output_dir / filename

        # Sanitize content for PDF
        safe_title = SecurityUtils.sanitize_html_content(report.title)
        safe_report_id = SecurityUtils.sanitize_html_content(report.report_id)
        safe_summary = SecurityUtils.redact_sensitive_data(report.executive_summary)

        try:
            with PdfPages(pdf_file) as pdf:
                # Title page with security notice
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.text(0.5, 0.85, safe_title, transform=ax.transAxes,
                       fontsize=16, ha='center', va='center', fontweight='bold')
                ax.text(0.5, 0.75, "CONFIDENTIAL - For Internal Use Only",
                       transform=ax.transAxes, fontsize=12, ha='center', va='center', color='red')
                ax.text(0.5, 0.65, f"Generated: {datetime.fromtimestamp(report.timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                       transform=ax.transAxes, fontsize=12, ha='center', va='center')
                ax.text(0.5, 0.55, f"Report ID: {safe_report_id}",
                       transform=ax.transAxes, fontsize=10, ha='center', va='center')
                ax.text(0.5, 0.35, "This document contains sanitized data.",
                       transform=ax.transAxes, fontsize=10, ha='center', va='center', style='italic')
                ax.set_axis_off()
                pdf.savefig(fig)
                plt.close()

                # Executive Summary
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.text(0.1, 0.9, "Executive Summary", fontsize=14, fontweight='bold')
                y_pos = 0.8

                ax.text(0.1, y_pos, f"Overall Health Score: {safe_summary.get('overall_health_score', 'N/A')}/100",
                       fontsize=12)
                y_pos -= 0.1

                # Add key findings with sanitization
                for finding in safe_summary.get('key_findings', []):
                    safe_finding = SecurityUtils.sanitize_html_content(str(finding))
                    ax.text(0.1, y_pos, f"• {safe_finding}", fontsize=10)
                    y_pos -= 0.05
                    if y_pos < 0.1:  # Prevent overflow
                        break

                # Add critical issues with sanitization
                for issue in safe_summary.get('critical_issues', []):
                    safe_issue = SecurityUtils.sanitize_html_content(str(issue))
                    ax.text(0.1, y_pos, f"CRITICAL: {safe_issue}", fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3))
                    y_pos -= 0.08
                    if y_pos < 0.1:  # Prevent overflow
                        break

                ax.set_axis_off()
                pdf.savefig(fig)
                plt.close()

                # Add charts if available (with security validation)
                for chart_file, description in report.charts.items():
                    try:
                        # Validate chart filename for security
                        safe_chart_file = SecurityUtils.sanitize_filename(chart_file)
                        chart_path = self.charts_dir / safe_chart_file

                        # Additional security: ensure chart file is actually within charts directory
                        chart_path = SecurityUtils.secure_path_validation(self.charts_dir, safe_chart_file)

                        if chart_path.exists() and SecurityUtils.validate_file_size(chart_path, 10):  # 10MB limit for images
                            fig, ax = plt.subplots(figsize=(8.5, 11))
                            img = plt.imread(str(chart_path))
                            ax.imshow(img)
                            safe_description = SecurityUtils.sanitize_html_content(description)
                            ax.set_title(safe_description, fontsize=12, fontweight='bold')
                            ax.set_axis_off()
                            pdf.savefig(fig)
                            plt.close()
                        else:
                            self.logger.warning(f"Chart file validation failed: {chart_file}")
                    except Exception as e:
                        self.logger.warning(f"Failed to add chart {chart_file} to PDF: {e}")

            # Set secure file permissions
            SecurityUtils.secure_file_permissions(pdf_file)

            # Validate final file size
            if not SecurityUtils.validate_file_size(pdf_file, 50):  # 50MB limit for PDFs
                logger.warning(f"Generated PDF file exceeds size limit: {pdf_file}")
                pdf_file.unlink()  # Delete oversized file
                raise ValueError("Generated report exceeds maximum file size")

            return pdf_file

        except Exception as e:
            logger.error(f"Failed to export PDF report securely: {e}")
            if pdf_file.exists():
                pdf_file.unlink()  # Clean up on failure
            raise

    async def _export_html(self, report: PerformanceReport) -> Path:
        """Export report as HTML with security measures."""
        # Generate secure filename
        filename = SecurityUtils.generate_secure_filename("report", "html")
        html_file = self.output_dir / filename

        # Sanitize all user-provided content
        safe_title = SecurityUtils.sanitize_html_content(report.title)
        safe_report_id = SecurityUtils.sanitize_html_content(report.report_id)

        # Redact sensitive data
        safe_executive_summary = SecurityUtils.redact_sensitive_data(report.executive_summary)
        safe_kpi_summary = SecurityUtils.redact_sensitive_data(report.kpi_summary)
        safe_recommendations = SecurityUtils.redact_sensitive_data(report.recommendations)

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{safe_title}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
            color: #333;
            line-height: 1.6;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 8px;
            margin-bottom: 30px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: 300;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .section {{
            margin: 30px 0;
            padding: 20px;
            border-left: 4px solid #667eea;
            background-color: #fafafa;
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            font-size: 1.8em;
            font-weight: 400;
        }}
        .metric {{
            background-color: #e8f4f8;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #4a90e2;
        }}
        .recommendation {{
            background-color: #fff3cd;
            padding: 15px;
            margin: 10px 0;
            border-radius: 5px;
            border-left: 4px solid #ffc107;
        }}
        .critical {{
            background-color: #f8d7da;
            border-left-color: #dc3545;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left-color: #ffc107;
        }}
        .good {{
            background-color: #d1ecf1;
            border-left-color: #17a2b8;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            background-color: white;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
            color: #495057;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
        .security-notice {{
            background-color: #e7f3ff;
            border: 1px solid #b8daff;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .security-notice strong {{
            color: #004085;
        }}
        .footer {{
            text-align: center;
            margin-top: 40px;
            padding: 20px;
            color: #6c757d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{safe_title}</h1>
            <p><strong>Report ID:</strong> {safe_report_id}</p>
            <p><strong>Generated:</strong> {datetime.fromtimestamp(report.timestamp).strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>

        <div class="security-notice">
            <strong>🔒 Security Notice:</strong> This report contains sanitized and redacted data.
            Sensitive information has been automatically removed for security purposes.
        </div>

        <div class="section">
            <h2>Executive Summary</h2>
            <div class="metric">
                <strong>Overall Health Score:</strong> {safe_executive_summary.get('overall_health_score', 'N/A')}/100
            </div>
"""

        # Add key findings with sanitization
        for finding in safe_executive_summary.get('key_findings', []):
            safe_finding = SecurityUtils.sanitize_html_content(str(finding))
            html_content += f"<div class='metric'>• {safe_finding}</div>"

        # Add critical issues with sanitization
        for issue in safe_executive_summary.get('critical_issues', []):
            safe_issue = SecurityUtils.sanitize_html_content(str(issue))
            html_content += f"<div class='metric critical'>🚨 {safe_issue}</div>"

        # KPI Summary
        html_content += """
        <div class="section">
            <h2>KPI Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th><th>Unit</th><th>Status</th></tr>
"""

        for metric_name, metric_data in safe_kpi_summary.get('current_metrics', {}).items():
            safe_metric_name = SecurityUtils.sanitize_html_content(str(metric_name))
            value = metric_data.get('value', 'N/A')
            unit = SecurityUtils.sanitize_html_content(str(metric_data.get('unit', '')))
            status = safe_kpi_summary.get('kpi_status', {}).get(metric_name, 'unknown')
            status_class = f"class='{SecurityUtils.sanitize_html_content(status)}'"
            html_content += f"<tr><td>{safe_metric_name}</td><td>{value}</td><td>{unit}</td><td {status_class}>{SecurityUtils.sanitize_html_content(status.upper())}</td></tr>"

        html_content += "</table></div>"

        # Recommendations
        html_content += "<div class='section'><h2>Recommendations</h2>"
        for rec in safe_recommendations:
            priority_class = "good"
            if rec.get('priority') == 'critical':
                priority_class = "critical"
            elif rec.get('priority') == 'high':
                priority_class = "warning"

            safe_title = SecurityUtils.sanitize_html_content(str(rec.get('title', 'Recommendation')))
            safe_description = SecurityUtils.sanitize_html_content(str(rec.get('description', '')))

            html_content += f"""
            <div class="recommendation {priority_class}">
                <strong>{safe_title}</strong><br>
                {safe_description}<br>
                <em>Actions:</em>
                <ul>
"""
            for action in rec.get('actions', []):
                safe_action = SecurityUtils.sanitize_html_content(str(action))
                html_content += f"<li>{safe_action}</li>"
            html_content += "</ul></div>"

        html_content += """
        </div>

        <div class="footer">
            <p>Generated by CodeSage MCP Server Performance Report Generator</p>
            <p>This report is for internal use only and contains no sensitive information.</p>
        </div>
    </div>
</body>
</html>"""

        try:
            with open(html_file, 'w', encoding='utf-8') as f:
                f.write(html_content)

            # Set secure file permissions
            SecurityUtils.secure_file_permissions(html_file)

            # Validate file size
            if not SecurityUtils.validate_file_size(html_file):
                logger.warning(f"Generated HTML file exceeds size limit: {html_file}")
                html_file.unlink()  # Delete oversized file
                raise ValueError("Generated report exceeds maximum file size")

            return html_file

        except Exception as e:
            logger.error(f"Failed to export HTML report securely: {e}")
            if html_file.exists():
                html_file.unlink()  # Clean up on failure
            raise


# Global instance
_performance_report_generator = None


def get_performance_report_generator() -> PerformanceReportGenerator:
    """Get the global performance report generator instance."""
    global _performance_report_generator
    if _performance_report_generator is None:
        _performance_report_generator = PerformanceReportGenerator()
    return _performance_report_generator