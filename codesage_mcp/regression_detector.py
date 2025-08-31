"""
Performance Regression Detection System for CodeSage MCP Server.

This module provides comprehensive regression detection capabilities including
statistical analysis, configurable thresholds, alerting integration, and
automated rollback/issue creation mechanisms.
"""

import json
import os
import time
import asyncio
import statistics
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
from datetime import datetime
import scipy.stats as stats

from .logging_config import get_logger
from .error_reporting import error_reporter
from .config import get_optional_env_var


@dataclass
class RegressionConfig:
    """Configuration for regression detection."""
    # Statistical thresholds
    significance_level: float = 0.05  # p-value threshold for statistical significance
    minimum_sample_size: int = 5      # Minimum samples for statistical tests

    # Performance thresholds (percentage changes)
    latency_regression_threshold: float = 10.0    # 10% increase in latency
    throughput_regression_threshold: float = 15.0  # 15% decrease in throughput
    memory_regression_threshold: float = 20.0     # 20% increase in memory usage
    error_rate_regression_threshold: float = 5.0   # 5% absolute increase in error rate
    cache_hit_regression_threshold: float = 10.0   # 10% decrease in cache hit rate

    # Alerting configuration
    enable_alerting: bool = True
    alert_cooldown_minutes: int = 30  # Minimum time between alerts for same issue

    # Rollback configuration
    enable_auto_rollback: bool = False
    rollback_trigger_severity: str = "critical"  # critical, high, medium, low

    # Issue creation
    enable_github_issues: bool = False
    github_token: Optional[str] = None
    github_repo: Optional[str] = None

    # Baseline management
    baseline_retention_days: int = 30
    auto_update_baseline: bool = False
    baseline_update_threshold: float = 5.0  # Only update if improvement > 5%

    def __post_init__(self):
        """Load configuration from environment variables."""
        self.significance_level = float(get_optional_env_var("REGRESSION_SIGNIFICANCE_LEVEL") or self.significance_level)
        self.latency_regression_threshold = float(get_optional_env_var("LATENCY_REGRESSION_THRESHOLD") or self.latency_regression_threshold)
        self.throughput_regression_threshold = float(get_optional_env_var("THROUGHPUT_REGRESSION_THRESHOLD") or self.throughput_regression_threshold)
        self.memory_regression_threshold = float(get_optional_env_var("MEMORY_REGRESSION_THRESHOLD") or self.memory_regression_threshold)
        self.error_rate_regression_threshold = float(get_optional_env_var("ERROR_RATE_REGRESSION_THRESHOLD") or self.error_rate_regression_threshold)
        self.cache_hit_regression_threshold = float(get_optional_env_var("CACHE_HIT_REGRESSION_THRESHOLD") or self.cache_hit_regression_threshold)
        alerting_env = get_optional_env_var("ENABLE_REGRESSION_ALERTING")
        self.enable_alerting = (alerting_env or "true").lower() == "true"
        rollback_env = get_optional_env_var("ENABLE_AUTO_ROLLBACK")
        self.enable_auto_rollback = (rollback_env or "false").lower() == "true"
        issues_env = get_optional_env_var("ENABLE_GITHUB_ISSUES")
        self.enable_github_issues = (issues_env or "false").lower() == "true"
        self.github_token = get_optional_env_var("GITHUB_TOKEN")
        self.github_repo = get_optional_env_var("GITHUB_REPO")


@dataclass
class RegressionResult:
    """Result of a regression analysis."""
    metric_name: str
    baseline_value: float
    current_value: float
    percentage_change: float
    is_regression: bool
    severity: str  # critical, high, medium, low
    statistical_significance: bool
    p_value: Optional[float]
    confidence_interval: Optional[Tuple[float, float]]
    sample_size: int
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionReport:
    """Comprehensive regression detection report."""
    test_run_id: str
    timestamp: float
    regressions_detected: List[RegressionResult]
    summary: Dict[str, Any]
    recommendations: List[str]


class StatisticalAnalyzer:
    """Handles statistical analysis for regression detection."""

    def __init__(self, config: RegressionConfig):
        self.config = config
        self.logger = get_logger("statistical_analyzer")

    def perform_t_test(self, baseline_samples: List[float], current_samples: List[float]) -> Tuple[bool, float, Optional[Tuple[float, float]]]:
        """
        Perform t-test to determine if there's a statistically significant difference.

        Returns:
            Tuple of (is_significant, p_value, confidence_interval)
        """
        if len(baseline_samples) < self.config.minimum_sample_size or len(current_samples) < self.config.minimum_sample_size:
            return False, 1.0, None

        try:
            # Perform two-sample t-test
            t_stat, p_value = stats.ttest_ind(baseline_samples, current_samples, equal_var=False)

            # Calculate confidence interval for the difference
            baseline_mean = statistics.mean(baseline_samples)
            current_mean = statistics.mean(current_samples)
            diff_mean = current_mean - baseline_mean

            # Standard error of the difference
            se_diff = ((statistics.stdev(baseline_samples) ** 2 / len(baseline_samples)) +
                      (statistics.stdev(current_samples) ** 2 / len(current_samples))) ** 0.5

            # 95% confidence interval
            ci_lower = diff_mean - 1.96 * se_diff
            ci_upper = diff_mean + 1.96 * se_diff

            is_significant = p_value < self.config.significance_level

            return is_significant, p_value, (ci_lower, ci_upper)

        except Exception as e:
            self.logger.error("Statistical analysis failed", error=str(e))
            return False, 1.0, None

    def calculate_percentage_change(self, baseline: float, current: float) -> float:
        """Calculate percentage change from baseline to current."""
        if baseline == 0:
            return float('inf') if current > 0 else 0.0
        return ((current - baseline) / baseline) * 100.0


class RegressionDetector:
    """Main regression detection engine."""

    def __init__(self, config: Optional[RegressionConfig] = None):
        self.config = config or RegressionConfig()
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        self.logger = get_logger("regression_detector")
        self.baseline_dir = Path("benchmark_results/baselines")
        self.baseline_dir.mkdir(exist_ok=True)

        # Alert cooldown tracking
        self.last_alert_times: Dict[str, float] = {}

    def detect_regressions(self, current_results: Dict[str, Any], baseline_results: Optional[Dict[str, Any]] = None) -> RegressionReport:
        """
        Detect performance regressions by comparing current results against baseline.

        Args:
            current_results: Current benchmark results
            baseline_results: Baseline results (if None, loads from file)

        Returns:
            Comprehensive regression report
        """
        test_run_id = f"regression_check_{int(time.time())}"

        # Load baseline if not provided
        if baseline_results is None:
            baseline_results = self._load_baseline_results()

        if not baseline_results:
            self.logger.warning("No baseline results available for comparison")
            return RegressionReport(
                test_run_id=test_run_id,
                timestamp=time.time(),
                regressions_detected=[],
                summary={"message": "No baseline available"},
                recommendations=["Run baseline benchmarks first"]
            )

        # Analyze each metric
        regressions = []
        analyzed_metrics = 0

        for metric_name, current_data in current_results.items():
            if not isinstance(current_data, dict) or "value" not in current_data:
                continue

            baseline_data = baseline_results.get(metric_name)
            if not baseline_data or "value" not in baseline_data:
                continue

            regression_result = self._analyze_metric(metric_name, baseline_data, current_data)
            if regression_result:
                regressions.append(regression_result)
            analyzed_metrics += 1

        # Generate summary and recommendations
        summary = self._generate_summary(regressions, analyzed_metrics)
        recommendations = self._generate_recommendations(regressions)

        report = RegressionReport(
            test_run_id=test_run_id,
            timestamp=time.time(),
            regressions_detected=regressions,
            summary=summary,
            recommendations=recommendations
        )

        # Handle regressions (alerting, rollback, issue creation)
        if regressions and (self.config.enable_alerting or self.config.enable_auto_rollback or self.config.enable_github_issues):
            try:
                asyncio.create_task(self._handle_regressions_async(report))
            except RuntimeError:
                # No event loop available, skip async handling
                pass

        return report

    def _analyze_metric(self, metric_name: str, baseline_data: Dict[str, Any], current_data: Dict[str, Any]) -> Optional[RegressionResult]:
        """Analyze a single metric for regression."""
        try:
            baseline_value = baseline_data["value"]
            current_value = current_data["value"]

            # Get sample data for statistical analysis
            baseline_samples = baseline_data.get("samples", [baseline_value])
            current_samples = current_data.get("samples", [current_value])

            # Perform statistical analysis
            is_significant, p_value, confidence_interval = self.statistical_analyzer.perform_t_test(
                baseline_samples, current_samples
            )

            # Calculate percentage change
            percentage_change = self.statistical_analyzer.calculate_percentage_change(
                baseline_value, current_value
            )

            # Determine if this is a regression
            is_regression, severity = self._check_regression_criteria(
                metric_name, percentage_change, is_significant
            )

            if is_regression:
                return RegressionResult(
                    metric_name=metric_name,
                    baseline_value=baseline_value,
                    current_value=current_value,
                    percentage_change=percentage_change,
                    is_regression=True,
                    severity=severity,
                    statistical_significance=is_significant,
                    p_value=p_value,
                    confidence_interval=confidence_interval,
                    sample_size=len(current_samples),
                    timestamp=time.time(),
                    metadata={
                        "baseline_unit": baseline_data.get("unit"),
                        "current_unit": current_data.get("unit"),
                        "baseline_timestamp": baseline_data.get("timestamp"),
                        "current_timestamp": current_data.get("timestamp")
                    }
                )

        except Exception as e:
            self.logger.error(f"Failed to analyze metric {metric_name}", error=str(e))

        return None

    def _check_regression_criteria(self, metric_name: str, percentage_change: float, is_significant: bool) -> Tuple[bool, str]:
        """Check if the change meets regression criteria."""
        # Define regression criteria based on metric type
        criteria = {
            "latency": (abs(percentage_change) > self.config.latency_regression_threshold and percentage_change > 0, "high"),
            "response_time": (abs(percentage_change) > self.config.latency_regression_threshold and percentage_change > 0, "high"),
            "throughput": (abs(percentage_change) > self.config.throughput_regression_threshold and percentage_change < 0, "high"),
            "memory": (abs(percentage_change) > self.config.memory_regression_threshold and percentage_change > 0, "medium"),
            "error_rate": (abs(percentage_change) > self.config.error_rate_regression_threshold and percentage_change > 0, "critical"),
            "cache_hit_rate": (abs(percentage_change) > self.config.cache_hit_regression_threshold and percentage_change < 0, "medium"),
        }

        # Check if metric matches any criteria
        for pattern, (condition, severity) in criteria.items():
            if pattern.lower() in metric_name.lower():
                if condition and is_significant:
                    return True, severity
                break

        # Default: significant change in any direction
        if abs(percentage_change) > 5.0 and is_significant:
            return True, "low"

        return False, "none"

    def _generate_summary(self, regressions: List[RegressionResult], total_metrics: int) -> Dict[str, Any]:
        """Generate summary statistics for the regression analysis."""
        severity_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}

        for regression in regressions:
            severity_counts[regression.severity] = severity_counts.get(regression.severity, 0) + 1

        return {
            "total_metrics_analyzed": total_metrics,
            "regressions_detected": len(regressions),
            "severity_breakdown": severity_counts,
            "most_severe_regression": max(regressions, key=lambda r: ["low", "medium", "high", "critical"].index(r.severity)).severity if regressions else "none",
            "analysis_timestamp": time.time()
        }

    def _generate_recommendations(self, regressions: List[RegressionResult]) -> List[str]:
        """Generate recommendations based on detected regressions."""
        recommendations = []

        if not regressions:
            recommendations.append("No performance regressions detected. System performance is stable.")
            return recommendations

        # Group by severity
        critical_regs = [r for r in regressions if r.severity == "critical"]
        high_regs = [r for r in regressions if r.severity == "high"]

        if critical_regs:
            recommendations.append("🚨 CRITICAL: Immediate action required for critical regressions:")
            for reg in critical_regs:
                recommendations.append(f"  - {reg.metric_name}: {reg.percentage_change:.1f}% change")

        if high_regs:
            recommendations.append("⚠️ HIGH PRIORITY: Address high-priority regressions:")
            for reg in high_regs:
                recommendations.append(f"  - {reg.metric_name}: {reg.percentage_change:.1f}% change")

        # General recommendations
        recommendations.extend([
            "Investigate recent code changes that may have caused these regressions",
            "Consider rolling back recent deployments if regressions are severe",
            "Run additional benchmarks to confirm regression stability",
            "Review system configuration and resource allocation"
        ])

        return recommendations

    async def _handle_regressions_async(self, report: RegressionReport):
        """Handle detected regressions asynchronously."""
        try:
            # Send alerts
            if self.config.enable_alerting:
                await self._send_alerts(report)

            # Auto-rollback if enabled and critical regressions detected
            if self.config.enable_auto_rollback:
                critical_regs = [r for r in report.regressions_detected if r.severity == "critical"]
                if critical_regs:
                    await self._trigger_rollback(report)

            # Create GitHub issues
            if self.config.enable_github_issues:
                await self._create_github_issues(report)

        except Exception as e:
            self.logger.error("Failed to handle regressions", error=str(e))

    async def _send_alerts(self, report: RegressionReport):
        """Send alerts for detected regressions."""
        # Check cooldown
        current_time = time.time()
        last_alert = self.last_alert_times.get("regressions", 0)
        if current_time - last_alert < self.config.alert_cooldown_minutes * 60:
            return

        # Create alert message
        alert_message = self._format_alert_message(report)

        # Send via error reporter (which handles Slack, email, etc.)
        await error_reporter.report_error(
            Exception("Performance Regression Detected"),
            {
                "regression_report": {
                    "test_run_id": report.test_run_id,
                    "regressions_count": len(report.regressions_detected),
                    "severity_breakdown": report.summary.get("severity_breakdown", {}),
                    "most_severe": report.summary.get("most_severe_regression", "none")
                },
                "alert_message": alert_message
            },
            "warning"
        )

        self.last_alert_times["regressions"] = current_time

    def _format_alert_message(self, report: RegressionReport) -> str:
        """Format regression report for alerts."""
        lines = ["🚨 PERFORMANCE REGRESSION ALERT 🚨", ""]

        for regression in report.regressions_detected:
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(regression.severity, "⚪")
            lines.append(f"{emoji} {regression.metric_name}: {regression.percentage_change:+.1f}%")

        lines.extend(["", "Recommendations:"] + report.recommendations)
        return "\n".join(lines)

    async def _trigger_rollback(self, report: RegressionReport):
        """Trigger automated rollback for critical regressions."""
        self.logger.warning("Automated rollback triggered for critical regressions")

        # This would integrate with deployment system
        # For now, just log the intent
        await error_reporter.report_error(
            Exception("Automated Rollback Triggered"),
            {
                "reason": "Critical performance regressions detected",
                "regression_count": len([r for r in report.regressions_detected if r.severity == "critical"]),
                "rollback_trigger_severity": self.config.rollback_trigger_severity
            },
            "critical"
        )

    async def _create_github_issues(self, report: RegressionReport):
        """Create GitHub issues for detected regressions."""
        if not self.config.github_token or not self.config.github_repo:
            return

        try:
            import requests

            headers = {
                "Authorization": f"token {self.config.github_token}",
                "Accept": "application/vnd.github.v3+json"
            }

            for regression in report.regressions_detected:
                issue_title = f"🚨 Performance Regression: {regression.metric_name}"
                issue_body = self._format_github_issue_body(regression, report)

                payload = {
                    "title": issue_title,
                    "body": issue_body,
                    "labels": ["performance-regression", f"severity-{regression.severity}"]
                }

                response = requests.post(
                    f"https://api.github.com/repos/{self.config.github_repo}/issues",
                    headers=headers,
                    json=payload
                )

                if response.status_code == 201:
                    self.logger.info(f"Created GitHub issue for {regression.metric_name}")
                else:
                    self.logger.error(f"Failed to create GitHub issue: {response.text}")

        except Exception as e:
            self.logger.error("Failed to create GitHub issues", error=str(e))

    def _format_github_issue_body(self, regression: RegressionResult, report: RegressionReport) -> str:
        """Format regression details for GitHub issue."""
        return f"""## Performance Regression Detected

**Metric:** {regression.metric_name}
**Severity:** {regression.severity.upper()}
**Change:** {regression.percentage_change:+.1f}%
**Baseline:** {regression.baseline_value}
**Current:** {regression.current_value}

### Statistical Analysis
- **Statistically Significant:** {"Yes" if regression.statistical_significance else "No"}
- **P-value:** {regression.p_value:.4f if regression.p_value else "N/A"}
- **Sample Size:** {regression.sample_size}

### Details
- **Test Run ID:** {report.test_run_id}
- **Timestamp:** {datetime.fromtimestamp(regression.timestamp).isoformat()}

### Recommended Actions
{chr(10).join(f"- {rec}" for rec in report.recommendations)}

---
*This issue was automatically created by the CodeSage MCP regression detection system.*
"""

    def _load_baseline_results(self) -> Dict[str, Any]:
        """Load baseline results from file."""
        baseline_file = self.baseline_dir / "baseline_results.json"
        if not baseline_file.exists():
            return {}

        try:
            with open(baseline_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error("Failed to load baseline results", error=str(e))
            return {}

    def save_baseline_results(self, results: Dict[str, Any]):
        """Save results as new baseline."""
        baseline_file = self.baseline_dir / "baseline_results.json"
        try:
            with open(baseline_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            self.logger.info("Baseline results saved")
        except Exception as e:
            self.logger.error("Failed to save baseline results", error=str(e))

    def update_baseline_if_improved(self, current_results: Dict[str, Any]):
        """Update baseline if current results show significant improvements."""
        if not self.config.auto_update_baseline:
            return

        baseline_results = self._load_baseline_results()
        if not baseline_results:
            self.save_baseline_results(current_results)
            return

        # Check if current results are significantly better
        improvements = []
        for metric_name, current_data in current_results.items():
            if not isinstance(current_data, dict) or "value" not in current_data:
                continue

            baseline_data = baseline_results.get(metric_name)
            if not baseline_data or "value" not in baseline_data:
                continue

            baseline_value = baseline_data["value"]
            current_value = current_data["value"]
            percentage_change = self.statistical_analyzer.calculate_percentage_change(baseline_value, current_value)

            # For improvement metrics (lower is better for latency/error, higher for throughput/cache)
            if "latency" in metric_name.lower() or "error" in metric_name.lower():
                if percentage_change < -self.config.baseline_update_threshold:
                    improvements.append((metric_name, percentage_change))
            elif "throughput" in metric_name.lower() or "cache_hit" in metric_name.lower():
                if percentage_change > self.config.baseline_update_threshold:
                    improvements.append((metric_name, percentage_change))

        if improvements:
            self.logger.info(f"Significant improvements detected: {len(improvements)} metrics")
            self.save_baseline_results(current_results)


# Global regression detector instance
_regression_detector = None

def get_regression_detector() -> RegressionDetector:
    """Get the global regression detector instance."""
    global _regression_detector
    if _regression_detector is None:
        _regression_detector = RegressionDetector()
    return _regression_detector