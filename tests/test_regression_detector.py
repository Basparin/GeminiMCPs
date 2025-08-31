#!/usr/bin/env python3
"""
Unit Tests for CodeSage MCP Regression Detection System.

This module provides comprehensive unit tests for the regression detection
functionality, including statistical analysis, configuration, and alerting.
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import time
import statistics
from dataclasses import asdict

from codesage_mcp.regression_detector import (
    RegressionConfig,
    RegressionResult,
    RegressionReport,
    StatisticalAnalyzer,
    RegressionDetector,
    get_regression_detector
)


class TestRegressionConfig:
    """Test cases for RegressionConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = RegressionConfig()

        assert config.significance_level == 0.05
        assert config.latency_regression_threshold == 10.0
        assert config.throughput_regression_threshold == 15.0
        assert config.enable_alerting is True
        assert config.enable_auto_rollback is False
        assert config.enable_github_issues is False

    @patch.dict('os.environ', {
        'REGRESSION_SIGNIFICANCE_LEVEL': '0.01',
        'LATENCY_REGRESSION_THRESHOLD': '5.0',
        'ENABLE_REGRESSION_ALERTING': 'false'
    })
    def test_config_from_environment(self):
        """Test loading configuration from environment variables."""
        config = RegressionConfig()

        assert config.significance_level == 0.01
        assert config.latency_regression_threshold == 5.0
        assert config.enable_alerting is False


class TestStatisticalAnalyzer:
    """Test cases for StatisticalAnalyzer."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RegressionConfig()
        self.analyzer = StatisticalAnalyzer(self.config)

    def test_calculate_percentage_change(self):
        """Test percentage change calculation."""
        # Normal case
        assert self.analyzer.calculate_percentage_change(100, 110) == 10.0
        assert self.analyzer.calculate_percentage_change(100, 90) == -10.0

        # Edge case: baseline is zero
        assert self.analyzer.calculate_percentage_change(0, 10) == float('inf')
        assert self.analyzer.calculate_percentage_change(0, 0) == 0.0

    def test_perform_t_test_insufficient_samples(self):
        """Test t-test with insufficient sample size."""
        baseline = [1, 2, 3]  # Less than minimum_sample_size (5)
        current = [4, 5, 6]

        is_significant, p_value, ci = self.analyzer.perform_t_test(baseline, current)

        assert is_significant is False
        assert p_value == 1.0
        assert ci is None

    def test_perform_t_test_significant_difference(self):
        """Test t-test with significant difference."""
        baseline = [10, 10, 10, 10, 10]  # Mean = 10
        current = [15, 15, 15, 15, 15]   # Mean = 15, clear difference

        is_significant, p_value, ci = self.analyzer.perform_t_test(baseline, current)

        assert is_significant is True
        assert p_value < 0.05
        assert ci is not None
        assert len(ci) == 2

    def test_perform_t_test_no_significant_difference(self):
        """Test t-test with no significant difference."""
        baseline = [10.1, 9.9, 10.0, 10.2, 9.8]  # Mean ≈ 10
        current = [10.0, 10.1, 9.9, 10.1, 10.0]   # Mean ≈ 10

        is_significant, p_value, ci = self.analyzer.perform_t_test(baseline, current)

        assert is_significant is False
        assert p_value >= 0.05


class TestRegressionDetector:
    """Test cases for RegressionDetector."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RegressionConfig()
        self.detector = RegressionDetector(self.config)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.detector.baseline_dir = self.temp_dir / "baselines"
        self.detector.baseline_dir.mkdir()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_detect_regressions_no_baseline(self):
        """Test regression detection with no baseline available."""
        current_results = {"latency": {"value": 100}}

        report = self.detector.detect_regressions(current_results)

        assert isinstance(report, RegressionReport)
        assert len(report.regressions_detected) == 0
        assert "No baseline available" in report.summary.get("message", "")

    def test_detect_regressions_with_baseline(self):
        """Test regression detection with baseline data."""
        # Create baseline
        baseline_data = {
            "avg_response_time_ms": {"value": 100.0, "samples": [95, 100, 105, 98, 102]},
            "throughput_rps": {"value": 50.0, "samples": [48, 50, 52, 49, 51]},
            "memory_usage_percent": {"value": 60.0, "samples": [58, 60, 62, 59, 61]}
        }
        self._save_baseline(baseline_data)

        # Current results with regression in latency
        current_results = {
            "avg_response_time_ms": {"value": 120.0, "samples": [115, 120, 125, 118, 122]},  # 20% increase
            "throughput_rps": {"value": 48.0, "samples": [46, 48, 50, 47, 49]},  # 4% decrease
            "memory_usage_percent": {"value": 65.0, "samples": [63, 65, 67, 64, 66]}  # 8.3% increase
        }

        report = self.detector.detect_regressions(current_results)

        assert len(report.regressions_detected) == 1  # Only latency should be flagged
        regression = report.regressions_detected[0]
        assert regression.metric_name == "avg_response_time_ms"
        assert regression.percentage_change == 20.0
        assert regression.is_regression is True
        assert regression.severity == "high"

    def test_check_regression_criteria_latency(self):
        """Test regression criteria checking for latency metrics."""
        # Significant increase in latency
        is_regression, severity = self.detector._check_regression_criteria(
            "avg_response_time_ms", 15.0, True
        )
        assert is_regression is True
        assert severity == "high"

        # Insignificant change
        is_regression, severity = self.detector._check_regression_criteria(
            "avg_response_time_ms", 5.0, True
        )
        assert is_regression is False

        # Decrease in latency (improvement)
        is_regression, severity = self.detector._check_regression_criteria(
            "avg_response_time_ms", -10.0, True
        )
        assert is_regression is False

    def test_check_regression_criteria_throughput(self):
        """Test regression criteria checking for throughput metrics."""
        # Significant decrease in throughput
        is_regression, severity = self.detector._check_regression_criteria(
            "throughput_rps", -20.0, True
        )
        assert is_regression is True
        assert severity == "high"

        # Increase in throughput (improvement)
        is_regression, severity = self.detector._check_regression_criteria(
            "throughput_rps", 10.0, True
        )
        assert is_regression is False

    def test_check_regression_criteria_error_rate(self):
        """Test regression criteria checking for error rate metrics."""
        # Significant increase in error rate
        is_regression, severity = self.detector._check_regression_criteria(
            "error_rate_percent", 8.0, True
        )
        assert is_regression is True
        assert severity == "critical"

    def test_generate_summary(self):
        """Test summary generation."""
        regressions = [
            RegressionResult("latency", 100, 120, 20.0, True, "high", True, 0.01, (15, 25), 5, time.time()),
            RegressionResult("throughput", 50, 45, -10.0, True, "medium", True, 0.03, (-15, -5), 5, time.time()),
            RegressionResult("memory", 60, 63, 5.0, True, "low", True, 0.04, (2, 8), 5, time.time())
        ]

        summary = self.detector._generate_summary(regressions, 10)

        assert summary["total_metrics_analyzed"] == 10
        assert summary["regressions_detected"] == 3
        assert summary["severity_breakdown"]["high"] == 1
        assert summary["severity_breakdown"]["medium"] == 1
        assert summary["severity_breakdown"]["low"] == 1
        assert summary["most_severe_regression"] == "high"

    def test_generate_recommendations_no_regressions(self):
        """Test recommendations when no regressions detected."""
        recommendations = self.detector._generate_recommendations([])

        assert len(recommendations) == 1
        assert "No performance regressions detected" in recommendations[0]

    def test_generate_recommendations_with_regressions(self):
        """Test recommendations when regressions are detected."""
        regressions = [
            RegressionResult("error_rate", 1.0, 8.0, 700.0, True, "critical", True, 0.001, (6, 10), 5, time.time()),
            RegressionResult("latency", 100, 120, 20.0, True, "high", True, 0.01, (15, 25), 5, time.time())
        ]

        recommendations = self.detector._generate_recommendations(regressions)

        assert len(recommendations) > 1
        assert any("CRITICAL" in rec for rec in recommendations)
        assert any("HIGH PRIORITY" in rec for rec in recommendations)

    def test_save_and_load_baseline(self):
        """Test saving and loading baseline results."""
        baseline_data = {
            "latency": {"value": 100, "unit": "ms"},
            "throughput": {"value": 50, "unit": "rps"}
        }

        # Save baseline
        self.detector.save_baseline_results(baseline_data)

        # Load baseline
        loaded_data = self.detector._load_baseline_results()

        assert loaded_data == baseline_data

    def test_update_baseline_if_improved(self):
        """Test automatic baseline updates for improvements."""
        # Enable auto-update
        self.config.auto_update_baseline = True
        self.config.baseline_update_threshold = 5.0

        # Create baseline
        baseline_data = {
            "latency": {"value": 100.0, "samples": [95, 100, 105]},
            "throughput": {"value": 50.0, "samples": [48, 50, 52]}
        }
        self._save_baseline(baseline_data)

        # Current results with significant improvement in latency
        current_results = {
            "latency": {"value": 85.0, "samples": [80, 85, 90]},  # 15% improvement
            "throughput": {"value": 52.0, "samples": [50, 52, 54]}  # 4% improvement (below threshold)
        }

        self.detector.update_baseline_if_improved(current_results)

        # Check if baseline was updated (only latency should be updated)
        loaded_baseline = self.detector._load_baseline_results()
        assert loaded_baseline["latency"]["value"] == 85.0
        assert loaded_baseline["throughput"]["value"] == 50.0  # Should remain unchanged

    def _save_baseline(self, data):
        """Helper to save baseline data."""
        baseline_file = self.detector.baseline_dir / "baseline_results.json"
        with open(baseline_file, 'w') as f:
            json.dump(data, f)

    @patch('codesage_mcp.regression_detector.error_reporter')
    @pytest.mark.asyncio
    async def test_send_alerts(self, mock_error_reporter):
        """Test sending alerts for regressions."""
        mock_error_reporter.report_error = AsyncMock()

        # Create a regression report
        regression = RegressionResult(
            "latency", 100, 120, 20.0, True, "high", True, 0.01, (15, 25), 5, time.time()
        )
        report = RegressionReport(
            test_run_id="test_123",
            timestamp=time.time(),
            regressions_detected=[regression],
            summary={"severity_breakdown": {"high": 1}},
            recommendations=["Test recommendation"]
        )

        await self.detector._send_alerts(report)

        # Verify alert was sent
        mock_error_reporter.report_error.assert_called_once()
        call_args = mock_error_reporter.report_error.call_args
        assert "Performance Regression Detected" in str(call_args[0][0])
        assert call_args[0][1]["regression_report"]["regressions_count"] == 1

    @patch('codesage_mcp.regression_detector.requests.post')
    @pytest.mark.asyncio
    async def test_create_github_issues(self, mock_post):
        """Test creating GitHub issues for regressions."""
        # Configure GitHub integration
        self.config.enable_github_issues = True
        self.config.github_token = "test_token"
        self.config.github_repo = "test/repo"

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"number": 123}
        mock_post.return_value = mock_response

        # Create a regression report
        regression = RegressionResult(
            "latency", 100, 120, 20.0, True, "high", True, 0.01, (15, 25), 5, time.time()
        )
        report = RegressionReport(
            test_run_id="test_123",
            timestamp=time.time(),
            regressions_detected=[regression],
            summary={},
            recommendations=[]
        )

        await self.detector._create_github_issues(report)

        # Verify GitHub API was called
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        assert "https://api.github.com/repos/test/repo/issues" in call_args[0][0]
        assert call_args[1]["headers"]["Authorization"] == "token test_token"
        assert "Performance Regression" in call_args[1]["json"]["title"]

    def test_format_alert_message(self):
        """Test formatting alert messages."""
        regression = RegressionResult(
            "latency", 100, 120, 20.0, True, "high", True, 0.01, (15, 25), 5, time.time()
        )
        report = RegressionReport(
            test_run_id="test_123",
            timestamp=time.time(),
            regressions_detected=[regression],
            summary={},
            recommendations=["Check recent changes", "Consider rollback"]
        )

        message = self.detector._format_alert_message(report)

        assert "PERFORMANCE REGRESSION ALERT" in message
        assert "+20.0%" in message
        assert "Check recent changes" in message
        assert "Consider rollback" in message

    def test_format_github_issue_body(self):
        """Test formatting GitHub issue body."""
        regression = RegressionResult(
            "latency", 100, 120, 20.0, True, "high", True, 0.01, (15, 25), 5, time.time()
        )
        report = RegressionReport(
            test_run_id="test_123",
            timestamp=time.time(),
            regressions_detected=[regression],
            summary={},
            recommendations=["Investigate code changes"]
        )

        body = self.detector._format_github_issue_body(regression, report)

        assert "## Performance Regression Detected" in body
        assert "latency" in body
        assert "HIGH" in body
        assert "+20.0%" in body
        assert "Investigate code changes" in body
        assert "automatically created" in body


class TestRegressionResult:
    """Test cases for RegressionResult dataclass."""

    def test_regression_result_creation(self):
        """Test creating a RegressionResult instance."""
        result = RegressionResult(
            metric_name="latency",
            baseline_value=100.0,
            current_value=120.0,
            percentage_change=20.0,
            is_regression=True,
            severity="high",
            statistical_significance=True,
            p_value=0.01,
            confidence_interval=(15.0, 25.0),
            sample_size=5,
            timestamp=time.time()
        )

        assert result.metric_name == "latency"
        assert result.baseline_value == 100.0
        assert result.current_value == 120.0
        assert result.percentage_change == 20.0
        assert result.is_regression is True
        assert result.severity == "high"
        assert result.statistical_significance is True
        assert result.p_value == 0.01
        assert result.confidence_interval == (15.0, 25.0)
        assert result.sample_size == 5


class TestRegressionReport:
    """Test cases for RegressionReport dataclass."""

    def test_regression_report_creation(self):
        """Test creating a RegressionReport instance."""
        regression = RegressionResult(
            "latency", 100, 120, 20.0, True, "high", True, 0.01, (15, 25), 5, time.time()
        )

        report = RegressionReport(
            test_run_id="test_123",
            timestamp=time.time(),
            regressions_detected=[regression],
            summary={"total_metrics": 5, "regressions": 1},
            recommendations=["Check code changes"]
        )

        assert report.test_run_id == "test_123"
        assert len(report.regressions_detected) == 1
        assert report.summary["regressions"] == 1
        assert len(report.recommendations) == 1


class TestGlobalFunctions:
    """Test cases for global functions."""

    def test_get_regression_detector(self):
        """Test getting the global regression detector instance."""
        detector1 = get_regression_detector()
        detector2 = get_regression_detector()

        # Should return the same instance
        assert detector1 is detector2
        assert isinstance(detector1, RegressionDetector)


class TestIntegrationScenarios:
    """Integration test scenarios for regression detection."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = RegressionConfig()
        self.detector = RegressionDetector(self.config)
        self.temp_dir = Path(tempfile.mkdtemp())
        self.detector.baseline_dir = self.temp_dir / "baselines"
        self.detector.baseline_dir.mkdir()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_end_to_end_regression_detection(self):
        """Test complete regression detection workflow."""
        # Step 1: Establish baseline
        baseline_results = {
            "avg_response_time_ms": {"value": 100.0, "samples": [95, 100, 105, 98, 102], "unit": "ms"},
            "throughput_rps": {"value": 50.0, "samples": [48, 50, 52, 49, 51], "unit": "rps"},
            "error_rate_percent": {"value": 1.0, "samples": [0.8, 1.0, 1.2, 0.9, 1.1], "unit": "percent"}
        }
        self.detector.save_baseline_results(baseline_results)

        # Step 2: Run current benchmarks (with regressions)
        current_results = {
            "avg_response_time_ms": {"value": 125.0, "samples": [120, 125, 130, 123, 127], "unit": "ms"},  # 25% increase
            "throughput_rps": {"value": 40.0, "samples": [38, 40, 42, 39, 41], "unit": "rps"},  # 20% decrease
            "error_rate_percent": {"value": 1.5, "samples": [1.3, 1.5, 1.7, 1.4, 1.6], "unit": "percent"}  # 50% increase
        }

        # Step 3: Detect regressions
        report = self.detector.detect_regressions(current_results)

        # Step 4: Verify results
        assert len(report.regressions_detected) == 3  # All three should be flagged

        # Check specific regressions
        latency_reg = next(r for r in report.regressions_detected if "response_time" in r.metric_name)
        throughput_reg = next(r for r in report.regressions_detected if "throughput" in r.metric_name)
        error_reg = next(r for r in report.regressions_detected if "error_rate" in r.metric_name)

        assert latency_reg.severity == "high"
        assert throughput_reg.severity == "high"
        assert error_reg.severity == "critical"

        # Check summary
        assert report.summary["regressions_detected"] == 3
        assert report.summary["most_severe_regression"] == "critical"

        # Check recommendations
        assert len(report.recommendations) > 0
        assert any("CRITICAL" in rec for rec in report.recommendations)

    def test_baseline_update_scenario(self):
        """Test baseline update when improvements are detected."""
        # Enable auto-update
        self.config.auto_update_baseline = True
        self.config.baseline_update_threshold = 10.0

        # Initial baseline
        baseline_results = {
            "latency": {"value": 100.0, "samples": [95, 100, 105]},
            "throughput": {"value": 50.0, "samples": [48, 50, 52]}
        }
        self.detector.save_baseline_results(baseline_results)

        # Improved results
        improved_results = {
            "latency": {"value": 80.0, "samples": [75, 80, 85]},  # 20% improvement
            "throughput": {"value": 65.0, "samples": [63, 65, 67]}  # 30% improvement
        }

        # Update baseline
        self.detector.update_baseline_if_improved(improved_results)

        # Verify baseline was updated
        updated_baseline = self.detector._load_baseline_results()
        assert updated_baseline["latency"]["value"] == 80.0
        assert updated_baseline["throughput"]["value"] == 65.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])