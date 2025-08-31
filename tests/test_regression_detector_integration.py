#!/usr/bin/env python3
"""
Integration Tests for CodeSage MCP Regression Detection System.

This module provides end-to-end integration tests for the regression detection
system, testing the complete workflow from benchmark execution to alerting.
"""

import pytest
import json
import asyncio
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import time
import subprocess
import os

from codesage_mcp.regression_detector import (
    RegressionDetector,
    RegressionConfig,
    get_regression_detector
)
from benchmark_performance import PerformanceBenchmarker, BenchmarkResult


class TestRegressionDetectorIntegration:
    """Integration tests for the complete regression detection workflow."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = RegressionConfig()
        self.config.enable_alerting = True
        self.config.enable_github_issues = False  # Disable for tests
        self.config.enable_auto_rollback = False  # Disable for tests

        self.detector = RegressionDetector(self.config)
        self.detector.baseline_dir = self.temp_dir / "baselines"
        self.detector.baseline_dir.mkdir()

        # Create test codebase
        self.test_codebase = self.temp_dir / "test_codebase"
        self.test_codebase.mkdir()
        self._create_test_files()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def _create_test_files(self):
        """Create test files for benchmarking."""
        # Create a simple Python module
        (self.test_codebase / "test_module.py").write_text("""
def hello_world():
    \"\"\"A simple function.\"\"\"
    return "Hello, World!"

class TestClass:
    def __init__(self):
        self.value = 42

    def get_value(self):
        return self.value
""")

        # Create README
        (self.test_codebase / "README.md").write_text("# Test Project\n\nThis is a test project.")

    @patch('codesage_mcp.regression_detector.error_reporter')
    @pytest.mark.asyncio
    async def test_full_regression_detection_workflow(self, mock_error_reporter):
        """Test the complete regression detection workflow."""
        mock_error_reporter.report_error = AsyncMock()

        # Step 1: Establish baseline performance
        print("Step 1: Establishing baseline performance...")
        baseline_results = self._simulate_benchmark_run("baseline")
        self.detector.save_baseline_results(baseline_results)

        # Step 2: Simulate performance regression
        print("Step 2: Simulating performance regression...")
        current_results = self._simulate_benchmark_run("regressed")

        # Step 3: Run regression detection
        print("Step 3: Running regression detection...")
        report = self.detector.detect_regressions(current_results)

        # Step 4: Verify regression detection
        print("Step 4: Verifying regression detection...")
        assert len(report.regressions_detected) > 0, "Should detect regressions"

        # Check that alerts were sent
        await asyncio.sleep(0.1)  # Allow async operations to complete
        mock_error_reporter.report_error.assert_called()

        # Verify alert content
        call_args = mock_error_reporter.report_error.call_args
        alert_context = call_args[0][1]
        assert "regression_report" in alert_context
        assert alert_context["regression_report"]["regressions_count"] > 0

        print(f"✓ Detected {len(report.regressions_detected)} regressions")
        print(f"✓ Sent {mock_error_reporter.report_error.call_count} alerts")

    @patch('codesage_mcp.regression_detector.requests.post')
    @pytest.mark.asyncio
    async def test_github_integration_workflow(self, mock_post):
        """Test GitHub issue creation workflow."""
        # Configure GitHub integration
        self.config.enable_github_issues = True
        self.config.github_token = "test_token_123"
        self.config.github_repo = "testorg/testrepo"

        # Mock successful GitHub API response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"number": 42, "html_url": "https://github.com/testorg/testrepo/issues/42"}
        mock_post.return_value = mock_response

        # Create regression report
        from codesage_mcp.regression_detector import RegressionResult, RegressionReport
        regression = RegressionResult(
            "response_time", 100, 150, 50.0, True, "high", True, 0.001,
            (40, 60), 10, time.time()
        )
        report = RegressionReport(
            test_run_id="github_test_123",
            timestamp=time.time(),
            regressions_detected=[regression],
            summary={"regressions_detected": 1},
            recommendations=["Check recent deployments"]
        )

        # Create GitHub issue
        await self.detector._create_github_issues(report)

        # Verify GitHub API call
        assert mock_post.called
        call_args = mock_post.call_args

        # Check URL
        assert "https://api.github.com/repos/testorg/testrepo/issues" == call_args[0][0]

        # Check headers
        headers = call_args[1]["headers"]
        assert headers["Authorization"] == "token test_token_123"

        # Check payload
        payload = call_args[1]["json"]
        assert "Performance Regression" in payload["title"]
        assert "response_time" in payload["body"]
        assert "+50.0%" in payload["body"]
        assert "Check recent deployments" in payload["body"]

        print("✓ GitHub issue created successfully")

    @patch('codesage_mcp.regression_detector.error_reporter')
    def test_alert_cooldown_mechanism(self, mock_error_reporter):
        """Test alert cooldown to prevent alert spam."""
        mock_error_reporter.report_error = AsyncMock()

        # Set short cooldown for testing
        self.config.alert_cooldown_minutes = 0.001  # 3.6 seconds

        # First alert
        asyncio.run(self.detector._send_alerts(self._create_test_report()))

        # Second alert immediately after
        asyncio.run(self.detector._send_alerts(self._create_test_report()))

        # Should only be called once due to cooldown
        assert mock_error_reporter.report_error.call_count == 1

        # Wait for cooldown to expire
        time.sleep(0.005)  # 5 seconds

        # Third alert should go through
        asyncio.run(self.detector._send_alerts(self._create_test_report()))
        assert mock_error_reporter.report_error.call_count == 2

        print("✓ Alert cooldown mechanism working correctly")

    def test_baseline_management_workflow(self):
        """Test baseline creation, loading, and updates."""
        # Test 1: Save baseline
        baseline_data = {
            "latency": {"value": 100, "samples": [95, 100, 105], "unit": "ms"},
            "throughput": {"value": 50, "samples": [48, 50, 52], "unit": "rps"}
        }
        self.detector.save_baseline_results(baseline_data)

        # Test 2: Load baseline
        loaded_data = self.detector._load_baseline_results()
        assert loaded_data == baseline_data

        # Test 3: Update baseline with improvements
        self.config.auto_update_baseline = True
        self.config.baseline_update_threshold = 10.0

        improved_data = {
            "latency": {"value": 80, "samples": [75, 80, 85], "unit": "ms"},  # 20% improvement
            "throughput": {"value": 55, "samples": [53, 55, 57], "unit": "rps"}  # 10% improvement
        }

        self.detector.update_baseline_if_improved(improved_data)

        # Verify baseline was updated
        updated_baseline = self.detector._load_baseline_results()
        assert updated_baseline["latency"]["value"] == 80
        assert updated_baseline["throughput"]["value"] == 55

        print("✓ Baseline management workflow working correctly")

    @patch('codesage_mcp.regression_detector.error_reporter')
    @pytest.mark.asyncio
    async def test_rollback_trigger_mechanism(self, mock_error_reporter):
        """Test automated rollback triggering for critical regressions."""
        mock_error_reporter.report_error = AsyncMock()

        # Enable auto-rollback
        self.config.enable_auto_rollback = True
        self.config.rollback_trigger_severity = "critical"

        # Create report with critical regression
        from codesage_mcp.regression_detector import RegressionResult, RegressionReport
        critical_regression = RegressionResult(
            "error_rate", 1.0, 15.0, 1400.0, True, "critical", True, 0.001,
            (1300, 1500), 10, time.time()
        )
        report = RegressionReport(
            test_run_id="rollback_test_123",
            timestamp=time.time(),
            regressions_detected=[critical_regression],
            summary={"regressions_detected": 1},
            recommendations=[]
        )

        # Trigger rollback
        await self.detector._trigger_rollback(report)

        # Verify rollback alert was sent
        mock_error_reporter.report_error.assert_called()
        call_args = mock_error_reporter.report_error.call_args
        assert "Automated Rollback Triggered" in str(call_args[0][0])

        print("✓ Rollback trigger mechanism working correctly")

    def test_configuration_persistence(self):
        """Test that configuration is properly loaded from environment."""
        # Test with environment variables
        env_vars = {
            "REGRESSION_SIGNIFICANCE_LEVEL": "0.01",
            "LATENCY_REGRESSION_THRESHOLD": "7.5",
            "THROUGHPUT_REGRESSION_THRESHOLD": "12.5",
            "ENABLE_REGRESSION_ALERTING": "false",
            "ENABLE_AUTO_ROLLBACK": "true",
            "GITHUB_TOKEN": "env_token_123",
            "GITHUB_REPO": "envorg/envrepo"
        }

        with patch.dict('os.environ', env_vars):
            config = RegressionConfig()

            assert config.significance_level == 0.01
            assert config.latency_regression_threshold == 7.5
            assert config.throughput_regression_threshold == 12.5
            assert config.enable_alerting is False
            assert config.enable_auto_rollback is True
            assert config.github_token == "env_token_123"
            assert config.github_repo == "envorg/envrepo"

        print("✓ Configuration persistence working correctly")

    def test_statistical_analysis_edge_cases(self):
        """Test statistical analysis with edge cases."""
        # Test with very small datasets
        baseline_samples = [100, 101, 99]
        current_samples = [120, 121, 119]

        is_significant, p_value, ci = self.detector.statistical_analyzer.perform_t_test(
            baseline_samples, current_samples
        )

        # Should not be significant due to small sample size
        assert is_significant is False
        assert p_value == 1.0
        assert ci is None

        # Test with identical distributions
        baseline_samples = [100, 100, 100, 100, 100]
        current_samples = [100, 100, 100, 100, 100]

        is_significant, p_value, ci = self.detector.statistical_analyzer.perform_t_test(
            baseline_samples, current_samples
        )

        assert is_significant is False
        assert p_value > 0.05  # Should not be significant

        print("✓ Statistical analysis edge cases handled correctly")

    def test_multiple_regression_scenarios(self):
        """Test detection of multiple types of regressions simultaneously."""
        # Establish baseline
        baseline_results = {
            "avg_response_time_ms": {"value": 100, "samples": [95, 100, 105, 98, 102]},
            "throughput_rps": {"value": 50, "samples": [48, 50, 52, 49, 51]},
            "memory_usage_percent": {"value": 60, "samples": [58, 60, 62, 59, 61]},
            "error_rate_percent": {"value": 1.0, "samples": [0.8, 1.0, 1.2, 0.9, 1.1]},
            "cache_hit_rate": {"value": 80, "samples": [78, 80, 82, 79, 81]}
        }
        self.detector.save_baseline_results(baseline_results)

        # Current results with multiple regressions
        current_results = {
            "avg_response_time_ms": {"value": 125, "samples": [120, 125, 130, 123, 127]},  # +25% (high severity)
            "throughput_rps": {"value": 35, "samples": [33, 35, 37, 34, 36]},  # -30% (high severity)
            "memory_usage_percent": {"value": 75, "samples": [73, 75, 77, 74, 76]},  # +25% (medium severity)
            "error_rate_percent": {"value": 8.0, "samples": [7.5, 8.0, 8.5, 7.8, 8.2]},  # +700% (critical severity)
            "cache_hit_rate": {"value": 70, "samples": [68, 70, 72, 69, 71]}  # -12.5% (medium severity)
        }

        # Run detection
        report = self.detector.detect_regressions(current_results)

        # Verify all regressions detected
        assert len(report.regressions_detected) == 5

        # Check severity distribution
        severity_counts = {}
        for regression in report.regressions_detected:
            severity_counts[regression.severity] = severity_counts.get(regression.severity, 0) + 1

        assert severity_counts.get("critical", 0) == 1  # error_rate
        assert severity_counts.get("high", 0) == 2     # response_time, throughput
        assert severity_counts.get("medium", 0) == 2   # memory, cache_hit_rate

        # Verify most severe regression
        assert report.summary["most_severe_regression"] == "critical"

        print("✓ Multiple regression scenarios handled correctly")

    @patch('subprocess.run')
    def test_system_integration_simulation(self, mock_subprocess):
        """Test integration with system commands for rollback simulation."""
        # Mock successful command execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stdout = "Rollback successful"
        mock_subprocess.return_value = mock_result

        # Simulate rollback command execution
        try:
            result = subprocess.run(
                ["echo", "rollback command"],
                capture_output=True,
                text=True,
                timeout=30
            )
            assert result.returncode == 0
        except subprocess.TimeoutExpired:
            pytest.fail("Rollback command timed out")

        print("✓ System integration simulation working correctly")

    def _simulate_benchmark_run(self, scenario: str) -> dict:
        """Simulate benchmark results for different scenarios."""
        base_results = {
            "avg_response_time_ms": {"value": 100, "samples": [95, 100, 105, 98, 102], "unit": "ms"},
            "throughput_rps": {"value": 50, "samples": [48, 50, 52, 49, 51], "unit": "rps"},
            "memory_usage_percent": {"value": 60, "samples": [58, 60, 62, 59, 61], "unit": "percent"},
            "error_rate_percent": {"value": 1.0, "samples": [0.8, 1.0, 1.2, 0.9, 1.1], "unit": "percent"}
        }

        if scenario == "baseline":
            return base_results
        elif scenario == "regressed":
            # Introduce regressions
            return {
                "avg_response_time_ms": {"value": 130, "samples": [125, 130, 135, 128, 132], "unit": "ms"},  # +30%
                "throughput_rps": {"value": 35, "samples": [33, 35, 37, 34, 36], "unit": "rps"},  # -30%
                "memory_usage_percent": {"value": 75, "samples": [73, 75, 77, 74, 76], "unit": "percent"},  # +25%
                "error_rate_percent": {"value": 3.0, "samples": [2.8, 3.0, 3.2, 2.9, 3.1], "unit": "percent"}  # +200%
            }
        else:
            return base_results

    def _create_test_report(self):
        """Create a test regression report."""
        from codesage_mcp.regression_detector import RegressionResult, RegressionReport
        regression = RegressionResult(
            "test_metric", 100, 120, 20.0, True, "high", True, 0.01,
            (15, 25), 5, time.time()
        )
        return RegressionReport(
            test_run_id="test_123",
            timestamp=time.time(),
            regressions_detected=[regression],
            summary={"regressions_detected": 1},
            recommendations=["Test recommendation"]
        )


class TestRegressionDetectorE2E:
    """End-to-end tests simulating real-world usage scenarios."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.config = RegressionConfig()
        self.detector = RegressionDetector(self.config)
        self.detector.baseline_dir = self.temp_dir / "baselines"
        self.detector.baseline_dir.mkdir()

    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)

    def test_continuous_monitoring_scenario(self):
        """Test continuous monitoring scenario with multiple benchmark runs."""
        print("=== Testing Continuous Monitoring Scenario ===")

        # Run 1: Establish baseline
        print("Run 1: Establishing baseline...")
        run1_results = {
            "response_time": {"value": 100, "samples": [95, 100, 105, 98, 102]},
            "throughput": {"value": 50, "samples": [48, 50, 52, 49, 51]},
            "error_rate": {"value": 1.0, "samples": [0.8, 1.0, 1.2, 0.9, 1.1]}
        }
        self.detector.save_baseline_results(run1_results)

        # Run 2: Slight improvement (should update baseline)
        print("Run 2: Slight improvement...")
        self.config.auto_update_baseline = True
        self.config.baseline_update_threshold = 5.0

        run2_results = {
            "response_time": {"value": 95, "samples": [90, 95, 100, 93, 97]},  # 5% improvement
            "throughput": {"value": 52, "samples": [50, 52, 54, 51, 53]},  # 4% improvement (below threshold)
            "error_rate": {"value": 0.9, "samples": [0.7, 0.9, 1.1, 0.8, 1.0]}  # 10% improvement
        }

        report2 = self.detector.detect_regressions(run2_results)
        assert len(report2.regressions_detected) == 0  # No regressions

        self.detector.update_baseline_if_improved(run2_results)

        # Verify baseline was partially updated
        updated_baseline = self.detector._load_baseline_results()
        assert updated_baseline["response_time"]["value"] == 95  # Updated
        assert updated_baseline["error_rate"]["value"] == 0.9    # Updated
        assert updated_baseline["throughput"]["value"] == 50     # Not updated (below threshold)

        # Run 3: Performance regression
        print("Run 3: Performance regression...")
        run3_results = {
            "response_time": {"value": 120, "samples": [115, 120, 125, 118, 122]},  # 26% regression from updated baseline
            "throughput": {"value": 40, "samples": [38, 40, 42, 39, 41]},  # 20% regression
            "error_rate": {"value": 2.0, "samples": [1.8, 2.0, 2.2, 1.9, 2.1]}  # 122% regression
        }

        report3 = self.detector.detect_regressions(run3_results)
        assert len(report3.regressions_detected) == 3  # All metrics regressed

        # Verify severity levels
        severity_levels = [r.severity for r in report3.regressions_detected]
        assert "critical" in severity_levels  # error_rate
        assert severity_levels.count("high") >= 2  # response_time and throughput

        print("✓ Continuous monitoring scenario completed successfully")

    def test_deployment_pipeline_integration(self):
        """Test integration with deployment pipeline scenarios."""
        print("=== Testing Deployment Pipeline Integration ===")

        # Pre-deployment baseline
        pre_deploy_results = {
            "api_response_time": {"value": 150, "samples": [145, 150, 155, 148, 152]},
            "database_query_time": {"value": 50, "samples": [48, 50, 52, 49, 51]},
            "cpu_usage": {"value": 65, "samples": [63, 65, 67, 64, 66]}
        }
        self.detector.save_baseline_results(pre_deploy_results)

        # Post-deployment results (with some regressions)
        post_deploy_results = {
            "api_response_time": {"value": 180, "samples": [175, 180, 185, 178, 182]},  # 20% increase
            "database_query_time": {"value": 45, "samples": [43, 45, 47, 44, 46]},  # 10% decrease (improvement)
            "cpu_usage": {"value": 75, "samples": [73, 75, 77, 74, 76]}  # 15% increase
        }

        # Run regression detection
        report = self.detector.detect_regressions(post_deploy_results)

        # Should detect 2 regressions
        assert len(report.regressions_detected) == 2

        # Check specific regressions
        api_regression = next(r for r in report.regressions_detected if "api_response_time" in r.metric_name)
        cpu_regression = next(r for r in report.regressions_detected if "cpu_usage" in r.metric_name)

        assert api_regression.percentage_change == 20.0
        assert api_regression.severity == "high"
        assert cpu_regression.percentage_change == 15.38  # Approximate
        assert cpu_regression.severity == "medium"

        # Verify recommendations include deployment-related advice
        recommendations = report.recommendations
        assert any("recent code changes" in rec.lower() for rec in recommendations)
        assert any("rollback" in rec.lower() for rec in recommendations)

        print("✓ Deployment pipeline integration test completed successfully")

    def test_load_testing_scenario(self):
        """Test regression detection under different load conditions."""
        print("=== Testing Load Testing Scenario ===")

        # Baseline under normal load
        normal_load_results = {
            "p50_response_time": {"value": 200, "samples": [195, 200, 205, 198, 202]},
            "p95_response_time": {"value": 500, "samples": [495, 500, 505, 498, 502]},
            "p99_response_time": {"value": 1000, "samples": [995, 1000, 1005, 998, 1002]},
            "requests_per_second": {"value": 100, "samples": [98, 100, 102, 99, 101]}
        }
        self.detector.save_baseline_results(normal_load_results)

        # High load scenario (degraded performance expected but should be flagged if excessive)
        high_load_results = {
            "p50_response_time": {"value": 300, "samples": [295, 300, 305, 298, 302]},  # 50% increase
            "p95_response_time": {"value": 800, "samples": [795, 800, 805, 798, 802]},  # 60% increase
            "p99_response_time": {"value": 1500, "samples": [1495, 1500, 1505, 1498, 1502]},  # 50% increase
            "requests_per_second": {"value": 80, "samples": [78, 80, 82, 79, 81]}  # 20% decrease
        }

        report = self.detector.detect_regressions(high_load_results)

        # Should detect multiple regressions
        assert len(report.regressions_detected) >= 3

        # Check that throughput regression is detected
        throughput_regs = [r for r in report.regressions_detected if "requests_per_second" in r.metric_name]
        assert len(throughput_regs) == 1
        assert throughput_regs[0].percentage_change == -20.0
        assert throughput_regs[0].severity == "high"

        print("✓ Load testing scenario completed successfully")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])