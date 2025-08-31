"""
Integration tests for Performance Report Generation System.

This module contains end-to-end integration tests for the performance report
generation system, testing the interaction between all components including
performance monitoring, regression detection, Prometheus integration, and
report generation.
"""

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch, AsyncMock, Mock

from codesage_mcp.performance_report_generator import (
    PerformanceReportGenerator,
    ReportConfig,
    get_performance_report_generator
)
from codesage_mcp.performance_monitor import get_performance_monitor
from codesage_mcp.regression_detector import get_regression_detector
from codesage_mcp.prometheus_client import get_prometheus_client


class TestPerformanceReportIntegration(unittest.TestCase):
    """Integration tests for the complete performance report system."""

    def setUp(self):
        """Set up integration test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create test configuration
        self.config = ReportConfig(
            benchmark_results_dir=self.temp_dir,
            historical_reports_dir=self.temp_dir,
            include_charts=False,  # Disable for faster testing
            prometheus_url="http://test-prometheus:9090",
            grafana_url="http://test-grafana:3000"
        )

        # Create mock benchmark data
        self.benchmark_data = {
            "test_suite": "Integration Test Suite",
            "timestamp": time.time(),
            "results": [
                {
                    "test_name": "response_time_test",
                    "metric_name": "response_time_ms",
                    "value": 125.0,
                    "unit": "ms",
                    "target": 200.0,
                    "achieved": True,
                    "metadata": {"test_type": "latency"}
                },
                {
                    "test_name": "throughput_test",
                    "metric_name": "throughput_rps",
                    "value": 28.5,
                    "unit": "requests/sec",
                    "target": 25.0,
                    "achieved": True,
                    "metadata": {"test_type": "scalability"}
                },
                {
                    "test_name": "memory_test",
                    "metric_name": "memory_usage_percent",
                    "value": 72.3,
                    "unit": "percent",
                    "target": 80.0,
                    "achieved": True,
                    "metadata": {"test_type": "resource"}
                },
                {
                    "test_name": "cache_test",
                    "metric_name": "cache_hit_rate",
                    "value": 89.5,
                    "unit": "percent",
                    "target": 85.0,
                    "achieved": True,
                    "metadata": {"test_type": "efficiency"}
                }
            ],
            "summary": {
                "total_tests": 4,
                "passed_tests": 4,
                "failed_tests": 0,
                "performance_score": 100.0,
                "categories": {
                    "latency": {"passed": 1, "failed": 0, "total": 1},
                    "scalability": {"passed": 1, "failed": 0, "total": 1},
                    "resource": {"passed": 1, "failed": 0, "total": 1},
                    "efficiency": {"passed": 1, "failed": 0, "total": 1}
                },
                "overall_status": "PASS"
            }
        }

        # Create mock historical data
        self.historical_data = {
            "baseline": {
                "response_time_ms": {"value": 115.0, "timestamp": time.time() - 86400},
                "throughput_rps": {"value": 25.0, "timestamp": time.time() - 86400},
                "memory_usage_percent": {"value": 68.0, "timestamp": time.time() - 86400},
                "cache_hit_rate": {"value": 87.0, "timestamp": time.time() - 86400}
            },
            "historical": [
                {
                    "timestamp": time.time() - 7200,  # 2 hours ago
                    "metrics": {
                        "response_time_ms": {"value": 118.0},
                        "throughput_rps": {"value": 26.2},
                        "memory_usage_percent": {"value": 69.5},
                        "cache_hit_rate": {"value": 88.2}
                    }
                },
                {
                    "timestamp": time.time() - 3600,  # 1 hour ago
                    "metrics": {
                        "response_time_ms": {"value": 122.0},
                        "throughput_rps": {"value": 27.1},
                        "memory_usage_percent": {"value": 71.0},
                        "cache_hit_rate": {"value": 89.0}
                    }
                }
            ]
        }

    def tearDown(self):
        """Clean up integration test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _setup_test_data_files(self):
        """Set up test data files in temporary directory."""
        # Create benchmark results file
        benchmark_file = Path(self.temp_dir) / "benchmark_report_integration.json"
        with open(benchmark_file, 'w') as f:
            json.dump(self.benchmark_data, f)

        # Create baseline file
        baseline_file = Path(self.temp_dir) / "baseline_results.json"
        with open(baseline_file, 'w') as f:
            json.dump(self.historical_data["baseline"], f)

        # Create historical report files
        for i, hist_entry in enumerate(self.historical_data["historical"]):
            hist_file = Path(self.temp_dir) / f"historical_report_{i}.json"
            with open(hist_file, 'w') as f:
                json.dump({
                    "timestamp": hist_entry["timestamp"],
                    "kpi_summary": {"current_metrics": hist_entry["metrics"]}
                }, f)

    @patch('codesage_mcp.performance_report_generator.get_performance_monitor')
    @patch('codesage_mcp.performance_report_generator.get_regression_detector')
    @patch('codesage_mcp.performance_report_generator.get_prometheus_client')
    async def test_full_report_generation_workflow(self, mock_prometheus_client,
                                                 mock_regression_detector, mock_performance_monitor):
        """Test the complete report generation workflow from data collection to export."""
        # Setup test data
        self._setup_test_data_files()

        # Mock current metrics
        current_metrics = {
            "response_time_ms": {"value": 125.0, "timestamp": time.time(), "unit": "ms"},
            "throughput_rps": {"value": 28.5, "timestamp": time.time(), "unit": "requests/sec"},
            "memory_usage_percent": {"value": 72.3, "timestamp": time.time(), "unit": "percent"},
            "cpu_usage_percent": {"value": 45.2, "timestamp": time.time(), "unit": "percent"},
            "error_rate_percent": {"value": 1.2, "timestamp": time.time(), "unit": "percent"},
            "cache_hit_rate": {"value": 89.5, "timestamp": time.time(), "unit": "percent"}
        }

        # Setup mocks
        mock_pm = Mock()
        mock_pm.get_current_metrics.return_value = current_metrics

        mock_rd = Mock()
        mock_regression_report = Mock()
        mock_regression_report.test_run_id = "integration_test_run"
        mock_regression_report.timestamp = time.time()
        mock_regression_report.regressions_detected = []
        mock_regression_report.summary = {"total_metrics_analyzed": 6, "regressions_detected": 0}
        mock_regression_report.recommendations = ["System performance is stable"]
        mock_rd.detect_regressions.return_value = mock_regression_report

        mock_pc = Mock()
        mock_pc.get_current_metrics = AsyncMock(return_value={
            "prometheus_metric": {"value": 100.0, "timestamp": time.time(), "unit": "count"}
        })

        mock_performance_monitor.return_value = mock_pm
        mock_regression_detector.return_value = mock_rd
        mock_prometheus_client.return_value = mock_pc

        # Create generator and generate report
        generator = PerformanceReportGenerator(self.config)
        report = await generator.generate_comprehensive_report()

        # Verify report structure and content
        self.assertIsNotNone(report)
        self.assertIn("integration_test_", report.report_id)
        self.assertEqual(report.title, self.config.report_title)

        # Verify executive summary
        exec_summary = report.executive_summary
        self.assertIn("overall_health_score", exec_summary)
        self.assertIn("key_findings", exec_summary)
        self.assertGreater(exec_summary["overall_health_score"], 0)
        self.assertLessEqual(exec_summary["overall_health_score"], 100)

        # Verify KPI summary
        kpi_summary = report.kpi_summary
        self.assertIn("current_metrics", kpi_summary)
        self.assertIn("benchmark_kpis", kpi_summary)
        self.assertIn("kpi_status", kpi_summary)

        # Verify all expected metrics are present
        for metric in ["response_time_ms", "throughput_rps", "memory_usage_percent", "cache_hit_rate"]:
            self.assertIn(metric, kpi_summary["current_metrics"])
            self.assertIn(metric, kpi_summary["benchmark_kpis"])

        # Verify trend analysis
        trend_analysis = report.trend_analysis
        self.assertIn("metric_trends", trend_analysis)
        self.assertIn("trend_summary", trend_analysis)

        # Verify historical comparison
        hist_comparison = report.historical_comparison
        self.assertIn("baseline_comparison", hist_comparison)
        self.assertIn("performance_changes", hist_comparison)

        # Verify recommendations are generated
        self.assertIsInstance(report.recommendations, list)
        self.assertGreater(len(report.recommendations), 0)

        # Verify metadata
        self.assertIn("metadata", report.__dict__)
        metadata = report.metadata
        self.assertIn("data_sources", metadata)
        self.assertTrue(metadata["data_sources"]["performance_monitor"])
        self.assertTrue(metadata["data_sources"]["benchmark_data"])
        self.assertTrue(metadata["data_sources"]["prometheus"])

    @patch('codesage_mcp.performance_report_generator.get_performance_monitor')
    @patch('codesage_mcp.performance_report_generator.get_regression_detector')
    @patch('codesage_mcp.performance_report_generator.get_prometheus_client')
    async def test_report_generation_with_regressions(self, mock_prometheus_client,
                                                    mock_regression_detector, mock_performance_monitor):
        """Test report generation when performance regressions are detected."""
        # Setup test data
        self._setup_test_data_files()

        # Mock current metrics with performance issues
        current_metrics = {
            "response_time_ms": {"value": 350.0, "timestamp": time.time(), "unit": "ms"},  # High latency
            "throughput_rps": {"value": 15.0, "timestamp": time.time(), "unit": "requests/sec"},  # Low throughput
            "memory_usage_percent": {"value": 92.0, "timestamp": time.time(), "unit": "percent"},  # High memory
            "cpu_usage_percent": {"value": 88.0, "timestamp": time.time(), "unit": "percent"},  # High CPU
            "error_rate_percent": {"value": 8.5, "timestamp": time.time(), "unit": "percent"},  # High error rate
        }

        # Mock regression detection with critical regressions
        mock_pm = Mock()
        mock_pm.get_current_metrics.return_value = current_metrics

        mock_rd = Mock()
        mock_regression_report = Mock()
        mock_regression_report.test_run_id = "regression_test_run"
        mock_regression_report.timestamp = time.time()
        mock_regression_report.regressions_detected = [
            Mock(
                metric_name="response_time_ms",
                percentage_change=203.45,
                severity="critical",
                statistical_significance=True,
                p_value=0.001
            ),
            Mock(
                metric_name="memory_usage_percent",
                percentage_change=35.29,
                severity="high",
                statistical_significance=True,
                p_value=0.01
            ),
            Mock(
                metric_name="error_rate_percent",
                percentage_change=608.33,
                severity="critical",
                statistical_significance=True,
                p_value=0.0001
            )
        ]
        mock_regression_report.summary = {
            "total_metrics_analyzed": 5,
            "regressions_detected": 3,
            "severity_breakdown": {"critical": 2, "high": 1, "medium": 0, "low": 0}
        }
        mock_regression_report.recommendations = [
            "🚨 CRITICAL: Immediate action required for critical regressions",
            "Investigate recent code changes that may have caused these regressions",
            "Consider rolling back recent deployments"
        ]
        mock_rd.detect_regressions.return_value = mock_regression_report

        mock_pc = Mock()
        mock_pc.get_current_metrics = AsyncMock(return_value={})

        mock_performance_monitor.return_value = mock_pm
        mock_regression_detector.return_value = mock_rd
        mock_prometheus_client.return_value = mock_pc

        # Generate report
        generator = PerformanceReportGenerator(self.config)
        report = await generator.generate_comprehensive_report()

        # Verify critical issues are flagged
        exec_summary = report.executive_summary
        self.assertIn("critical_issues", exec_summary)
        self.assertGreater(len(exec_summary["critical_issues"]), 0)

        # Verify low health score due to regressions
        self.assertLess(exec_summary["overall_health_score"], 50)

        # Verify recommendations include regression mitigation
        regression_recs = [r for r in report.recommendations if "regression" in r.get("type", "")]
        self.assertGreater(len(regression_recs), 0)

        # Verify KPI status reflects issues
        kpi_status = report.kpi_summary["kpi_status"]
        self.assertEqual(kpi_status["response_time_ms"], "critical")
        self.assertEqual(kpi_status["memory_usage_percent"], "critical")
        self.assertEqual(kpi_status["error_rate_percent"], "critical")

    @patch('codesage_mcp.performance_report_generator.get_performance_monitor')
    @patch('codesage_mcp.performance_report_generator.get_regression_detector')
    @patch('codesage_mcp.performance_report_generator.get_prometheus_client')
    async def test_end_to_end_export_workflow(self, mock_prometheus_client,
                                            mock_regression_detector, mock_performance_monitor):
        """Test the complete workflow from report generation to multiple format exports."""
        # Setup test data
        self._setup_test_data_files()

        # Mock components
        mock_pm = Mock()
        mock_pm.get_current_metrics.return_value = {
            "response_time_ms": {"value": 125.0, "timestamp": time.time(), "unit": "ms"},
            "throughput_rps": {"value": 28.5, "timestamp": time.time(), "unit": "requests/sec"}
        }

        mock_rd = Mock()
        mock_regression_report = Mock()
        mock_regression_report.regressions_detected = []
        mock_regression_report.summary = {"total_metrics_analyzed": 2, "regressions_detected": 0}
        mock_regression_report.recommendations = []
        mock_rd.detect_regressions.return_value = mock_regression_report

        mock_pc = Mock()
        mock_pc.get_current_metrics = AsyncMock(return_value={})

        mock_performance_monitor.return_value = mock_pm
        mock_regression_detector.return_value = mock_rd
        mock_prometheus_client.return_value = mock_pc

        # Generate report
        generator = PerformanceReportGenerator(self.config)
        report = await generator.generate_comprehensive_report()

        # Test export to all supported formats
        export_formats = ["json", "html"]
        exported_files = await generator.export_report(report, export_formats)

        # Verify all formats were exported
        self.assertEqual(len(exported_files), len(export_formats))
        for format_type in export_formats:
            self.assertIn(format_type, exported_files)
            export_path = Path(exported_files[format_type])
            self.assertTrue(export_path.exists(), f"Export file for {format_type} was not created")

        # Verify JSON export content
        json_path = Path(exported_files["json"])
        with open(json_path, 'r') as f:
            json_data = json.load(f)

        self.assertEqual(json_data["report_id"], report.report_id)
        self.assertIn("executive_summary", json_data)
        self.assertIn("kpi_summary", json_data)
        self.assertIn("recommendations", json_data)

        # Verify HTML export content
        html_path = Path(exported_files["html"])
        with open(html_path, 'r') as f:
            html_content = f.read()

        self.assertIn(report.title, html_content)
        self.assertIn("Executive Summary", html_content)
        self.assertIn("KPI Summary", html_content)

    def test_global_generator_instance(self):
        """Test the global performance report generator instance."""
        generator1 = get_performance_report_generator()
        generator2 = get_performance_report_generator()

        # Should return the same instance
        self.assertIs(generator1, generator2)
        self.assertIsInstance(generator1, PerformanceReportGenerator)

    async def test_prometheus_integration_error_handling(self):
        """Test error handling when Prometheus is unavailable."""
        # Setup test data
        self._setup_test_data_files()

        with patch('codesage_mcp.performance_report_generator.get_performance_monitor') as mock_pm, \
             patch('codesage_mcp.performance_report_generator.get_regression_detector') as mock_rd, \
             patch('codesage_mcp.performance_report_generator.get_prometheus_client') as mock_pc:

            # Mock successful components
            mock_pm_instance = Mock()
            mock_pm_instance.get_current_metrics.return_value = {
                "response_time_ms": {"value": 125.0, "timestamp": time.time(), "unit": "ms"}
            }
            mock_pm.return_value = mock_pm_instance

            mock_rd_instance = Mock()
            mock_regression_report = Mock()
            mock_regression_report.regressions_detected = []
            mock_regression_report.summary = {"total_metrics_analyzed": 1, "regressions_detected": 0}
            mock_regression_report.recommendations = []
            mock_rd_instance.detect_regressions.return_value = mock_regression_report
            mock_rd.return_value = mock_rd_instance

            # Mock Prometheus client to raise exception
            mock_pc_instance = Mock()
            mock_pc_instance.get_current_metrics = AsyncMock(side_effect=Exception("Prometheus unavailable"))
            mock_pc.return_value = mock_pc_instance

            # Generate report - should not fail despite Prometheus error
            generator = PerformanceReportGenerator(self.config)
            report = await generator.generate_comprehensive_report()

            # Verify report was still generated successfully
            self.assertIsNotNone(report)
            self.assertIn("response_time_ms", report.kpi_summary["current_metrics"])

            # Verify Prometheus error was logged but didn't break the process
            # (In a real scenario, we'd check the logs, but for this test we just ensure completion)

    async def test_report_generation_with_missing_data(self):
        """Test report generation when some data sources are unavailable."""
        # Don't setup test data files - simulate missing data

        with patch('codesage_mcp.performance_report_generator.get_performance_monitor') as mock_pm, \
             patch('codesage_mcp.performance_report_generator.get_regression_detector') as mock_rd, \
             patch('codesage_mcp.performance_report_generator.get_prometheus_client') as mock_pc:

            # Mock components
            mock_pm_instance = Mock()
            mock_pm_instance.get_current_metrics.return_value = {
                "response_time_ms": {"value": 125.0, "timestamp": time.time(), "unit": "ms"}
            }
            mock_pm.return_value = mock_pm_instance

            mock_rd_instance = Mock()
            mock_regression_report = Mock()
            mock_regression_report.regressions_detected = []
            mock_regression_report.summary = {"total_metrics_analyzed": 1, "regressions_detected": 0}
            mock_regression_report.recommendations = []
            mock_rd_instance.detect_regressions.return_value = mock_regression_report
            mock_rd.return_value = mock_rd_instance

            mock_pc_instance = Mock()
            mock_pc_instance.get_current_metrics = AsyncMock(return_value={})
            mock_pc.return_value = mock_pc_instance

            # Generate report with missing benchmark and historical data
            generator = PerformanceReportGenerator(self.config)
            report = await generator.generate_comprehensive_report()

            # Verify report was generated despite missing data
            self.assertIsNotNone(report)
            self.assertIn("executive_summary", report.__dict__)
            self.assertIn("kpi_summary", report.__dict__)

            # Verify recommendations are still generated
            self.assertIsInstance(report.recommendations, list)


if __name__ == '__main__':
    unittest.main()