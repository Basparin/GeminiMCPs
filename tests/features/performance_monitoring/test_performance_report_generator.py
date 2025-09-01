"""
Unit tests for Performance Report Generator.

This module contains comprehensive unit tests for the performance report generation
system, including tests for report generation, data processing, chart creation,
and export functionality.
"""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock


from codesage_mcp.features.performance_monitoring.performance_report_generator import (
    PerformanceReportGenerator,
    ReportConfig,
    PerformanceReport
)
from codesage_mcp.features.performance_monitoring.regression_detector import RegressionReport


class TestReportConfig(unittest.TestCase):
    """Test cases for ReportConfig."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ReportConfig()
        self.assertEqual(config.report_title, "CodeSage MCP Server Performance Report")
        self.assertTrue(config.include_charts)
        self.assertTrue(config.include_historical_comparison)
        self.assertTrue(config.include_recommendations)
        self.assertEqual(config.chart_dpi, 150)
        self.assertEqual(config.historical_window_days, 30)

    @patch.dict(os.environ, {
        'REPORT_INCLUDE_CHARTS': 'false',
        'REPORT_INCLUDE_HISTORICAL': 'false',
        'REPORT_INCLUDE_RECOMMENDATIONS': 'false',
        'REPORT_CHART_DPI': '300',
        'PROMETHEUS_URL': 'http://test-prometheus:9090',
        'GRAFANA_URL': 'http://test-grafana:3000'
    })
    def test_config_from_environment(self):
        """Test configuration loading from environment variables."""
        config = ReportConfig()
        self.assertFalse(config.include_charts)
        self.assertFalse(config.include_historical_comparison)
        self.assertFalse(config.include_recommendations)
        self.assertEqual(config.chart_dpi, 300)
        self.assertEqual(config.prometheus_url, 'http://test-prometheus:9090')
        self.assertEqual(config.grafana_url, 'http://test-grafana:3000')


class TestPerformanceReportGenerator(unittest.TestCase):
    """Test cases for PerformanceReportGenerator."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = ReportConfig(
            include_charts=False,  # Disable charts for faster testing
            prometheus_url="http://test-prometheus:9090",
            grafana_url="http://test-grafana:3000"
        )
        self.generator = PerformanceReportGenerator(self.config)

        # Mock current metrics
        self.mock_current_metrics = {
            "response_time_ms": {
                "value": 150.5,
                "timestamp": time.time(),
                "unit": "ms"
            },
            "throughput_rps": {
                "value": 25.3,
                "timestamp": time.time(),
                "unit": "requests/sec"
            },
            "memory_usage_percent": {
                "value": 65.2,
                "timestamp": time.time(),
                "unit": "percent"
            },
            "cpu_usage_percent": {
                "value": 45.8,
                "timestamp": time.time(),
                "unit": "percent"
            },
            "error_rate_percent": {
                "value": 2.1,
                "timestamp": time.time(),
                "unit": "percent"
            }
        }

        # Mock benchmark data
        self.mock_benchmark_data = {
            "test_suite": "Test Suite",
            "timestamp": time.time(),
            "results": [
                {
                    "test_name": "test_response_time",
                    "metric_name": "response_time_ms",
                    "value": 120.0,
                    "unit": "ms",
                    "target": 200.0,
                    "achieved": True
                },
                {
                    "test_name": "test_throughput",
                    "metric_name": "throughput_rps",
                    "value": 30.0,
                    "unit": "requests/sec",
                    "target": 20.0,
                    "achieved": True
                }
            ],
            "summary": {
                "total_tests": 2,
                "passed_tests": 2,
                "failed_tests": 0,
                "performance_score": 100.0
            }
        }

    def tearDown(self):
        """Clean up test fixtures."""
        # Clean up any generated files
        import shutil
        if self.generator.output_dir.exists():
            shutil.rmtree(self.generator.output_dir)
        if self.generator.charts_dir.exists():
            shutil.rmtree(self.generator.charts_dir)
        if self.generator.historical_dir.exists():
            shutil.rmtree(self.generator.historical_dir)

    @patch('codesage_mcp.performance_report_generator.get_performance_monitor')
    @patch('codesage_mcp.performance_report_generator.get_regression_detector')
    @patch('codesage_mcp.performance_report_generator.get_prometheus_client')
    def test_initialization(self, mock_prometheus_client, mock_regression_detector, mock_performance_monitor):
        """Test generator initialization."""
        mock_pm = Mock()
        mock_rd = Mock()
        mock_pc = Mock()

        mock_performance_monitor.return_value = mock_pm
        mock_regression_detector.return_value = mock_rd
        mock_prometheus_client.return_value = mock_pc

        generator = PerformanceReportGenerator(self.config)

        self.assertEqual(generator.config, self.config)
        self.assertEqual(generator.performance_monitor, mock_pm)
        self.assertEqual(generator.regression_detector, mock_rd)
        self.assertEqual(generator.prometheus_client, mock_pc)

    def test_generate_executive_summary(self):
        """Test executive summary generation."""
        regression_report = RegressionReport(
            test_run_id="test_run",
            timestamp=time.time(),
            regressions_detected=[],
            summary={"total_metrics_analyzed": 5, "regressions_detected": 0},
            recommendations=[]
        )

        summary = self.generator._generate_executive_summary(
            self.mock_current_metrics, self.mock_benchmark_data, regression_report
        )

        self.assertIn("overall_health_score", summary)
        self.assertIn("key_findings", summary)
        self.assertIn("critical_issues", summary)
        self.assertIn("performance_trends", summary)
        self.assertGreater(summary["overall_health_score"], 0)
        self.assertLessEqual(summary["overall_health_score"], 100)

    def test_generate_kpi_summary(self):
        """Test KPI summary generation."""
        kpi_summary = self.generator._generate_kpi_summary(
            self.mock_current_metrics, self.mock_benchmark_data
        )

        self.assertIn("current_metrics", kpi_summary)
        self.assertIn("benchmark_kpis", kpi_summary)
        self.assertIn("kpi_status", kpi_summary)

        # Check that all current metrics are included
        for metric_name in self.mock_current_metrics.keys():
            self.assertIn(metric_name, kpi_summary["current_metrics"])
            self.assertIn(metric_name, kpi_summary["kpi_status"])

        # Check benchmark KPIs
        self.assertIn("response_time_ms", kpi_summary["benchmark_kpis"])
        self.assertIn("throughput_rps", kpi_summary["benchmark_kpis"])

    def test_generate_trend_analysis(self):
        """Test trend analysis generation."""
        historical_data = {
            "historical": [
                {
                    "timestamp": time.time() - 3600,  # 1 hour ago
                    "metrics": {
                        "response_time_ms": {"value": 140.0},
                        "throughput_rps": {"value": 22.0}
                    }
                },
                {
                    "timestamp": time.time() - 1800,  # 30 minutes ago
                    "metrics": {
                        "response_time_ms": {"value": 145.0},
                        "throughput_rps": {"value": 24.0}
                    }
                }
            ]
        }

        trend_analysis = self.generator._generate_trend_analysis(
            self.mock_current_metrics, historical_data, 24
        )

        self.assertIn("analysis_window_hours", trend_analysis)
        self.assertIn("metric_trends", trend_analysis)
        self.assertIn("trend_summary", trend_analysis)

        # Check that trends are calculated for each metric
        for metric_name in self.mock_current_metrics.keys():
            self.assertIn(metric_name, trend_analysis["metric_trends"])
            trend_data = trend_analysis["metric_trends"][metric_name]
            self.assertIn("current_value", trend_data)
            self.assertIn("trend_direction", trend_data)
            self.assertIn("trend_magnitude", trend_data)

    def test_generate_historical_comparison(self):
        """Test historical comparison generation."""
        historical_data = {
            "baseline": {
                "response_time_ms": {"value": 130.0},
                "throughput_rps": {"value": 20.0}
            }
        }

        comparison = self.generator._generate_historical_comparison(
            self.mock_current_metrics, historical_data
        )

        self.assertIn("baseline_comparison", comparison)
        self.assertIn("performance_changes", comparison)

        # Check baseline comparison calculations
        baseline_comp = comparison["baseline_comparison"]
        self.assertIn("response_time_ms", baseline_comp)
        self.assertIn("throughput_rps", baseline_comp)

        rt_comp = baseline_comp["response_time_ms"]
        self.assertIn("current_value", rt_comp)
        self.assertIn("baseline_value", rt_comp)
        self.assertIn("percentage_change", rt_comp)
        self.assertIn("improvement", rt_comp)

    def test_generate_recommendations(self):
        """Test recommendations generation."""
        # Test with high memory usage
        high_memory_metrics = self.mock_current_metrics.copy()
        high_memory_metrics["memory_usage_percent"]["value"] = 95.0

        regression_report = RegressionReport(
            test_run_id="test_run",
            timestamp=time.time(),
            regressions_detected=[],
            summary={},
            recommendations=[]
        )

        recommendations = self.generator._generate_recommendations(
            high_memory_metrics, self.mock_benchmark_data, regression_report
        )

        # Should include memory optimization recommendation
        memory_rec = next((r for r in recommendations if "memory" in r.get("type", "")), None)
        self.assertIsNotNone(memory_rec)
        self.assertEqual(memory_rec["priority"], "critical")
        self.assertIn("High Memory Usage", memory_rec["title"])

    def test_get_grafana_dashboard_info(self):
        """Test Grafana dashboard information retrieval."""
        grafana_info = self.generator._get_grafana_dashboard_info()

        self.assertIsNotNone(grafana_info)
        self.assertIn("grafana_url", grafana_info)
        self.assertIn("available_dashboards", grafana_info)
        self.assertIn("time_range", grafana_info)
        self.assertIn("refresh_interval", grafana_info)

        dashboards = grafana_info["available_dashboards"]
        self.assertIn("performance_report", dashboards)
        self.assertIn("overview", dashboards)
        self.assertIn("user_experience", dashboards)

    def test_get_grafana_dashboard_info_no_config(self):
        """Test Grafana dashboard info when no Grafana URL configured."""
        generator = PerformanceReportGenerator(ReportConfig(grafana_url=None))
        grafana_info = generator._get_grafana_dashboard_info()
        self.assertIsNone(grafana_info)

    @patch('codesage_mcp.performance_report_generator.PerformanceReportGenerator._load_benchmark_data')
    @patch('codesage_mcp.performance_report_generator.PerformanceReportGenerator._load_historical_data')
    @patch('codesage_mcp.performance_report_generator.PerformanceReportGenerator._save_report_to_history')
    @patch('codesage_mcp.performance_report_generator.get_performance_monitor')
    @patch('codesage_mcp.performance_report_generator.get_regression_detector')
    @patch('codesage_mcp.performance_report_generator.get_prometheus_client')
    async def test_generate_comprehensive_report(self, mock_prometheus_client, mock_regression_detector,
                                               mock_performance_monitor, mock_save_history,
                                               mock_load_historical, mock_load_benchmark):
        """Test comprehensive report generation."""
        # Setup mocks
        mock_pm = Mock()
        mock_pm.get_current_metrics.return_value = self.mock_current_metrics

        mock_rd = Mock()
        mock_regression_report = RegressionReport(
            test_run_id="test_run",
            timestamp=time.time(),
            regressions_detected=[],
            summary={"total_metrics_analyzed": 5, "regressions_detected": 0},
            recommendations=[]
        )
        mock_rd.detect_regressions.return_value = mock_regression_report

        mock_pc = Mock()
        mock_pc.get_current_metrics = AsyncMock(return_value={})

        mock_performance_monitor.return_value = mock_pm
        mock_regression_detector.return_value = mock_rd
        mock_prometheus_client.return_value = mock_pc

        mock_load_benchmark.return_value = self.mock_benchmark_data
        mock_load_historical.return_value = {"historical": [], "baseline": {}}
        mock_save_history.return_value = None

        # Create generator and generate report
        generator = PerformanceReportGenerator(self.config)
        report = await generator.generate_comprehensive_report()

        # Verify report structure
        self.assertIsInstance(report, PerformanceReport)
        self.assertIn("perf_report_", report.report_id)
        self.assertEqual(report.title, self.config.report_title)

        # Verify all sections are present
        self.assertIn("executive_summary", report.__dict__)
        self.assertIn("kpi_summary", report.__dict__)
        self.assertIn("trend_analysis", report.__dict__)
        self.assertIn("historical_comparison", report.__dict__)
        self.assertIn("recommendations", report.__dict__)

        # Verify metadata
        self.assertIn("metadata", report.__dict__)
        metadata = report.metadata
        self.assertIn("generator_version", metadata)
        self.assertIn("data_sources", metadata)

    async def test_export_json(self):
        """Test JSON export functionality."""
        report = PerformanceReport(
            report_id="test_report",
            timestamp=time.time(),
            title="Test Report",
            executive_summary={"test": "data"},
            kpi_summary={"test": "data"},
            trend_analysis={"test": "data"},
            regression_analysis=None,
            historical_comparison={"test": "data"},
            recommendations=[],
            charts={}
        )

        exported_files = await self.generator.export_report(report, ["json"])

        self.assertIn("json", exported_files)
        json_file = Path(exported_files["json"])
        self.assertTrue(json_file.exists())

        # Verify JSON content
        with open(json_file, 'r') as f:
            data = json.load(f)

        self.assertEqual(data["report_id"], "test_report")
        self.assertEqual(data["title"], "Test Report")
        self.assertIn("executive_summary", data)
        self.assertIn("kpi_summary", data)

    async def test_export_html(self):
        """Test HTML export functionality."""
        report = PerformanceReport(
            report_id="test_report",
            timestamp=time.time(),
            title="Test Report",
            executive_summary={"overall_health_score": 85.5, "key_findings": ["Test finding"]},
            kpi_summary={"current_metrics": {}, "benchmark_kpis": {}, "kpi_status": {}},
            trend_analysis={"metric_trends": {}, "trend_summary": {}},
            regression_analysis=None,
            historical_comparison={},
            recommendations=[{
                "type": "test",
                "priority": "medium",
                "title": "Test Recommendation",
                "description": "Test description",
                "actions": ["Action 1", "Action 2"]
            }],
            charts={}
        )

        exported_files = await self.generator.export_report(report, ["html"])

        self.assertIn("html", exported_files)
        html_file = Path(exported_files["html"])
        self.assertTrue(html_file.exists())

        # Verify HTML content
        with open(html_file, 'r') as f:
            html_content = f.read()

        self.assertIn("Test Report", html_content)
        self.assertIn("85.5", html_content)
        self.assertIn("Test finding", html_content)
        self.assertIn("Test Recommendation", html_content)

    def test_load_benchmark_data(self):
        """Test benchmark data loading."""
        # Create a temporary benchmark file
        with tempfile.TemporaryDirectory() as temp_dir:
            benchmark_file = Path(temp_dir) / "benchmark_report_test.json"
            with open(benchmark_file, 'w') as f:
                json.dump(self.mock_benchmark_data, f)

            # Create generator with temp directory
            config = ReportConfig(benchmark_results_dir=temp_dir)
            generator = PerformanceReportGenerator(config)

            data = generator._load_benchmark_data()

            self.assertEqual(data["test_suite"], "Test Suite")
            self.assertEqual(len(data["results"]), 2)
            self.assertEqual(data["summary"]["total_tests"], 2)

    def test_load_historical_data(self):
        """Test historical data loading."""
        # Create temporary historical files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create baseline file
            baseline_file = Path(temp_dir) / "baseline_results.json"
            baseline_data = {"response_time_ms": {"value": 100.0}}
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f)

            # Create historical report
            hist_file = Path(temp_dir) / "hist_report.json"
            hist_data = {
                "timestamp": time.time() - 3600,
                "kpi_summary": {"current_metrics": {"response_time_ms": {"value": 120.0}}}
            }
            with open(hist_file, 'w') as f:
                json.dump(hist_data, f)

            # Create generator with temp directory
            config = ReportConfig(historical_reports_dir=temp_dir)
            generator = PerformanceReportGenerator(config)

            data = generator._load_historical_data()

            self.assertIn("baseline", data)
            self.assertIn("historical", data)
            self.assertEqual(data["baseline"]["response_time_ms"]["value"], 100.0)
            self.assertEqual(len(data["historical"]), 1)


class TestPerformanceReport(unittest.TestCase):
    """Test cases for PerformanceReport dataclass."""

    def test_report_creation(self):
        """Test PerformanceReport creation and attributes."""
        report = PerformanceReport(
            report_id="test_report_123",
            timestamp=1234567890.0,
            title="Test Performance Report",
            executive_summary={"score": 95.5},
            kpi_summary={"metrics": {}},
            trend_analysis={"trends": {}},
            regression_analysis=None,
            historical_comparison={"comparison": {}},
            recommendations=[],
            charts={"chart1.png": "Test Chart"}
        )

        self.assertEqual(report.report_id, "test_report_123")
        self.assertEqual(report.timestamp, 1234567890.0)
        self.assertEqual(report.title, "Test Performance Report")
        self.assertEqual(report.executive_summary["score"], 95.5)
        self.assertIn("chart1.png", report.charts)


if __name__ == '__main__':
    unittest.main()