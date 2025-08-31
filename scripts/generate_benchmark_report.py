#!/usr/bin/env python3
"""
Benchmark Report Generator for CodeSage MCP Server.

This script generates comprehensive benchmark reports from raw benchmark data,
including performance summaries, trend analysis, and recommendations.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import statistics
from datetime import datetime
import glob


class BenchmarkReportGenerator:
    """Generates comprehensive benchmark reports."""

    def __init__(self, results_dir: str = "benchmark_results"):
        self.results_dir = Path(results_dir)
        self.output_dir = Path("reports")
        self.output_dir.mkdir(exist_ok=True)

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive benchmark report."""
        print("Generating comprehensive benchmark report...")

        # Load all benchmark results
        results = self._load_all_results()

        if not results:
            return "No benchmark results found to generate report."

        # Generate report sections
        report = []
        report.append("# CodeSage MCP Server - Comprehensive Benchmark Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")

        # Executive Summary
        report.extend(self._generate_executive_summary(results))

        # Performance Metrics Summary
        report.extend(self._generate_performance_summary(results))

        # Detailed Results by Category
        report.extend(self._generate_detailed_results(results))

        # Recommendations
        report.extend(self._generate_recommendations(results))

        # Raw Data Summary
        report.extend(self._generate_raw_data_summary(results))

        return "\n".join(report)

    def _load_all_results(self) -> List[Dict[str, Any]]:
        """Load all benchmark result files."""
        results = []

        # Load JSON result files
        pattern = str(self.results_dir / "benchmark_report_*.json")
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")

        # Load parameterized results
        pattern = str(self.results_dir / "parameterized_benchmark_*.json")
        for filepath in glob.glob(pattern):
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Failed to load {filepath}: {e}")

        return results

    def _generate_executive_summary(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate executive summary section."""
        section = []
        section.append("## Executive Summary")
        section.append("")

        if not results:
            section.append("No benchmark results available.")
            return section

        # Calculate overall statistics
        total_tests = 0
        passed_tests = 0
        failed_tests = 0
        avg_response_times = []
        success_rates = []

        for result in results:
            if "results" in result:
                # Handle different result formats
                if isinstance(result["results"], list):
                    # Standard benchmark format
                    for benchmark_result in result["results"]:
                        total_tests += 1
                        if benchmark_result.get("achieved", False):
                            passed_tests += 1
                        else:
                            failed_tests += 1
                elif isinstance(result["results"], dict):
                    # Parameterized format
                    for benchmark_type, benchmark_data in result["results"].items():
                        if isinstance(benchmark_data, dict) and "success" in benchmark_data:
                            total_tests += 1
                            if benchmark_data["success"]:
                                passed_tests += 1
                            else:
                                failed_tests += 1

        if total_tests > 0:
            success_rate = (passed_tests / total_tests) * 100
            section.append(f"- **Total Tests Run**: {total_tests}")
            section.append(f"- **Tests Passed**: {passed_tests}")
            section.append(f"- **Tests Failed**: {failed_tests}")
            section.append(f"- **Overall Success Rate**: {success_rate:.1f}%")
        else:
            section.append("No test results found.")

        section.append("")
        section.append("### Key Findings")
        section.append("")

        # Add key findings based on results
        if success_rate >= 90:
            section.append("✅ **Excellent Performance**: System is performing within acceptable parameters.")
        elif success_rate >= 75:
            section.append("⚠️ **Good Performance**: System performance is acceptable but has room for improvement.")
        else:
            section.append("❌ **Performance Issues**: System requires attention to meet performance targets.")

        section.append("")
        return section

    def _generate_performance_summary(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate performance metrics summary."""
        section = []
        section.append("## Performance Metrics Summary")
        section.append("")

        # Group results by category
        categories = self._categorize_results(results)

        for category, category_results in categories.items():
            if not category_results:
                continue

            section.append(f"### {category.replace('_', ' ').title()}")
            section.append("")

            # Calculate category statistics
            passed = sum(1 for r in category_results if r.get("achieved", False))
            total = len(category_results)
            success_rate = (passed / total * 100) if total > 0 else 0

            section.append(f"- **Tests**: {passed}/{total} passed ({success_rate:.1f}%)")

            # Add specific metrics for the category
            if category == "jsonrpc_latency":
                latencies = [r.get("value", 0) for r in category_results if "latency" in r.get("metric_name", "")]
                if latencies:
                    avg_latency = statistics.mean(latencies)
                    section.append(f"- **Average Latency**: {avg_latency:.2f} ms")

            elif category == "tool_execution":
                times = [r.get("value", 0) for r in category_results if "time" in r.get("metric_name", "")]
                if times:
                    avg_time = statistics.mean(times)
                    section.append(f"- **Average Execution Time**: {avg_time:.2f} ms")

            elif category == "cache_performance":
                hit_rates = [r.get("value", 0) for r in category_results if "hit_rate" in r.get("metric_name", "")]
                if hit_rates:
                    avg_hit_rate = statistics.mean(hit_rates)
                    section.append(f"- **Average Cache Hit Rate**: {avg_hit_rate:.1f}%")

            section.append("")

        return section

    def _generate_detailed_results(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate detailed results section."""
        section = []
        section.append("## Detailed Results")
        section.append("")

        categories = self._categorize_results(results)

        for category, category_results in categories.items():
            if not category_results:
                continue

            section.append(f"### {category.replace('_', ' ').title()}")
            section.append("")
            section.append("| Test Name | Metric | Value | Target | Status |")
            section.append("|-----------|--------|-------|--------|--------|")

            for result in category_results:
                test_name = result.get("test_name", "Unknown")
                metric = result.get("metric_name", "Unknown")
                value = result.get("value", "N/A")
                target = result.get("target", "N/A")
                status = "✅ PASS" if result.get("achieved", False) else "❌ FAIL"

                if isinstance(value, (int, float)):
                    if "percent" in result.get("unit", "").lower():
                        value_str = f"{value:.1f}%"
                    elif "time" in metric.lower() or "latency" in metric.lower():
                        value_str = f"{value:.2f} {result.get('unit', '')}"
                    else:
                        value_str = f"{value:.2f} {result.get('unit', '')}"
                else:
                    value_str = str(value)

                section.append(f"| {test_name} | {metric} | {value_str} | {target} | {status} |")

            section.append("")

        return section

    def _generate_recommendations(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on results."""
        section = []
        section.append("## Recommendations")
        section.append("")

        categories = self._categorize_results(results)
        recommendations = []

        # Analyze cache performance
        cache_results = categories.get("cache_performance", [])
        if cache_results:
            hit_rates = [r.get("value", 0) for r in cache_results if "hit_rate" in r.get("metric_name", "")]
            if hit_rates:
                avg_hit_rate = statistics.mean(hit_rates)
                if avg_hit_rate < 70:
                    recommendations.append("⚠️ **Improve Cache Performance**: Cache hit rate is below target. Consider increasing cache size or optimizing cache policies.")

        # Analyze latency
        latency_results = categories.get("jsonrpc_latency", [])
        if latency_results:
            latencies = [r.get("value", 0) for r in latency_results if "latency" in r.get("metric_name", "")]
            if latencies:
                avg_latency = statistics.mean(latencies)
                if avg_latency > 1000:  # Over 1 second
                    recommendations.append("⚠️ **Reduce Latency**: Average response time is high. Consider optimizing database queries or implementing caching.")

        # Analyze tool execution
        tool_results = categories.get("tool_execution", [])
        if tool_results:
            slow_tools = [r for r in tool_results if r.get("value", 0) > 5000]  # Over 5 seconds
            if slow_tools:
                recommendations.append("⚠️ **Optimize Tool Performance**: Some tools are running slowly. Review and optimize tool implementations.")

        # General recommendations
        if not recommendations:
            recommendations.append("✅ **Good Performance**: All metrics are within acceptable ranges. Continue monitoring for any regressions.")

        for rec in recommendations:
            section.append(f"- {rec}")

        section.append("")
        return section

    def _generate_raw_data_summary(self, results: List[Dict[str, Any]]) -> List[str]:
        """Generate raw data summary section."""
        section = []
        section.append("## Raw Data Summary")
        section.append("")
        section.append(f"- **Total Result Files**: {len(results)}")
        section.append(f"- **Results Directory**: {self.results_dir}")
        section.append("")

        # List all result files
        if results:
            section.append("### Result Files")
            for i, result in enumerate(results, 1):
                timestamp = result.get("timestamp", "Unknown")
                if isinstance(timestamp, (int, float)):
                    timestamp_str = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                else:
                    timestamp_str = str(timestamp)
                section.append(f"{i}. {timestamp_str}")

        return section

    def _categorize_results(self, results: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize results by type."""
        categories = {
            "jsonrpc_latency": [],
            "tool_execution": [],
            "cache_performance": [],
            "resource_utilization": [],
            "throughput_scalability": [],
            "edge_cases": [],
            "indexing_performance": [],
            "search_performance": [],
            "memory_management": [],
            "chunking_performance": []
        }

        for result in results:
            if "results" in result:
                if isinstance(result["results"], list):
                    # Standard format
                    for benchmark_result in result["results"]:
                        test_name = benchmark_result.get("test_name", "")
                        if "jsonrpc" in test_name.lower() or "latency" in test_name.lower():
                            categories["jsonrpc_latency"].append(benchmark_result)
                        elif "tool" in test_name.lower() and "execution" in test_name.lower():
                            categories["tool_execution"].append(benchmark_result)
                        elif "cache" in test_name.lower():
                            categories["cache_performance"].append(benchmark_result)
                        elif "resource" in test_name.lower():
                            categories["resource_utilization"].append(benchmark_result)
                        elif "throughput" in test_name.lower() or "scalability" in test_name.lower():
                            categories["throughput_scalability"].append(benchmark_result)
                        elif "edge" in test_name.lower():
                            categories["edge_cases"].append(benchmark_result)
                        elif "indexing" in test_name.lower():
                            categories["indexing_performance"].append(benchmark_result)
                        elif "search" in test_name.lower():
                            categories["search_performance"].append(benchmark_result)
                        elif "memory" in test_name.lower():
                            categories["memory_management"].append(benchmark_result)
                        elif "chunking" in test_name.lower():
                            categories["chunking_performance"].append(benchmark_result)

        return categories

    def save_report(self, report: str, filename: str = "benchmark_report.md"):
        """Save the report to a file."""
        filepath = self.output_dir / filename
        with open(filepath, 'w') as f:
            f.write(report)

        print(f"Report saved to: {filepath}")
        return filepath


def main():
    """Main entry point."""
    generator = BenchmarkReportGenerator()
    report = generator.generate_comprehensive_report()
    generator.save_report(report)

    print("Benchmark report generation completed!")


if __name__ == "__main__":
    main()