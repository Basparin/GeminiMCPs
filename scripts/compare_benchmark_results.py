#!/usr/bin/env python3
"""
Benchmark Results Comparator for CodeSage MCP Server.

This script compares current benchmark results against baseline results
to detect performance regressions and improvements.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import statistics
from datetime import datetime
import glob


class BenchmarkComparator:
    """Compares benchmark results against baseline."""

    def __init__(self, current_results_dir: str = "benchmark_results",
                 baseline_file: str = "baseline_results.json"):
        self.current_results_dir = Path(current_results_dir)
        self.baseline_file = Path(baseline_file)
        self.comparison_tolerance = 0.05  # 5% tolerance for changes

    def compare_with_baseline(self) -> Dict[str, Any]:
        """Compare current results with baseline."""
        print("Comparing benchmark results with baseline...")

        # Load current results
        current_results = self._load_current_results()
        if not current_results:
            return {"error": "No current benchmark results found"}

        # Load baseline results
        baseline_results = self._load_baseline_results()
        if not baseline_results:
            print("No baseline results found. Creating new baseline...")
            self._save_as_baseline(current_results)
            return {"message": "New baseline created", "results": current_results}

        # Perform comparison
        comparison = self._perform_comparison(current_results, baseline_results)

        # Save comparison results
        self._save_comparison_results(comparison)

        return comparison

    def _load_current_results(self) -> List[Dict[str, Any]]:
        """Load current benchmark results."""
        results = []

        # Load all recent benchmark files
        patterns = [
            "benchmark_report_*.json",
            "parameterized_benchmark_*.json"
        ]

        for pattern in patterns:
            for filepath in glob.glob(str(self.current_results_dir / pattern)):
                try:
                    with open(filepath, 'r') as f:
                        data = json.load(f)
                        results.append(data)
                except Exception as e:
                    print(f"Failed to load {filepath}: {e}")

        return results

    def _load_baseline_results(self) -> Optional[Dict[str, Any]]:
        """Load baseline results."""
        if not self.baseline_file.exists():
            return None

        try:
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load baseline: {e}")
            return None

    def _perform_comparison(self, current: List[Dict[str, Any]],
                          baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed comparison between current and baseline results."""
        comparison = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_current_tests": 0,
                "total_baseline_tests": 0,
                "improvements": 0,
                "regressions": 0,
                "unchanged": 0
            },
            "detailed_comparison": [],
            "recommendations": []
        }

        # Extract metrics from current results
        current_metrics = self._extract_metrics(current)
        baseline_metrics = self._extract_metrics_from_baseline(baseline)

        comparison["summary"]["total_current_tests"] = len(current_metrics)
        comparison["summary"]["total_baseline_tests"] = len(baseline_metrics)

        # Compare each metric
        for metric_name, current_data in current_metrics.items():
            baseline_data = baseline_metrics.get(metric_name)

            if baseline_data:
                comparison_result = self._compare_metric(
                    metric_name, current_data, baseline_data
                )
                comparison["detailed_comparison"].append(comparison_result)

                # Update summary counters
                if comparison_result["change_type"] == "improvement":
                    comparison["summary"]["improvements"] += 1
                elif comparison_result["change_type"] == "regression":
                    comparison["summary"]["regressions"] += 1
                else:
                    comparison["summary"]["unchanged"] += 1
            else:
                # New metric
                comparison["detailed_comparison"].append({
                    "metric": metric_name,
                    "change_type": "new",
                    "current_value": current_data["value"],
                    "baseline_value": None,
                    "change_percent": None,
                    "status": "new_metric"
                })

        # Generate recommendations
        comparison["recommendations"] = self._generate_recommendations(comparison)

        return comparison

    def _extract_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract metrics from results."""
        metrics = {}

        for result in results:
            if "results" in result:
                if isinstance(result["results"], list):
                    # Standard format
                    for benchmark_result in result["results"]:
                        metric_name = benchmark_result.get("metric_name", "")
                        test_name = benchmark_result.get("test_name", "")
                        key = f"{test_name}_{metric_name}"

                        metrics[key] = {
                            "value": benchmark_result.get("value"),
                            "unit": benchmark_result.get("unit"),
                            "target": benchmark_result.get("target"),
                            "achieved": benchmark_result.get("achieved", False)
                        }
                elif isinstance(result["results"], dict):
                    # Parameterized format - extract success/failure
                    for benchmark_type, benchmark_data in result["results"].items():
                        if isinstance(benchmark_data, dict) and "success" in benchmark_data:
                            key = f"{result.get('config_name', 'unknown')}_{benchmark_type}"
                            metrics[key] = {
                                "value": 1.0 if benchmark_data["success"] else 0.0,
                                "unit": "boolean",
                                "target": 1.0,
                                "achieved": benchmark_data["success"]
                            }

        return metrics

    def _extract_metrics_from_baseline(self, baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Extract metrics from baseline format."""
        metrics = {}

        # Handle different baseline formats
        if "results" in baseline:
            if isinstance(baseline["results"], list):
                for result in baseline["results"]:
                    metric_name = result.get("metric_name", "")
                    test_name = result.get("test_name", "")
                    key = f"{test_name}_{metric_name}"

                    metrics[key] = {
                        "value": result.get("value"),
                        "unit": result.get("unit"),
                        "target": result.get("target"),
                        "achieved": result.get("achieved", False)
                    }

        return metrics

    def _compare_metric(self, metric_name: str, current: Dict[str, Any],
                       baseline: Dict[str, Any]) -> Dict[str, Any]:
        """Compare a single metric."""
        current_value = current.get("value")
        baseline_value = baseline.get("value")

        if current_value is None or baseline_value is None:
            return {
                "metric": metric_name,
                "change_type": "unknown",
                "current_value": current_value,
                "baseline_value": baseline_value,
                "change_percent": None,
                "status": "data_missing"
            }

        # Calculate change
        if isinstance(current_value, (int, float)) and isinstance(baseline_value, (int, float)):
            if baseline_value != 0:
                change_percent = ((current_value - baseline_value) / baseline_value) * 100
            else:
                change_percent = float('inf') if current_value > 0 else 0.0
        else:
            # For non-numeric values, compare as boolean (success/failure)
            change_percent = None

        # Determine change type
        change_type = "unchanged"
        status = "stable"

        if isinstance(change_percent, (int, float)):
            if change_percent > self.comparison_tolerance * 100:
                if self._is_improvement(metric_name, current_value, baseline_value):
                    change_type = "improvement"
                    status = "improved"
                else:
                    change_type = "regression"
                    status = "regressed"
            elif change_percent < -self.comparison_tolerance * 100:
                if self._is_improvement(metric_name, current_value, baseline_value):
                    change_type = "regression"
                    status = "regressed"
                else:
                    change_type = "improvement"
                    status = "improved"

        return {
            "metric": metric_name,
            "change_type": change_type,
            "current_value": current_value,
            "baseline_value": baseline_value,
            "change_percent": change_percent,
            "status": status,
            "unit": current.get("unit"),
            "target": current.get("target")
        }

    def _is_improvement(self, metric_name: str, current_value: float,
                       baseline_value: float) -> bool:
        """Determine if a change represents an improvement."""
        # For most metrics, lower values are better (latency, memory usage, etc.)
        # For some metrics, higher values are better (throughput, hit rates, etc.)

        improvement_metrics = {
            "throughput", "requests_per_second", "hit_rate", "cache_hit_rate",
            "success_rate", "performance_score"
        }

        if any(term in metric_name.lower() for term in improvement_metrics):
            return current_value > baseline_value
        else:
            # For other metrics (latency, memory, etc.), lower is better
            return current_value < baseline_value

    def _generate_recommendations(self, comparison: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on comparison results."""
        recommendations = []

        summary = comparison.get("summary", {})
        regressions = summary.get("regressions", 0)
        improvements = summary.get("improvements", 0)

        if regressions > 0:
            recommendations.append(
                f"⚠️ **Performance Regression Detected**: {regressions} metrics have regressed. "
                "Review the detailed comparison for specific issues."
            )

        if improvements > 0:
            recommendations.append(
                f"✅ **Performance Improvements**: {improvements} metrics have improved. "
                "Consider updating the baseline with these improvements."
            )

        # Check for specific types of regressions
        detailed = comparison.get("detailed_comparison", [])
        latency_regressions = [
            d for d in detailed
            if d.get("change_type") == "regression" and "latency" in d.get("metric", "").lower()
        ]

        if latency_regressions:
            recommendations.append(
                "⚠️ **Latency Regression**: Response times have increased. "
                "Check for database query optimizations or caching issues."
            )

        throughput_regressions = [
            d for d in detailed
            if d.get("change_type") == "regression" and "throughput" in d.get("metric", "").lower()
        ]

        if throughput_regressions:
            recommendations.append(
                "⚠️ **Throughput Regression**: Request processing capacity has decreased. "
                "Review resource utilization and scaling configuration."
            )

        if not recommendations:
            recommendations.append(
                "✅ **Stable Performance**: No significant performance changes detected."
            )

        return recommendations

    def _save_as_baseline(self, results: List[Dict[str, Any]]):
        """Save current results as new baseline."""
        baseline_data = {
            "timestamp": datetime.now().isoformat(),
            "description": "Auto-generated baseline from current results",
            "results": []
        }

        # Flatten results for baseline
        for result in results:
            if "results" in result:
                if isinstance(result["results"], list):
                    baseline_data["results"].extend(result["results"])

        with open(self.baseline_file, 'w') as f:
            json.dump(baseline_data, f, indent=2, default=str)

        print(f"New baseline saved to: {self.baseline_file}")

    def _save_comparison_results(self, comparison: Dict[str, Any]):
        """Save comparison results to file."""
        timestamp = int(datetime.now().timestamp())
        filename = f"benchmark_comparison_{timestamp}.json"

        output_file = self.current_results_dir / filename
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)

        print(f"Comparison results saved to: {output_file}")

    def print_comparison_summary(self, comparison: Dict[str, Any]):
        """Print a human-readable comparison summary."""
        print("\n" + "="*60)
        print("BENCHMARK COMPARISON SUMMARY")
        print("="*60)

        summary = comparison.get("summary", {})
        print(f"Current Tests: {summary.get('total_current_tests', 0)}")
        print(f"Baseline Tests: {summary.get('total_baseline_tests', 0)}")
        print(f"Improvements: {summary.get('improvements', 0)}")
        print(f"Regressions: {summary.get('regressions', 0)}")
        print(f"Unchanged: {summary.get('unchanged', 0)}")

        print("\nRECOMMENDATIONS:")
        for rec in comparison.get("recommendations", []):
            print(f"- {rec}")

        print("\nDETAILED CHANGES:")
        for detail in comparison.get("detailed_comparison", []):
            if detail.get("change_type") != "unchanged":
                metric = detail.get("metric", "Unknown")
                change_type = detail.get("change_type", "unknown")
                change_percent = detail.get("change_percent")

                if change_percent is not None:
                    print(".1f")
                else:
                    print(f"- {metric}: {change_type.upper()}")


def main():
    """Main entry point."""
    if len(sys.argv) < 3:
        print("Usage: python compare_benchmark_results.py <current_results_dir> <baseline_file>")
        sys.exit(1)

    current_dir = sys.argv[1]
    baseline_file = sys.argv[2]

    comparator = BenchmarkComparator(current_dir, baseline_file)
    comparison = comparator.compare_with_baseline()
    comparator.print_comparison_summary(comparison)


if __name__ == "__main__":
    main()