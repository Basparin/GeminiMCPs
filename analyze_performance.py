#!/usr/bin/env python3
"""
CES Performance Analysis Script

Analyzes performance test results and generates comprehensive reports.
"""

import json
import statistics
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Any
import numpy as np

def load_performance_results(filename: str) -> Dict[str, Any]:
    """Load performance test results from JSON file."""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def analyze_response_times(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze response time data for different test types."""
    analysis = {}

    for test_type, data in results.get("response_times", {}).items():
        if "results" in data:
            response_times = [r["response_time_ms"] for r in data["results"] if r.get("success", False)]
            if response_times:
                analysis[test_type] = {
                    "count": len(response_times),
                    "mean": statistics.mean(response_times),
                    "median": statistics.median(response_times),
                    "min": min(response_times),
                    "max": max(response_times),
                    "p95": np.percentile(response_times, 95),
                    "p99": np.percentile(response_times, 99),
                    "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0
                }

    return analysis

def analyze_throughput(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze throughput and concurrent request handling."""
    analysis = {}

    for test_type, data in results.get("throughput", {}).items():
        if isinstance(data, dict) and "results" in data:
            successful_requests = len([r for r in data["results"] if r.get("success", False)])
            total_requests = len(data["results"])
            duration = data.get("duration_seconds", 0)
            concurrent_users = data.get("concurrent_users", 1)

            analysis[test_type] = {
                "total_requests": total_requests,
                "successful_requests": successful_requests,
                "success_rate": (successful_requests / total_requests * 100) if total_requests > 0 else 0,
                "throughput_rps": data.get("throughput_rps", 0),
                "concurrent_users": concurrent_users,
                "duration_seconds": duration,
                "requests_per_user": total_requests / concurrent_users if concurrent_users > 0 else 0
            }

    return analysis

def analyze_resource_usage(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze memory and CPU usage patterns."""
    analysis = {}

    memory_data = results.get("memory_usage", [])
    cpu_data = results.get("cpu_usage", [])

    if memory_data:
        memory_percents = [m["percent"] for m in memory_data]
        memory_used_mb = [m["used_mb"] for m in memory_data]

        analysis["memory"] = {
            "count": len(memory_data),
            "avg_percent": statistics.mean(memory_percents),
            "max_percent": max(memory_percents),
            "min_percent": min(memory_percents),
            "avg_used_mb": statistics.mean(memory_used_mb),
            "max_used_mb": max(memory_used_mb),
            "memory_trend": "stable" if statistics.stdev(memory_percents) < 5 else "variable"
        }

    if cpu_data:
        cpu_percents = [c["percent"] for c in cpu_data]

        analysis["cpu"] = {
            "count": len(cpu_data),
            "avg_percent": statistics.mean(cpu_percents),
            "max_percent": max(cpu_percents),
            "min_percent": min(cpu_percents),
            "cpu_trend": "stable" if statistics.stdev(cpu_percents) < 10 else "variable"
        }

    return analysis

def generate_performance_report(results: Dict[str, Any]) -> str:
    """Generate a comprehensive performance report."""
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("CES SERVER PERFORMANCE ANALYSIS REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Test Timestamp: {results.get('test_timestamp', 'Unknown')}")
    report_lines.append(f"Server URL: {results.get('server_url', 'Unknown')}")
    report_lines.append("")

    # Overall Summary
    summary = results.get("summary", {})
    report_lines.append("OVERALL PERFORMANCE SUMMARY")
    report_lines.append("-" * 40)
    report_lines.append(".2f")
    report_lines.append(".2f")
    report_lines.append(".2f")
    report_lines.append(".2f")
    report_lines.append(f"Total Errors: {summary.get('total_errors', 0)}")
    report_lines.append("")

    # Response Time Analysis
    response_analysis = analyze_response_times(results)
    if response_analysis:
        report_lines.append("RESPONSE TIME ANALYSIS (by Test Type)")
        report_lines.append("-" * 40)

        for test_type, stats in response_analysis.items():
            report_lines.append(f"\n{test_type.upper()} REQUESTS:")
            report_lines.append(f"  Sample Size: {stats['count']} requests")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append(".2f")

    # Throughput Analysis
    throughput_analysis = analyze_throughput(results)
    if throughput_analysis:
        report_lines.append("\nTHROUGHPUT ANALYSIS")
        report_lines.append("-" * 40)

        for test_type, stats in throughput_analysis.items():
            report_lines.append(f"\n{test_type.upper()} TEST:")
            report_lines.append(f"  Total Requests: {stats['total_requests']}")
            report_lines.append(f"  Successful Requests: {stats['successful_requests']}")
            report_lines.append(".2f")
            report_lines.append(".2f")
            report_lines.append(f"  Concurrent Users: {stats['concurrent_users']}")
            report_lines.append(f"  Duration: {stats['duration_seconds']} seconds")
            report_lines.append(".2f")

    # Resource Usage Analysis
    resource_analysis = analyze_resource_usage(results)
    if resource_analysis:
        report_lines.append("\nRESOURCE USAGE ANALYSIS")
        report_lines.append("-" * 40)

        if "memory" in resource_analysis:
            mem = resource_analysis["memory"]
            report_lines.append("\nMEMORY USAGE:")
            report_lines.append(f"  Average: {mem['avg_percent']:.1f}%")
            report_lines.append(f"  Peak: {mem['max_percent']:.1f}%")
            report_lines.append(f"  Average Used: {mem['avg_used_mb']:.0f} MB")
            report_lines.append(f"  Peak Used: {mem['max_used_mb']:.0f} MB")
            report_lines.append(f"  Trend: {mem['memory_trend']}")

        if "cpu" in resource_analysis:
            cpu = resource_analysis["cpu"]
            report_lines.append("\nCPU USAGE:")
            report_lines.append(f"  Average: {cpu['avg_percent']:.1f}%")
            report_lines.append(f"  Peak: {cpu['max_percent']:.1f}%")
            report_lines.append(f"  Trend: {cpu['cpu_trend']}")

    # Recommendations
    recommendations = results.get("recommendations", [])
    if recommendations:
        report_lines.append("\nPERFORMANCE RECOMMENDATIONS")
        report_lines.append("-" * 40)
        for i, rec in enumerate(recommendations, 1):
            report_lines.append(f"{i}. {rec}")

    # Error Analysis
    errors = results.get("errors", [])
    if errors:
        report_lines.append(f"\nERROR ANALYSIS ({len(errors)} errors detected)")
        report_lines.append("-" * 40)

        # Group errors by type
        error_types = {}
        for error in errors:
            error_type = error.split(":")[0] if ":" in error else "Unknown"
            error_types[error_type] = error_types.get(error_type, 0) + 1

        for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
            report_lines.append(f"  {error_type}: {count} occurrences")

    report_lines.append("\n" + "=" * 80)
    report_lines.append("END OF PERFORMANCE REPORT")
    report_lines.append("=" * 80)

    return "\n".join(report_lines)

def create_performance_charts(results: Dict[str, Any], output_prefix: str = "ces_performance"):
    """Create performance visualization charts."""
    try:
        # Response time distribution chart
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Chart 1: Response times by test type
        test_types = []
        avg_response_times = []

        for test_type, data in results.get("response_times", {}).items():
            if "results" in data:
                response_times = [r["response_time_ms"] for r in data["results"] if r.get("success", False)]
                if response_times:
                    test_types.append(test_type)
                    avg_response_times.append(statistics.mean(response_times))

        if test_types:
            ax1.bar(test_types, avg_response_times)
            ax1.set_title("Average Response Times by Test Type")
            ax1.set_ylabel("Response Time (ms)")
            ax1.tick_params(axis='x', rotation=45)

        # Chart 2: Memory usage over time
        memory_data = results.get("memory_usage", [])
        if memory_data:
            timestamps = [datetime.fromisoformat(m["timestamp"]) for m in memory_data]
            memory_percents = [m["percent"] for m in memory_data]

            ax2.plot(timestamps, memory_percents)
            ax2.set_title("Memory Usage Over Time")
            ax2.set_ylabel("Memory Usage (%)")
            ax2.tick_params(axis='x', rotation=45)

        # Chart 3: CPU usage over time
        cpu_data = results.get("cpu_usage", [])
        if cpu_data:
            timestamps = [datetime.fromisoformat(c["timestamp"]) for c in cpu_data]
            cpu_percents = [c["percent"] for c in cpu_data]

            ax3.plot(timestamps, cpu_percents)
            ax3.set_title("CPU Usage Over Time")
            ax3.set_ylabel("CPU Usage (%)")
            ax3.tick_params(axis='x', rotation=45)

        # Chart 4: Throughput comparison
        throughput_data = results.get("throughput", {})
        if throughput_data:
            test_names = []
            throughputs = []

            for test_type, data in throughput_data.items():
                if isinstance(data, dict) and "throughput_rps" in data:
                    test_names.append(test_type)
                    throughputs.append(data["throughput_rps"])

            if test_names:
                ax4.bar(test_names, throughputs)
                ax4.set_title("Throughput by Test Type")
                ax4.set_ylabel("Requests per Second")
                ax4.tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_charts.png", dpi=300, bbox_inches='tight')
        plt.close()

        print(f"Performance charts saved to {output_prefix}_charts.png")

    except Exception as e:
        print(f"Error creating charts: {e}")

def main():
    """Main analysis function."""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_performance.py <results_file.json>")
        sys.exit(1)

    results_file = sys.argv[1]

    print("Loading performance results...")
    results = load_performance_results(results_file)

    if not results:
        print("Failed to load results")
        sys.exit(1)

    print("Generating performance report...")
    report = generate_performance_report(results)

    # Save report to file
    report_filename = results_file.replace('.json', '_report.txt')
    with open(report_filename, 'w') as f:
        f.write(report)

    print(f"Performance report saved to: {report_filename}")

    # Create charts
    print("Creating performance charts...")
    chart_prefix = results_file.replace('.json', '')
    create_performance_charts(results, chart_prefix)

    # Print summary to console
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS SUMMARY")
    print("="*60)

    summary = results.get("summary", {})
    print(".2f")
    print(".2f")
    print(".2f")
    print(".2f")
    print(f"Total Errors: {summary.get('total_errors', 0)}")

    # Response time highlights
    response_analysis = analyze_response_times(results)
    if response_analysis:
        print("\nRESPONSE TIME HIGHLIGHTS:")
        for test_type, stats in response_analysis.items():
            print(".2f")

    # Throughput highlights
    throughput_analysis = analyze_throughput(results)
    if throughput_analysis:
        print("\nTHROUGHPUT HIGHLIGHTS:")
        for test_type, stats in throughput_analysis.items():
            print(".2f")

    print("\nDetailed analysis completed successfully!")

if __name__ == "__main__":
    main()