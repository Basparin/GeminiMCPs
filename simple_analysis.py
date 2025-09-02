#!/usr/bin/env python3
"""
Simple CES Performance Analysis Script
"""

import json
import statistics
import numpy as np
from typing import Dict, List, Any

def analyze_results(filename: str):
    """Simple analysis of performance results."""
    with open(filename, 'r') as f:
        data = json.load(f)

    print("=" * 60)
    print("CES PERFORMANCE ANALYSIS")
    print("=" * 60)

    # Overall summary
    summary = data.get("summary", {})
    print("\nOVERALL SUMMARY:")
    print(f"  Total Requests: {summary.get('total_requests', 0)}")
    print(f"  Successful Requests: {summary.get('successful_requests', 0)}")
    print(f"  Overall Success Rate: {summary.get('overall_success_rate', 0):.2f}%")
    print(f"  Average Response Time: {summary.get('average_response_time_ms', 0):.2f}ms")
    print(f"  P95 Response Time: {summary.get('p95_response_time_ms', 0):.2f}ms")
    print(f"  P99 Response Time: {summary.get('p99_response_time_ms', 0):.2f}ms")

    # Response times by type
    response_times = data.get("response_times", {})
    print("\nRESPONSE TIMES BY TYPE:")

    for test_type, test_data in response_times.items():
        if "statistics" in test_data:
            stats = test_data["statistics"]
            print(f"\n  {test_type.upper()}:")
            print(f"    Count: {stats.get('count', 0)}")
            print(f"    Mean: {stats.get('mean', 0):.2f}ms")
            print(f"    Median: {stats.get('median', 0):.2f}ms")
            print(f"    Min: {stats.get('min', 0):.2f}ms")
            print(f"    Max: {stats.get('max', 0):.2f}ms")
            print(f"    P95: {stats.get('p95', 0):.2f}ms")

    # Throughput analysis
    throughput = data.get("throughput", {})
    print("\nTHROUGHPUT ANALYSIS:")

    for test_type, test_data in throughput.items():
        if isinstance(test_data, dict) and "throughput_rps" in test_data:
            print(f"\n  {test_type.upper()}:")
            print(f"    Throughput: {test_data.get('throughput_rps', 0):.2f} RPS")
            print(f"    Concurrent Users: {test_data.get('concurrent_users', 0)}")
            print(f"    Duration: {test_data.get('duration_seconds', 0)}s")
            print(f"    Success Rate: {test_data.get('success_rate', 0):.2f}%")

    # Resource usage
    memory_usage = data.get("memory_usage", [])
    cpu_usage = data.get("cpu_usage", [])

    if memory_usage:
        memory_percents = [m["percent"] for m in memory_usage]
        print("\nMEMORY USAGE:")
        print(f"  Average: {statistics.mean(memory_percents):.2f}%")
        print(f"  Peak: {max(memory_percents):.2f}%")
        print(f"  Min: {min(memory_percents):.2f}%")

    if cpu_usage:
        cpu_percents = [c["percent"] for c in cpu_usage]
        print("\nCPU USAGE:")
        print(f"  Average: {statistics.mean(cpu_percents):.2f}%")
        print(f"  Peak: {max(cpu_percents):.2f}%")
        print(f"  Min: {min(cpu_percents):.2f}%")

    # Error analysis
    errors = data.get("errors", [])
    if errors:
        print(f"\nERRORS ({len(errors)} total):")
        error_counts = {}
        for error in errors:
            error_type = error.split(":")[0] if ":" in error else "Other"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1

        for error_type, count in sorted(error_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {error_type}: {count}")

    print("\n" + "=" * 60)

if __name__ == "__main__":
    analyze_results("ces_performance_test_20250901_232604.json")