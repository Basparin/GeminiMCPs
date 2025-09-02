#!/usr/bin/env python3
"""
Detailed CES Performance Analysis Script
"""

import json
import statistics
from collections import Counter
from typing import Dict, List, Any

def analyze_detailed_performance(filename: str):
    """Detailed analysis of performance test results."""
    with open(filename, 'r') as f:
        data = json.load(f)

    print("=" * 80)
    print("DETAILED CES PERFORMANCE ANALYSIS")
    print("=" * 80)

    # Analyze errors in detail
    analyze_errors(data)

    # Analyze throughput patterns
    analyze_throughput_patterns(data)

    # Analyze response time distributions
    analyze_response_distributions(data)

    # Analyze system resource patterns
    analyze_resource_patterns(data)

    # Generate recommendations
    generate_recommendations(data)

    print("\n" + "=" * 80)

def analyze_errors(data: Dict[str, Any]):
    """Analyze error patterns in detail."""
    errors = data.get("errors", [])
    print(f"\nERROR ANALYSIS ({len(errors)} total errors)")

    if errors:
        # Group errors by type
        error_types = Counter()
        for error in errors:
            if ":" in error:
                error_type = error.split(":")[0].strip()
            else:
                error_type = "Other"
            error_types[error_type] += 1

        print("\nError Types:")
        for error_type, count in error_types.most_common():
            percentage = (count / len(errors)) * 100
            print(f"  {error_type}: {count} ({percentage:.1f}%)")

        # Show sample errors
        print("\nSample Errors:")
        for i, error in enumerate(errors[:5]):
            print(f"  {i+1}. {error}")
        if len(errors) > 5:
            print(f"  ... and {len(errors) - 5} more errors")
    else:
        print("No errors detected in the test run.")

def analyze_throughput_patterns(data: Dict[str, Any]):
    """Analyze throughput patterns."""
    throughput = data.get("throughput", {})
    print("\nTHROUGHPUT ANALYSIS")

    for test_type, test_data in throughput.items():
        if isinstance(test_data, dict) and "results" in test_data:
            results = test_data["results"]
            successful = [r for r in results if r.get("success", False)]
            failed = [r for r in results if not r.get("success", False)]

            print(f"\n{test_type.upper()} TEST:")
            print(f"  Total Requests: {len(results)}")
            print(f"  Successful: {len(successful)}")
            print(f"  Failed: {len(failed)}")
            print(f"  Success Rate: {(len(successful) / len(results) * 100):.2f}%" if results else "  Success Rate: N/A")
            print(f"  Throughput: {test_data.get('throughput_rps', 0):.2f} RPS")

            if successful:
                response_times = [r["response_time_ms"] for r in successful]
                print(f"  Avg Response Time: {statistics.mean(response_times):.2f}ms")
                print(f"  Max Response Time: {max(response_times):.2f}ms")

            # Error analysis for this test
            if failed:
                error_codes = Counter()
                for r in failed:
                    status = r.get("status_code", "unknown")
                    error_codes[status] += 1

                print("  Error Status Codes:")
                for status, count in error_codes.most_common():
                    print(f"    {status}: {count}")

def analyze_response_distributions(data: Dict[str, Any]):
    """Analyze response time distributions."""
    response_times = data.get("response_times", {})
    print("\nRESPONSE TIME DISTRIBUTION ANALYSIS")

    for test_type, test_data in response_times.items():
        if "results" in test_data:
            results = test_data["results"]
            successful_times = [r["response_time_ms"] for r in results if r.get("success", False)]

            if successful_times:
                print(f"\n{test_type.upper()} REQUESTS:")
                print(f"  Total Successful: {len(successful_times)}")

                # Create time buckets
                buckets = {
                    "< 1ms": len([t for t in successful_times if t < 1]),
                    "1-5ms": len([t for t in successful_times if 1 <= t < 5]),
                    "5-10ms": len([t for t in successful_times if 5 <= t < 10]),
                    "10-50ms": len([t for t in successful_times if 10 <= t < 50]),
                    "50-100ms": len([t for t in successful_times if 50 <= t < 100]),
                    "> 100ms": len([t for t in successful_times if t >= 100])
                }

                print("  Response Time Distribution:")
                for bucket, count in buckets.items():
                    if count > 0:
                        percentage = (count / len(successful_times)) * 100
                        print(f"    {bucket}: {count} ({percentage:.1f}%)")

def analyze_resource_patterns(data: Dict[str, Any]):
    """Analyze system resource usage patterns."""
    memory_data = data.get("memory_usage", [])
    cpu_data = data.get("cpu_usage", [])

    print("\nRESOURCE USAGE PATTERNS")

    if memory_data:
        memory_percents = [m["percent"] for m in memory_data]
        print("\nMemory Usage:")
        print(f"  Average: {statistics.mean(memory_percents):.2f}%")
        print(f"  Peak: {max(memory_percents):.2f}%")
        print(f"  Min: {min(memory_percents):.2f}%")

        # Memory trend analysis
        if len(memory_percents) > 1:
            memory_trend = "stable"
            if memory_percents[-1] > memory_percents[0] * 1.1:
                memory_trend = "increasing"
            elif memory_percents[-1] < memory_percents[0] * 0.9:
                memory_trend = "decreasing"
            print(f"  Memory Trend: {memory_trend}")

    if cpu_data:
        cpu_percents = [c["percent"] for c in cpu_data]
        print("\nCPU Usage:")
        print(f"  Average: {statistics.mean(cpu_percents):.2f}%")
        print(f"  Peak: {max(cpu_percents):.2f}%")
        print(f"  Min: {min(cpu_percents):.2f}%")

        # CPU trend analysis
        if len(cpu_percents) > 1:
            cpu_trend = "stable"
            if cpu_percents[-1] > cpu_percents[0] * 1.1:
                cpu_trend = "increasing"
            elif cpu_percents[-1] < cpu_percents[0] * 0.9:
                cpu_trend = "decreasing"
            print(f"  CPU Trend: {cpu_trend}")

def generate_recommendations(data: Dict[str, Any]):
    """Generate performance recommendations."""
    print("\nPERFORMANCE RECOMMENDATIONS")

    summary = data.get("summary", {})
    success_rate = summary.get("overall_success_rate", 100)
    avg_response_time = summary.get("average_response_time_ms", 0)

    recommendations = []

    # Success rate recommendations
    if success_rate < 80:
        recommendations.append("CRITICAL: Low success rate detected. Immediate investigation required.")
    elif success_rate < 95:
        recommendations.append("WARNING: Moderate success rate. Error handling improvements needed.")

    # Response time recommendations
    if avg_response_time > 100:
        recommendations.append("High average response time. Consider database query optimization.")
    elif avg_response_time > 50:
        recommendations.append("Moderate response time. Consider implementing caching.")
    else:
        recommendations.append("Excellent response times achieved.")

    # Error analysis recommendations
    errors = data.get("errors", [])
    if errors:
        recommendations.append(f"Found {len(errors)} errors. Review error logs for root causes.")

    # Resource recommendations
    memory_data = data.get("memory_usage", [])
    if memory_data:
        max_memory = max(m["percent"] for m in memory_data)
        if max_memory > 85:
            recommendations.append("High memory usage detected. Monitor for memory leaks.")
        elif max_memory > 70:
            recommendations.append("Moderate memory usage. Consider memory optimization.")

    cpu_data = data.get("cpu_usage", [])
    if cpu_data:
        max_cpu = max(c["percent"] for c in cpu_data)
        if max_cpu > 80:
            recommendations.append("High CPU usage detected. Consider load balancing.")
        elif max_cpu > 60:
            recommendations.append("Moderate CPU usage. Monitor performance under higher load.")

    # Throughput recommendations
    throughput = data.get("throughput", {})
    if throughput:
        for test_type, test_data in throughput.items():
            if isinstance(test_data, dict) and "throughput_rps" in test_data:
                rps = test_data["throughput_rps"]
                if rps < 10:
                    recommendations.append(f"Low throughput ({rps:.1f} RPS) for {test_type}. Consider optimization.")
                elif rps > 50:
                    recommendations.append(f"Good throughput ({rps:.1f} RPS) for {test_type}.")

    if not recommendations:
        recommendations.append("Overall performance is excellent. No major issues detected.")

    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

if __name__ == "__main__":
    analyze_detailed_performance("ces_performance_test_20250901_232604.json")