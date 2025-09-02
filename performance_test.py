#!/usr/bin/env python3
"""
CES Server Performance Testing Script

Comprehensive performance testing for CES (Cognitive Enhancement System) server.
Tests response times, throughput, memory usage, and system stability.
"""

import asyncio
import aiohttp
import time
import json
import psutil
import threading
import statistics
from datetime import datetime
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor
import requests
import matplotlib.pyplot as plt
import numpy as np

class CESPerformanceTester:
    def __init__(self, base_url: str = "http://127.0.0.1:8001"):
        self.base_url = base_url
        self.session = None
        self.results = {
            "response_times": {},
            "throughput": {},
            "memory_usage": [],
            "cpu_usage": [],
            "errors": [],
            "timestamps": []
        }

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    def record_system_metrics(self):
        """Record current system resource usage."""
        try:
            memory = psutil.virtual_memory()
            cpu = psutil.cpu_percent(interval=0.1)

            self.results["memory_usage"].append({
                "timestamp": datetime.now().isoformat(),
                "percent": memory.percent,
                "used_mb": memory.used / 1024 / 1024,
                "available_mb": memory.available / 1024 / 1024
            })

            self.results["cpu_usage"].append({
                "timestamp": datetime.now().isoformat(),
                "percent": cpu
            })

            self.results["timestamps"].append(datetime.now().isoformat())

        except Exception as e:
            self.results["errors"].append(f"System metrics error: {e}")

    async def make_request(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict[str, Any]:
        """Make a single HTTP request and measure response time."""
        url = f"{self.base_url}{endpoint}"

        start_time = time.time()
        try:
            if method == "GET":
                async with self.session.get(url) as response:
                    result = await response.text()
                    status = response.status
            elif method == "POST":
                async with self.session.post(url, json=data) as response:
                    result = await response.text()
                    status = response.status
            else:
                raise ValueError(f"Unsupported method: {method}")

            response_time = (time.time() - start_time) * 1000  # Convert to milliseconds

            return {
                "endpoint": endpoint,
                "method": method,
                "response_time_ms": response_time,
                "status_code": status,
                "success": status < 400,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            self.results["errors"].append(f"Request error for {endpoint}: {e}")
            return {
                "endpoint": endpoint,
                "method": method,
                "response_time_ms": response_time,
                "status_code": None,
                "success": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def test_simple_requests(self, iterations: int = 100) -> Dict[str, Any]:
        """Test simple, low-complexity endpoints."""
        print(f"Testing simple requests ({iterations} iterations)...")

        endpoints = [
            "/api/health",
            "/api/users/active",
            "/api/sessions"
        ]

        results = []

        for i in range(iterations):
            for endpoint in endpoints:
                result = await self.make_request(endpoint)
                results.append(result)
                self.record_system_metrics()

                if (i + 1) % 10 == 0:
                    print(f"  Completed {i + 1}/{iterations} iterations")

        # Calculate statistics
        response_times = [r["response_time_ms"] for r in results if r["success"]]
        stats = self.calculate_statistics(response_times)

        self.results["response_times"]["simple"] = {
            "results": results,
            "statistics": stats,
            "total_requests": len(results),
            "successful_requests": len([r for r in results if r["success"]]),
            "error_rate": (len(results) - len([r for r in results if r["success"]])) / len(results) * 100
        }

        return self.results["response_times"]["simple"]

    async def test_complex_requests(self, iterations: int = 50) -> Dict[str, Any]:
        """Test complex endpoints that may involve AI processing."""
        print(f"Testing complex requests ({iterations} iterations)...")

        endpoints = [
            "/api/system/status",
            "/api/analytics/overview",
            "/api/monitoring/realtime/metrics"
        ]

        results = []

        for i in range(iterations):
            for endpoint in endpoints:
                result = await self.make_request(endpoint)
                results.append(result)
                self.record_system_metrics()

                if (i + 1) % 5 == 0:
                    print(f"  Completed {i + 1}/{iterations} iterations")

        # Calculate statistics
        response_times = [r["response_time_ms"] for r in results if r["success"]]
        stats = self.calculate_statistics(response_times)

        self.results["response_times"]["complex"] = {
            "results": results,
            "statistics": stats,
            "total_requests": len(results),
            "successful_requests": len([r for r in results if r["success"]]),
            "error_rate": (len(results) - len([r for r in results if r["success"]])) / len(results) * 100
        }

        return self.results["response_times"]["complex"]

    async def test_ai_assisted_requests(self, iterations: int = 20) -> Dict[str, Any]:
        """Test AI-assisted endpoints."""
        print(f"Testing AI-assisted requests ({iterations} iterations)...")

        # Test task creation and analysis
        task_data = {
            "description": "Implement a machine learning model for image classification using TensorFlow",
            "priority": "high",
            "tags": ["ml", "tensorflow", "computer-vision"],
            "user_id": "test_user"
        }

        results = []

        for i in range(iterations):
            # Create task
            create_result = await self.make_request("/api/tasks", "POST", task_data)
            results.append(create_result)

            if create_result["success"]:
                task_response = json.loads(create_result.get("result", "{}"))
                task_id = task_response.get("task_id")

                if task_id:
                    # Get task details
                    get_result = await self.make_request(f"/api/tasks/{task_id}")
                    results.append(get_result)

                    # Test AI specialization analysis
                    analyze_result = await self.make_request("/api/ai/specialization/analyze", "POST", task_data)
                    results.append(analyze_result)

            self.record_system_metrics()

            if (i + 1) % 5 == 0:
                print(f"  Completed {i + 1}/{iterations} iterations")

        # Calculate statistics
        response_times = [r["response_time_ms"] for r in results if r["success"]]
        stats = self.calculate_statistics(response_times)

        self.results["response_times"]["ai_assisted"] = {
            "results": results,
            "statistics": stats,
            "total_requests": len(results),
            "successful_requests": len([r for r in results if r["success"]]),
            "error_rate": (len(results) - len([r for r in results if r["success"]])) / len(results) * 100
        }

        return self.results["response_times"]["ai_assisted"]

    async def test_concurrent_requests(self, concurrent_users: int = 10, duration_seconds: int = 60) -> Dict[str, Any]:
        """Test concurrent request handling."""
        print(f"Testing concurrent requests ({concurrent_users} users, {duration_seconds}s)...")

        async def user_simulation(user_id: int):
            """Simulate a single user's behavior."""
            user_results = []
            start_time = time.time()

            while time.time() - start_time < duration_seconds:
                # Randomly select endpoint
                endpoints = [
                    "/api/health",
                    "/api/users/active",
                    "/api/system/status",
                    "/api/analytics/overview"
                ]

                import random
                endpoint = random.choice(endpoints)

                result = await self.make_request(endpoint)
                user_results.append(result)

                # Small delay between requests
                await asyncio.sleep(random.uniform(0.1, 0.5))

            return user_results

        # Run concurrent users
        start_time = time.time()
        tasks = [user_simulation(i) for i in range(concurrent_users)]
        all_results = await asyncio.gather(*tasks)
        total_time = time.time() - start_time

        # Flatten results
        results = []
        for user_results in all_results:
            results.extend(user_results)

        # Calculate throughput
        total_requests = len(results)
        throughput_rps = total_requests / total_time

        # Calculate statistics
        response_times = [r["response_time_ms"] for r in results if r["success"]]
        stats = self.calculate_statistics(response_times)

        self.results["throughput"]["concurrent"] = {
            "results": results,
            "statistics": stats,
            "total_requests": total_requests,
            "successful_requests": len([r for r in results if r["success"]]),
            "throughput_rps": throughput_rps,
            "concurrent_users": concurrent_users,
            "duration_seconds": duration_seconds,
            "error_rate": (total_requests - len([r for r in results if r["success"]])) / total_requests * 100
        }

        return self.results["throughput"]["concurrent"]

    async def test_load_stability(self, load_levels: List[int] = [5, 10, 20, 50], duration_per_level: int = 30) -> Dict[str, Any]:
        """Test system stability under different load levels."""
        print("Testing system stability under load...")

        stability_results = {}

        for load_level in load_levels:
            print(f"  Testing with {load_level} concurrent users...")

            result = await self.test_concurrent_requests(load_level, duration_per_level)
            stability_results[f"load_{load_level}"] = result

            # Brief cooldown
            await asyncio.sleep(5)

        self.results["throughput"]["stability"] = stability_results
        return stability_results

    def calculate_statistics(self, response_times: List[float]) -> Dict[str, Any]:
        """Calculate statistical measures for response times."""
        if not response_times:
            return {"error": "No valid response times"}

        return {
            "count": len(response_times),
            "mean": statistics.mean(response_times),
            "median": statistics.median(response_times),
            "min": min(response_times),
            "max": max(response_times),
            "std_dev": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "p95": np.percentile(response_times, 95),
            "p99": np.percentile(response_times, 99)
        }

    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        report = {
            "test_timestamp": datetime.now().isoformat(),
            "server_url": self.base_url,
            "summary": {},
            "recommendations": []
        }

        # Overall summary
        total_requests = 0
        total_successful = 0
        all_response_times = []

        for test_type, data in self.results["response_times"].items():
            total_requests += data["total_requests"]
            total_successful += data["successful_requests"]
            all_response_times.extend([r["response_time_ms"] for r in data["results"] if r["success"]])

        for test_type, data in self.results["throughput"].items():
            if "results" in data:
                total_requests += data["total_requests"]
                total_successful += data["successful_requests"]
                all_response_times.extend([r["response_time_ms"] for r in data["results"] if r["success"]])

        report["summary"] = {
            "total_requests": total_requests,
            "successful_requests": total_successful,
            "overall_success_rate": (total_successful / total_requests * 100) if total_requests > 0 else 0,
            "average_response_time_ms": statistics.mean(all_response_times) if all_response_times else 0,
            "p95_response_time_ms": np.percentile(all_response_times, 95) if all_response_times else 0,
            "p99_response_time_ms": np.percentile(all_response_times, 99) if all_response_times else 0,
            "total_errors": len(self.results["errors"])
        }

        # Generate recommendations
        avg_response_time = report["summary"]["average_response_time_ms"]
        success_rate = report["summary"]["overall_success_rate"]

        if avg_response_time > 1000:
            report["recommendations"].append("High average response time detected. Consider optimizing database queries and caching.")
        elif avg_response_time > 500:
            report["recommendations"].append("Moderate response time. Consider implementing response caching.")

        if success_rate < 95:
            report["recommendations"].append("Low success rate detected. Investigate error patterns and implement better error handling.")
        elif success_rate < 99:
            report["recommendations"].append("Acceptable success rate but could be improved with better error recovery.")

        # Memory usage analysis
        if self.results["memory_usage"]:
            memory_percents = [m["percent"] for m in self.results["memory_usage"]]
            avg_memory = statistics.mean(memory_percents)
            max_memory = max(memory_percents)

            if max_memory > 90:
                report["recommendations"].append("High memory usage detected. Consider memory optimization and monitoring.")
            elif avg_memory > 70:
                report["recommendations"].append("Moderate memory usage. Monitor for potential memory leaks.")

        report["detailed_results"] = self.results

        return report

    def save_results(self, filename: str = None):
        """Save test results to file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"ces_performance_test_{timestamp}.json"

        report = self.generate_report()

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"Results saved to {filename}")
        return filename

async def main():
    """Main performance testing function."""
    print("CES Server Performance Testing")
    print("=" * 50)

    async with CESPerformanceTester() as tester:
        try:
            # Test simple requests
            print("\n1. Testing Simple Requests...")
            await tester.test_simple_requests(50)

            # Test complex requests
            print("\n2. Testing Complex Requests...")
            await tester.test_complex_requests(25)

            # Test AI-assisted requests
            print("\n3. Testing AI-Assisted Requests...")
            await tester.test_ai_assisted_requests(10)

            # Test concurrent requests
            print("\n4. Testing Concurrent Request Handling...")
            await tester.test_concurrent_requests(5, 30)

            # Test load stability
            print("\n5. Testing System Stability Under Load...")
            await tester.test_load_stability([3, 5, 8], 20)

            # Generate and save report
            print("\n6. Generating Performance Report...")
            report = tester.generate_report()

            print("\n" + "=" * 50)
            print("PERFORMANCE TEST SUMMARY")
            print("=" * 50)
            print(".2f")
            print(".2f")
            print(".2f")
            print(".2f")
            print(f"Total Errors: {report['summary']['total_errors']}")

            # Save detailed results
            filename = tester.save_results()
            print(f"\nDetailed results saved to: {filename}")

            return report

        except Exception as e:
            print(f"Error during testing: {e}")
            return None

if __name__ == "__main__":
    # Run the performance tests
    result = asyncio.run(main())

    if result:
        print("\nPerformance testing completed successfully!")
    else:
        print("\nPerformance testing failed!")