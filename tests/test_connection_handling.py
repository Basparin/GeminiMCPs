#!/usr/bin/env python3
"""
Connection Handling Test Script for CodeSage MCP Server

This script tests the connection handling improvements under various load scenarios
to validate that the implemented fixes work correctly.
"""

import asyncio
import aiohttp
import time
import statistics
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionLoadTester:
    """Test connection handling under various load scenarios."""

    def __init__(self, server_url: str = "http://localhost:8000"):
        self.server_url = server_url
        self.session = None

    async def __aenter__(self):
        connector = aiohttp.TCPConnector(
            limit=100,  # Connection pool size
            limit_per_host=10,  # Max connections per host
            ttl_dns_cache=30,
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True,
        )
        timeout = aiohttp.ClientTimeout(
            total=30,  # Total timeout
            connect=10,  # Connection timeout
            sock_read=10,  # Socket read timeout
            sock_connect=5,  # Socket connect timeout
        )
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={"Content-Type": "application/json"}
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    async def make_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make a single MCP request."""
        request_data = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params or {},
            "id": int(time.time() * 1000)
        }

        start_time = time.time()
        try:
            async with self.session.post(f"{self.server_url}/mcp", json=request_data) as response:
                response_time = (time.time() - start_time) * 1000
                if response.status == 200:
                    result = await response.json()
                    return {
                        "success": True,
                        "response_time": response_time,
                        "data": result
                    }
                else:
                    error_text = await response.text()
                    return {
                        "success": False,
                        "response_time": response_time,
                        "error": f"HTTP {response.status}: {error_text}"
                    }
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return {
                "success": False,
                "response_time": response_time,
                "error": str(e)
            }

    async def test_basic_connectivity(self) -> Dict[str, Any]:
        """Test basic server connectivity."""
        logger.info("Testing basic connectivity...")

        # Test root endpoint
        try:
            async with self.session.get(f"{self.server_url}/") as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Root endpoint response: {data}")
                else:
                    logger.error(f"Root endpoint failed: HTTP {response.status}")
                    return {"success": False, "error": f"Root endpoint failed: HTTP {response.status}"}
        except Exception as e:
            logger.error(f"Root endpoint error: {e}")
            return {"success": False, "error": str(e)}

        # Test MCP initialize
        result = await self.make_request("initialize")
        if not result["success"]:
            logger.error(f"Initialize failed: {result['error']}")
            return {"success": False, "error": result["error"]}

        logger.info("Basic connectivity test passed")
        return {"success": True}

    async def test_connection_pooling(self, num_requests: int = 50) -> Dict[str, Any]:
        """Test connection pooling under moderate load."""
        logger.info(f"Testing connection pooling with {num_requests} concurrent requests...")

        start_time = time.time()

        # Create concurrent requests
        tasks = []
        for i in range(num_requests):
            task = self.make_request("tools/list")
            tasks.append(task)

        # Execute all requests concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        total_time = time.time() - start_time

        # Analyze results
        successful_requests = 0
        failed_requests = 0
        response_times = []

        for result in results:
            if isinstance(result, Exception):
                failed_requests += 1
                logger.error(f"Request exception: {result}")
            elif result["success"]:
                successful_requests += 1
                response_times.append(result["response_time"])
            else:
                failed_requests += 1
                logger.error(f"Request failed: {result['error']}")

        success_rate = (successful_requests / num_requests) * 100

        analysis = {
            "total_requests": num_requests,
            "successful_requests": successful_requests,
            "failed_requests": failed_requests,
            "success_rate_percent": success_rate,
            "total_time_seconds": total_time,
            "requests_per_second": num_requests / total_time if total_time > 0 else 0,
            "avg_response_time_ms": statistics.mean(response_times) if response_times else 0,
            "median_response_time_ms": statistics.median(response_times) if response_times else 0,
            "p95_response_time_ms": statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else 0,
            "min_response_time_ms": min(response_times) if response_times else 0,
            "max_response_time_ms": max(response_times) if response_times else 0,
        }

        logger.info(f"Connection pooling test completed: {success_rate:.1f}% success rate")
        return analysis

    async def test_connection_recovery(self) -> Dict[str, Any]:
        """Test connection recovery after failures."""
        logger.info("Testing connection recovery...")

        # First, make some successful requests
        logger.info("Making initial successful requests...")
        results = []
        for i in range(5):
            result = await self.make_request("tools/list")
            results.append(result)
            await asyncio.sleep(0.1)  # Small delay between requests

        successful_initial = sum(1 for r in results if r["success"])

        # Simulate connection issues by making requests with invalid data
        logger.info("Testing error handling...")
        error_results = []
        for i in range(3):
            # Make request with invalid method to trigger error
            result = await self.make_request("invalid_method")
            error_results.append(result)

        # Make more successful requests to test recovery
        logger.info("Testing recovery after errors...")
        recovery_results = []
        for i in range(5):
            result = await self.make_request("tools/list")
            recovery_results.append(result)
            await asyncio.sleep(0.1)

        successful_recovery = sum(1 for r in recovery_results if r["success"])

        analysis = {
            "initial_success_count": successful_initial,
            "error_handling_tested": len(error_results),
            "recovery_success_count": successful_recovery,
            "recovery_success_rate": (successful_recovery / 5) * 100,
            "connection_recovery_successful": successful_recovery >= 4  # 80% success rate
        }

        logger.info(f"Connection recovery test: {successful_recovery}/5 successful after errors")
        return analysis

    async def test_metrics_endpoint(self) -> Dict[str, Any]:
        """Test metrics endpoint for connection pool monitoring."""
        logger.info("Testing metrics endpoint...")

        try:
            async with self.session.get(f"{self.server_url}/metrics") as response:
                if response.status == 200:
                    metrics_text = await response.text()

                    # Check for connection pool metrics
                    has_connection_pool_metrics = "codesage_mcp_connection_pool_max_connections" in metrics_text
                    has_provider_metrics = any(provider in metrics_text for provider in ["groq", "openrouter", "google"])

                    analysis = {
                        "metrics_available": True,
                        "connection_pool_metrics_present": has_connection_pool_metrics,
                        "provider_metrics_present": has_provider_metrics,
                        "metrics_length": len(metrics_text),
                    }

                    logger.info(f"Metrics endpoint working: {len(metrics_text)} characters")
                    return analysis
                else:
                    return {
                        "metrics_available": False,
                        "error": f"HTTP {response.status}"
                    }
        except Exception as e:
            return {
                "metrics_available": False,
                "error": str(e)
            }

    async def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run comprehensive connection handling tests."""
        logger.info("Starting comprehensive connection handling test...")

        results = {}

        # Test 1: Basic connectivity
        results["basic_connectivity"] = await self.test_basic_connectivity()

        if not results["basic_connectivity"]["success"]:
            logger.error("Basic connectivity test failed, aborting further tests")
            return results

        # Test 2: Connection pooling
        results["connection_pooling"] = await self.test_connection_pooling(30)

        # Test 3: Connection recovery
        results["connection_recovery"] = await self.test_connection_recovery()

        # Test 4: Metrics endpoint
        results["metrics_endpoint"] = await self.test_metrics_endpoint()

        # Overall assessment
        all_tests_passed = all(
            test_result.get("success", False) if "success" in test_result
            else test_result.get("connection_recovery_successful", False) if "connection_recovery_successful" in test_result
            else test_result.get("metrics_available", False) if "metrics_available" in test_result
            else test_result.get("success_rate_percent", 0) > 80 if "success_rate_percent" in test_result
            else False
            for test_result in results.values()
        )

        results["overall_assessment"] = {
            "all_tests_passed": all_tests_passed,
            "tests_run": len(results),
            "timestamp": time.time(),
        }

        logger.info(f"Comprehensive test completed: {'PASSED' if all_tests_passed else 'FAILED'}")
        return results


async def main():
    """Main test function."""
    print("CodeSage MCP Server - Connection Handling Test")
    print("=" * 50)

    async with ConnectionLoadTester() as tester:
        results = await tester.run_comprehensive_test()

        # Print results
        print("\nTest Results:")
        print("-" * 30)

        for test_name, test_result in results.items():
            if test_name == "overall_assessment":
                continue

            print(f"\n{test_name.upper()}:")
            if isinstance(test_result, dict):
                for key, value in test_result.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
            else:
                print(f"  Result: {test_result}")

        # Overall assessment
        assessment = results["overall_assessment"]
        print("\nOVERALL ASSESSMENT:")
        print(f"  Tests Passed: {assessment['all_tests_passed']}")
        print(f"  Tests Run: {assessment['tests_run']}")

        if assessment["all_tests_passed"]:
            print("  ✅ All connection handling tests PASSED!")
        else:
            print("  ❌ Some connection handling tests FAILED!")


if __name__ == "__main__":
    asyncio.run(main())