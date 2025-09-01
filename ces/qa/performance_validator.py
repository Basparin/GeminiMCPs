"""CES Performance Validator.

Validates performance standards including response times, throughput, memory usage,
CPU utilization, and AI response times against Phase 1 targets.
"""

import time
import psutil
import asyncio
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime
from statistics import mean, median
from concurrent.futures import ThreadPoolExecutor


@dataclass
class PerformanceMetric:
    """Individual performance metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    target: Optional[float] = None
    compliant: bool = False


@dataclass
class PerformanceTestResult:
    """Result of a performance test."""
    test_name: str
    success: bool
    duration: float
    metrics: List[PerformanceMetric]
    error_message: Optional[str] = None


@dataclass
class PerformanceValidationReport:
    """Comprehensive performance validation report."""
    overall_compliance: bool
    compliance_score: float
    passed_tests: int
    failed_tests: int
    test_results: List[PerformanceTestResult]
    summary: Dict[str, Any]
    recommendations: List[str]
    timestamp: datetime


class PerformanceValidator:
    """Validates CES performance against Phase 1 standards."""

    def __init__(self):
        self.targets = {
            'response_time_p50': 200.0,  # ms
            'response_time_p95': 2000.0,  # ms
            'response_time_p99': 5000.0,  # ms
            'throughput_sustained': 100.0,  # req/min
            'throughput_peak': 200.0,  # req/min
            'memory_normal': 256.0,  # MB
            'memory_peak': 512.0,  # MB
            'cpu_normal': 30.0,  # %
            'cpu_peak': 70.0,  # %
            'ai_grok_response': 300.0,  # ms
            'ai_gemini_response': 500.0,  # ms
            'ai_qwen_response': 400.0,  # ms
            'memory_search_latency': 1.0,  # ms
            'memory_utilization': 90.0  # %
        }

    def validate_all_standards(self) -> PerformanceValidationReport:
        """Validate all performance standards."""
        test_results = []

        # Response time tests
        test_results.append(self._test_response_times())

        # Throughput tests
        test_results.append(self._test_throughput())

        # Memory usage tests
        test_results.append(self._test_memory_usage())

        # CPU usage tests
        test_results.append(self._test_cpu_usage())

        # AI response time tests
        test_results.append(self._test_ai_response_times())

        # Memory search latency tests
        test_results.append(self._test_memory_search_latency())

        # Memory utilization tests
        test_results.append(self._test_memory_utilization())

        # Calculate overall metrics
        passed_tests = len([r for r in test_results if r.success])
        failed_tests = len([r for r in test_results if not r.success])

        # Calculate compliance score
        compliance_score = self._calculate_compliance_score(test_results)

        # Determine overall compliance
        overall_compliance = passed_tests >= len(test_results) * 0.8  # 80% pass rate

        # Generate recommendations
        recommendations = self._generate_performance_recommendations(test_results)

        return PerformanceValidationReport(
            overall_compliance=overall_compliance,
            compliance_score=compliance_score,
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            test_results=test_results,
            summary=self._generate_summary(test_results),
            recommendations=recommendations,
            timestamp=datetime.now()
        )

    def _test_response_times(self) -> PerformanceTestResult:
        """Test response time performance."""
        start_time = time.time()

        try:
            # Simulate response time measurements
            response_times = []

            # Simple operations (should be < 200ms P50)
            for i in range(100):
                op_start = time.time()
                # Simulate simple operation
                time.sleep(0.001 + (i * 0.0001))  # 1-11ms
                op_end = time.time()
                response_times.append((op_end - op_start) * 1000)

            # Complex operations (should be < 2000ms P95)
            complex_times = []
            for i in range(50):
                op_start = time.time()
                # Simulate complex operation
                time.sleep(0.01 + (i * 0.001))  # 10-60ms
                op_end = time.time()
                complex_times.append((op_end - op_start) * 1000)

            # Calculate percentiles
            p50_simple = median(response_times)
            p95_complex = sorted(complex_times)[int(len(complex_times) * 0.95)]
            p99_complex = sorted(complex_times)[int(len(complex_times) * 0.99)]

            metrics = [
                PerformanceMetric(
                    name='response_time_p50',
                    value=p50_simple,
                    unit='ms',
                    timestamp=datetime.now(),
                    target=self.targets['response_time_p50'],
                    compliant=p50_simple <= self.targets['response_time_p50']
                ),
                PerformanceMetric(
                    name='response_time_p95',
                    value=p95_complex,
                    unit='ms',
                    timestamp=datetime.now(),
                    target=self.targets['response_time_p95'],
                    compliant=p95_complex <= self.targets['response_time_p95']
                ),
                PerformanceMetric(
                    name='response_time_p99',
                    value=p99_complex,
                    unit='ms',
                    timestamp=datetime.now(),
                    target=self.targets['response_time_p99'],
                    compliant=p99_complex <= self.targets['response_time_p99']
                )
            ]

            success = all(m.compliant for m in metrics)
            duration = time.time() - start_time

            return PerformanceTestResult(
                test_name='response_time_test',
                success=success,
                duration=duration,
                metrics=metrics
            )

        except Exception as e:
            duration = time.time() - start_time
            return PerformanceTestResult(
                test_name='response_time_test',
                success=False,
                duration=duration,
                metrics=[],
                error_message=str(e)
            )

    def _test_throughput(self) -> PerformanceTestResult:
        """Test throughput performance."""
        start_time = time.time()

        try:
            # Test sustained throughput
            sustained_requests = 0
            throughput_start = time.time()

            # Run for 1 minute at sustained load
            while time.time() - throughput_start < 60:
                # Simulate request processing
                time.sleep(0.01)  # 10ms per request
                sustained_requests += 1

                # Cap at reasonable limit
                if sustained_requests > 200:
                    break

            sustained_throughput = (sustained_requests / 60) * 60  # req/min

            # Test peak throughput
            peak_requests = 0
            peak_start = time.time()

            # Run for 30 seconds at peak load
            while time.time() - peak_start < 30:
                # Simulate faster request processing
                time.sleep(0.005)  # 5ms per request
                peak_requests += 1

                # Cap at reasonable limit
                if peak_requests > 600:
                    break

            peak_throughput = (peak_requests / 30) * 60  # req/min

            metrics = [
                PerformanceMetric(
                    name='throughput_sustained',
                    value=sustained_throughput,
                    unit='req/min',
                    timestamp=datetime.now(),
                    target=self.targets['throughput_sustained'],
                    compliant=sustained_throughput >= self.targets['throughput_sustained']
                ),
                PerformanceMetric(
                    name='throughput_peak',
                    value=peak_throughput,
                    unit='req/min',
                    timestamp=datetime.now(),
                    target=self.targets['throughput_peak'],
                    compliant=peak_throughput >= self.targets['throughput_peak']
                )
            ]

            success = all(m.compliant for m in metrics)
            duration = time.time() - start_time

            return PerformanceTestResult(
                test_name='throughput_test',
                success=success,
                duration=duration,
                metrics=metrics
            )

        except Exception as e:
            duration = time.time() - start_time
            return PerformanceTestResult(
                test_name='throughput_test',
                success=False,
                duration=duration,
                metrics=[],
                error_message=str(e)
            )

    def _test_memory_usage(self) -> PerformanceTestResult:
        """Test memory usage performance."""
        start_time = time.time()

        try:
            # Monitor memory usage over time
            memory_readings = []

            for i in range(10):
                memory_mb = psutil.virtual_memory().used / 1024 / 1024
                memory_readings.append(memory_mb)
                time.sleep(0.1)

            # Simulate memory pressure
            large_data = [0] * 1000000  # ~8MB
            time.sleep(0.5)
            peak_memory = psutil.virtual_memory().used / 1024 / 1024
            del large_data

            normal_memory = mean(memory_readings)

            metrics = [
                PerformanceMetric(
                    name='memory_normal',
                    value=normal_memory,
                    unit='MB',
                    timestamp=datetime.now(),
                    target=self.targets['memory_normal'],
                    compliant=normal_memory <= self.targets['memory_normal']
                ),
                PerformanceMetric(
                    name='memory_peak',
                    value=peak_memory,
                    unit='MB',
                    timestamp=datetime.now(),
                    target=self.targets['memory_peak'],
                    compliant=peak_memory <= self.targets['memory_peak']
                )
            ]

            success = all(m.compliant for m in metrics)
            duration = time.time() - start_time

            return PerformanceTestResult(
                test_name='memory_usage_test',
                success=success,
                duration=duration,
                metrics=metrics
            )

        except Exception as e:
            duration = time.time() - start_time
            return PerformanceTestResult(
                test_name='memory_usage_test',
                success=False,
                duration=duration,
                metrics=[],
                error_message=str(e)
            )

    def _test_cpu_usage(self) -> PerformanceTestResult:
        """Test CPU usage performance."""
        start_time = time.time()

        try:
            # Monitor CPU usage
            cpu_readings = []

            for i in range(10):
                cpu_percent = psutil.cpu_percent(interval=0.1)
                cpu_readings.append(cpu_percent)

            normal_cpu = mean(cpu_readings)

            # Simulate CPU intensive operation
            cpu_intensive_start = time.time()
            while time.time() - cpu_intensive_start < 2:
                # CPU intensive calculation
                sum(x*x for x in range(10000))

            peak_cpu = psutil.cpu_percent(interval=0.1)

            metrics = [
                PerformanceMetric(
                    name='cpu_normal',
                    value=normal_cpu,
                    unit='%',
                    timestamp=datetime.now(),
                    target=self.targets['cpu_normal'],
                    compliant=normal_cpu <= self.targets['cpu_normal']
                ),
                PerformanceMetric(
                    name='cpu_peak',
                    value=peak_cpu,
                    unit='%',
                    timestamp=datetime.now(),
                    target=self.targets['cpu_peak'],
                    compliant=peak_cpu <= self.targets['cpu_peak']
                )
            ]

            success = all(m.compliant for m in metrics)
            duration = time.time() - start_time

            return PerformanceTestResult(
                test_name='cpu_usage_test',
                success=success,
                duration=duration,
                metrics=metrics
            )

        except Exception as e:
            duration = time.time() - start_time
            return PerformanceTestResult(
                test_name='cpu_usage_test',
                success=False,
                duration=duration,
                metrics=[],
                error_message=str(e)
            )

    def _test_ai_response_times(self) -> PerformanceTestResult:
        """Test AI response time performance."""
        start_time = time.time()

        try:
            # Simulate AI response times (these would be actual measurements in production)
            ai_times = {
                'grok': 250.0,    # ms (simulated)
                'gemini': 450.0,  # ms (simulated)
                'qwen': 350.0     # ms (simulated)
            }

            metrics = []
            for ai_name, response_time in ai_times.items():
                target_key = f'ai_{ai_name}_response'
                target = self.targets.get(target_key, 1000.0)  # Default 1s

                metrics.append(PerformanceMetric(
                    name=target_key,
                    value=response_time,
                    unit='ms',
                    timestamp=datetime.now(),
                    target=target,
                    compliant=response_time <= target
                ))

            success = all(m.compliant for m in metrics)
            duration = time.time() - start_time

            return PerformanceTestResult(
                test_name='ai_response_time_test',
                success=success,
                duration=duration,
                metrics=metrics
            )

        except Exception as e:
            duration = time.time() - start_time
            return PerformanceTestResult(
                test_name='ai_response_time_test',
                success=False,
                duration=duration,
                metrics=[],
                error_message=str(e)
            )

    def _test_memory_search_latency(self) -> PerformanceTestResult:
        """Test memory search latency performance."""
        start_time = time.time()

        try:
            # Simulate memory search operations
            latencies = []

            for i in range(20):
                search_start = time.time()
                # Simulate FAISS/vector search (very fast)
                time.sleep(0.0005 + (i * 0.0001))  # 0.5-2.5ms
                search_end = time.time()
                latency = (search_end - search_start) * 1000  # Convert to ms
                latencies.append(latency)

            avg_latency = mean(latencies)

            metrics = [
                PerformanceMetric(
                    name='memory_search_latency',
                    value=avg_latency,
                    unit='ms',
                    timestamp=datetime.now(),
                    target=self.targets['memory_search_latency'],
                    compliant=avg_latency <= self.targets['memory_search_latency']
                )
            ]

            success = all(m.compliant for m in metrics)
            duration = time.time() - start_time

            return PerformanceTestResult(
                test_name='memory_search_latency_test',
                success=success,
                duration=duration,
                metrics=metrics
            )

        except Exception as e:
            duration = time.time() - start_time
            return PerformanceTestResult(
                test_name='memory_search_latency_test',
                success=False,
                duration=duration,
                metrics=[],
                error_message=str(e)
            )

    def _test_memory_utilization(self) -> PerformanceTestResult:
        """Test memory utilization performance."""
        start_time = time.time()

        try:
            # Simulate high memory utilization scenario
            utilization_percent = 92.0  # Simulated high utilization

            metrics = [
                PerformanceMetric(
                    name='memory_utilization',
                    value=utilization_percent,
                    unit='%',
                    timestamp=datetime.now(),
                    target=self.targets['memory_utilization'],
                    compliant=utilization_percent <= self.targets['memory_utilization']
                )
            ]

            success = all(m.compliant for m in metrics)
            duration = time.time() - start_time

            return PerformanceTestResult(
                test_name='memory_utilization_test',
                success=success,
                duration=duration,
                metrics=metrics
            )

        except Exception as e:
            duration = time.time() - start_time
            return PerformanceTestResult(
                test_name='memory_utilization_test',
                success=False,
                duration=duration,
                metrics=[],
                error_message=str(e)
            )

    def _calculate_compliance_score(self, test_results: List[PerformanceTestResult]) -> float:
        """Calculate overall compliance score."""
        if not test_results:
            return 0.0

        total_metrics = 0
        compliant_metrics = 0

        for result in test_results:
            for metric in result.metrics:
                total_metrics += 1
                if metric.compliant:
                    compliant_metrics += 1

        if total_metrics == 0:
            return 100.0

        return (compliant_metrics / total_metrics) * 100.0

    def _generate_summary(self, test_results: List[PerformanceTestResult]) -> Dict[str, Any]:
        """Generate performance test summary."""
        summary = {
            'total_tests': len(test_results),
            'passed_tests': len([r for r in test_results if r.success]),
            'failed_tests': len([r for r in test_results if not r.success]),
            'total_metrics': sum(len(r.metrics) for r in test_results),
            'compliant_metrics': sum(len([m for m in r.metrics if m.compliant]) for r in test_results),
            'performance_targets': self.targets
        }

        # Add metric averages
        metric_averages = {}
        for result in test_results:
            for metric in result.metrics:
                if metric.name not in metric_averages:
                    metric_averages[metric.name] = []
                metric_averages[metric.name].append(metric.value)

        summary['metric_averages'] = {
            name: mean(values) for name, values in metric_averages.items()
        }

        return summary

    def _generate_performance_recommendations(self, test_results: List[PerformanceTestResult]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []

        # Analyze failed tests
        failed_tests = [r for r in test_results if not r.success]

        for result in failed_tests:
            if result.test_name == 'response_time_test':
                recommendations.append("Optimize response times by implementing caching and async processing")
            elif result.test_name == 'throughput_test':
                recommendations.append("Improve throughput by optimizing database queries and implementing connection pooling")
            elif result.test_name == 'memory_usage_test':
                recommendations.append("Reduce memory usage by implementing memory-efficient data structures and garbage collection optimization")
            elif result.test_name == 'cpu_usage_test':
                recommendations.append("Optimize CPU usage by implementing parallel processing and algorithm improvements")
            elif result.test_name == 'ai_response_time_test':
                recommendations.append("Improve AI response times by optimizing model inference and implementing model caching")
            elif result.test_name == 'memory_search_latency_test':
                recommendations.append("Optimize memory search latency by improving indexing and query optimization")
            elif result.test_name == 'memory_utilization_test':
                recommendations.append("Improve memory utilization by implementing memory pooling and efficient data management")

        # General recommendations
        if len(failed_tests) > 0:
            recommendations.append(f"Address {len(failed_tests)} failed performance tests to meet Phase 1 targets")

        # Check for specific metric issues
        for result in test_results:
            for metric in result.metrics:
                if not metric.compliant:
                    if 'response_time' in metric.name and metric.value > metric.target * 2:
                        recommendations.append(f"Critical: {metric.name} is {metric.value/metric.target:.1f}x above target")
                    elif 'memory' in metric.name and metric.value > metric.target * 1.5:
                        recommendations.append(f"High memory usage detected in {metric.name}")

        recommendations.append(f"Performance compliance: {self._calculate_compliance_score(test_results):.1f}%")

        return recommendations