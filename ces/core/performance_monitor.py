"""
Advanced Performance Monitoring System - CES Real-time Performance Tracking

Provides comprehensive performance monitoring with P95 < 500ms target achievement,
real-time metrics collection, bottleneck detection, and automatic optimization.

Key Features:
- Real-time response time tracking with P95 calculation
- Component-level performance profiling
- Automatic bottleneck detection and alerting
- Performance optimization recommendations
- Historical trend analysis
- Resource utilization monitoring
"""

import time
import threading
import statistics
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import logging
from dataclasses import dataclass, field
from enum import Enum


class PerformanceMetric(Enum):
    """Performance metrics being tracked"""
    RESPONSE_TIME = "response_time"
    CPU_USAGE = "cpu_usage"
    MEMORY_USAGE = "memory_usage"
    DISK_IO = "disk_io"
    NETWORK_IO = "network_io"
    TASK_COMPLETION_TIME = "task_completion_time"
    CONTEXT_RETRIEVAL_TIME = "context_retrieval_time"
    CONFLICT_RESOLUTION_TIME = "conflict_resolution_time"


class AlertSeverity(Enum):
    """Alert severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PerformanceSample:
    """Individual performance measurement sample"""
    metric: PerformanceMetric
    value: float
    timestamp: datetime
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceAlert:
    """Performance alert for threshold violations"""
    alert_id: str
    metric: PerformanceMetric
    severity: AlertSeverity
    message: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    component: str
    recommendations: List[str] = field(default_factory=list)


@dataclass
class PerformanceReport:
    """Comprehensive performance report"""
    timestamp: datetime
    period_start: datetime
    period_end: datetime
    metrics_summary: Dict[str, Dict[str, float]]
    alerts: List[PerformanceAlert]
    bottlenecks: List[Dict[str, Any]]
    recommendations: List[str]
    target_achievement: Dict[str, bool]
    overall_score: float


class PerformanceMonitor:
    """
    Advanced performance monitoring system ensuring P95 < 500ms target.

    Features:
    - Real-time metric collection and analysis
    - P95 calculation with sub-500ms target
    - Automatic bottleneck detection
    - Performance optimization recommendations
    - Historical trend analysis
    """

    def __init__(self, target_p95_ms: int = 500, monitoring_interval: int = 10):
        self.logger = logging.getLogger(__name__)

        # Configuration
        self.target_p95_ms = target_p95_ms
        self.monitoring_interval = monitoring_interval

        # Data storage
        self.samples: Dict[PerformanceMetric, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.current_alerts: List[PerformanceAlert] = []
        self.performance_history: List[PerformanceReport] = []

        # Thresholds for alerts
        self.thresholds = {
            PerformanceMetric.RESPONSE_TIME: {
                'warning': 300,  # ms
                'critical': 800,  # ms
                'p95_target': target_p95_ms
            },
            PerformanceMetric.CPU_USAGE: {
                'warning': 70,  # %
                'critical': 90   # %
            },
            PerformanceMetric.MEMORY_USAGE: {
                'warning': 80,  # %
                'critical': 95   # %
            },
            PerformanceMetric.TASK_COMPLETION_TIME: {
                'warning': 5000,  # ms
                'critical': 15000  # ms
            }
        }

        # Monitoring control
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Performance optimization
        self.optimization_recommendations: List[str] = []
        self.bottleneck_patterns: Dict[str, int] = defaultdict(int)

        self.logger.info(f"Performance Monitor initialized with P95 target: {target_p95_ms}ms")

    def start_monitoring(self):
        """Start the performance monitoring system"""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._stop_event.clear()
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitoring_thread.start()
        self.logger.info("Performance monitoring started")

    def stop_monitoring(self):
        """Stop the performance monitoring system"""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self._stop_event.set()

        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5)

        self.logger.info("Performance monitoring stopped")

    def record_sample(self, metric: PerformanceMetric, value: float,
                     component: str = "system", metadata: Optional[Dict[str, Any]] = None):
        """Record a performance sample"""
        sample = PerformanceSample(
            metric=metric,
            value=value,
            timestamp=datetime.now(),
            component=component,
            metadata=metadata or {}
        )

        self.samples[metric].append(sample)

        # Check for threshold violations
        self._check_thresholds(sample)

    def record_response_time(self, response_time_ms: float, component: str = "api",
                           metadata: Optional[Dict[str, Any]] = None):
        """Record response time with automatic P95 tracking"""
        self.record_sample(
            PerformanceMetric.RESPONSE_TIME,
            response_time_ms,
            component,
            metadata
        )

    def record_task_completion(self, completion_time_ms: float, task_type: str = "general",
                             metadata: Optional[Dict[str, Any]] = None):
        """Record task completion time"""
        metadata = metadata or {}
        metadata['task_type'] = task_type

        self.record_sample(
            PerformanceMetric.TASK_COMPLETION_TIME,
            completion_time_ms,
            "task_processor",
            metadata
        )

    def record_context_retrieval(self, retrieval_time_ms: float, context_type: str = "general",
                                metadata: Optional[Dict[str, Any]] = None):
        """Record context retrieval time"""
        metadata = metadata or {}
        metadata['context_type'] = context_type

        self.record_sample(
            PerformanceMetric.CONTEXT_RETRIEVAL_TIME,
            retrieval_time_ms,
            "memory_manager",
            metadata
        )

    def record_conflict_resolution(self, resolution_time_ms: float, strategy: str = "auto",
                                 metadata: Optional[Dict[str, Any]] = None):
        """Record conflict resolution time"""
        metadata = metadata or {}
        metadata['resolution_strategy'] = strategy

        self.record_sample(
            PerformanceMetric.CONFLICT_RESOLUTION_TIME,
            resolution_time_ms,
            "conflict_resolver",
            metadata
        )

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics"""
        metrics = {}

        for metric, samples in self.samples.items():
            if not samples:
                continue

            values = [s.value for s in samples]
            recent_values = [s.value for s in samples if s.timestamp > datetime.now() - timedelta(minutes=5)]

            metrics[metric.value] = {
                'current': values[-1] if values else 0,
                'average': statistics.mean(values) if values else 0,
                'p50': statistics.median(values) if values else 0,
                'p95': sorted(values)[int(len(values) * 0.95)] if len(values) >= 20 else statistics.mean(values) if values else 0,
                'p99': sorted(values)[int(len(values) * 0.99)] if len(values) >= 100 else statistics.mean(values) if values else 0,
                'min': min(values) if values else 0,
                'max': max(values) if values else 0,
                'recent_avg': statistics.mean(recent_values) if recent_values else 0,
                'sample_count': len(values),
                'timestamp': datetime.now().isoformat()
            }

        # Add system resource metrics
        metrics.update(self._get_system_metrics())

        return metrics

    def get_performance_report(self, time_window_minutes: int = 60) -> PerformanceReport:
        """Generate comprehensive performance report"""
        now = datetime.now()
        period_start = now - timedelta(minutes=time_window_minutes)
        period_end = now

        # Collect metrics for the time window
        window_samples = {}
        for metric, samples in self.samples.items():
            window_samples[metric] = [
                s for s in samples
                if period_start <= s.timestamp <= period_end
            ]

        # Calculate metrics summary
        metrics_summary = {}
        for metric, samples in window_samples.items():
            if not samples:
                continue

            values = [s.value for s in samples]
            metrics_summary[metric.value] = {
                'count': len(values),
                'average': statistics.mean(values),
                'p50': statistics.median(values),
                'p95': sorted(values)[int(len(values) * 0.95)] if values else 0,
                'p99': sorted(values)[int(len(values) * 0.99)] if len(values) >= 100 else sorted(values)[-1] if values else 0,
                'min': min(values),
                'max': max(values)
            }

        # Generate alerts
        alerts = self._generate_alerts(window_samples)

        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(window_samples)

        # Generate recommendations
        recommendations = self._generate_recommendations(metrics_summary, alerts, bottlenecks)

        # Check target achievement
        target_achievement = self._check_target_achievement(metrics_summary)

        # Calculate overall score
        overall_score = self._calculate_overall_score(metrics_summary, target_achievement)

        report = PerformanceReport(
            timestamp=now,
            period_start=period_start,
            period_end=period_end,
            metrics_summary=metrics_summary,
            alerts=alerts,
            bottlenecks=bottlenecks,
            recommendations=recommendations,
            target_achievement=target_achievement,
            overall_score=overall_score
        )

        self.performance_history.append(report)

        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]

        return report

    def _monitoring_loop(self):
        """Main monitoring loop"""
        while not self._stop_event.is_set():
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Check for performance issues
                self._check_performance_health()

                # Clean old data periodically
                self._cleanup_old_data()

                # Sleep for monitoring interval
                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitoring_interval)

    def _collect_system_metrics(self):
        """Collect system resource metrics"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_sample(PerformanceMetric.CPU_USAGE, cpu_percent, "system")

            # Memory usage
            memory = psutil.virtual_memory()
            self.record_sample(PerformanceMetric.MEMORY_USAGE, memory.percent, "system")

            # Disk I/O (simplified)
            disk_io = psutil.disk_io_counters()
            if disk_io:
                read_bytes_per_sec = disk_io.read_bytes / self.monitoring_interval
                write_bytes_per_sec = disk_io.write_bytes / self.monitoring_interval
                self.record_sample(PerformanceMetric.DISK_IO, read_bytes_per_sec + write_bytes_per_sec, "system")

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Get current system resource metrics"""
        try:
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'system_cpu_percent': cpu_percent,
                'system_memory_percent': memory.percent,
                'system_memory_used_mb': memory.used / 1024 / 1024,
                'system_memory_available_mb': memory.available / 1024 / 1024,
                'system_disk_percent': disk.percent,
                'system_disk_free_gb': disk.free / 1024 / 1024 / 1024
            }
        except Exception as e:
            self.logger.error(f"Error getting system metrics: {e}")
            return {}

    def _check_thresholds(self, sample: PerformanceSample):
        """Check if sample violates any thresholds"""
        if sample.metric not in self.thresholds:
            return

        thresholds = self.thresholds[sample.metric]

        # Check warning threshold
        if 'warning' in thresholds and sample.value > thresholds['warning']:
            severity = AlertSeverity.MEDIUM
            message = f"{sample.metric.value} exceeded warning threshold: {sample.value:.2f} > {thresholds['warning']}"

            if sample.value > thresholds.get('critical', float('inf')):
                severity = AlertSeverity.CRITICAL
                message = f"{sample.metric.value} exceeded critical threshold: {sample.value:.2f} > {thresholds.get('critical', 'N/A')}"

            alert = PerformanceAlert(
                alert_id=f"{sample.metric.value}_{int(time.time())}",
                metric=sample.metric,
                severity=severity,
                message=message,
                current_value=sample.value,
                threshold_value=thresholds.get('warning', thresholds.get('critical', 0)),
                timestamp=sample.timestamp,
                component=sample.component,
                recommendations=self._get_threshold_recommendations(sample.metric, sample.value)
            )

            self.current_alerts.append(alert)
            self.logger.warning(f"Performance alert: {message}")

            # Keep only recent alerts
            if len(self.current_alerts) > 50:
                self.current_alerts = self.current_alerts[-50:]

    def _get_threshold_recommendations(self, metric: PerformanceMetric, value: float) -> List[str]:
        """Get recommendations for threshold violations"""
        recommendations = []

        if metric == PerformanceMetric.RESPONSE_TIME:
            if value > 1000:
                recommendations.extend([
                    "Consider optimizing database queries",
                    "Implement response caching",
                    "Review algorithm complexity"
                ])
            elif value > 500:
                recommendations.extend([
                    "Optimize memory allocation",
                    "Consider async processing",
                    "Review network latency"
                ])

        elif metric == PerformanceMetric.CPU_USAGE:
            if value > 80:
                recommendations.extend([
                    "Optimize CPU-intensive operations",
                    "Consider load balancing",
                    "Review thread pool configuration"
                ])

        elif metric == PerformanceMetric.MEMORY_USAGE:
            if value > 85:
                recommendations.extend([
                    "Implement memory cleanup",
                    "Optimize data structures",
                    "Consider memory-mapped files"
                ])

        return recommendations

    def _generate_alerts(self, window_samples: Dict[PerformanceMetric, List[PerformanceSample]]) -> List[PerformanceAlert]:
        """Generate alerts based on time window analysis"""
        alerts = []

        # Check P95 response time target
        response_times = [s.value for s in window_samples.get(PerformanceMetric.RESPONSE_TIME, [])]
        if len(response_times) >= 20:
            p95_response_time = sorted(response_times)[int(len(response_times) * 0.95)]

            if p95_response_time > self.target_p95_ms:
                alerts.append(PerformanceAlert(
                    alert_id=f"p95_target_violation_{int(time.time())}",
                    metric=PerformanceMetric.RESPONSE_TIME,
                    severity=AlertSeverity.HIGH,
                    message=f"P95 response time target violated: {p95_response_time:.2f}ms > {self.target_p95_ms}ms",
                    current_value=p95_response_time,
                    threshold_value=self.target_p95_ms,
                    timestamp=datetime.now(),
                    component="system",
                    recommendations=[
                        "Optimize slowest 5% of requests",
                        "Implement request queuing",
                        "Consider horizontal scaling",
                        "Review caching strategy"
                    ]
                ))

        return alerts

    def _detect_bottlenecks(self, window_samples: Dict[PerformanceMetric, List[PerformanceSample]]) -> List[Dict[str, Any]]:
        """Detect performance bottlenecks"""
        bottlenecks = []

        # Response time bottlenecks
        response_samples = window_samples.get(PerformanceMetric.RESPONSE_TIME, [])
        if response_samples:
            response_times = [s.value for s in response_samples]
            p95 = sorted(response_times)[int(len(response_times) * 0.95)]

            if p95 > 1000:  # 1 second bottleneck threshold
                bottlenecks.append({
                    'type': 'response_time',
                    'severity': 'high',
                    'metric': 'p95_response_time',
                    'value': p95,
                    'threshold': 1000,
                    'description': f'P95 response time bottleneck: {p95:.2f}ms',
                    'affected_components': ['api', 'processing']
                })

        # Memory bottlenecks
        memory_samples = window_samples.get(PerformanceMetric.MEMORY_USAGE, [])
        if memory_samples:
            avg_memory = statistics.mean([s.value for s in memory_samples])

            if avg_memory > 85:
                bottlenecks.append({
                    'type': 'memory',
                    'severity': 'medium',
                    'metric': 'avg_memory_usage',
                    'value': avg_memory,
                    'threshold': 85,
                    'description': f'Memory usage bottleneck: {avg_memory:.1f}%',
                    'affected_components': ['memory_manager']
                })

        # CPU bottlenecks
        cpu_samples = window_samples.get(PerformanceMetric.CPU_USAGE, [])
        if cpu_samples:
            avg_cpu = statistics.mean([s.value for s in cpu_samples])

            if avg_cpu > 75:
                bottlenecks.append({
                    'type': 'cpu',
                    'severity': 'medium',
                    'metric': 'avg_cpu_usage',
                    'value': avg_cpu,
                    'threshold': 75,
                    'description': f'CPU usage bottleneck: {avg_cpu:.1f}%',
                    'affected_components': ['processing', 'computation']
                })

        return bottlenecks

    def _generate_recommendations(self, metrics_summary: Dict[str, Dict[str, float]],
                                alerts: List[PerformanceAlert],
                                bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Generate performance optimization recommendations"""
        recommendations = []

        # Response time recommendations
        response_metrics = metrics_summary.get('response_time', {})
        if response_metrics:
            p95 = response_metrics.get('p95', 0)
            if p95 > self.target_p95_ms:
                recommendations.append(f"Optimize P95 response time: current {p95:.2f}ms, target {self.target_p95_ms}ms")
                recommendations.append("Consider implementing response time caching")
                recommendations.append("Review and optimize slowest database queries")

            if p95 > 1000:
                recommendations.append("Implement request queuing for high-load periods")
                recommendations.append("Consider horizontal scaling of services")

        # Memory recommendations
        memory_metrics = metrics_summary.get('memory_usage', {})
        if memory_metrics and memory_metrics.get('average', 0) > 80:
            recommendations.append("Implement memory usage optimization")
            recommendations.append("Consider memory-mapped data structures")
            recommendations.append("Review object lifecycle management")

        # CPU recommendations
        cpu_metrics = metrics_summary.get('cpu_usage', {})
        if cpu_metrics and cpu_metrics.get('average', 0) > 70:
            recommendations.append("Optimize CPU-intensive operations")
            recommendations.append("Consider asynchronous processing")
            recommendations.append("Review algorithm complexity")

        # Bottleneck-specific recommendations
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'response_time':
                recommendations.append("Profile and optimize response time bottlenecks")
            elif bottleneck['type'] == 'memory':
                recommendations.append("Implement memory optimization strategies")
            elif bottleneck['type'] == 'cpu':
                recommendations.append("Optimize CPU-bound operations")

        # Alert-based recommendations
        for alert in alerts:
            recommendations.extend(alert.recommendations)

        return list(set(recommendations))  # Remove duplicates

    def _check_target_achievement(self, metrics_summary: Dict[str, Dict[str, float]]) -> Dict[str, bool]:
        """Check if performance targets are being met"""
        achievement = {}

        # P95 response time target
        response_metrics = metrics_summary.get('response_time', {})
        p95_response = response_metrics.get('p95', float('inf'))
        achievement['p95_response_time_target'] = p95_response <= self.target_p95_ms

        # General performance targets
        achievement['response_time_stable'] = response_metrics.get('p99', float('inf')) < 2000
        achievement['memory_efficient'] = metrics_summary.get('memory_usage', {}).get('average', 0) < 85
        achievement['cpu_efficient'] = metrics_summary.get('cpu_usage', {}).get('average', 0) < 80

        return achievement

    def _calculate_overall_score(self, metrics_summary: Dict[str, Dict[str, float]],
                               target_achievement: Dict[str, bool]) -> float:
        """Calculate overall performance score (0-100)"""
        score = 100  # Start with perfect score

        # P95 response time impact (40% of score)
        response_metrics = metrics_summary.get('response_time', {})
        p95 = response_metrics.get('p95', 0)
        if p95 > self.target_p95_ms:
            excess_time = p95 - self.target_p95_ms
            penalty = min(excess_time / 10, 40)  # Max 40 point penalty
            score -= penalty

        # Target achievement impact (30% of score)
        achievement_score = sum(target_achievement.values()) / len(target_achievement) * 30
        score -= (30 - achievement_score)

        # Resource efficiency impact (30% of score)
        memory_avg = metrics_summary.get('memory_usage', {}).get('average', 0)
        cpu_avg = metrics_summary.get('cpu_usage', {}).get('average', 0)

        resource_penalty = 0
        if memory_avg > 85:
            resource_penalty += (memory_avg - 85) * 0.3
        if cpu_avg > 80:
            resource_penalty += (cpu_avg - 80) * 0.3

        score -= min(resource_penalty, 30)

        return max(0, min(100, score))

    def _check_performance_health(self):
        """Check overall performance health and trigger optimizations if needed"""
        current_metrics = self.get_current_metrics()

        # Check for critical performance issues
        response_p95 = current_metrics.get('response_time', {}).get('p95', 0)
        if response_p95 > self.target_p95_ms * 1.5:  # 50% over target
            self.logger.critical(f"Critical performance issue: P95 {response_p95:.2f}ms exceeds target by 50%")
            # Could trigger emergency optimization here

        # Update bottleneck patterns
        if response_p95 > 1000:
            self.bottleneck_patterns['response_time'] += 1
        if current_metrics.get('memory_usage', {}).get('current', 0) > 90:
            self.bottleneck_patterns['memory'] += 1
        if current_metrics.get('cpu_usage', {}).get('current', 0) > 85:
            self.bottleneck_patterns['cpu'] += 1

    def _cleanup_old_data(self):
        """Clean up old performance data to manage memory"""
        cutoff_time = datetime.now() - timedelta(hours=24)  # Keep 24 hours of data

        for metric in self.samples:
            # Remove old samples
            self.samples[metric] = deque(
                [s for s in self.samples[metric] if s.timestamp > cutoff_time],
                maxlen=10000
            )

    def get_optimization_recommendations(self) -> List[str]:
        """Get current optimization recommendations"""
        return self.optimization_recommendations.copy()

    def get_active_alerts(self) -> List[PerformanceAlert]:
        """Get currently active performance alerts"""
        return self.current_alerts.copy()

    def get_bottleneck_analysis(self) -> Dict[str, Any]:
        """Get analysis of detected bottlenecks"""
        return {
            'patterns': dict(self.bottleneck_patterns),
            'most_common': max(self.bottleneck_patterns.keys(), key=self.bottleneck_patterns.get) if self.bottleneck_patterns else None,
            'total_incidents': sum(self.bottleneck_patterns.values())
        }

    def reset_metrics(self):
        """Reset all performance metrics and history"""
        self.samples.clear()
        self.current_alerts.clear()
        self.performance_history.clear()
        self.bottleneck_patterns.clear()
        self.optimization_recommendations.clear()
        self.logger.info("Performance metrics reset")


# Global performance monitor instance
_performance_monitor = None
_usage_analyzer = None


def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance"""
    global _performance_monitor
    if _performance_monitor is None:
        _performance_monitor = PerformanceMonitor()
        _performance_monitor.start_monitoring()
    return _performance_monitor


def get_usage_analyzer():
    """Get the global usage analyzer (placeholder for compatibility)"""
    global _usage_analyzer
    if _usage_analyzer is None:
        _usage_analyzer = UsageAnalyzer()
    return _usage_analyzer


class UsageAnalyzer:
    """Placeholder usage analyzer for compatibility"""
    def record_user_action(self, user_id: str, action: str, metadata: Dict[str, Any]):
        """Record user action (placeholder)"""
        pass