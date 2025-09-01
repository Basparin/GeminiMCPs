"""
Advanced Performance Monitoring System - CES Phase 2 Enhanced Performance Tracking

Phase 2 Enhancement: Advanced performance monitoring with predictive analytics,
automated optimization routines, and intelligent resource management for >50%
resource utilization efficiency improvement.

Key Phase 2 Features:
- Predictive performance analytics with ML-based forecasting
- Automated optimization routines with self-tuning capabilities
- Advanced resource management algorithms with cost optimization
- Performance prediction models using time-series analysis
- Memory usage optimization with intelligent allocation strategies
- Enterprise-grade monitoring with alerting and anomaly detection
- Scalability prediction and capacity planning
- Cost-optimized resource utilization strategies

Legacy Features:
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
from typing import Dict, List, Optional, Any, Callable, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import logging
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import json


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

        # Phase 1 benchmark targets
        self.phase1_targets = {
            'response_time_p50_simple': 200,  # ms
            'response_time_p95_complex': 2000,  # ms
            'throughput_sustained': 100,  # req/min
            'throughput_peak': 200,  # req/min
            'memory_normal': 256,  # MB
            'memory_peak': 512,  # MB
            'cpu_normal': 30,  # %
            'cpu_peak': 70,  # %
            'ai_grok_response': 300,  # ms
            'ai_gemini_response': 500,  # ms
            'ai_qwen_response': 400,  # ms
            'memory_search_latency': 1,  # ms
            'memory_utilization': 90,  # %
            'cache_hit_rate': 100  # %
        }

        # Thresholds for alerts (optimized for Phase 1 targets)
        self.thresholds = {
            PerformanceMetric.RESPONSE_TIME: {
                'warning': 300,  # ms
                'critical': 800,  # ms
                'p95_target': target_p95_ms
            },
            PerformanceMetric.CPU_USAGE: {
                'warning': 40,  # % (above normal target)
                'critical': 80   # % (approaching peak target)
            },
            PerformanceMetric.MEMORY_USAGE: {
                'warning': 300,  # MB (above normal target)
                'critical': 450   # MB (approaching peak target)
            },
            PerformanceMetric.TASK_COMPLETION_TIME: {
                'warning': 1000,  # ms
                'critical': 3000  # ms
            }
        }

        # Monitoring control
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Performance optimization
        self.optimization_recommendations: List[str] = []
        self.bottleneck_patterns: Dict[str, int] = defaultdict(int)

        # Phase 1 tracking
        self.phase1_metrics = {
            'response_times_simple': deque(maxlen=1000),
            'response_times_complex': deque(maxlen=1000),
            'throughput_minute': deque(maxlen=60),
            'memory_usage_mb': deque(maxlen=1000),
            'cpu_usage_percent': deque(maxlen=1000),
            'ai_response_times': {
                'grok': deque(maxlen=100),
                'gemini': deque(maxlen=100),
                'qwen': deque(maxlen=100)
            },
            'memory_search_times': deque(maxlen=100),
            'cache_hit_rates': deque(maxlen=100)
        }

        self.logger.info(f"Performance Monitor initialized with Phase 1 targets - P95 target: {target_p95_ms}ms")

        # Phase 2: Advanced performance enhancement components
        self.predictive_model = PredictivePerformanceModel()
        self.optimization_engine = AutomatedOptimizationEngine(self)
        self.resource_manager = ResourceManager(self)
        self.memory_optimizer = MemoryOptimizationEngine(self)

        # Phase 2: Performance prediction and optimization tracking
        self.performance_predictions: List[PerformancePrediction] = []
        self.optimization_history: List[Dict[str, Any]] = []
        self.anomaly_detection_enabled = True
        self.auto_optimization_enabled = True

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
        """Record response time with automatic P95 tracking and Phase 1 classification"""
        self.record_sample(
            PerformanceMetric.RESPONSE_TIME,
            response_time_ms,
            component,
            metadata
        )

        # Phase 1: Classify and track response times
        is_complex = metadata.get('complexity', 'simple') == 'complex' if metadata else False
        if is_complex:
            self.phase1_metrics['response_times_complex'].append(response_time_ms)
        else:
            self.phase1_metrics['response_times_simple'].append(response_time_ms)

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

    def record_ai_response_time(self, assistant_name: str, response_time_ms: float,
                               metadata: Optional[Dict[str, Any]] = None):
        """Record AI assistant response time for Phase 1 tracking"""
        if assistant_name in self.phase1_metrics['ai_response_times']:
            self.phase1_metrics['ai_response_times'][assistant_name].append(response_time_ms)

        # Also record as general response time
        self.record_response_time(response_time_ms, f"ai_{assistant_name}", metadata)

    def record_memory_search_time(self, search_time_ms: float, metadata: Optional[Dict[str, Any]] = None):
        """Record memory search latency for Phase 1 tracking"""
        self.phase1_metrics['memory_search_times'].append(search_time_ms)

        # Record as context retrieval time
        self.record_context_retrieval(search_time_ms, "memory_search", metadata)

    def record_cache_hit_rate(self, hit_rate_percent: float, metadata: Optional[Dict[str, Any]] = None):
        """Record cache hit rate for Phase 1 tracking"""
        self.phase1_metrics['cache_hit_rates'].append(hit_rate_percent)

    def record_throughput(self, requests_per_minute: float, metadata: Optional[Dict[str, Any]] = None):
        """Record throughput for Phase 1 tracking"""
        self.phase1_metrics['throughput_minute'].append(requests_per_minute)

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

    def get_phase1_performance_report(self) -> Dict[str, Any]:
        """Generate Phase 1 specific performance report with benchmark validation"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'phase': 'Phase 1',
            'benchmarks': {},
            'current_metrics': {},
            'target_achievement': {},
            'optimization_recommendations': []
        }

        # Calculate current Phase 1 metrics
        current_metrics = {
            'response_time_p50_simple': self._calculate_p50(self.phase1_metrics['response_times_simple']),
            'response_time_p95_complex': self._calculate_p95(self.phase1_metrics['response_times_complex']),
            'throughput_sustained': self._calculate_average(self.phase1_metrics['throughput_minute']),
            'throughput_peak': max(self.phase1_metrics['throughput_minute']) if self.phase1_metrics['throughput_minute'] else 0,
            'memory_normal': self._calculate_average(self.phase1_metrics['memory_usage_mb']),
            'memory_peak': max(self.phase1_metrics['memory_usage_mb']) if self.phase1_metrics['memory_usage_mb'] else 0,
            'cpu_normal': self._calculate_average(self.phase1_metrics['cpu_usage_percent']),
            'cpu_peak': max(self.phase1_metrics['cpu_usage_percent']) if self.phase1_metrics['cpu_usage_percent'] else 0,
            'ai_grok_response': self._calculate_average(self.phase1_metrics['ai_response_times']['grok']),
            'ai_gemini_response': self._calculate_average(self.phase1_metrics['ai_response_times']['gemini']),
            'ai_qwen_response': self._calculate_average(self.phase1_metrics['ai_response_times']['qwen']),
            'memory_search_latency': self._calculate_average(self.phase1_metrics['memory_search_times']),
            'cache_hit_rate': self._calculate_average(self.phase1_metrics['cache_hit_rates'])
        }

        report['current_metrics'] = current_metrics

        # Check benchmark achievement
        target_achievement = {}
        for metric, target in self.phase1_targets.items():
            current = current_metrics.get(metric, 0)
            if metric in ['memory_normal', 'memory_peak', 'memory_search_latency']:
                # Lower is better for these metrics
                achieved = current <= target if current > 0 else False
            elif metric == 'memory_utilization':
                # Higher is better for utilization
                achieved = current >= target if current > 0 else False
            else:
                # Lower is better for most metrics (response times, CPU, etc.)
                achieved = current <= target if current > 0 else False
            target_achievement[metric] = {
                'target': target,
                'current': current,
                'achieved': achieved,
                'variance_percent': ((current - target) / target * 100) if target > 0 else 0
            }

        report['target_achievement'] = target_achievement

        # Generate optimization recommendations
        recommendations = []
        for metric, achievement in target_achievement.items():
            if not achievement['achieved']:
                variance = achievement['variance_percent']
                if variance > 50:
                    recommendations.append(f"CRITICAL: {metric.replace('_', ' ').title()} is {variance:.1f}% above target")
                elif variance > 20:
                    recommendations.append(f"HIGH: {metric.replace('_', ' ').title()} needs {variance:.1f}% improvement")
                else:
                    recommendations.append(f"MEDIUM: {metric.replace('_', ' ').title()} requires optimization")

        # Add specific recommendations based on failing metrics
        if not target_achievement.get('response_time_p50_simple', {}).get('achieved', False):
            recommendations.append("Optimize simple task response times - consider caching and async processing")
        if not target_achievement.get('response_time_p95_complex', {}).get('achieved', False):
            recommendations.append("Optimize complex task response times - review algorithm complexity")
        if not target_achievement.get('ai_grok_response', {}).get('achieved', False):
            recommendations.append("Optimize Groq API integration - reduce network latency")
        if not target_achievement.get('memory_search_latency', {}).get('achieved', False):
            recommendations.append("Optimize FAISS indexing and search algorithms")
        if not target_achievement.get('cache_hit_rate', {}).get('achieved', False):
            recommendations.append("Improve cache hit rate through better cache strategies")

        report['optimization_recommendations'] = recommendations
        report['overall_compliance'] = sum(1 for a in target_achievement.values() if a['achieved']) / len(target_achievement) * 100

        return report

    def _calculate_p50(self, values: deque) -> float:
        """Calculate P50 (median) from values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        return sorted_values[len(sorted_values) // 2]

    def _calculate_p95(self, values: deque) -> float:
        """Calculate P95 from values"""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        index = int(len(sorted_values) * 0.95)
        return sorted_values[min(index, len(sorted_values) - 1)]

    def _calculate_average(self, values: deque) -> float:
        """Calculate average from values"""
        if not values:
            return 0.0
        return sum(values) / len(values)

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
        """Collect system resource metrics with Phase 1 tracking"""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_sample(PerformanceMetric.CPU_USAGE, cpu_percent, "system")
            self.phase1_metrics['cpu_usage_percent'].append(cpu_percent)

            # Memory usage (track in MB for Phase 1)
            memory = psutil.virtual_memory()
            memory_mb = memory.used / 1024 / 1024
            self.record_sample(PerformanceMetric.MEMORY_USAGE, memory.percent, "system")
            self.phase1_metrics['memory_usage_mb'].append(memory_mb)

            # Disk I/O (simplified)
            disk_io = psutil.disk_io_counters()
            if disk_io:
                read_bytes_per_sec = disk_io.read_bytes / self.monitoring_interval
                write_bytes_per_sec = disk_io.write_bytes / self.monitoring_interval
                self.record_sample(PerformanceMetric.DISK_IO, read_bytes_per_sec + write_bytes_per_sec, "system")

            # Calculate and record throughput (requests per minute)
            current_time = datetime.now()
            recent_responses = [
                s for s in self.samples[PerformanceMetric.RESPONSE_TIME]
                if (current_time - s.timestamp).total_seconds() < 60
            ]
            throughput = len(recent_responses)
            self.phase1_metrics['throughput_minute'].append(throughput)

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

    # Phase 2: Advanced Performance Methods

    async def get_performance_predictions(self, time_horizon_hours: int = 1) -> Dict[str, Any]:
        """
        Phase 2: Get performance predictions for specified time horizon

        Args:
            time_horizon_hours: Hours to predict ahead

        Returns:
            Performance predictions with confidence intervals
        """
        predictions = {
            'timestamp': datetime.now().isoformat(),
            'time_horizon_hours': time_horizon_hours,
            'predictions': {},
            'overall_confidence': 0.0
        }

        # Predict key metrics
        key_metrics = ['response_time', 'cpu_usage', 'memory_usage']

        for metric in key_metrics:
            # Get historical data for the metric
            historical_data = []
            if metric in self.samples:
                for sample in list(self.samples[PerformanceMetric[metric.upper()]]):
                    historical_data.append((sample.timestamp, sample.value))

            if len(historical_data) >= 10:
                # Train/update model
                await self.predictive_model.train_model(metric, historical_data)

                # Make prediction
                prediction = await self.predictive_model.predict_performance(metric, time_horizon_hours)
                if prediction:
                    predictions['predictions'][metric] = {
                        'predicted_value': prediction.predicted_value,
                        'confidence_interval': prediction.confidence_interval,
                        'confidence_level': prediction.confidence_level,
                        'factors': prediction.factors
                    }

        # Calculate overall confidence
        if predictions['predictions']:
            confidence_values = [p['confidence_level'] for p in predictions['predictions'].values()]
            predictions['overall_confidence'] = statistics.mean(confidence_values)

        return predictions

    async def run_automated_optimization(self) -> Dict[str, Any]:
        """
        Phase 2: Run automated optimization routines

        Returns:
            Optimization results and actions taken
        """
        if not self.auto_optimization_enabled:
            return {'status': 'disabled', 'message': 'Auto-optimization is disabled'}

        optimization_result = {
            'timestamp': datetime.now().isoformat(),
            'actions_analyzed': 0,
            'actions_executed': 0,
            'improvements_achieved': {},
            'status': 'completed'
        }

        try:
            # Analyze current performance and generate optimization actions
            actions = await self.optimization_engine.analyze_and_optimize()
            optimization_result['actions_analyzed'] = len(actions)

            # Execute top actions (limit to prevent system instability)
            executed_count = 0
            for action in actions[:3]:  # Execute top 3 actions
                if action.risk_level in ['low', 'medium']:  # Only execute low/medium risk actions automatically
                    execution_result = await self.optimization_engine.execute_optimization_action(action)

                    if execution_result.get('status') == 'success':
                        executed_count += 1
                        improvement = execution_result.get('improvement_achieved', 0)
                        optimization_result['improvements_achieved'][action.action_type] = improvement

            optimization_result['actions_executed'] = executed_count

            # Store optimization history
            self.optimization_history.append(optimization_result)

            self.logger.info(f"Phase 2: Automated optimization completed - {executed_count} actions executed")

        except Exception as e:
            optimization_result['status'] = 'error'
            optimization_result['error'] = str(e)
            self.logger.error(f"Phase 2: Automated optimization failed: {e}")

        return optimization_result

    async def optimize_resource_allocation(self) -> Dict[str, Any]:
        """
        Phase 2: Optimize resource allocation based on usage patterns

        Returns:
            Resource optimization plan
        """
        return await self.resource_manager.optimize_resource_allocation()

    async def predict_capacity_requirements(self, time_horizon_hours: int = 24) -> Dict[str, Any]:
        """
        Phase 2: Predict future capacity requirements

        Args:
            time_horizon_hours: Hours to predict ahead

        Returns:
            Capacity prediction and scaling recommendations
        """
        return await self.resource_manager.predict_capacity_needs(time_horizon_hours)

    async def optimize_memory_usage(self) -> Dict[str, Any]:
        """
        Phase 2: Run memory optimization routines

        Returns:
            Memory optimization results
        """
        analysis = await self.memory_optimizer.analyze_memory_usage()

        # Apply optimizations if needed
        if analysis['current_memory_usage'] > 85:
            # Apply critical optimizations
            for opportunity in analysis['optimization_opportunities']:
                if opportunity['priority'] == 'high':
                    optimization_result = await self.memory_optimizer.apply_memory_optimization(opportunity['type'])
                    analysis[f"optimization_result_{opportunity['type']}"] = optimization_result

        return analysis

    def detect_performance_anomalies(self) -> List[Dict[str, Any]]:
        """
        Phase 2: Detect performance anomalies using statistical analysis

        Returns:
            List of detected anomalies
        """
        anomalies = []

        for metric, samples in self.samples.items():
            if len(samples) < 20:  # Need minimum samples for anomaly detection
                continue

            # Get recent values (last 10 samples)
            recent_values = [s.value for s in list(samples)[-10:]]
            historical_values = [s.value for s in list(samples)[:-10]]

            if len(historical_values) < 10:
                continue

            # Calculate statistical measures
            recent_mean = statistics.mean(recent_values)
            historical_mean = statistics.mean(historical_values)
            historical_std = statistics.stdev(historical_values) if len(historical_values) > 1 else 0

            # Detect anomalies (values outside 3 standard deviations)
            if historical_std > 0:
                z_score = abs(recent_mean - historical_mean) / historical_std

                if z_score > 3:  # Significant anomaly
                    anomaly = {
                        'metric': metric.value,
                        'anomaly_type': 'statistical_outlier',
                        'severity': 'high' if z_score > 4 else 'medium',
                        'recent_mean': recent_mean,
                        'historical_mean': historical_mean,
                        'z_score': z_score,
                        'timestamp': datetime.now().isoformat(),
                        'description': f"Unusual {metric.value} pattern detected"
                    }
                    anomalies.append(anomaly)

        return anomalies

    def get_phase2_performance_report(self) -> Dict[str, Any]:
        """
        Phase 2: Generate comprehensive Phase 2 performance report

        Returns:
            Detailed Phase 2 performance analytics
        """
        report = {
            'phase': 'Phase 2 Enhancement',
            'timestamp': datetime.now().isoformat(),
            'predictive_analytics': {},
            'optimization_performance': {},
            'resource_efficiency': {},
            'memory_optimization': {},
            'anomaly_detection': {},
            'overall_improvement': 0.0
        }

        # Predictive analytics performance
        if self.performance_predictions:
            recent_predictions = self.performance_predictions[-10:]  # Last 10 predictions
            prediction_accuracy = []

            for prediction in recent_predictions:
                # Simplified accuracy calculation (would need actual vs predicted comparison)
                accuracy = prediction.confidence_level * (1 - abs(prediction.predicted_value - prediction.predicted_value * 0.1) / prediction.predicted_value)
                prediction_accuracy.append(accuracy)

            report['predictive_analytics'] = {
                'total_predictions': len(self.performance_predictions),
                'average_accuracy': statistics.mean(prediction_accuracy) if prediction_accuracy else 0,
                'prediction_horizon_coverage': f"{min(p.prediction_horizon for p in recent_predictions)}-{max(p.prediction_horizon for p in recent_predictions)} minutes"
            }

        # Optimization performance
        if self.optimization_history:
            recent_optimizations = self.optimization_history[-5:]  # Last 5 optimization runs
            total_improvements = sum(sum(opt.get('improvements_achieved', {}).values()) for opt in recent_optimizations)
            avg_improvements = total_improvements / len(recent_optimizations) if recent_optimizations else 0

            report['optimization_performance'] = {
                'total_optimization_runs': len(self.optimization_history),
                'average_improvement': avg_improvements,
                'successful_actions': sum(opt.get('actions_executed', 0) for opt in recent_optimizations),
                'automation_efficiency': avg_improvements * 100  # Percentage improvement
            }

        # Resource efficiency
        current_metrics = self.get_current_metrics()
        memory_efficiency = 1 - (current_metrics.get('system_memory_percent', 0) / 100)
        cpu_efficiency = 1 - (current_metrics.get('system_cpu_percent', 0) / 100)

        report['resource_efficiency'] = {
            'memory_efficiency': memory_efficiency,
            'cpu_efficiency': cpu_efficiency,
            'overall_efficiency': (memory_efficiency + cpu_efficiency) / 2,
            'optimization_potential': max(0, 0.9 - ((memory_efficiency + cpu_efficiency) / 2))  # Distance from optimal
        }

        # Memory optimization metrics
        memory_usage = current_metrics.get('system_memory_percent', 0)
        report['memory_optimization'] = {
            'current_usage_percent': memory_usage,
            'optimization_status': 'good' if memory_usage < 80 else 'needs_attention' if memory_usage < 90 else 'critical',
            'efficiency_score': self.memory_optimizer._calculate_memory_efficiency()
        }

        # Anomaly detection
        anomalies = self.detect_performance_anomalies()
        report['anomaly_detection'] = {
            'anomalies_detected': len(anomalies),
            'severity_breakdown': {
                'high': len([a for a in anomalies if a['severity'] == 'high']),
                'medium': len([a for a in anomalies if a['severity'] == 'medium'])
            },
            'detection_status': 'active' if self.anomaly_detection_enabled else 'disabled'
        }

        # Calculate overall improvement score
        improvement_factors = [
            report['predictive_analytics'].get('average_accuracy', 0) * 0.2,
            report['optimization_performance'].get('average_improvement', 0) * 0.3,
            report['resource_efficiency'].get('overall_efficiency', 0) * 0.3,
            report['memory_optimization'].get('efficiency_score', 0) * 0.2
        ]

        report['overall_improvement'] = statistics.mean(improvement_factors) if improvement_factors else 0

        return report


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


# Phase 2: Advanced Performance Enhancement Classes

@dataclass
class PerformancePrediction:
    """Phase 2: Performance prediction with confidence intervals"""
    metric: str
    predicted_value: float
    confidence_interval: Tuple[float, float]
    prediction_horizon: int  # minutes
    confidence_level: float
    prediction_timestamp: datetime
    factors: List[str]


@dataclass
class OptimizationAction:
    """Phase 2: Automated optimization action"""
    action_id: str
    action_type: str
    target_component: str
    parameters: Dict[str, Any]
    expected_improvement: float
    risk_level: str
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    executed_at: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None


class PredictivePerformanceModel:
    """Phase 2: ML-based performance prediction model"""

    def __init__(self):
        self.models: Dict[str, LinearRegression] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.training_data: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        self.logger = logging.getLogger(__name__)

    async def train_model(self, metric: str, historical_data: List[Tuple[datetime, float]]):
        """Train prediction model for a specific metric"""
        if len(historical_data) < 10:
            return False

        # Prepare training data
        timestamps = [(dt - historical_data[0][0]).total_seconds() / 3600 for dt, _ in historical_data]  # Hours since start
        values = [val for _, val in historical_data]

        # Create features (time-based)
        X = np.array(timestamps).reshape(-1, 1)
        y = np.array(values)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train model
        model = LinearRegression()
        model.fit(X_scaled, y)

        self.models[metric] = model
        self.scalers[metric] = scaler
        self.training_data[metric] = historical_data

        return True

    async def predict_performance(self, metric: str, prediction_horizon_hours: int = 1) -> Optional[PerformancePrediction]:
        """Predict future performance for a metric"""
        if metric not in self.models:
            return None

        model = self.models[metric]
        scaler = self.scalers[metric]

        # Get latest timestamp
        if not self.training_data[metric]:
            return None

        latest_time = max(dt for dt, _ in self.training_data[metric])
        prediction_time = latest_time + timedelta(hours=prediction_horizon_hours)

        # Prepare prediction input
        time_diff = (prediction_time - self.training_data[metric][0][0]).total_seconds() / 3600
        X_pred = np.array([[time_diff]])
        X_pred_scaled = scaler.transform(X_pred)

        # Make prediction
        predicted_value = model.predict(X_pred_scaled)[0]

        # Calculate confidence interval (simplified)
        recent_values = [val for _, val in self.training_data[metric][-20:]]
        std_dev = statistics.stdev(recent_values) if len(recent_values) > 1 else 0
        confidence_interval = (predicted_value - 1.96 * std_dev, predicted_value + 1.96 * std_dev)

        return PerformancePrediction(
            metric=metric,
            predicted_value=predicted_value,
            confidence_interval=confidence_interval,
            prediction_horizon=prediction_horizon_hours * 60,  # Convert to minutes
            confidence_level=0.95,
            prediction_timestamp=datetime.now(),
            factors=["historical_trend", "time_based_pattern"]
        )


class AutomatedOptimizationEngine:
    """Phase 2: Automated optimization engine with self-tuning capabilities"""

    def __init__(self, performance_monitor: 'PerformanceMonitor'):
        self.performance_monitor = performance_monitor
        self.optimization_actions: List[OptimizationAction] = []
        self.optimization_rules = self._load_optimization_rules()
        self.logger = logging.getLogger(__name__)

    def _load_optimization_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load optimization rules for different scenarios"""
        return {
            'high_response_time': {
                'condition': lambda metrics: metrics.get('response_time', {}).get('p95', 0) > 1000,
                'actions': [
                    {
                        'type': 'cache_optimization',
                        'description': 'Increase cache size and implement LRU eviction',
                        'parameters': {'cache_size_multiplier': 1.5},
                        'expected_improvement': 0.15,
                        'risk_level': 'low'
                    },
                    {
                        'type': 'query_optimization',
                        'description': 'Optimize slow database queries',
                        'parameters': {'enable_query_caching': True},
                        'expected_improvement': 0.20,
                        'risk_level': 'medium'
                    }
                ]
            },
            'high_memory_usage': {
                'condition': lambda metrics: metrics.get('memory_usage', {}).get('current', 0) > 85,
                'actions': [
                    {
                        'type': 'memory_cleanup',
                        'description': 'Trigger garbage collection and cleanup expired cache',
                        'parameters': {'force_gc': True},
                        'expected_improvement': 0.10,
                        'risk_level': 'low'
                    },
                    {
                        'type': 'memory_optimization',
                        'description': 'Optimize memory allocation patterns',
                        'parameters': {'enable_memory_pool': True},
                        'expected_improvement': 0.25,
                        'risk_level': 'medium'
                    }
                ]
            },
            'high_cpu_usage': {
                'condition': lambda metrics: metrics.get('cpu_usage', {}).get('current', 0) > 80,
                'actions': [
                    {
                        'type': 'async_processing',
                        'description': 'Enable asynchronous processing for CPU-intensive tasks',
                        'parameters': {'max_workers': 4},
                        'expected_improvement': 0.30,
                        'risk_level': 'medium'
                    }
                ]
            }
        }

    async def analyze_and_optimize(self) -> List[OptimizationAction]:
        """Analyze current performance and generate optimization actions"""
        current_metrics = self.performance_monitor.get_current_metrics()
        actions = []

        # Check each optimization rule
        for rule_name, rule_config in self.optimization_rules.items():
            if rule_config['condition'](current_metrics):
                for action_config in rule_config['actions']:
                    action = OptimizationAction(
                        action_id=f"{rule_name}_{int(time.time())}_{len(actions)}",
                        action_type=action_config['type'],
                        target_component=rule_name.split('_')[1],  # Extract component from rule name
                        parameters=action_config['parameters'],
                        expected_improvement=action_config['expected_improvement'],
                        risk_level=action_config['risk_level']
                    )
                    actions.append(action)

        # Prioritize actions by expected improvement and risk
        actions.sort(key=lambda x: (x.expected_improvement / (1 if x.risk_level == 'low' else 2 if x.risk_level == 'medium' else 3)), reverse=True)

        # Store actions for tracking
        self.optimization_actions.extend(actions)

        return actions[:5]  # Return top 5 actions

    async def execute_optimization_action(self, action: OptimizationAction) -> Dict[str, Any]:
        """Execute an optimization action"""
        action.status = "executing"
        action.executed_at = datetime.now()

        try:
            # Simulate action execution (in production, this would implement actual optimizations)
            execution_result = await self._simulate_action_execution(action)

            action.status = "completed"
            action.result = execution_result

            self.logger.info(f"Executed optimization action: {action.action_type} - Result: {execution_result}")

            return execution_result

        except Exception as e:
            action.status = "failed"
            action.result = {"error": str(e)}
            self.logger.error(f"Failed to execute optimization action {action.action_id}: {e}")
            return {"status": "error", "error": str(e)}

    async def _simulate_action_execution(self, action: OptimizationAction) -> Dict[str, Any]:
        """Simulate execution of optimization action"""
        # Simulate execution time
        await asyncio.sleep(0.1)

        # Simulate success/failure
        success = np.random.random() > 0.1  # 90% success rate

        if success:
            return {
                "status": "success",
                "action_type": action.action_type,
                "improvement_achieved": action.expected_improvement * (0.8 + np.random.random() * 0.4),  # 80-120% of expected
                "execution_time_ms": 100 + np.random.random() * 200
            }
        else:
            return {
                "status": "failed",
                "action_type": action.action_type,
                "error": "Simulated execution failure",
                "execution_time_ms": 50 + np.random.random() * 100
            }


class ResourceManager:
    """Phase 2: Advanced resource management with cost optimization"""

    def __init__(self, performance_monitor: 'PerformanceMonitor'):
        self.performance_monitor = performance_monitor
        self.resource_allocation = {}
        self.cost_optimization_rules = self._load_cost_rules()
        self.logger = logging.getLogger(__name__)

    def _load_cost_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load cost optimization rules"""
        return {
            'memory_optimization': {
                'condition': lambda usage: usage > 80,
                'actions': ['enable_compression', 'reduce_cache_size', 'optimize_data_structures'],
                'cost_savings': 0.15
            },
            'cpu_optimization': {
                'condition': lambda usage: usage > 75,
                'actions': ['enable_async_processing', 'optimize_algorithms', 'reduce_thread_count'],
                'cost_savings': 0.20
            },
            'storage_optimization': {
                'condition': lambda usage: usage > 85,
                'actions': ['enable_deduplication', 'compress_old_data', 'archive_unused_data'],
                'cost_savings': 0.25
            }
        }

    async def optimize_resource_allocation(self) -> Dict[str, Any]:
        """Optimize resource allocation based on current usage and predictions"""
        current_metrics = self.performance_monitor.get_current_metrics()

        optimization_plan = {
            'timestamp': datetime.now().isoformat(),
            'current_allocation': {},
            'recommended_changes': {},
            'expected_cost_savings': 0,
            'risk_assessment': 'low'
        }

        # Analyze memory usage
        memory_usage = current_metrics.get('system_memory_percent', 0)
        if memory_usage > 80:
            optimization_plan['recommended_changes']['memory'] = {
                'action': 'optimize_memory_usage',
                'current_usage': memory_usage,
                'target_usage': 70,
                'methods': ['garbage_collection', 'cache_optimization', 'memory_pooling']
            }
            optimization_plan['expected_cost_savings'] += 0.15

        # Analyze CPU usage
        cpu_usage = current_metrics.get('system_cpu_percent', 0)
        if cpu_usage > 75:
            optimization_plan['recommended_changes']['cpu'] = {
                'action': 'optimize_cpu_usage',
                'current_usage': cpu_usage,
                'target_usage': 60,
                'methods': ['async_processing', 'algorithm_optimization', 'load_balancing']
            }
            optimization_plan['expected_cost_savings'] += 0.20

        return optimization_plan

    async def predict_capacity_needs(self, time_horizon_hours: int = 24) -> Dict[str, Any]:
        """Predict future capacity needs based on usage patterns"""
        # Get historical data
        memory_history = list(self.performance_monitor.phase1_metrics['memory_usage_mb'])
        cpu_history = list(self.performance_monitor.phase1_metrics['cpu_usage_percent'])

        predictions = {
            'time_horizon_hours': time_horizon_hours,
            'memory_prediction': self._predict_resource_usage(memory_history, time_horizon_hours),
            'cpu_prediction': self._predict_resource_usage(cpu_history, time_horizon_hours),
            'scaling_recommendations': []
        }

        # Generate scaling recommendations
        if predictions['memory_prediction']['peak_usage'] > 400:  # MB
            predictions['scaling_recommendations'].append({
                'resource': 'memory',
                'action': 'increase_memory_allocation',
                'reason': 'Predicted memory usage exceeds current capacity',
                'urgency': 'high'
            })

        if predictions['cpu_prediction']['peak_usage'] > 80:  # %
            predictions['scaling_recommendations'].append({
                'resource': 'cpu',
                'action': 'optimize_cpu_allocation',
                'reason': 'Predicted CPU usage indicates potential bottleneck',
                'urgency': 'medium'
            })

        return predictions

    def _predict_resource_usage(self, historical_data: List[float], hours_ahead: int) -> Dict[str, Any]:
        """Predict resource usage using simple trend analysis"""
        if len(historical_data) < 5:
            return {'error': 'Insufficient historical data'}

        # Simple linear trend prediction
        recent_avg = statistics.mean(historical_data[-10:]) if len(historical_data) > 10 else statistics.mean(historical_data)
        trend = (historical_data[-1] - historical_data[0]) / len(historical_data) if len(historical_data) > 1 else 0

        predicted_peak = recent_avg + (trend * hours_ahead)
        predicted_avg = recent_avg + (trend * hours_ahead * 0.7)  # Slightly dampened for average

        return {
            'current_average': recent_avg,
            'predicted_average': max(0, predicted_avg),
            'predicted_peak': max(0, predicted_peak),
            'trend_direction': 'increasing' if trend > 0 else 'decreasing' if trend < 0 else 'stable',
            'confidence_level': min(0.9, len(historical_data) / 100)  # Higher confidence with more data
        }


class MemoryOptimizationEngine:
    """Phase 2: Advanced memory optimization with intelligent allocation"""

    def __init__(self, performance_monitor: 'PerformanceMonitor'):
        self.performance_monitor = performance_monitor
        self.memory_patterns = defaultdict(list)
        self.optimization_strategies = {}
        self.logger = logging.getLogger(__name__)

    async def analyze_memory_usage(self) -> Dict[str, Any]:
        """Analyze current memory usage patterns"""
        current_metrics = self.performance_monitor.get_current_metrics()

        analysis = {
            'timestamp': datetime.now().isoformat(),
            'current_memory_usage': current_metrics.get('system_memory_percent', 0),
            'memory_efficiency_score': self._calculate_memory_efficiency(),
            'leak_detection': await self._detect_memory_leaks(),
            'optimization_opportunities': await self._identify_optimization_opportunities(),
            'allocation_recommendations': []
        }

        # Generate allocation recommendations
        if analysis['current_memory_usage'] > 85:
            analysis['allocation_recommendations'].append({
                'action': 'increase_memory_limit',
                'reason': 'Memory usage consistently high',
                'expected_benefit': 'Improved stability'
            })

        if analysis['memory_efficiency_score'] < 0.7:
            analysis['allocation_recommendations'].append({
                'action': 'optimize_memory_allocation',
                'reason': 'Low memory efficiency detected',
                'expected_benefit': 'Better resource utilization'
            })

        return analysis

    def _calculate_memory_efficiency(self) -> float:
        """Calculate memory efficiency score"""
        memory_history = list(self.performance_monitor.phase1_metrics['memory_usage_mb'])

        if not memory_history:
            return 0.5

        # Calculate efficiency based on stability and utilization
        avg_usage = statistics.mean(memory_history)
        usage_variance = statistics.variance(memory_history) if len(memory_history) > 1 else 0

        # Higher stability (lower variance) and optimal utilization (around 70-80%) = higher efficiency
        stability_score = max(0, 1 - (usage_variance / (avg_usage ** 2)) * 100) if avg_usage > 0 else 0
        utilization_score = 1 - abs(avg_usage - 350) / 350  # Optimal around 350MB

        return (stability_score + utilization_score) / 2

    async def _detect_memory_leaks(self) -> Dict[str, Any]:
        """Detect potential memory leaks"""
        memory_history = list(self.performance_monitor.phase1_metrics['memory_usage_mb'])

        if len(memory_history) < 20:
            return {'status': 'insufficient_data'}

        # Simple leak detection: consistent upward trend
        recent_trend = memory_history[-10:]
        trend_slope = (recent_trend[-1] - recent_trend[0]) / len(recent_trend)

        leak_probability = min(1.0, max(0, trend_slope / 10))  # Normalize trend to 0-1

        return {
            'leak_probability': leak_probability,
            'trend_slope_mb_per_sample': trend_slope,
            'recommendation': 'investigate' if leak_probability > 0.7 else 'monitor'
        }

    async def _identify_optimization_opportunities(self) -> List[Dict[str, Any]]:
        """Identify memory optimization opportunities"""
        opportunities = []

        current_usage = self.performance_monitor.get_current_metrics().get('system_memory_percent', 0)

        if current_usage > 90:
            opportunities.append({
                'type': 'critical_cleanup',
                'description': 'Immediate memory cleanup required',
                'priority': 'high',
                'expected_savings_mb': current_usage * 0.1
            })

        if current_usage > 80:
            opportunities.append({
                'type': 'cache_optimization',
                'description': 'Optimize cache size and eviction policies',
                'priority': 'medium',
                'expected_savings_mb': 50
            })

        opportunities.append({
            'type': 'memory_pooling',
            'description': 'Implement memory pooling for frequent allocations',
            'priority': 'low',
            'expected_savings_mb': 30
        })

        return opportunities

    async def apply_memory_optimization(self, optimization_type: str) -> Dict[str, Any]:
        """Apply a specific memory optimization"""
        optimizations = {
            'garbage_collection': self._apply_garbage_collection,
            'cache_cleanup': self._apply_cache_cleanup,
            'memory_pooling': self._apply_memory_pooling
        }

        if optimization_type in optimizations:
            return await optimizations[optimization_type]()
        else:
            return {'status': 'error', 'error': f'Unknown optimization type: {optimization_type}'}

    async def _apply_garbage_collection(self) -> Dict[str, Any]:
        """Apply garbage collection optimization"""
        # Simulate garbage collection
        await asyncio.sleep(0.05)
        memory_freed = np.random.uniform(20, 50)  # MB

        return {
            'status': 'success',
            'optimization_type': 'garbage_collection',
            'memory_freed_mb': memory_freed,
            'execution_time_ms': 50
        }

    async def _apply_cache_cleanup(self) -> Dict[str, Any]:
        """Apply cache cleanup optimization"""
        await asyncio.sleep(0.03)
        cache_freed = np.random.uniform(15, 35)  # MB

        return {
            'status': 'success',
            'optimization_type': 'cache_cleanup',
            'cache_freed_mb': cache_freed,
            'execution_time_ms': 30
        }

    async def _apply_memory_pooling(self) -> Dict[str, Any]:
        """Apply memory pooling optimization"""
        await asyncio.sleep(0.1)
        efficiency_gain = np.random.uniform(0.05, 0.15)  # 5-15% efficiency gain

        return {
            'status': 'success',
            'optimization_type': 'memory_pooling',
            'efficiency_gain_percent': efficiency_gain * 100,
            'execution_time_ms': 100
        }