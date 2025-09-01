"""
Enterprise Monitoring and Alerting System for CodeSage MCP Server.

This module provides enterprise-grade monitoring, alerting, and predictive analytics
for Phase 4, including real-time dashboards, anomaly detection, and automated responses.

Features:
- Real-time performance monitoring with custom metrics
- Predictive analytics for performance degradation
- Automated alerting with escalation policies
- Anomaly detection using statistical and ML methods
- Enterprise dashboards with custom reporting
- Automated incident response and remediation
"""

import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import statistics
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


class MetricType(Enum):
    """Types of metrics to monitor."""
    COUNTER = "counter"      # Monotonically increasing value
    GAUGE = "gauge"         # Value that can go up or down
    HISTOGRAM = "histogram" # Distribution of values
    SUMMARY = "summary"     # Quantiles over sliding time window


@dataclass
class MetricValue:
    """A single metric measurement."""
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: MetricType = MetricType.GAUGE


@dataclass
class AlertRule:
    """Alert rule definition."""
    rule_id: str
    name: str
    description: str
    metric_name: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    threshold: float
    duration_seconds: int = 300  # How long condition must be true
    cooldown_seconds: int = 3600  # Minimum time between alerts
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    last_triggered: float = 0
    enabled: bool = True


@dataclass
class Alert:
    """Alert instance."""
    alert_id: str
    rule_id: str
    severity: AlertSeverity
    status: AlertStatus
    summary: str
    description: str
    labels: Dict[str, str]
    annotations: Dict[str, str]
    starts_at: float
    ends_at: Optional[float] = None
    generator_url: Optional[str] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None


@dataclass
class AnomalyDetectionResult:
    """Result of anomaly detection."""
    metric_name: str
    is_anomaly: bool
    anomaly_score: float
    confidence: float
    detection_method: str
    expected_value: float
    actual_value: float
    timestamp: float = field(default_factory=time.time)


class MetricsCollector:
    """Collects and stores metrics data."""

    def __init__(self, retention_hours: int = 24, max_metrics_per_hour: int = 10000):
        self.retention_hours = retention_hours
        self.max_metrics_per_hour = max_metrics_per_hour

        # Storage: metric_name -> deque of MetricValue
        self.metrics_storage: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_metrics_per_hour))

        # Current gauge values
        self.gauge_values: Dict[str, float] = {}

        # Counters
        self.counter_values: Dict[str, float] = {}

        # Histograms and summaries storage
        self.histogram_data: Dict[str, List[float]] = defaultdict(list)
        self.summary_data: Dict[str, List[float]] = defaultdict(list)

        self.lock = threading.RLock()

    def record_metric(self, metric: MetricValue):
        """Record a metric value."""
        with self.lock:
            if metric.metric_type == MetricType.GAUGE:
                self.gauge_values[metric.name] = metric.value
            elif metric.metric_type == MetricType.COUNTER:
                if metric.name not in self.counter_values:
                    self.counter_values[metric.name] = 0
                self.counter_values[metric.name] += metric.value
            elif metric.metric_type == MetricType.HISTOGRAM:
                self.histogram_data[metric.name].append(metric.value)
            elif metric.metric_type == MetricType.SUMMARY:
                self.summary_data[metric.name].append(metric.value)

            # Store in time-series storage
            self.metrics_storage[metric.name].append(metric)

            # Clean old data
            self._cleanup_old_data()

    def get_metric_values(self, metric_name: str, hours: int = 1) -> List[MetricValue]:
        """Get metric values for the last N hours."""
        with self.lock:
            if metric_name not in self.metrics_storage:
                return []

            cutoff_time = time.time() - (hours * 3600)
            return [m for m in self.metrics_storage[metric_name] if m.timestamp >= cutoff_time]

    def get_current_value(self, metric_name: str) -> Optional[float]:
        """Get the current value of a metric."""
        with self.lock:
            if metric_name in self.gauge_values:
                return self.gauge_values[metric_name]
            elif metric_name in self.counter_values:
                return self.counter_values[metric_name]
            else:
                # Get latest from time-series
                values = list(self.metrics_storage[metric_name])
                if values:
                    return values[-1].value
                return None

    def get_metric_stats(self, metric_name: str, hours: int = 1) -> Dict[str, Any]:
        """Get statistics for a metric over the last N hours."""
        values = self.get_metric_values(metric_name, hours)
        if not values:
            return {"count": 0, "min": None, "max": None, "avg": None, "stddev": None}

        metric_values = [v.value for v in values]

        return {
            "count": len(metric_values),
            "min": min(metric_values),
            "max": max(metric_values),
            "avg": statistics.mean(metric_values),
            "stddev": statistics.stdev(metric_values) if len(metric_values) > 1 else 0,
            "latest": metric_values[-1] if metric_values else None
        }

    def _cleanup_old_data(self):
        """Clean up data older than retention period."""
        cutoff_time = time.time() - (self.retention_hours * 3600)

        for metric_name, metrics_queue in self.metrics_storage.items():
            # Remove old entries
            while metrics_queue and metrics_queue[0].timestamp < cutoff_time:
                metrics_queue.popleft()


class AlertManager:
    """Manages alerts and alert rules."""

    def __init__(self):
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.max_history_size = 1000

        self.lock = threading.RLock()

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self.lock:
            self.alert_rules[rule.rule_id] = rule
            logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_id: str):
        """Remove an alert rule."""
        with self.lock:
            if rule_id in self.alert_rules:
                del self.alert_rules[rule_id]
                logger.info(f"Removed alert rule: {rule_id}")

    def evaluate_alerts(self, metrics_collector: MetricsCollector) -> List[Alert]:
        """Evaluate all alert rules and return triggered alerts."""
        new_alerts = []

        with self.lock:
            for rule in self.alert_rules.values():
                if not rule.enabled:
                    continue

                # Check cooldown
                if time.time() - rule.last_triggered < rule.cooldown_seconds:
                    continue

                # Evaluate condition
                if self._evaluate_condition(rule, metrics_collector):
                    alert = self._create_alert(rule)
                    new_alerts.append(alert)
                    rule.last_triggered = time.time()

        return new_alerts

    def _evaluate_condition(self, rule: AlertRule, metrics_collector: MetricsCollector) -> bool:
        """Evaluate an alert condition."""
        try:
            # Get current metric value
            current_value = metrics_collector.get_current_value(rule.metric_name)
            if current_value is None:
                return False

            # Simple condition evaluation (can be extended with expression parser)
            if ">" in rule.condition:
                threshold = float(rule.condition.split(">")[1].strip())
                return current_value > threshold
            elif "<" in rule.condition:
                threshold = float(rule.condition.split("<")[1].strip())
                return current_value < threshold
            elif ">=" in rule.condition:
                threshold = float(rule.condition.split(">=")[1].strip())
                return current_value >= threshold
            elif "<=" in rule.condition:
                threshold = float(rule.condition.split("<=")[1].strip())
                return current_value <= threshold
            elif "==" in rule.condition:
                threshold = float(rule.condition.split("==")[1].strip())
                return abs(current_value - threshold) < 0.001

            return False

        except Exception as e:
            logger.error(f"Error evaluating alert condition for rule {rule.rule_id}: {e}")
            return False

    def _create_alert(self, rule: AlertRule) -> Alert:
        """Create an alert from a rule."""
        alert_id = f"{rule.rule_id}_{int(time.time())}"

        alert = Alert(
            alert_id=alert_id,
            rule_id=rule.rule_id,
            severity=rule.severity,
            status=AlertStatus.ACTIVE,
            summary=rule.name,
            description=rule.description,
            labels=rule.labels.copy(),
            annotations=rule.annotations.copy(),
            starts_at=time.time()
        )

        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Maintain history size
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]

        return alert

    def acknowledge_alert(self, alert_id: str, user: str):
        """Acknowledge an alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = user
                alert.acknowledged_at = time.time()
                logger.info(f"Alert {alert_id} acknowledged by {user}")

    def resolve_alert(self, alert_id: str):
        """Resolve an alert."""
        with self.lock:
            if alert_id in self.active_alerts:
                alert = self.active_alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.ends_at = time.time()
                logger.info(f"Alert {alert_id} resolved")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self.lock:
            return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the last N hours."""
        with self.lock:
            cutoff_time = time.time() - (hours * 3600)
            return [alert for alert in self.alert_history if alert.starts_at >= cutoff_time]


class AnomalyDetector:
    """Detects anomalies in metrics using statistical and ML methods."""

    def __init__(self, sensitivity: float = 0.8, min_data_points: int = 50):
        self.sensitivity = sensitivity
        self.min_data_points = min_data_points

        # Statistical models for each metric
        self.metric_models: Dict[str, Dict[str, Any]] = {}

        # Anomaly history
        self.anomaly_history: List[AnomalyDetectionResult] = []
        self.max_history_size = 1000

    def detect_anomalies(self, metrics_collector: MetricsCollector) -> List[AnomalyDetectionResult]:
        """Detect anomalies in all monitored metrics."""
        anomalies = []

        for metric_name in metrics_collector.metrics_storage.keys():
            result = self._detect_metric_anomaly(metric_name, metrics_collector)
            if result and result.is_anomaly:
                anomalies.append(result)
                self.anomaly_history.append(result)

                # Maintain history size
                if len(self.anomaly_history) > self.max_history_size:
                    self.anomaly_history = self.anomaly_history[-self.max_history_size:]

        return anomalies

    def _detect_metric_anomaly(self, metric_name: str, metrics_collector: MetricsCollector) -> Optional[AnomalyDetectionResult]:
        """Detect anomaly in a specific metric."""
        values = metrics_collector.get_metric_values(metric_name, hours=1)
        if len(values) < self.min_data_points:
            return None

        metric_values = [v.value for v in values]
        current_value = metric_values[-1]

        # Method 1: Statistical outlier detection (Z-score)
        anomaly_result = self._zscore_anomaly_detection(metric_name, metric_values, current_value)
        if anomaly_result and anomaly_result.is_anomaly:
            return anomaly_result

        # Method 2: Moving average deviation
        anomaly_result = self._moving_average_anomaly_detection(metric_name, metric_values, current_value)
        if anomaly_result and anomaly_result.is_anomaly:
            return anomaly_result

        # Method 3: Seasonal decomposition (if enough data)
        if len(metric_values) >= 100:  # Need more data for seasonal analysis
            anomaly_result = self._seasonal_anomaly_detection(metric_name, metric_values, current_value)
            if anomaly_result and anomaly_result.is_anomaly:
                return anomaly_result

        return None

    def _zscore_anomaly_detection(self, metric_name: str, values: List[float], current_value: float) -> Optional[AnomalyDetectionResult]:
        """Detect anomalies using Z-score method."""
        if len(values) < 10:
            return None

        mean = statistics.mean(values[:-1])  # Exclude current value
        stddev = statistics.stdev(values[:-1]) if len(values) > 1 else 0

        if stddev == 0:
            return None

        z_score = abs(current_value - mean) / stddev
        is_anomaly = z_score > (3 - self.sensitivity)  # Adjust threshold based on sensitivity

        return AnomalyDetectionResult(
            metric_name=metric_name,
            is_anomaly=is_anomaly,
            anomaly_score=z_score,
            confidence=min(z_score / 3, 1.0),
            detection_method="zscore",
            expected_value=mean,
            actual_value=current_value
        )

    def _moving_average_anomaly_detection(self, metric_name: str, values: List[float], current_value: float) -> Optional[AnomalyDetectionResult]:
        """Detect anomalies using moving average method."""
        if len(values) < 20:
            return None

        # Calculate moving average of last 10 values
        window_size = min(10, len(values) - 1)
        moving_avg = statistics.mean(values[-window_size-1:-1])

        deviation = abs(current_value - moving_avg)
        relative_deviation = deviation / moving_avg if moving_avg != 0 else 0

        threshold = 0.5 - (self.sensitivity * 0.3)  # Adjust threshold based on sensitivity
        is_anomaly = relative_deviation > threshold

        return AnomalyDetectionResult(
            metric_name=metric_name,
            is_anomaly=is_anomaly,
            anomaly_score=relative_deviation,
            confidence=min(relative_deviation / threshold, 1.0),
            detection_method="moving_average",
            expected_value=moving_avg,
            actual_value=current_value
        )

    def _seasonal_anomaly_detection(self, metric_name: str, values: List[float], current_value: float) -> Optional[AnomalyDetectionResult]:
        """Detect anomalies using seasonal decomposition."""
        # Simplified seasonal analysis - in production would use more sophisticated methods
        if len(values) < 100:
            return None

        # Simple pattern: check if current value deviates from recent trend
        recent_trend = np.polyfit(range(len(values[-50:-1])), values[-50:-1], 1)
        expected_value = np.polyval(recent_trend, len(values))

        deviation = abs(current_value - expected_value)
        relative_deviation = deviation / expected_value if expected_value != 0 else 0

        threshold = 0.3 - (self.sensitivity * 0.2)
        is_anomaly = relative_deviation > threshold

        return AnomalyDetectionResult(
            metric_name=metric_name,
            is_anomaly=is_anomaly,
            anomaly_score=relative_deviation,
            confidence=min(relative_deviation / threshold, 1.0),
            detection_method="seasonal",
            expected_value=expected_value,
            actual_value=current_value
        )


class EnterpriseMonitor:
    """Main enterprise monitoring and alerting system."""

    def __init__(self, monitoring_interval_seconds: int = 60,
                 enable_anomaly_detection: bool = True,
                 enable_predictive_alerts: bool = True):
        self.monitoring_interval_seconds = monitoring_interval_seconds
        self.enable_anomaly_detection = enable_anomaly_detection
        self.enable_predictive_alerts = enable_predictive_alerts

        # Core components
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager()
        self.anomaly_detector = AnomalyDetector() if enable_anomaly_detection else None

        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []

        # Control
        self._monitoring_active = False
        self._monitoring_thread: Optional[threading.Thread] = None

        # Start monitoring
        self._start_monitoring()

        # Initialize default alert rules
        self._initialize_default_alert_rules()

    def _start_monitoring(self):
        """Start the monitoring system."""
        if self._monitoring_active:
            return

        self._monitoring_active = True
        self._monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="EnterpriseMonitor"
        )
        self._monitoring_thread.start()
        logger.info("Enterprise monitoring started")

    def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._monitoring_active:
            try:
                self._perform_monitoring_cycle()
                time.sleep(self.monitoring_interval_seconds)
            except Exception as e:
                logger.error(f"Monitoring cycle error: {e}")
                time.sleep(10)

    def _perform_monitoring_cycle(self):
        """Perform one monitoring cycle."""
        try:
            # Evaluate alert rules
            new_alerts = self.alert_manager.evaluate_alerts(self.metrics_collector)

            # Trigger alert callbacks
            for alert in new_alerts:
                self._trigger_alert_callbacks(alert)

            # Detect anomalies
            if self.anomaly_detector:
                anomalies = self.anomaly_detector.detect_anomalies(self.metrics_collector)
                if anomalies:
                    self._handle_anomalies(anomalies)

        except Exception as e:
            logger.error(f"Error in monitoring cycle: {e}")

    def _trigger_alert_callbacks(self, alert: Alert):
        """Trigger alert callbacks."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Error in alert callback: {e}")

    def _handle_anomalies(self, anomalies: List[AnomalyDetectionResult]):
        """Handle detected anomalies."""
        for anomaly in anomalies:
            logger.warning(f"Anomaly detected in {anomaly.metric_name}: "
                         f"expected={anomaly.expected_value:.2f}, "
                         f"actual={anomaly.actual_value:.2f}, "
                         f"score={anomaly.anomaly_score:.2f}")

            # Create alert for critical anomalies
            if anomaly.confidence > 0.8:
                self._create_anomaly_alert(anomaly)

    def _create_anomaly_alert(self, anomaly: AnomalyDetectionResult):
        """Create an alert for a detected anomaly."""
        alert_rule = AlertRule(
            rule_id=f"anomaly_{anomaly.metric_name}_{int(time.time())}",
            name=f"Anomaly in {anomaly.metric_name}",
            description=f"Anomaly detected using {anomaly.detection_method} method",
            metric_name=anomaly.metric_name,
            condition=f"> {anomaly.expected_value * 1.5}",  # Simplified condition
            severity=AlertSeverity.WARNING,
            threshold=anomaly.expected_value * 1.5,
            labels={"anomaly_type": anomaly.detection_method},
            annotations={
                "expected_value": str(anomaly.expected_value),
                "actual_value": str(anomaly.actual_value),
                "anomaly_score": str(anomaly.anomaly_score)
            }
        )

        self.alert_manager.add_alert_rule(alert_rule)

    def _initialize_default_alert_rules(self):
        """Initialize default alert rules."""
        default_rules = [
            AlertRule(
                rule_id="high_cpu_usage",
                name="High CPU Usage",
                description="CPU usage is above 80%",
                metric_name="system_cpu_percent",
                condition="> 80",
                severity=AlertSeverity.WARNING,
                threshold=80.0,
                labels={"component": "system"},
                annotations={"summary": "High CPU usage detected"}
            ),
            AlertRule(
                rule_id="high_memory_usage",
                name="High Memory Usage",
                description="Memory usage is above 90%",
                metric_name="system_memory_percent",
                condition="> 90",
                severity=AlertSeverity.ERROR,
                threshold=90.0,
                labels={"component": "system"},
                annotations={"summary": "High memory usage detected"}
            ),
            AlertRule(
                rule_id="cache_miss_rate_high",
                name="High Cache Miss Rate",
                description="Cache miss rate is above 30%",
                metric_name="cache_miss_rate",
                condition="> 0.3",
                severity=AlertSeverity.WARNING,
                threshold=0.3,
                labels={"component": "cache"},
                annotations={"summary": "High cache miss rate detected"}
            ),
            AlertRule(
                rule_id="response_time_high",
                name="High Response Time",
                description="Average response time is above 2 seconds",
                metric_name="response_time_avg",
                condition="> 2000",
                severity=AlertSeverity.ERROR,
                threshold=2000.0,
                labels={"component": "api"},
                annotations={"summary": "High response time detected"}
            )
        ]

        for rule in default_rules:
            self.alert_manager.add_alert_rule(rule)

    def record_metric(self, name: str, value: float, metric_type: MetricType = MetricType.GAUGE,
                     labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        metric = MetricValue(
            name=name,
            value=value,
            metric_type=metric_type,
            labels=labels or {}
        )
        self.metrics_collector.record_metric(metric)

    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)

    def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        return {
            "monitoring_active": self._monitoring_active,
            "monitoring_interval_seconds": self.monitoring_interval_seconds,
            "anomaly_detection_enabled": self.enable_anomaly_detection,
            "predictive_alerts_enabled": self.enable_predictive_alerts,
            "metrics_count": len(self.metrics_collector.metrics_storage),
            "active_alerts_count": len(self.alert_manager.active_alerts),
            "alert_rules_count": len(self.alert_manager.alert_rules),
            "anomaly_history_count": len(self.anomaly_detector.anomaly_history) if self.anomaly_detector else 0,
            "alert_callbacks_count": len(self.alert_callbacks)
        }

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for monitoring dashboard."""
        # Get key metrics
        cpu_usage = self.metrics_collector.get_current_value("system_cpu_percent")
        memory_usage = self.metrics_collector.get_current_value("system_memory_percent")
        cache_hit_rate = self.metrics_collector.get_current_value("cache_hit_rate")
        response_time = self.metrics_collector.get_current_value("response_time_avg")

        # Get recent alerts
        recent_alerts = self.alert_manager.get_alert_history(hours=1)

        # Get metric trends (last hour)
        cpu_trend = self.metrics_collector.get_metric_values("system_cpu_percent", hours=1)
        memory_trend = self.metrics_collector.get_metric_values("system_memory_percent", hours=1)
        response_trend = self.metrics_collector.get_metric_values("response_time_avg", hours=1)

        return {
            "current_metrics": {
                "cpu_usage_percent": cpu_usage,
                "memory_usage_percent": memory_usage,
                "cache_hit_rate": cache_hit_rate,
                "response_time_avg_ms": response_time
            },
            "trends": {
                "cpu": [{"timestamp": v.timestamp, "value": v.value} for v in cpu_trend[-20:]],
                "memory": [{"timestamp": v.timestamp, "value": v.value} for v in memory_trend[-20:]],
                "response_time": [{"timestamp": v.timestamp, "value": v.value} for v in response_trend[-20:]]
            },
            "alerts": {
                "active": len(self.alert_manager.active_alerts),
                "recent": [
                    {
                        "id": alert.alert_id,
                        "severity": alert.severity.value,
                        "summary": alert.summary,
                        "starts_at": alert.starts_at
                    }
                    for alert in recent_alerts[-10:]
                ]
            },
            "anomalies": [
                {
                    "metric": anomaly.metric_name,
                    "score": anomaly.anomaly_score,
                    "timestamp": anomaly.timestamp
                }
                for anomaly in (self.anomaly_detector.anomaly_history[-10:] if self.anomaly_detector else [])
            ]
        }

    def stop_monitoring(self):
        """Stop the monitoring system."""
        self._monitoring_active = False
        if self._monitoring_thread:
            self._monitoring_thread.join(timeout=5.0)
        logger.info("Enterprise monitoring stopped")


# Global instance
_enterprise_monitor_instance: Optional[EnterpriseMonitor] = None
_monitor_lock = threading.Lock()


def get_enterprise_monitor(config: Optional[Dict[str, Any]] = None) -> EnterpriseMonitor:
    """Get the global enterprise monitor instance."""
    global _enterprise_monitor_instance

    if _enterprise_monitor_instance is None:
        with _monitor_lock:
            if _enterprise_monitor_instance is None:
                default_config = {
                    "monitoring_interval_seconds": 60,
                    "enable_anomaly_detection": True,
                    "enable_predictive_alerts": True
                }
                config = {**default_config, **(config or {})}

                _enterprise_monitor_instance = EnterpriseMonitor(
                    monitoring_interval_seconds=config["monitoring_interval_seconds"],
                    enable_anomaly_detection=config["enable_anomaly_detection"],
                    enable_predictive_alerts=config["enable_predictive_alerts"]
                )

    return _enterprise_monitor_instance