"""CES Quality Monitor.

Provides real-time quality monitoring and alerting for CES Phase 1,
including performance metrics, error tracking, and automated notifications.
"""

import time
import threading
import psutil
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import defaultdict
import logging


@dataclass
class QualityMetric:
    """Represents a quality metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    threshold: Optional[float] = None
    status: str = 'normal'  # 'normal', 'warning', 'critical'


@dataclass
class QualityAlert:
    """Represents a quality alert."""
    alert_id: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    title: str
    description: str
    metric_name: str
    current_value: float
    threshold_value: Optional[float]
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class QualityMonitoringReport:
    """Comprehensive quality monitoring report."""
    system_health_score: float
    total_alerts: int
    active_alerts: int
    resolved_alerts: int
    critical_alerts: int
    metrics: Dict[str, QualityMetric]
    recent_alerts: List[QualityAlert]
    trends: Dict[str, Any]
    timestamp: datetime


class QualityMonitor:
    """Monitors CES quality metrics and provides alerting capabilities."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Monitoring configuration
        self.monitoring_interval = 60  # seconds
        self.alert_history_size = 1000
        self.metric_history_size = 100

        # Quality thresholds
        self.thresholds = {
            'cpu_usage_percent': {'warning': 70.0, 'critical': 90.0},
            'memory_usage_percent': {'warning': 80.0, 'critical': 95.0},
            'response_time_ms': {'warning': 1500.0, 'critical': 2500.0},
            'error_rate_percent': {'warning': 5.0, 'critical': 15.0},
            'disk_usage_percent': {'warning': 85.0, 'critical': 95.0}
        }

        # Monitoring data
        self.metrics_history: Dict[str, List[QualityMetric]] = defaultdict(list)
        self.active_alerts: Dict[str, QualityAlert] = {}
        self.alert_history: List[QualityAlert] = []

        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

        # Alert callbacks
        self.alert_callbacks: List[Callable] = []

    def start_monitoring(self) -> None:
        """Start quality monitoring."""
        if self.monitoring_active:
            self.logger.warning("Quality monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("Quality monitoring started")

    def stop_monitoring(self) -> None:
        """Stop quality monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)

        self.logger.info("Quality monitoring stopped")

    def add_alert_callback(self, callback: Callable) -> None:
        """Add a callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def record_metric(self, name: str, value: float, unit: str = '') -> None:
        """Record a quality metric."""
        metric = QualityMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            threshold=self.thresholds.get(name, {}).get('warning'),
            status=self._determine_metric_status(name, value)
        )

        self.metrics_history[name].append(metric)

        # Keep only recent metrics
        if len(self.metrics_history[name]) > self.metric_history_size:
            self.metrics_history[name] = self.metrics_history[name][-self.metric_history_size:]

        # Check for alerts
        self._check_metric_alerts(metric)

    def get_monitoring_report(self) -> QualityMonitoringReport:
        """Generate comprehensive monitoring report."""
        # Calculate system health score
        health_score = self._calculate_system_health_score()

        # Alert statistics
        total_alerts = len(self.alert_history)
        active_alerts = len(self.active_alerts)
        resolved_alerts = len([a for a in self.alert_history if a.resolved])
        critical_alerts = len([a for a in self.active_alerts.values() if a.severity == 'critical'])

        # Get current metrics
        current_metrics = {}
        for name, metrics in self.metrics_history.items():
            if metrics:
                current_metrics[name] = metrics[-1]

        # Get recent alerts
        recent_alerts = self.alert_history[-10:]

        # Calculate trends
        trends = self._calculate_metric_trends()

        return QualityMonitoringReport(
            system_health_score=health_score,
            total_alerts=total_alerts,
            active_alerts=active_alerts,
            resolved_alerts=resolved_alerts,
            critical_alerts=critical_alerts,
            metrics=current_metrics,
            recent_alerts=recent_alerts,
            trends=trends,
            timestamp=datetime.now()
        )

    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an active alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = datetime.now()

            # Move to history
            self.alert_history.append(alert)
            del self.active_alerts[alert_id]

            # Keep history size manageable
            if len(self.alert_history) > self.alert_history_size:
                self.alert_history = self.alert_history[-self.alert_history_size:]

            self.logger.info(f"Alert resolved: {alert_id}")
            return True

        return False

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info("Quality monitoring loop started")

        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Collect application metrics
                self._collect_application_metrics()

                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying

        self.logger.info("Quality monitoring loop stopped")

    def _collect_system_metrics(self) -> None:
        """Collect system-level quality metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('cpu_usage_percent', cpu_percent, '%')

            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.record_metric('memory_usage_percent', memory_percent, '%')

            # Disk usage
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            self.record_metric('disk_usage_percent', disk_percent, '%')

            # Network connections
            net_connections = len(psutil.net_connections())
            self.record_metric('network_connections', net_connections, 'count')

        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")

    def _collect_application_metrics(self) -> None:
        """Collect application-level quality metrics."""
        try:
            # Response time (simulated - would come from actual application metrics)
            # In a real implementation, this would collect from application instrumentation
            response_time = 250.0  # ms (simulated)
            self.record_metric('response_time_ms', response_time, 'ms')

            # Error rate (simulated)
            error_rate = 2.5  # % (simulated)
            self.record_metric('error_rate_percent', error_rate, '%')

            # Active connections (simulated)
            active_connections = 15  # (simulated)
            self.record_metric('active_connections', active_connections, 'count')

        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {str(e)}")

    def _determine_metric_status(self, name: str, value: float) -> str:
        """Determine the status of a metric based on thresholds."""
        thresholds = self.thresholds.get(name, {})

        if value >= thresholds.get('critical', float('inf')):
            return 'critical'
        elif value >= thresholds.get('warning', float('inf')):
            return 'warning'
        else:
            return 'normal'

    def _check_metric_alerts(self, metric: QualityMetric) -> None:
        """Check if a metric triggers any alerts."""
        if metric.status == 'normal':
            # Check if there's an existing alert that should be resolved
            alert_key = f"{metric.name}_alert"
            if alert_key in self.active_alerts:
                self.resolve_alert(alert_key)
            return

        # Create or update alert
        alert_id = f"{metric.name}_alert"
        severity = 'warning' if metric.status == 'warning' else 'critical'

        if alert_id not in self.active_alerts:
            # Create new alert
            alert = QualityAlert(
                alert_id=alert_id,
                severity=severity,
                title=f"{metric.name.replace('_', ' ').title()} Alert",
                description=f"{metric.name} is {metric.status}: {metric.value}{metric.unit}",
                metric_name=metric.name,
                current_value=metric.value,
                threshold_value=metric.threshold,
                timestamp=datetime.now()
            )

            self.active_alerts[alert_id] = alert

            # Trigger alert callbacks
            self._trigger_alert_callbacks(alert)

            self.logger.warning(f"Alert triggered: {alert.title}")
        else:
            # Update existing alert
            existing_alert = self.active_alerts[alert_id]
            existing_alert.current_value = metric.value
            existing_alert.severity = severity

    def _trigger_alert_callbacks(self, alert: QualityAlert) -> None:
        """Trigger alert callback functions."""
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Error in alert callback: {str(e)}")

    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score."""
        if not self.metrics_history:
            return 100.0

        total_score = 0
        metric_count = 0

        for name, metrics in self.metrics_history.items():
            if metrics:
                latest_metric = metrics[-1]
                metric_score = self._calculate_metric_health_score(latest_metric)
                total_score += metric_score
                metric_count += 1

        if metric_count == 0:
            return 100.0

        # Factor in active alerts
        alert_penalty = len(self.active_alerts) * 5  # 5 points per active alert

        health_score = max(0.0, (total_score / metric_count) - alert_penalty)

        return health_score

    def _calculate_metric_health_score(self, metric: QualityMetric) -> float:
        """Calculate health score for a single metric."""
        if metric.status == 'normal':
            return 100.0
        elif metric.status == 'warning':
            return 75.0
        elif metric.status == 'critical':
            return 50.0
        else:
            return 100.0

    def _calculate_metric_trends(self) -> Dict[str, Any]:
        """Calculate metric trends over time."""
        trends = {}

        for name, metrics in self.metrics_history.items():
            if len(metrics) < 2:
                continue

            # Calculate trend direction
            recent_values = [m.value for m in metrics[-10:]]  # Last 10 measurements
            if len(recent_values) >= 2:
                first_half = recent_values[:len(recent_values)//2]
                second_half = recent_values[len(recent_values)//2:]

                first_avg = sum(first_half) / len(first_half)
                second_avg = sum(second_half) / len(second_half)

                if second_avg > first_avg * 1.05:  # 5% increase
                    trend = 'increasing'
                elif second_avg < first_avg * 0.95:  # 5% decrease
                    trend = 'decreasing'
                else:
                    trend = 'stable'

                trends[name] = {
                    'direction': trend,
                    'change_percent': ((second_avg - first_avg) / first_avg) * 100 if first_avg != 0 else 0,
                    'data_points': len(recent_values)
                }

        return trends

    def get_metric_history(self, metric_name: str, hours: int = 24) -> List[QualityMetric]:
        """Get historical data for a specific metric."""
        if metric_name not in self.metrics_history:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history[metric_name] if m.timestamp >= cutoff_time]

    def get_alert_history(self, hours: int = 24) -> List[QualityAlert]:
        """Get historical alert data."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alert_history if a.timestamp >= cutoff_time]

    def export_monitoring_data(self, file_path: str) -> None:
        """Export monitoring data to a file."""
        try:
            data = {
                'metrics_history': {
                    name: [
                        {
                            'value': m.value,
                            'unit': m.unit,
                            'timestamp': m.timestamp.isoformat(),
                            'status': m.status
                        }
                        for m in metrics
                    ]
                    for name, metrics in self.metrics_history.items()
                },
                'alert_history': [
                    {
                        'alert_id': a.alert_id,
                        'severity': a.severity,
                        'title': a.title,
                        'description': a.description,
                        'metric_name': a.metric_name,
                        'current_value': a.current_value,
                        'threshold_value': a.threshold_value,
                        'timestamp': a.timestamp.isoformat(),
                        'resolved': a.resolved,
                        'resolved_at': a.resolved_at.isoformat() if a.resolved_at else None
                    }
                    for a in self.alert_history
                ],
                'active_alerts': {
                    alert_id: {
                        'severity': a.severity,
                        'title': a.title,
                        'description': a.description,
                        'metric_name': a.metric_name,
                        'current_value': a.current_value,
                        'threshold_value': a.threshold_value,
                        'timestamp': a.timestamp.isoformat()
                    }
                    for alert_id, a in self.active_alerts.items()
                },
                'export_timestamp': datetime.now().isoformat()
            }

            import json
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)

            self.logger.info(f"Monitoring data exported to {file_path}")

        except Exception as e:
            self.logger.error(f"Error exporting monitoring data: {str(e)}")

    def import_monitoring_data(self, file_path: str) -> None:
        """Import monitoring data from a file."""
        try:
            import json
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Import metrics history
            for name, metrics_data in data.get('metrics_history', {}).items():
                for metric_data in metrics_data:
                    metric = QualityMetric(
                        name=name,
                        value=metric_data['value'],
                        unit=metric_data['unit'],
                        timestamp=datetime.fromisoformat(metric_data['timestamp']),
                        status=metric_data['status']
                    )
                    self.metrics_history[name].append(metric)

            # Import alert history
            for alert_data in data.get('alert_history', []):
                alert = QualityAlert(
                    alert_id=alert_data['alert_id'],
                    severity=alert_data['severity'],
                    title=alert_data['title'],
                    description=alert_data['description'],
                    metric_name=alert_data['metric_name'],
                    current_value=alert_data['current_value'],
                    threshold_value=alert_data['threshold_value'],
                    timestamp=datetime.fromisoformat(alert_data['timestamp']),
                    resolved=alert_data['resolved'],
                    resolved_at=datetime.fromisoformat(alert_data['resolved_at']) if alert_data['resolved_at'] else None
                )
                self.alert_history.append(alert)

            # Import active alerts
            for alert_id, alert_data in data.get('active_alerts', {}).items():
                alert = QualityAlert(
                    alert_id=alert_id,
                    severity=alert_data['severity'],
                    title=alert_data['title'],
                    description=alert_data['description'],
                    metric_name=alert_data['metric_name'],
                    current_value=alert_data['current_value'],
                    threshold_value=alert_data['threshold_value'],
                    timestamp=datetime.fromisoformat(alert_data['timestamp'])
                )
                self.active_alerts[alert_id] = alert

            self.logger.info(f"Monitoring data imported from {file_path}")

        except Exception as e:
            self.logger.error(f"Error importing monitoring data: {str(e)}")

    def get_health_summary(self) -> Dict[str, Any]:
        """Get a quick health summary."""
        report = self.get_monitoring_report()

        return {
            'health_score': report.system_health_score,
            'status': 'healthy' if report.system_health_score >= 80 else ('warning' if report.system_health_score >= 60 else 'critical'),
            'active_alerts': report.active_alerts,
            'critical_alerts': report.critical_alerts,
            'last_updated': report.timestamp.isoformat()
        }