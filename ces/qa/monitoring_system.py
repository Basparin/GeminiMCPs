"""CES Quality Monitoring and Alerting System.

Continuous monitoring of system quality metrics with automated alerting
and trend analysis for proactive issue detection.
"""

import time
import threading
import json
import smtplib
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


@dataclass
class QualityMetric:
    """Individual quality metric measurement."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    component: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None


@dataclass
class Alert:
    """Quality alert notification."""
    alert_id: str
    severity: str  # 'info', 'warning', 'critical'
    title: str
    description: str
    component: str
    metric_name: str
    current_value: float
    threshold_value: float
    timestamp: datetime
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class MonitoringReport:
    """Quality monitoring report."""
    total_metrics: int
    active_alerts: int
    critical_alerts: int
    warning_alerts: int
    resolved_alerts: int
    metric_trends: Dict[str, List[Dict[str, Any]]]
    system_health_score: float
    recommendations: List[str]
    timestamp: datetime


class QualityMonitor:
    """Continuous quality monitoring and alerting system."""

    def __init__(self, project_root: Optional[str] = None):
        self.project_root = Path(project_root or Path(__file__).parent.parent.parent)
        self.logger = logging.getLogger(__name__)

        # Monitoring configuration
        self.monitoring_interval = 300  # 5 minutes
        self.metric_history_size = 100  # Keep last 100 measurements per metric
        self.alert_history_size = 500  # Keep last 500 alerts

        # Metric storage
        self.metric_history: Dict[str, deque] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.resolved_alerts: deque = deque(maxlen=self.alert_history_size)

        # Alert thresholds
        self.thresholds = {
            'response_time_p95': {'warning': 1500.0, 'critical': 2000.0},  # ms
            'cpu_usage': {'warning': 70.0, 'critical': 90.0},  # %
            'memory_usage': {'warning': 80.0, 'critical': 95.0},  # %
            'error_rate': {'warning': 5.0, 'critical': 15.0},  # %
            'test_coverage': {'warning': 85.0, 'critical': 75.0},  # %
            'security_score': {'warning': 85.0, 'critical': 70.0},  # %
            'accessibility_score': {'warning': 90.0, 'critical': 80.0}  # %
        }

        # Alert channels
        self.alert_channels = {
            'console': self._alert_console,
            'file': self._alert_file,
            'email': self._alert_email
        }

        # Monitoring thread
        self.monitoring_thread: Optional[threading.Thread] = None
        self.monitoring_active = False

        # Email configuration (can be set via config)
        self.email_config = {
            'smtp_server': 'localhost',
            'smtp_port': 587,
            'sender_email': 'ces-monitor@localhost',
            'recipient_emails': ['admin@localhost'],
            'use_tls': True
        }

    def start_monitoring(self) -> None:
        """Start continuous quality monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()

        self.logger.info("Quality monitoring started")

    def stop_monitoring(self) -> None:
        """Stop continuous quality monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)

        self.logger.info("Quality monitoring stopped")

    def record_metric(self, name: str, value: float, unit: str, component: str) -> None:
        """Record a quality metric measurement."""
        metric = QualityMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            component=component,
            threshold_warning=self.thresholds.get(name, {}).get('warning'),
            threshold_critical=self.thresholds.get(name, {}).get('critical')
        )

        # Initialize metric history if needed
        if name not in self.metric_history:
            self.metric_history[name] = deque(maxlen=self.metric_history_size)

        # Add to history
        self.metric_history[name].append(metric)

        # Check for alerts
        self._check_metric_alerts(metric)

    def _check_metric_alerts(self, metric: QualityMetric) -> None:
        """Check if metric triggers any alerts."""
        if metric.threshold_critical and metric.value >= metric.threshold_critical:
            self._create_alert('critical', metric)
        elif metric.threshold_warning and metric.value >= metric.threshold_warning:
            self._create_alert('warning', metric)

    def _create_alert(self, severity: str, metric: QualityMetric) -> None:
        """Create and dispatch an alert."""
        alert_id = f"{metric.name}_{metric.component}_{int(time.time())}"

        # Check if similar alert already exists
        existing_alert = self.active_alerts.get(alert_id)
        if existing_alert and not existing_alert.resolved:
            return  # Don't create duplicate alerts

        threshold_value = (
            metric.threshold_critical if severity == 'critical'
            else metric.threshold_warning
        )

        alert = Alert(
            alert_id=alert_id,
            severity=severity,
            title=f"{severity.capitalize()}: {metric.name} threshold exceeded",
            description=f"{metric.name} in {metric.component} is {metric.value:.2f}{metric.unit}, "
                       f"exceeding {severity} threshold of {threshold_value:.2f}{metric.unit}",
            component=metric.component,
            metric_name=metric.name,
            current_value=metric.value,
            threshold_value=threshold_value,
            timestamp=datetime.now()
        )

        self.active_alerts[alert_id] = alert

        # Dispatch alert to all channels
        self._dispatch_alert(alert)

        self.logger.warning(f"Alert created: {alert.title}")

    def _dispatch_alert(self, alert: Alert) -> None:
        """Dispatch alert to all configured channels."""
        for channel_name, channel_func in self.alert_channels.items():
            try:
                channel_func(alert)
            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel_name}: {str(e)}")

    def _alert_console(self, alert: Alert) -> None:
        """Send alert to console."""
        severity_color = {
            'info': '\033[36m',  # Cyan
            'warning': '\033[33m',  # Yellow
            'critical': '\033[31m'  # Red
        }.get(alert.severity, '\033[0m')

        reset_color = '\033[0m'

        print(f"{severity_color}[{alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] "
              f"{alert.severity.upper()}: {alert.title}{reset_color}")
        print(f"  {alert.description}")

    def _alert_file(self, alert: Alert) -> None:
        """Write alert to log file."""
        log_dir = self.project_root / "logs"
        log_dir.mkdir(exist_ok=True)

        log_file = log_dir / "quality_alerts.log"

        with open(log_file, 'a') as f:
            f.write(f"[{alert.timestamp.isoformat()}] {alert.severity.upper()}: "
                   f"{alert.title}\n")
            f.write(f"  {alert.description}\n")
            f.write(f"  Component: {alert.component}, Metric: {alert.metric_name}\n")
            f.write(f"  Value: {alert.current_value}, Threshold: {alert.threshold_value}\n\n")

    def _alert_email(self, alert: Alert) -> None:
        """Send alert via email."""
        if not self.email_config.get('recipient_emails'):
            return

        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_config['sender_email']
            msg['To'] = ', '.join(self.email_config['recipient_emails'])
            msg['Subject'] = f"CES Quality Alert: {alert.title}"

            body = f"""
CES Quality Monitoring Alert

Severity: {alert.severity.upper()}
Title: {alert.title}
Description: {alert.description}

Component: {alert.component}
Metric: {alert.metric_name}
Current Value: {alert.current_value}
Threshold: {alert.threshold_value}

Timestamp: {alert.timestamp.isoformat()}

This is an automated message from the CES Quality Monitoring System.
            """

            msg.attach(MIMEText(body, 'plain'))

            server = smtplib.SMTP(
                self.email_config['smtp_server'],
                self.email_config['smtp_port']
            )

            if self.email_config.get('use_tls'):
                server.starttls()

            # Note: In production, you would need proper authentication
            # server.login(username, password)

            server.send_message(msg)
            server.quit()

        except Exception as e:
            self.logger.error(f"Failed to send email alert: {str(e)}")

    def resolve_alert(self, alert_id: str, resolution_note: Optional[str] = None) -> bool:
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return False

        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()

        # Move to resolved alerts
        self.resolved_alerts.append(alert)
        del self.active_alerts[alert_id]

        self.logger.info(f"Alert resolved: {alert_id}")

        # Send resolution notification
        resolution_alert = Alert(
            alert_id=f"{alert_id}_resolved",
            severity='info',
            title=f"Alert Resolved: {alert.title}",
            description=f"Alert has been resolved. {resolution_note or ''}",
            component=alert.component,
            metric_name=alert.metric_name,
            current_value=alert.current_value,
            threshold_value=alert.threshold_value,
            timestamp=datetime.now(),
            resolved=True,
            resolved_at=datetime.now()
        )

        self._dispatch_alert(resolution_alert)

        return True

    def get_monitoring_report(self) -> MonitoringReport:
        """Generate comprehensive monitoring report."""
        total_metrics = sum(len(history) for history in self.metric_history.values())
        active_alerts = len(self.active_alerts)
        critical_alerts = len([a for a in self.active_alerts.values() if a.severity == 'critical'])
        warning_alerts = len([a for a in self.active_alerts.values() if a.severity == 'warning'])
        resolved_alerts = len(self.resolved_alerts)

        # Generate metric trends
        metric_trends = {}
        for metric_name, history in self.metric_history.items():
            if len(history) > 1:
                trend_data = []
                for i, metric in enumerate(history):
                    trend_data.append({
                        'timestamp': metric.timestamp.isoformat(),
                        'value': metric.value,
                        'index': i
                    })
                metric_trends[metric_name] = trend_data

        # Calculate system health score
        system_health_score = self._calculate_system_health_score()

        # Generate recommendations
        recommendations = self._generate_monitoring_recommendations()

        return MonitoringReport(
            total_metrics=total_metrics,
            active_alerts=active_alerts,
            critical_alerts=critical_alerts,
            warning_alerts=warning_alerts,
            resolved_alerts=resolved_alerts,
            metric_trends=metric_trends,
            system_health_score=system_health_score,
            recommendations=recommendations,
            timestamp=datetime.now()
        )

    def _calculate_system_health_score(self) -> float:
        """Calculate overall system health score based on metrics and alerts."""
        if not self.metric_history:
            return 100.0  # No metrics = assume healthy

        # Base score from metric compliance
        total_score = 0.0
        metric_count = 0

        for metric_name, history in self.metric_history.items():
            if history:
                latest_metric = history[-1]
                metric_score = 100.0  # Start with perfect score

                # Deduct points for threshold violations
                if latest_metric.threshold_critical and latest_metric.value >= latest_metric.threshold_critical:
                    metric_score -= 30  # Critical violation
                elif latest_metric.threshold_warning and latest_metric.value >= latest_metric.threshold_warning:
                    metric_score -= 15  # Warning violation

                total_score += metric_score
                metric_count += 1

        if metric_count == 0:
            return 100.0

        base_score = total_score / metric_count

        # Deduct points for active alerts
        alert_penalty = len(self.active_alerts) * 5  # 5 points per active alert
        critical_penalty = self.get_monitoring_report().critical_alerts * 10  # Extra 10 points per critical alert

        final_score = max(0.0, base_score - alert_penalty - critical_penalty)

        return final_score

    def _generate_monitoring_recommendations(self) -> List[str]:
        """Generate monitoring and alerting recommendations."""
        recommendations = []

        report = self.get_monitoring_report()

        if report.critical_alerts > 0:
            recommendations.append(f"Address {report.critical_alerts} critical alerts immediately")

        if report.warning_alerts > 0:
            recommendations.append(f"Review {report.warning_alerts} warning alerts")

        if report.system_health_score < 70.0:
            recommendations.append("System health score is low - investigate root causes")

        # Check for metrics with concerning trends
        for metric_name, trend in report.metric_trends.items():
            if len(trend) >= 5:
                # Simple trend analysis - check if last 3 values are increasing
                recent_values = [point['value'] for point in trend[-3:]]
                if len(recent_values) == 3 and recent_values[0] < recent_values[1] < recent_values[2]:
                    recommendations.append(f"Investigate increasing trend in {metric_name}")

        if len(self.metric_history) < 5:
            recommendations.append("Consider monitoring more quality metrics for comprehensive coverage")

        recommendations.append(f"System health score: {report.system_health_score:.1f}%")

        return recommendations

    def _monitoring_loop(self) -> None:
        """Main monitoring loop."""
        self.logger.info("Monitoring loop started")

        while self.monitoring_active:
            try:
                # Collect system metrics
                self._collect_system_metrics()

                # Check for alert auto-resolution
                self._check_alert_auto_resolution()

                # Sleep until next monitoring cycle
                time.sleep(self.monitoring_interval)

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {str(e)}")
                time.sleep(60)  # Wait a minute before retrying

        self.logger.info("Monitoring loop stopped")

    def _collect_system_metrics(self) -> None:
        """Collect current system quality metrics."""
        try:
            import psutil

            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent

            self.record_metric('cpu_usage', cpu_percent, '%', 'system')
            self.record_metric('memory_usage', memory_percent, '%', 'system')

            # Disk usage
            disk_percent = psutil.disk_usage('/').percent
            self.record_metric('disk_usage', disk_percent, '%', 'system')

        except ImportError:
            # psutil not available
            pass
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {str(e)}")

    def _check_alert_auto_resolution(self) -> None:
        """Check for alerts that can be auto-resolved."""
        alerts_to_resolve = []

        for alert_id, alert in self.active_alerts.items():
            # Check if metric has returned to normal
            if alert.metric_name in self.metric_history:
                history = self.metric_history[alert.metric_name]
                if history:
                    latest_metric = history[-1]

                    # Auto-resolve if value is now below warning threshold
                    warning_threshold = self.thresholds.get(alert.metric_name, {}).get('warning')
                    if warning_threshold and latest_metric.value < warning_threshold:
                        alerts_to_resolve.append((alert_id, "Metric returned to normal range"))

        # Resolve alerts
        for alert_id, note in alerts_to_resolve:
            self.resolve_alert(alert_id, note)

    def configure_email_alerts(self, smtp_server: str, smtp_port: int,
                             sender_email: str, recipient_emails: List[str],
                             use_tls: bool = True) -> None:
        """Configure email alerting."""
        self.email_config.update({
            'smtp_server': smtp_server,
            'smtp_port': smtp_port,
            'sender_email': sender_email,
            'recipient_emails': recipient_emails,
            'use_tls': use_tls
        })

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_alerts = []
        for alert in list(self.resolved_alerts) + list(self.active_alerts.values()):
            if alert.timestamp >= cutoff_time:
                recent_alerts.append(alert)

        return sorted(recent_alerts, key=lambda x: x.timestamp, reverse=True)

    def export_monitoring_data(self, output_path: Optional[str] = None) -> str:
        """Export monitoring data to JSON file."""
        if output_path is None:
            output_path = self.project_root / "benchmark_results" / f"monitoring_data_{int(time.time())}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare export data
        export_data = {
            'timestamp': datetime.now().isoformat(),
            'metric_history': {},
            'active_alerts': {},
            'resolved_alerts': [],
            'thresholds': self.thresholds
        }

        # Export metric history
        for metric_name, history in self.metric_history.items():
            export_data['metric_history'][metric_name] = [
                {
                    'value': metric.value,
                    'unit': metric.unit,
                    'timestamp': metric.timestamp.isoformat(),
                    'component': metric.component
                }
                for metric in history
            ]

        # Export alerts
        for alert_id, alert in self.active_alerts.items():
            export_data['active_alerts'][alert_id] = {
                'severity': alert.severity,
                'title': alert.title,
                'description': alert.description,
                'component': alert.component,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'timestamp': alert.timestamp.isoformat(),
                'resolved': alert.resolved
            }

        for alert in self.resolved_alerts:
            export_data['resolved_alerts'].append({
                'alert_id': alert.alert_id,
                'severity': alert.severity,
                'title': alert.title,
                'description': alert.description,
                'component': alert.component,
                'metric_name': alert.metric_name,
                'current_value': alert.current_value,
                'threshold_value': alert.threshold_value,
                'timestamp': alert.timestamp.isoformat(),
                'resolved': alert.resolved,
                'resolved_at': alert.resolved_at.isoformat() if alert.resolved_at else None
            })

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        return str(output_path)