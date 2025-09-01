"""
Error reporting system for CodeSage MCP server.

This module provides comprehensive error reporting with integration hooks for monitoring tools,
webhook-based alerting, performance monitoring, and privacy-compliant notifications.
"""

import asyncio
import json
import os
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional, Callable

import aiohttp
import atexit
import psutil

from ..config.config import get_optional_env_var
from .logging_config import get_logger, log_exception


# Environment variable configuration
ERROR_REPORTING_ENABLED = get_optional_env_var("ERROR_REPORTING_ENABLED") == "true"
SENTRY_DSN = get_optional_env_var("SENTRY_DSN")
PROMETHEUS_PUSHGATEWAY_URL = get_optional_env_var("PROMETHEUS_PUSHGATEWAY_URL")
SLACK_WEBHOOK_URL = get_optional_env_var("SLACK_WEBHOOK_URL")
EMAIL_WEBHOOK_URL = get_optional_env_var("EMAIL_WEBHOOK_URL")
DASHBOARD_WEBHOOK_URL = get_optional_env_var("DASHBOARD_WEBHOOK_URL")
ERROR_LOG_FILE = get_optional_env_var("ERROR_LOG_FILE") or "logs/errors.log"
SLOW_OPERATION_THRESHOLD = float(get_optional_env_var("SLOW_OPERATION_THRESHOLD") or "5.0")
MEMORY_SPIKE_THRESHOLD_MB = int(get_optional_env_var("MEMORY_SPIKE_THRESHOLD_MB") or "500")
PRIVACY_SANITIZE_FIELDS = get_optional_env_var("PRIVACY_SANITIZE_FIELDS") or "password,token,key,secret,api_key"


class ErrorReporter:
    """Central error reporting system with monitoring integrations."""

    def __init__(self):
        self.logger = get_logger("error_reporting")
        self._sentry_client = None
        self._prometheus_client = None
        self._webhook_session = None
        self._privacy_filters = self._parse_privacy_filters()

        # Initialize integrations
        if ERROR_REPORTING_ENABLED:
            self._init_integrations()

    def _parse_privacy_filters(self) -> List[str]:
        """Parse privacy filter fields from environment variable."""
        if not PRIVACY_SANITIZE_FIELDS:
            return []
        return [field.strip().lower() for field in PRIVACY_SANITIZE_FIELDS.split(",")]

    def _init_integrations(self):
        """Initialize monitoring tool integrations."""
        try:
            if SENTRY_DSN:
                import sentry_sdk
                sentry_sdk.init(dsn=SENTRY_DSN)
                self._sentry_client = sentry_sdk
                self.logger.info("Sentry integration initialized")
        except ImportError:
            self.logger.warning("Sentry SDK not available")

        try:
            if PROMETHEUS_PUSHGATEWAY_URL:
                from prometheus_client import CollectorRegistry, push_to_gateway
                self._prometheus_registry = CollectorRegistry()
                self._prometheus_push = lambda: push_to_gateway(
                    PROMETHEUS_PUSHGATEWAY_URL,
                    job="codesage_mcp",
                    registry=self._prometheus_registry
                )
                self.logger.info("Prometheus integration initialized")
        except ImportError:
            self.logger.warning("Prometheus client not available")

    async def _get_webhook_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session for webhooks."""
        if self._webhook_session is None or self._webhook_session.closed:
            self._webhook_session = aiohttp.ClientSession()
        return self._webhook_session

    def _sanitize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize sensitive data from error context."""
        if not isinstance(data, dict):
            return data

        sanitized = {}
        for key, value in data.items():
            if any(filter_field in key.lower() for filter_field in self._privacy_filters):
                sanitized[key] = "[REDACTED]"
            elif isinstance(value, dict):
                sanitized[key] = self._sanitize_data(value)
            elif isinstance(value, list):
                sanitized[key] = [self._sanitize_data(item) if isinstance(item, dict) else item for item in value]
            else:
                sanitized[key] = value
        return sanitized

    async def report_error(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        severity: str = "error",
        request_id: Optional[str] = None,
        session_details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Report an error with full context and integrations."""
        if not ERROR_REPORTING_ENABLED:
            return

        # Sanitize context for privacy
        safe_context = self._sanitize_data(context or {})
        safe_session = self._sanitize_data(session_details or {})

        # Log to existing system
        log_exception(error, self.logger, request_id, safe_session, safe_context)

        # Write to error log file
        await self._write_error_log(error, safe_context, severity, request_id)

        # Send to monitoring tools
        await self._send_to_monitoring_tools(error, safe_context, severity)

        # Send webhook alerts
        await self._send_webhook_alerts(error, safe_context, severity, request_id)

    async def _write_error_log(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: str,
        request_id: Optional[str] = None
    ) -> None:
        """Write error to dedicated log file."""
        try:
            # Ensure log directory exists
            log_dir = os.path.dirname(ERROR_LOG_FILE)
            if log_dir and not os.path.exists(log_dir):
                os.makedirs(log_dir)

            error_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "severity": severity,
                "error_type": type(error).__name__,
                "message": str(error),
                "traceback": traceback.format_exc(),
                "context": context,
                "request_id": request_id,
                "environment": {
                    "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}",
                    "platform": os.sys.platform,
                }
            }

            with open(ERROR_LOG_FILE, "a", encoding="utf-8") as f:
                json.dump(error_entry, f, ensure_ascii=False)
                f.write("\n")

        except Exception as log_error:
            self.logger.error("Failed to write error log", error=str(log_error))

    async def _send_to_monitoring_tools(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: str
    ) -> None:
        """Send error to monitoring tools (Sentry, Prometheus)."""
        # Sentry integration
        if self._sentry_client:
            try:
                with self._sentry_client.configure_scope() as scope:
                    scope.set_context("error_context", context)
                    scope.set_tag("severity", severity)
                    self._sentry_client.capture_exception(error)
            except Exception as sentry_error:
                self.logger.error("Sentry reporting failed", error=str(sentry_error))

        # Prometheus integration
        if self._prometheus_client:
            try:
                from prometheus_client import Counter
                error_counter = Counter(
                    'codesage_error_total',
                    'Total number of errors',
                    ['error_type', 'severity'],
                    registry=self._prometheus_registry
                )
                error_counter.labels(
                    error_type=type(error).__name__,
                    severity=severity
                ).inc()
                self._prometheus_push()
            except Exception as prom_error:
                self.logger.error("Prometheus reporting failed", error=str(prom_error))

    async def _send_webhook_alerts(
        self,
        error: Exception,
        context: Dict[str, Any],
        severity: str,
        request_id: Optional[str] = None
    ) -> None:
        """Send error alerts via webhooks (Slack, Email, Dashboard)."""
        alert_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": severity,
            "error_type": type(error).__name__,
            "message": str(error),
            "context": context,
            "request_id": request_id,
            "service": "codesage_mcp"
        }

        webhooks = []
        if SLACK_WEBHOOK_URL:
            webhooks.append((SLACK_WEBHOOK_URL, self._format_slack_alert(alert_data)))
        if EMAIL_WEBHOOK_URL:
            webhooks.append((EMAIL_WEBHOOK_URL, alert_data))
        if DASHBOARD_WEBHOOK_URL:
            webhooks.append((DASHBOARD_WEBHOOK_URL, alert_data))

        if webhooks:
            session = await self._get_webhook_session()
            for url, payload in webhooks:
                try:
                    async with session.post(url, json=payload, timeout=5.0) as response:
                        if response.status >= 400:
                            self.logger.warning(f"Webhook failed: {url}, status: {response.status}")
                except Exception as webhook_error:
                    self.logger.error(f"Webhook error for {url}", error=str(webhook_error))

    def _format_slack_alert(self, alert_data: Dict[str, Any]) -> Dict[str, Any]:
        """Format alert data for Slack webhook."""
        severity_emoji = {
            "critical": "🚨",
            "error": "❌",
            "warning": "⚠️",
            "info": "ℹ️"
        }.get(alert_data["severity"], "❓")

        return {
            "text": f"{severity_emoji} *{alert_data['severity'].upper()}* in CodeSage MCP",
            "blocks": [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"{severity_emoji} *{alert_data['severity'].upper()}*: {alert_data['message']}"
                    }
                },
                {
                    "type": "section",
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Type:* {alert_data['error_type']}"},
                        {"type": "mrkdwn", "text": f"*Request ID:* {alert_data.get('request_id', 'N/A')}"},
                        {"type": "mrkdwn", "text": f"*Time:* {alert_data['timestamp']}"},
                        {"type": "mrkdwn", "text": f"*Service:* {alert_data['service']}"}
                    ]
                }
            ]
        }

    async def monitor_performance(
        self,
        operation_name: str,
        operation_func: Callable,
        *args,
        **kwargs
    ) -> Any:
        """Monitor operation performance and report anomalies."""
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        try:
            result = await operation_func(*args, **kwargs) if asyncio.iscoroutinefunction(operation_func) else operation_func(*args, **kwargs)
            return result
        except Exception as e:
            # Report the error
            await self.report_error(e, {"operation": operation_name}, "error")
            raise
        finally:
            # Check performance metrics
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

            duration = end_time - start_time
            memory_delta = end_memory - start_memory

            # Log slow operations
            if duration > SLOW_OPERATION_THRESHOLD:
                await self.report_error(
                    Exception(f"Slow operation: {operation_name} took {duration:.2f}s"),
                    {"operation": operation_name, "duration": duration},
                    "warning"
                )

            # Log memory spikes
            if memory_delta > MEMORY_SPIKE_THRESHOLD_MB:
                await self.report_error(
                    Exception(f"Memory spike: {operation_name} used {memory_delta:.2f}MB"),
                    {"operation": operation_name, "memory_delta": memory_delta},
                    "warning"
                )

    async def close(self):
        """Clean up resources."""
        if self._webhook_session and not self._webhook_session.closed:
            await self._webhook_session.close()


# Global error reporter instance
error_reporter = ErrorReporter()


async def report_critical_error(
    error: Exception,
    context: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None
) -> None:
    """Convenience function to report critical errors."""
    await error_reporter.report_error(error, context, "critical", request_id)


async def demonstrate_error_reporting():
    """Demonstration function showing error reporting capabilities."""
    print("=== CodeSage MCP Error Reporting Demonstration ===\n")

    # Example 1: Basic error reporting
    print("1. Reporting a sample error...")
    try:
        raise ValueError("This is a demonstration error")
    except Exception as e:
        await error_reporter.report_error(
            e,
            {"demo": True, "user_id": "demo_user", "api_key": "secret_key_123"},
            "error",
            "demo_request_123"
        )
    print("   ✓ Error logged to file and sent to configured webhooks\n")

    # Example 2: Performance monitoring
    print("2. Demonstrating performance monitoring...")

    async def slow_operation():
        await asyncio.sleep(2)  # Simulate slow operation
        return "completed"

    result = await error_reporter.monitor_performance("demo_slow_op", slow_operation)
    print(f"   ✓ Operation result: {result}")
    print("   ✓ Performance metrics logged if thresholds exceeded\n")

    # Example 3: Memory monitoring
    print("3. Demonstrating memory usage monitoring...")

    def memory_intensive_operation():
        # Simulate memory usage
        large_list = [i for i in range(100000)]
        return len(large_list)

    result = await error_reporter.monitor_performance("demo_memory_op", memory_intensive_operation)
    print(f"   ✓ Operation result: {result}")
    print("   ✓ Memory usage monitored\n")

    # Example 4: Privacy filtering
    print("4. Demonstrating privacy filtering...")
    sensitive_context = {
        "user_email": "user@example.com",
        "password": "secret123",
        "api_token": "token_abc123",
        "normal_field": "safe_value"
    }

    sanitized = error_reporter._sanitize_data(sensitive_context)
    print(f"   Original: {sensitive_context}")
    print(f"   Sanitized: {sanitized}")
    print("   ✓ Sensitive fields redacted\n")

    print("=== Demonstration Complete ===")
    print(f"Check {ERROR_LOG_FILE} for logged errors")
    print("Configured webhooks should have received notifications")


# Cleanup on module unload

@atexit.register
def cleanup():
    """Clean up error reporter resources on exit."""
    if hasattr(error_reporter, '_webhook_session') and error_reporter._webhook_session:
        asyncio.create_task(error_reporter.close())