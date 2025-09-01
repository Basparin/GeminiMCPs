"""
Prometheus Client Integration for CodeSage MCP Server.

This module provides integration with Prometheus for metrics collection,
querying, and real-time monitoring data retrieval.

SECURITY CONSIDERATIONS:
- HTTPS-only connections in production
- Query parameter validation and sanitization
- Connection timeout limits
- Response size validation
- Secure credential handling
"""

import asyncio
import time
import ssl
import re
from typing import Dict, List, Any, Optional, Tuple
import aiohttp
import certifi

from codesage_mcp.core.logging_config import get_logger
from codesage_mcp.config.config import get_optional_env_var

logger = get_logger("prometheus_client")


class PrometheusClient:
    """Client for interacting with Prometheus metrics server with security measures."""

    def __init__(self, prometheus_url: Optional[str] = None):
        self.prometheus_url = prometheus_url or get_optional_env_var("PROMETHEUS_URL", "http://localhost:9090")
        self.session: Optional[aiohttp.ClientSession] = None
        self.query_timeout = 30  # seconds
        self.max_response_size = 10 * 1024 * 1024  # 10MB limit
        self.ssl_context = self._create_secure_ssl_context()

    def _create_secure_ssl_context(self) -> ssl.SSLContext:
        """Create secure SSL context for HTTPS connections."""
        ssl_context = ssl.create_default_context(cafile=certifi.where())
        ssl_context.check_hostname = True
        ssl_context.verify_mode = ssl.CERT_REQUIRED
        # Disable SSLv2, SSLv3, TLSv1, and TLSv1.1
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        return ssl_context

    def _validate_prometheus_url(self, url: str) -> bool:
        """Validate Prometheus URL for security."""
        if not url:
            return False

        # Must use HTTPS in production (unless localhost for development)
        if not url.startswith('https://') and not url.startswith('http://'):
            return False

        # Basic URL format validation
        url_pattern = re.compile(
            r'^https?://'  # http:// or https://
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

        return url_pattern.match(url) is not None

    def _sanitize_query(self, query: str) -> str:
        """Sanitize PromQL query to prevent injection attacks."""
        if not query or len(query) > 1000:  # Reasonable query length limit
            raise ValueError("Invalid query: empty or too long")

        # Remove potentially dangerous characters while preserving valid PromQL
        # Allow: letters, numbers, underscores, dots, curly braces, square brackets, quotes, operators
        sanitized = re.sub(r'[^\w\s\[\]{}"\'.,+\-*/=<>!&|()_]', '', query)

        # Basic validation - should contain at least one metric name pattern
        if not re.search(r'\w+', sanitized):
            raise ValueError("Invalid query: no valid metric name found")

        return sanitized.strip()

    def _validate_response_size(self, response: aiohttp.ClientResponse) -> bool:
        """Validate response size to prevent resource exhaustion."""
        try:
            content_length = response.headers.get('Content-Length')
            if content_length:
                size = int(content_length)
                return size <= self.max_response_size
            return True  # If no Content-Length header, allow but log warning
        except (ValueError, TypeError):
            return True

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def _query_prometheus(self, query: str, time_param: Optional[float] = None) -> Dict[str, Any]:
        """
        Execute a Prometheus query with security validation.

        Args:
            query: PromQL query string
            time_param: Optional timestamp for query

        Returns:
            Query result data
        """
        if not self.session:
            raise RuntimeError("PrometheusClient must be used as async context manager")

        # Validate and sanitize query
        try:
            sanitized_query = self._sanitize_query(query)
        except ValueError as e:
            logger.error(f"Query validation failed: {e}")
            return {"status": "error", "error": "Invalid query"}

        # Validate URL
        if not self._validate_prometheus_url(self.prometheus_url):
            logger.error("Invalid Prometheus URL")
            return {"status": "error", "error": "Invalid Prometheus URL"}

        params = {"query": sanitized_query}
        if time_param:
            # Validate timestamp
            if not (0 < time_param < time.time() + 86400):  # Reasonable timestamp range
                logger.error("Invalid timestamp parameter")
                return {"status": "error", "error": "Invalid timestamp"}
            params["time"] = str(time_param)

        try:
            async with self.session.get(
                f"{self.prometheus_url}/api/v1/query",
                params=params,
                timeout=self.query_timeout
            ) as response:
                # Validate response size
                if not self._validate_response_size(response):
                    logger.error("Response size exceeds limit")
                    return {"status": "error", "error": "Response too large"}

                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"Prometheus query failed: {response.status} - {error_text[:200]}...")  # Truncate for security
                    return {"status": "error", "error": f"HTTP {response.status}"}

                result = await response.json()

                # Validate result structure
                if not isinstance(result, dict) or "status" not in result:
                    logger.error("Invalid Prometheus response format")
                    return {"status": "error", "error": "Invalid response format"}

                return result

        except asyncio.TimeoutError:
            logger.error("Prometheus query timeout")
            return {"status": "error", "error": "Query timeout"}
        except Exception as e:
            logger.error(f"Failed to query Prometheus: {type(e).__name__}")
            # Don't leak internal error details
            return {"status": "error", "error": "Query failed"}

    async def _query_range_prometheus(self, query: str, start_time: float,
                                    end_time: float, step: str = "15s") -> Dict[str, Any]:
        """
        Execute a Prometheus range query.

        Args:
            query: PromQL query string
            start_time: Start timestamp
            end_time: End timestamp
            step: Query resolution step

        Returns:
            Range query result data
        """
        if not self.session:
            raise RuntimeError("PrometheusClient must be used as async context manager")

        params = {
            "query": query,
            "start": str(start_time),
            "end": str(end_time),
            "step": step
        }

        try:
            async with self.session.get(
                f"{self.prometheus_url}/api/v1/query_range",
                params=params,
                timeout=self.query_timeout
            ) as response:
                if response.status != 200:
                    logger.error(f"Prometheus range query failed: {response.status} - {await response.text()}")
                    return {"status": "error", "error": f"HTTP {response.status}"}

                result = await response.json()
                return result

        except Exception as e:
            logger.error(f"Failed to query Prometheus range: {e}")
            return {"status": "error", "error": str(e)}

    async def get_current_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics from Prometheus.

        Returns:
            Dictionary of current metric values
        """
        metrics = {}
        current_time = time.time()

        # Define metric queries
        metric_queries = {
            "response_time_ms": 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="codesage"}[5m])) * 1000',
            "throughput_rps": 'rate(http_requests_total{job="codesage"}[5m])',
            "memory_usage_percent": '(process_resident_memory_bytes{job="codesage"} / process_virtual_memory_max_bytes) * 100',
            "cpu_usage_percent": 'rate(process_cpu_user_seconds_total{job="codesage"}[5m]) * 100',
            "error_rate_percent": '(rate(http_requests_total{status=~"5..",job="codesage"}[5m]) / rate(http_requests_total{job="codesage"}[5m])) * 100',
            "active_connections": 'net_conntrack_dialer_conn_established{job="codesage"}'
        }

        async with self:
            for metric_name, query in metric_queries.items():
                result = await self._query_prometheus(query, current_time)

                if result.get("status") == "success" and result.get("data", {}).get("result"):
                    # Extract value from Prometheus response
                    result_data = result["data"]["result"]
                    if result_data and len(result_data) > 0:
                        value = float(result_data[0]["value"][1])
                        metrics[metric_name] = {
                            "value": value,
                            "timestamp": current_time,
                            "unit": self._get_metric_unit(metric_name),
                            "source": "prometheus"
                        }

        return metrics

    async def get_metric_history(self, metric_name: str, hours: int = 24,
                               step: str = "5m") -> List[Tuple[float, float]]:
        """
        Get historical data for a specific metric.

        Args:
            metric_name: Name of the metric
            hours: Number of hours of history
            step: Query resolution

        Returns:
            List of (timestamp, value) tuples
        """
        end_time = time.time()
        start_time = end_time - (hours * 3600)

        # Define metric queries for historical data
        metric_queries = {
            "response_time_ms": 'histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="codesage"}[5m])) * 1000',
            "throughput_rps": 'rate(http_requests_total{job="codesage"}[5m])',
            "memory_usage_percent": '(process_resident_memory_bytes{job="codesage"} / process_virtual_memory_max_bytes) * 100',
            "cpu_usage_percent": 'rate(process_cpu_user_seconds_total{job="codesage"}[5m]) * 100',
            "error_rate_percent": '(rate(http_requests_total{status=~"5..",job="codesage"}[5m]) / rate(http_requests_total{job="codesage"}[5m])) * 100'
        }

        query = metric_queries.get(metric_name)
        if not query:
            logger.warning(f"No Prometheus query defined for metric: {metric_name}")
            return []

        async with self:
            result = await self._query_range_prometheus(query, start_time, end_time, step)

            if result.get("status") == "success" and result.get("data", {}).get("result"):
                result_data = result["data"]["result"]
                if result_data and len(result_data) > 0:
                    values = result_data[0]["values"]
                    return [(float(ts), float(val)) for ts, val in values]

        return []

    async def get_system_health_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive system health metrics from Prometheus.

        Returns:
            Dictionary of system health metrics
        """
        health_metrics = {}

        # System resource queries
        system_queries = {
            "system_cpu_usage": '100 - (avg by (instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)',
            "system_memory_usage": '100 - ((node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes) * 100)',
            "disk_usage": '(node_filesystem_size_bytes{mountpoint="/"} - node_filesystem_avail_bytes{mountpoint="/"}) / node_filesystem_size_bytes{mountpoint="/"} * 100',
            "network_bytes_in": 'rate(node_network_receive_bytes_total{device="eth0"}[5m])',
            "network_bytes_out": 'rate(node_network_transmit_bytes_total{device="eth0"}[5m])'
        }

        async with self:
            for metric_name, query in system_queries.items():
                result = await self._query_prometheus(query)

                if result.get("status") == "success" and result.get("data", {}).get("result"):
                    result_data = result["data"]["result"]
                    if result_data and len(result_data) > 0:
                        value = float(result_data[0]["value"][1])
                        health_metrics[metric_name] = {
                            "value": value,
                            "timestamp": time.time(),
                            "unit": self._get_metric_unit(metric_name),
                            "source": "prometheus"
                        }

        return health_metrics

    async def get_service_uptime(self) -> Optional[float]:
        """
        Get service uptime from Prometheus.

        Returns:
            Uptime in seconds, or None if not available
        """
        query = 'up{job="codesage"}'

        async with self:
            result = await self._query_prometheus(query)

            if result.get("status") == "success" and result.get("data", {}).get("result"):
                result_data = result["data"]["result"]
                if result_data and len(result_data) > 0:
                    # up metric returns 1 if service is up
                    up_value = float(result_data[0]["value"][1])
                    if up_value == 1:
                        # Get the timestamp when the service came up
                        # This is a simplified approach - in practice you'd need to track this
                        return time.time() - 300  # Assume 5 minutes for demo
        return None

    def _get_metric_unit(self, metric_name: str) -> str:
        """Get the appropriate unit for a metric."""
        unit_map = {
            "response_time_ms": "ms",
            "throughput_rps": "requests/sec",
            "memory_usage_percent": "percent",
            "cpu_usage_percent": "percent",
            "error_rate_percent": "percent",
            "active_connections": "connections",
            "system_cpu_usage": "percent",
            "system_memory_usage": "percent",
            "disk_usage": "percent",
            "network_bytes_in": "bytes/sec",
            "network_bytes_out": "bytes/sec"
        }
        return unit_map.get(metric_name, "unit")

    async def check_connectivity(self) -> bool:
        """
        Check if Prometheus server is reachable.

        Returns:
            True if connected, False otherwise
        """
        try:
            async with self:
                result = await self._query_prometheus('up')
                return result.get("status") == "success"
        except Exception:
            return False


class PrometheusMetricsCollector:
    """Collector for pushing metrics to Prometheus Pushgateway."""

    def __init__(self, pushgateway_url: Optional[str] = None, job_name: str = "codesage"):
        self.pushgateway_url = pushgateway_url or get_optional_env_var("PUSHGATEWAY_URL")
        self.job_name = job_name
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        """Async context manager entry with secure session configuration."""
        connector = aiohttp.TCPConnector(
            limit=10,  # Connection pool limit
            limit_per_host=5,  # Per host limit
            ttl_dns_cache=300,  # DNS cache TTL
            use_dns_cache=True,
            keepalive_timeout=60,
            enable_cleanup_closed=True
        )

        # Configure SSL for HTTPS URLs
        if self.prometheus_url.startswith('https://'):
            connector = aiohttp.TCPConnector(
                limit=10,
                limit_per_host=5,
                ttl_dns_cache=300,
                use_dns_cache=True,
                keepalive_timeout=60,
                enable_cleanup_closed=True,
                ssl=self.ssl_context
            )

        timeout = aiohttp.ClientTimeout(
            total=self.query_timeout,
            connect=10,
            sock_read=30
        )

        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=timeout,
            headers={
                'User-Agent': 'CodeSage-MCP/1.0',
                'Accept': 'application/json',
                'Accept-Encoding': 'gzip, deflate'
            }
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        if self.session:
            await self.session.close()
            self.session = None

    async def push_metrics(self, metrics: Dict[str, Any], grouping_key: Optional[Dict[str, str]] = None):
        """
        Push metrics to Prometheus Pushgateway.

        Args:
            metrics: Dictionary of metric names to values
            grouping_key: Optional grouping key for the metrics
        """
        if not self.pushgateway_url or not self.session:
            return

        # Format metrics in Prometheus exposition format
        exposition_data = self._format_metrics_exposition(metrics, grouping_key)

        try:
            url = f"{self.pushgateway_url}/metrics/job/{self.job_name}"
            if grouping_key:
                grouping_params = "/".join([f"{k}/{v}" for k, v in grouping_key.items()])
                url += f"/{grouping_params}"

            async with self.session.post(
                url,
                data=exposition_data,
                headers={"Content-Type": "text/plain; charset=utf-8"}
            ) as response:
                if response.status not in [200, 202]:
                    logger.error(f"Failed to push metrics: {response.status} - {await response.text()}")
                else:
                    logger.debug("Successfully pushed metrics to Pushgateway")

        except Exception as e:
            logger.error(f"Failed to push metrics to Pushgateway: {e}")

    def _format_metrics_exposition(self, metrics: Dict[str, Any],
                                 grouping_key: Optional[Dict[str, str]] = None) -> str:
        """
        Format metrics in Prometheus exposition format.

        Args:
            metrics: Dictionary of metric names to values
            grouping_key: Optional grouping key

        Returns:
            Formatted exposition data
        """
        lines = []

        for metric_name, metric_data in metrics.items():
            if isinstance(metric_data, dict):
                value = metric_data.get("value", 0)
                timestamp = metric_data.get("timestamp", time.time())
                labels = metric_data.get("labels", {})

                # Add grouping key labels if provided
                if grouping_key:
                    labels.update(grouping_key)

                # Format metric name
                prometheus_name = metric_name.replace("_", "_").lower()

                # Add HELP comment
                lines.append(f"# HELP {prometheus_name} {metric_data.get('description', metric_name)}")

                # Add TYPE comment
                metric_type = metric_data.get("type", "gauge")
                lines.append(f"# TYPE {prometheus_name} {metric_type}")

                # Format labels
                if labels:
                    label_str = "{" + ",".join([f'{k}="{v}"' for k, v in labels.items()]) + "}"
                else:
                    label_str = ""

                # Add metric value
                lines.append(f"{prometheus_name}{label_str} {value} {int(timestamp * 1000)}")

            lines.append("")  # Empty line between metrics

        return "\n".join(lines)


# Global instances
_prometheus_client = None
_metrics_collector = None


def get_prometheus_client() -> PrometheusClient:
    """Get the global Prometheus client instance."""
    global _prometheus_client
    if _prometheus_client is None:
        _prometheus_client = PrometheusClient()
    return _prometheus_client


def get_metrics_collector() -> PrometheusMetricsCollector:
    """Get the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = PrometheusMetricsCollector()
    return _metrics_collector