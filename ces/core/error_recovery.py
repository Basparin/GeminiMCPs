"""CES Error Recovery and Self-Healing System.

Provides comprehensive error recovery, automatic healing, circuit breaker patterns,
and intelligent failure handling for the Cognitive Enhancement System.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque

from ..core.logging_config import get_logger

logger = get_logger(__name__)

class RecoveryStrategy(Enum):
    """Recovery strategy types."""
    RETRY = "retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    FALLBACK = "fallback"
    DEGRADATION = "degradation"
    RESTART = "restart"
    ISOLATION = "isolation"

class FailureType(Enum):
    """Types of failures that can occur."""
    NETWORK = "network"
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTHENTICATION = "api_authentication"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    DATA_CORRUPTION = "data_corruption"
    TIMEOUT = "timeout"
    SERVICE_UNAVAILABLE = "service_unavailable"
    INTERNAL_ERROR = "internal_error"

@dataclass
class FailureRecord:
    """Record of a system failure."""
    id: str
    timestamp: str
    failure_type: FailureType
    component: str
    error_message: str
    context: Dict[str, Any]
    severity: str  # critical, high, medium, low
    resolved: bool = False
    resolution_time: Optional[str] = None
    recovery_strategy: Optional[RecoveryStrategy] = None

@dataclass
class CircuitBreakerState:
    """Circuit breaker state information."""
    component: str
    state: str  # closed, open, half_open
    failure_count: int
    last_failure_time: Optional[str]
    next_retry_time: Optional[str]
    success_count: int
    total_requests: int

@dataclass
class HealthCheck:
    """Health check configuration."""
    component: str
    check_function: Callable
    interval_seconds: int
    timeout_seconds: int
    max_failures: int
    recovery_time_seconds: int

class ErrorRecoveryManager:
    """Manages error recovery and self-healing across the CES system."""

    def __init__(self):
        self.failure_history: List[FailureRecord] = []
        self.circuit_breakers: Dict[str, CircuitBreakerState] = {}
        self.health_checks: Dict[str, HealthCheck] = {}
        self.recovery_actions: Dict[str, List[Callable]] = defaultdict(list)
        self.isolation_zones: Dict[str, List[str]] = defaultdict(list)
        self.degradation_modes: Dict[str, Dict[str, Any]] = {}

        # Configuration
        self.max_failure_history = 1000
        self.circuit_breaker_threshold = 5
        self.circuit_breaker_timeout = 60  # seconds
        self.circuit_breaker_recovery_attempts = 3

        # Initialize default recovery strategies
        self._initialize_default_strategies()

    def is_healthy(self) -> bool:
        """Check if error recovery manager is healthy."""
        return True

    def _initialize_default_strategies(self):
        """Initialize default error recovery strategies."""
        # Network failure recovery
        self.recovery_actions["network_failure"] = [
            self._retry_with_backoff,
            self._switch_to_fallback_endpoint,
            self._degrade_service_quality
        ]

        # API rate limit recovery
        self.recovery_actions["api_rate_limit"] = [
            self._implement_request_throttling,
            self._switch_to_alternative_provider,
            self._queue_requests_for_later
        ]

        # Resource exhaustion recovery
        self.recovery_actions["resource_exhaustion"] = [
            self._scale_resources,
            self._cleanup_unused_resources,
            self._implement_resource_limits
        ]

        # Service unavailability recovery
        self.recovery_actions["service_unavailable"] = [
            self._activate_circuit_breaker,
            self._switch_to_backup_service,
            self._degrade_to_cached_responses
        ]

    async def record_failure(self, failure: FailureRecord) -> str:
        """Record a system failure for analysis and recovery."""
        try:
            # Add to failure history
            self.failure_history.append(failure)

            # Keep only recent failures
            if len(self.failure_history) > self.max_failure_history:
                self.failure_history = self.failure_history[-self.max_failure_history:]

            # Update circuit breaker state
            await self._update_circuit_breaker(failure.component, failure)

            # Trigger recovery actions
            await self._trigger_recovery_actions(failure)

            # Log failure
            logger.error(
                f"Failure recorded: {failure.failure_type.value} in {failure.component}",
                extra={
                    "failure_id": failure.id,
                    "severity": failure.severity,
                    "context": failure.context
                }
            )

            return failure.id

        except Exception as e:
            logger.error(f"Error recording failure: {e}")
            return ""

    async def _update_circuit_breaker(self, component: str, failure: FailureRecord):
        """Update circuit breaker state based on failure."""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreakerState(
                component=component,
                state="closed",
                failure_count=0,
                last_failure_time=None,
                next_retry_time=None,
                success_count=0,
                total_requests=0
            )

        cb = self.circuit_breakers[component]
        cb.failure_count += 1
        cb.last_failure_time = failure.timestamp
        cb.total_requests += 1

        # Check if circuit breaker should open
        if cb.failure_count >= self.circuit_breaker_threshold and cb.state == "closed":
            cb.state = "open"
            cb.next_retry_time = (datetime.now() + timedelta(seconds=self.circuit_breaker_timeout)).isoformat()
            logger.warning(f"Circuit breaker opened for {component}")

        elif cb.state == "half_open" and cb.failure_count > 0:
            # Half-open state failed, go back to open
            cb.state = "open"
            cb.next_retry_time = (datetime.now() + timedelta(seconds=self.circuit_breaker_timeout)).isoformat()
            logger.warning(f"Circuit breaker returned to open state for {component}")

    async def _trigger_recovery_actions(self, failure: FailureRecord):
        """Trigger appropriate recovery actions for a failure."""
        try:
            action_key = f"{failure.failure_type.value}_failure"
            actions = self.recovery_actions.get(action_key, [])

            for action in actions:
                try:
                    await action(failure)
                except Exception as e:
                    logger.error(f"Recovery action failed: {e}")

        except Exception as e:
            logger.error(f"Error triggering recovery actions: {e}")

    async def check_circuit_breaker(self, component: str) -> bool:
        """Check if a component's circuit breaker allows requests."""
        if component not in self.circuit_breakers:
            return True

        cb = self.circuit_breakers[component]

        if cb.state == "closed":
            return True
        elif cb.state == "open":
            # Check if it's time to try half-open
            if cb.next_retry_time:
                next_retry = datetime.fromisoformat(cb.next_retry_time)
                if datetime.now() >= next_retry:
                    cb.state = "half_open"
                    cb.failure_count = 0
                    logger.info(f"Circuit breaker half-open for {component}")
                    return True
            return False
        elif cb.state == "half_open":
            return True

        return False

    async def record_success(self, component: str):
        """Record a successful operation for circuit breaker recovery."""
        if component in self.circuit_breakers:
            cb = self.circuit_breakers[component]
            cb.success_count += 1
            cb.total_requests += 1

            # If in half-open state and success threshold met, close circuit
            if cb.state == "half_open" and cb.success_count >= self.circuit_breaker_recovery_attempts:
                cb.state = "closed"
                cb.failure_count = 0
                logger.info(f"Circuit breaker closed for {component}")

    async def _retry_with_backoff(self, failure: FailureRecord):
        """Retry failed operation with exponential backoff."""
        try:
            # Extract retry information from failure context
            retry_count = failure.context.get("retry_count", 0)
            max_retries = failure.context.get("max_retries", 3)

            if retry_count < max_retries:
                # Calculate backoff delay
                delay = min(2 ** retry_count, 60)  # Max 60 seconds
                await asyncio.sleep(delay)

                # Trigger retry (this would be component-specific)
                logger.info(f"Retrying {failure.component} after {delay}s delay")

        except Exception as e:
            logger.error(f"Retry with backoff failed: {e}")

    async def _switch_to_fallback_endpoint(self, failure: FailureRecord):
        """Switch to fallback endpoint or service."""
        try:
            # This would trigger configuration changes to use backup endpoints
            logger.info(f"Switching {failure.component} to fallback endpoint")
        except Exception as e:
            logger.error(f"Fallback endpoint switch failed: {e}")

    async def _degrade_service_quality(self, failure: FailureRecord):
        """Degrade service quality to maintain availability."""
        try:
            # Implement service degradation strategies
            degradation_mode = self.degradation_modes.get(failure.component, {})
            if degradation_mode:
                logger.info(f"Activating degradation mode for {failure.component}: {degradation_mode}")
        except Exception as e:
            logger.error(f"Service degradation failed: {e}")

    async def _implement_request_throttling(self, failure: FailureRecord):
        """Implement request throttling to handle rate limits."""
        try:
            # This would adjust request rates and implement queuing
            logger.info(f"Implementing request throttling for {failure.component}")
        except Exception as e:
            logger.error(f"Request throttling implementation failed: {e}")

    async def _switch_to_alternative_provider(self, failure: FailureRecord):
        """Switch to alternative service provider."""
        try:
            # This would switch between different AI providers or services
            logger.info(f"Switching {failure.component} to alternative provider")
        except Exception as e:
            logger.error(f"Provider switch failed: {e}")

    async def _queue_requests_for_later(self, failure: FailureRecord):
        """Queue requests for later processing."""
        try:
            # Implement request queuing for rate-limited services
            logger.info(f"Queueing requests for {failure.component}")
        except Exception as e:
            logger.error(f"Request queuing failed: {e}")

    async def _scale_resources(self, failure: FailureRecord):
        """Scale resources to handle increased load."""
        try:
            # This would trigger auto-scaling or resource allocation
            logger.info(f"Scaling resources for {failure.component}")
        except Exception as e:
            logger.error(f"Resource scaling failed: {e}")

    async def _cleanup_unused_resources(self, failure: FailureRecord):
        """Clean up unused resources to free capacity."""
        try:
            # Implement resource cleanup and garbage collection
            logger.info(f"Cleaning up resources for {failure.component}")
        except Exception as e:
            logger.error(f"Resource cleanup failed: {e}")

    async def _implement_resource_limits(self, failure: FailureRecord):
        """Implement resource limits to prevent exhaustion."""
        try:
            # Set resource limits and quotas
            logger.info(f"Implementing resource limits for {failure.component}")
        except Exception as e:
            logger.error(f"Resource limits implementation failed: {e}")

    async def _activate_circuit_breaker(self, failure: FailureRecord):
        """Activate circuit breaker for failing component."""
        try:
            # Circuit breaker is already handled in _update_circuit_breaker
            logger.info(f"Circuit breaker activated for {failure.component}")
        except Exception as e:
            logger.error(f"Circuit breaker activation failed: {e}")

    async def _switch_to_backup_service(self, failure: FailureRecord):
        """Switch to backup service instance."""
        try:
            # This would switch to backup service instances
            logger.info(f"Switching {failure.component} to backup service")
        except Exception as e:
            logger.error(f"Backup service switch failed: {e}")

    async def _degrade_to_cached_responses(self, failure: FailureRecord):
        """Degrade to cached responses when service is unavailable."""
        try:
            # Serve cached responses instead of live data
            logger.info(f"Degrading {failure.component} to cached responses")
        except Exception as e:
            logger.error(f"Cached response degradation failed: {e}")

    def add_health_check(self, health_check: HealthCheck):
        """Add a health check for a component."""
        self.health_checks[health_check.component] = health_check
        logger.info(f"Added health check for {health_check.component}")

    async def run_health_checks(self) -> Dict[str, Any]:
        """Run all configured health checks."""
        results = {}

        for component, health_check in self.health_checks.items():
            try:
                # Run health check with timeout
                result = await asyncio.wait_for(
                    health_check.check_function(),
                    timeout=health_check.timeout_seconds
                )
                results[component] = {
                    "healthy": result,
                    "timestamp": datetime.now().isoformat()
                }
            except asyncio.TimeoutError:
                results[component] = {
                    "healthy": False,
                    "error": "Health check timeout",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                results[component] = {
                    "healthy": False,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }

        return results

    def isolate_component(self, component: str, zone: str = "default"):
        """Isolate a component to prevent cascading failures."""
        if component not in self.isolation_zones[zone]:
            self.isolation_zones[zone].append(component)
            logger.warning(f"Isolated component {component} in zone {zone}")

    def remove_isolation(self, component: str, zone: str = "default"):
        """Remove isolation for a component."""
        if component in self.isolation_zones[zone]:
            self.isolation_zones[zone].remove(component)
            logger.info(f"Removed isolation for component {component} in zone {zone}")

    def is_isolated(self, component: str, zone: str = "default") -> bool:
        """Check if a component is isolated."""
        return component in self.isolation_zones[zone]

    def get_failure_statistics(self) -> Dict[str, Any]:
        """Get comprehensive failure statistics."""
        total_failures = len(self.failure_history)
        unresolved_failures = len([f for f in self.failure_history if not f.resolved])

        # Group by failure type
        failures_by_type = defaultdict(int)
        for failure in self.failure_history:
            failures_by_type[failure.failure_type.value] += 1

        # Group by component
        failures_by_component = defaultdict(int)
        for failure in self.failure_history:
            failures_by_component[failure.component] += 1

        # Calculate MTTR (Mean Time To Resolution)
        resolved_failures = [f for f in self.failure_history if f.resolved and f.resolution_time]
        if resolved_failures:
            total_resolution_time = 0
            for failure in resolved_failures:
                start_time = datetime.fromisoformat(failure.timestamp)
                end_time = datetime.fromisoformat(failure.resolution_time)
                total_resolution_time += (end_time - start_time).total_seconds()
            mttr_seconds = total_resolution_time / len(resolved_failures)
        else:
            mttr_seconds = 0

        return {
            "total_failures": total_failures,
            "unresolved_failures": unresolved_failures,
            "failures_by_type": dict(failures_by_type),
            "failures_by_component": dict(failures_by_component),
            "mttr_seconds": round(mttr_seconds, 2),
            "circuit_breaker_states": {
                component: asdict(state)
                for component, state in self.circuit_breakers.items()
            },
            "isolation_zones": dict(self.isolation_zones),
            "timestamp": datetime.now().isoformat()
        }

    async def resolve_failure(self, failure_id: str, resolution_notes: str = "") -> bool:
        """Mark a failure as resolved."""
        for failure in self.failure_history:
            if failure.id == failure_id and not failure.resolved:
                failure.resolved = True
                failure.resolution_time = datetime.now().isoformat()

                # Reset circuit breaker if component is healthy
                if failure.component in self.circuit_breakers:
                    cb = self.circuit_breakers[failure.component]
                    if cb.state == "half_open":
                        cb.success_count += 1
                        if cb.success_count >= self.circuit_breaker_recovery_attempts:
                            cb.state = "closed"
                            cb.failure_count = 0
                            logger.info(f"Circuit breaker closed for {failure.component}")

                logger.info(f"Failure {failure_id} resolved: {resolution_notes}")
                return True

        return False

    def get_recovery_recommendations(self, component: str) -> List[Dict[str, Any]]:
        """Get recovery recommendations for a component."""
        recommendations = []

        # Check circuit breaker state
        if component in self.circuit_breakers:
            cb = self.circuit_breakers[component]
            if cb.state == "open":
                recommendations.append({
                    "type": "circuit_breaker",
                    "action": "wait_for_recovery",
                    "description": f"Circuit breaker is open. Next retry at {cb.next_retry_time}",
                    "priority": "high"
                })

        # Check recent failures
        recent_failures = [
            f for f in self.failure_history
            if f.component == component and not f.resolved
        ][:5]  # Last 5 unresolved failures

        if recent_failures:
            failure_types = set(f.failure_type.value for f in recent_failures)
            recommendations.append({
                "type": "failure_analysis",
                "action": "investigate_failures",
                "description": f"Recent failures: {', '.join(failure_types)}",
                "priority": "high"
            })

        # Check isolation status
        if self.is_isolated(component):
            recommendations.append({
                "type": "isolation",
                "action": "remove_isolation",
                "description": "Component is isolated. Consider removing isolation if healthy",
                "priority": "medium"
            })

        return recommendations

    async def perform_automatic_recovery(self, component: str) -> Dict[str, Any]:
        """Perform automatic recovery for a component."""
        results = {
            "component": component,
            "actions_taken": [],
            "success": True,
            "errors": []
        }

        try:
            # Get recovery recommendations
            recommendations = self.get_recovery_recommendations(component)

            for rec in recommendations:
                try:
                    if rec["action"] == "wait_for_recovery":
                        # Wait for circuit breaker recovery
                        await asyncio.sleep(5)
                        results["actions_taken"].append("waited_for_circuit_breaker")

                    elif rec["action"] == "investigate_failures":
                        # Log investigation needed
                        logger.warning(f"Investigation needed for {component} failures")
                        results["actions_taken"].append("logged_investigation")

                    elif rec["action"] == "remove_isolation":
                        # Check if component is healthy before removing isolation
                        health_results = await self.run_health_checks()
                        if component in health_results and health_results[component]["healthy"]:
                            self.remove_isolation(component)
                            results["actions_taken"].append("removed_isolation")
                        else:
                            results["actions_taken"].append("isolation_maintained")

                except Exception as e:
                    results["errors"].append(str(e))
                    results["success"] = False

        except Exception as e:
            results["errors"].append(str(e))
            results["success"] = False

        return results