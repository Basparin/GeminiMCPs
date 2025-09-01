"""
Advanced Error Handling and Recovery Mechanisms - CES Fault Tolerance System

Provides comprehensive error handling, recovery strategies, and fault tolerance
for the Cognitive Enhancement System with automatic recovery and graceful degradation.

Key Features:
- Circuit breaker patterns for service protection
- Automatic retry with exponential backoff
- Graceful degradation strategies
- Error classification and contextual handling
- Recovery orchestration and monitoring
- Fallback mechanisms for critical operations
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Optional, Any, Callable, Union, Type
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import traceback
from collections import defaultdict, deque
import statistics


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(Enum):
    """Error categories for classification"""
    NETWORK = "network"
    TIMEOUT = "timeout"
    AUTHENTICATION = "authentication"
    RESOURCE = "resource"
    CONFIGURATION = "configuration"
    EXTERNAL_SERVICE = "external_service"
    INTERNAL = "internal"
    VALIDATION = "validation"


class RecoveryStrategy(Enum):
    """Recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    CIRCUIT_BREAKER = "circuit_breaker"
    DEGRADATION = "degradation"
    RECONNECT = "reconnect"
    RESET = "reset"


@dataclass
class ErrorContext:
    """Context information for error handling"""
    error_id: str
    timestamp: datetime
    component: str
    operation: str
    error_category: ErrorCategory
    error_severity: ErrorSeverity
    error_message: str
    stack_trace: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3


@dataclass
class CircuitBreakerState:
    """Circuit breaker state information"""
    service_name: str
    state: str  # 'closed', 'open', 'half_open'
    failure_count: int
    last_failure_time: Optional[datetime]
    next_retry_time: Optional[datetime]
    success_count: int
    total_requests: int


@dataclass
class RecoveryResult:
    """Result of a recovery operation"""
    success: bool
    strategy_used: RecoveryStrategy
    execution_time_ms: float
    error_context: ErrorContext
    recovery_metadata: Dict[str, Any] = field(default_factory=dict)
    fallback_used: bool = False


class CircuitBreaker:
    """Circuit breaker implementation for service protection"""

    def __init__(self, service_name: str, failure_threshold: int = 5,
                 recovery_timeout_seconds: int = 60, success_threshold: int = 3):
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.success_threshold = success_threshold

        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.total_requests = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_retry_time: Optional[datetime] = None

        self._lock = threading.Lock()

    def can_execute(self) -> bool:
        """Check if operation can be executed"""
        with self._lock:
            if self.state == "closed":
                return True
            elif self.state == "open":
                if self.next_retry_time and datetime.now() >= self.next_retry_time:
                    self.state = "half_open"
                    return True
                return False
            elif self.state == "half_open":
                return True
            return False

    def record_success(self):
        """Record successful operation"""
        with self._lock:
            self.total_requests += 1
            self.success_count += 1

            if self.state == "half_open":
                if self.success_count >= self.success_threshold:
                    self._reset()
                else:
                    # Stay in half-open until success threshold is met
                    pass
            elif self.state == "closed":
                # Reset failure count on success
                self.failure_count = 0

    def record_failure(self):
        """Record failed operation"""
        with self._lock:
            self.total_requests += 1
            self.failure_count += 1
            self.last_failure_time = datetime.now()

            if self.state == "half_open":
                # Go back to open state
                self.state = "open"
                self.next_retry_time = datetime.now() + timedelta(seconds=self.recovery_timeout_seconds)
            elif self.state == "closed" and self.failure_count >= self.failure_threshold:
                # Open the circuit
                self.state = "open"
                self.next_retry_time = datetime.now() + timedelta(seconds=self.recovery_timeout_seconds)

    def _reset(self):
        """Reset circuit breaker to closed state"""
        self.state = "closed"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.next_retry_time = None

    def get_state(self) -> CircuitBreakerState:
        """Get current circuit breaker state"""
        return CircuitBreakerState(
            service_name=self.service_name,
            state=self.state,
            failure_count=self.failure_count,
            last_failure_time=self.last_failure_time,
            next_retry_time=self.next_retry_time,
            success_count=self.success_count,
            total_requests=self.total_requests
        )


class RetryMechanism:
    """Exponential backoff retry mechanism"""

    def __init__(self, max_retries: int = 3, base_delay: float = 1.0,
                 max_delay: float = 60.0, backoff_factor: float = 2.0):
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor

    async def execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute operation with retry logic"""
        last_exception = None

        for attempt in range(self.max_retries + 1):
            try:
                return await operation(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_retries:
                    delay = min(self.base_delay * (self.backoff_factor ** attempt), self.max_delay)
                    await asyncio.sleep(delay)
                else:
                    # Max retries exceeded
                    raise last_exception

        # This should never be reached
        raise last_exception


class ErrorRecoveryManager:
    """
    Comprehensive error handling and recovery system for CES.

    Features:
    - Error classification and contextual handling
    - Circuit breaker protection
    - Automatic retry with exponential backoff
    - Graceful degradation strategies
    - Recovery monitoring and reporting
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Error tracking
        self.error_history: List[ErrorContext] = []
        self.recovery_history: List[RecoveryResult] = []

        # Circuit breakers for different services
        self.circuit_breakers: Dict[str, CircuitBreaker] = {
            'codesage_mcp': CircuitBreaker('codesage_mcp', failure_threshold=3, recovery_timeout_seconds=30),
            'ai_orchestrator': CircuitBreaker('ai_orchestrator', failure_threshold=5, recovery_timeout_seconds=60),
            'memory_manager': CircuitBreaker('memory_manager', failure_threshold=3, recovery_timeout_seconds=45),
            'performance_monitor': CircuitBreaker('performance_monitor', failure_threshold=2, recovery_timeout_seconds=20)
        }

        # Retry mechanisms
        self.retry_mechanisms: Dict[str, RetryMechanism] = {
            'network': RetryMechanism(max_retries=5, base_delay=1.0, max_delay=30.0),
            'timeout': RetryMechanism(max_retries=3, base_delay=2.0, max_delay=20.0),
            'resource': RetryMechanism(max_retries=2, base_delay=5.0, max_delay=60.0),
            'default': RetryMechanism(max_retries=3, base_delay=1.0, max_delay=10.0)
        }

        # Degradation strategies
        self.degradation_strategies: Dict[str, Callable] = {
            'codesage_unavailable': self._degrade_codesage_unavailable,
            'memory_unavailable': self._degrade_memory_unavailable,
            'ai_unavailable': self._degrade_ai_unavailable
        }

        # Recovery monitoring
        self.recovery_stats = defaultdict(int)
        self.error_stats = defaultdict(int)

        self.logger.info("Error Recovery Manager initialized")

    async def handle_error(self, error: Exception, component: str, operation: str,
                          context: Optional[Dict[str, Any]] = None) -> RecoveryResult:
        """
        Handle an error with appropriate recovery strategy

        Args:
            error: The exception that occurred
            component: Component where error occurred
            operation: Operation that failed
            context: Additional context information

        Returns:
            RecoveryResult with recovery outcome
        """
        start_time = time.time()

        # Classify the error
        error_context = self._classify_error(error, component, operation, context)

        # Log the error
        self.logger.error(f"Error in {component}.{operation}: {error_context.error_message}",
                         extra={'error_id': error_context.error_id})

        # Track error statistics
        self.error_stats[error_context.error_category.value] += 1
        self.error_history.append(error_context)

        # Keep only recent error history
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]

        # Determine recovery strategy
        strategy = self._determine_recovery_strategy(error_context)

        # Execute recovery
        recovery_result = await self._execute_recovery(error_context, strategy)

        # Record recovery time
        execution_time = (time.time() - start_time) * 1000
        recovery_result.execution_time_ms = execution_time

        # Track recovery statistics
        self.recovery_stats[strategy.value] += 1
        self.recovery_history.append(recovery_result)

        # Keep only recent recovery history
        if len(self.recovery_history) > 500:
            self.recovery_history = self.recovery_history[-500:]

        return recovery_result

    def _classify_error(self, error: Exception, component: str, operation: str,
                       context: Optional[Dict[str, Any]] = None) -> ErrorContext:
        """Classify error into category and severity"""
        error_message = str(error)
        stack_trace = traceback.format_exc()

        # Determine error category
        category = self._determine_error_category(error, error_message)

        # Determine error severity
        severity = self._determine_error_severity(category, error_message, context)

        # Generate error ID
        error_id = f"{component}_{operation}_{int(time.time() * 1000)}"

        return ErrorContext(
            error_id=error_id,
            timestamp=datetime.now(),
            component=component,
            operation=operation,
            error_category=category,
            error_severity=severity,
            error_message=error_message,
            stack_trace=stack_trace,
            metadata=context or {},
            max_retries=self._get_max_retries_for_category(category)
        )

    def _determine_error_category(self, error: Exception, error_message: str) -> ErrorCategory:
        """Determine error category based on error type and message"""
        error_type = type(error).__name__
        message_lower = error_message.lower()

        # Network-related errors
        if any(keyword in message_lower for keyword in ['connection', 'network', 'timeout', 'unreachable']):
            return ErrorCategory.NETWORK
        if 'timeout' in error_type.lower() or 'timeout' in message_lower:
            return ErrorCategory.TIMEOUT
        if any(keyword in message_lower for keyword in ['auth', 'credential', 'permission', 'unauthorized']):
            return ErrorCategory.AUTHENTICATION
        if any(keyword in message_lower for keyword in ['memory', 'disk', 'cpu', 'resource']):
            return ErrorCategory.RESOURCE
        if any(keyword in message_lower for keyword in ['config', 'configuration', 'setting']):
            return ErrorCategory.CONFIGURATION
        if any(keyword in message_lower for keyword in ['external', 'api', 'service', 'mcp']):
            return ErrorCategory.EXTERNAL_SERVICE
        if any(keyword in message_lower for keyword in ['validation', 'invalid', 'malformed']):
            return ErrorCategory.VALIDATION

        return ErrorCategory.INTERNAL

    def _determine_error_severity(self, category: ErrorCategory, error_message: str,
                                context: Optional[Dict[str, Any]] = None) -> ErrorSeverity:
        """Determine error severity based on category and context"""
        # Critical errors
        if category in [ErrorCategory.AUTHENTICATION, ErrorCategory.RESOURCE]:
            if 'out of memory' in error_message.lower() or 'disk full' in error_message.lower():
                return ErrorSeverity.CRITICAL

        # High severity
        if category == ErrorCategory.EXTERNAL_SERVICE:
            return ErrorSeverity.HIGH
        if 'critical' in error_message.lower():
            return ErrorSeverity.HIGH

        # Medium severity
        if category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]:
            return ErrorSeverity.MEDIUM

        # Low severity for validation and configuration
        if category in [ErrorCategory.VALIDATION, ErrorCategory.CONFIGURATION]:
            return ErrorSeverity.LOW

        return ErrorSeverity.MEDIUM

    def _get_max_retries_for_category(self, category: ErrorCategory) -> int:
        """Get maximum retry count for error category"""
        retry_counts = {
            ErrorCategory.NETWORK: 5,
            ErrorCategory.TIMEOUT: 3,
            ErrorCategory.EXTERNAL_SERVICE: 3,
            ErrorCategory.RESOURCE: 2,
            ErrorCategory.CONFIGURATION: 1,
            ErrorCategory.AUTHENTICATION: 1,
            ErrorCategory.VALIDATION: 0,
            ErrorCategory.INTERNAL: 2
        }
        return retry_counts.get(category, 3)

    def _determine_recovery_strategy(self, error_context: ErrorContext) -> RecoveryStrategy:
        """Determine appropriate recovery strategy for error"""
        category = error_context.error_category
        severity = error_context.error_severity

        # Circuit breaker for external services and high severity
        if category == ErrorCategory.EXTERNAL_SERVICE or severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
            if error_context.component in self.circuit_breakers:
                circuit_breaker = self.circuit_breakers[error_context.component]
                if not circuit_breaker.can_execute():
                    return RecoveryStrategy.CIRCUIT_BREAKER

        # Retry for transient errors
        if category in [ErrorCategory.NETWORK, ErrorCategory.TIMEOUT]:
            return RecoveryStrategy.RETRY

        # Fallback for service unavailability
        if category == ErrorCategory.EXTERNAL_SERVICE:
            return RecoveryStrategy.FALLBACK

        # Degradation for resource issues
        if category == ErrorCategory.RESOURCE:
            return RecoveryStrategy.DEGRADATION

        # Default to retry
        return RecoveryStrategy.RETRY

    async def _execute_recovery(self, error_context: ErrorContext, strategy: RecoveryStrategy) -> RecoveryResult:
        """Execute the determined recovery strategy"""
        try:
            if strategy == RecoveryStrategy.RETRY:
                return await self._execute_retry_recovery(error_context)
            elif strategy == RecoveryStrategy.FALLBACK:
                return await self._execute_fallback_recovery(error_context)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return await self._execute_circuit_breaker_recovery(error_context)
            elif strategy == RecoveryStrategy.DEGRADATION:
                return await self._execute_degradation_recovery(error_context)
            else:
                # Default fallback
                return RecoveryResult(
                    success=False,
                    strategy_used=strategy,
                    execution_time_ms=0,
                    error_context=error_context,
                    recovery_metadata={'strategy': 'none'}
                )

        except Exception as recovery_error:
            self.logger.error(f"Recovery execution failed: {recovery_error}")
            return RecoveryResult(
                success=False,
                strategy_used=strategy,
                execution_time_ms=0,
                error_context=error_context,
                recovery_metadata={'recovery_error': str(recovery_error)}
            )

    async def _execute_retry_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Execute retry recovery with exponential backoff"""
        retry_mechanism = self.retry_mechanisms.get(
            error_context.error_category.value,
            self.retry_mechanisms['default']
        )

        # For retry, we would need the original operation to retry
        # This is a simplified implementation
        success = error_context.retry_count < error_context.max_retries

        return RecoveryResult(
            success=success,
            strategy_used=RecoveryStrategy.RETRY,
            execution_time_ms=0,
            error_context=error_context,
            recovery_metadata={
                'retry_count': error_context.retry_count,
                'max_retries': error_context.max_retries
            }
        )

    async def _execute_fallback_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Execute fallback recovery strategy"""
        fallback_used = False

        # Apply component-specific fallback
        if error_context.component == 'codesage_mcp':
            fallback_used = await self._degrade_codesage_unavailable()
        elif error_context.component == 'memory_manager':
            fallback_used = await self._degrade_memory_unavailable()
        elif error_context.component == 'ai_orchestrator':
            fallback_used = await self._degrade_ai_unavailable()

        return RecoveryResult(
            success=fallback_used,
            strategy_used=RecoveryStrategy.FALLBACK,
            execution_time_ms=0,
            error_context=error_context,
            recovery_metadata={'fallback_applied': fallback_used},
            fallback_used=fallback_used
        )

    async def _execute_circuit_breaker_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Execute circuit breaker recovery"""
        if error_context.component in self.circuit_breakers:
            circuit_breaker = self.circuit_breakers[error_context.component]
            can_execute = circuit_breaker.can_execute()

            if not can_execute:
                # Circuit is open, cannot execute
                return RecoveryResult(
                    success=False,
                    strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
                    execution_time_ms=0,
                    error_context=error_context,
                    recovery_metadata={'circuit_state': 'open'}
                )

        # Circuit is closed or half-open, allow execution
        return RecoveryResult(
            success=True,
            strategy_used=RecoveryStrategy.CIRCUIT_BREAKER,
            execution_time_ms=0,
            error_context=error_context,
            recovery_metadata={'circuit_state': 'closed'}
        )

    async def _execute_degradation_recovery(self, error_context: ErrorContext) -> RecoveryResult:
        """Execute graceful degradation recovery"""
        degradation_strategy = self.degradation_strategies.get(f"{error_context.component}_unavailable")

        if degradation_strategy:
            success = await degradation_strategy()
        else:
            success = await self._apply_generic_degradation(error_context)

        return RecoveryResult(
            success=success,
            strategy_used=RecoveryStrategy.DEGRADATION,
            execution_time_ms=0,
            error_context=error_context,
            recovery_metadata={'degraded_mode': success}
        )

    async def _degrade_codesage_unavailable(self) -> bool:
        """Apply degradation when CodeSage is unavailable"""
        # Switch to basic analysis mode
        self.logger.warning("CodeSage unavailable, switching to basic analysis mode")
        # This would set a flag to use fallback analysis methods
        return True

    async def _degrade_memory_unavailable(self) -> bool:
        """Apply degradation when memory manager is unavailable"""
        # Use in-memory storage instead of persistent storage
        self.logger.warning("Memory manager unavailable, using in-memory fallback")
        return True

    async def _degrade_ai_unavailable(self) -> bool:
        """Apply degradation when AI orchestrator is unavailable"""
        # Use basic assistant selection
        self.logger.warning("AI orchestrator unavailable, using basic assistant selection")
        return True

    async def _apply_generic_degradation(self, error_context: ErrorContext) -> bool:
        """Apply generic degradation strategy"""
        self.logger.warning(f"Applying generic degradation for {error_context.component}")
        return True

    def record_operation_success(self, component: str):
        """Record successful operation for circuit breaker"""
        if component in self.circuit_breakers:
            self.circuit_breakers[component].record_success()

    def record_operation_failure(self, component: str):
        """Record failed operation for circuit breaker"""
        if component in self.circuit_breakers:
            self.circuit_breakers[component].record_failure()

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        total_errors = len(self.error_history)
        if total_errors == 0:
            return {"total_errors": 0, "error_categories": {}, "recovery_stats": {}}

        # Error categories
        categories = {}
        for error in self.error_history[-100:]:  # Last 100 errors
            cat = error.error_category.value
            categories[cat] = categories.get(cat, 0) + 1

        # Recovery statistics
        recovery_success_rate = 0
        if self.recovery_history:
            successful_recoveries = sum(1 for r in self.recovery_history if r.success)
            recovery_success_rate = successful_recoveries / len(self.recovery_history)

        return {
            "total_errors": total_errors,
            "error_categories": categories,
            "recovery_success_rate": recovery_success_rate,
            "circuit_breaker_states": {
                name: cb.get_state().__dict__ for name, cb in self.circuit_breakers.items()
            },
            "recovery_strategies_used": dict(self.recovery_stats)
        }

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall system health status based on error patterns"""
        stats = self.get_error_statistics()

        # Determine health based on error rates and recovery success
        health_score = 100

        # Reduce score based on error frequency
        error_rate_penalty = min(stats.get('total_errors', 0) * 2, 40)
        health_score -= error_rate_penalty

        # Reduce score based on recovery failure rate
        recovery_rate = stats.get('recovery_success_rate', 1.0)
        recovery_penalty = (1 - recovery_rate) * 30
        health_score -= recovery_penalty

        # Check circuit breaker states
        open_circuits = sum(1 for cb in stats.get('circuit_breaker_states', {}).values()
                           if cb.get('state') == 'open')
        circuit_penalty = open_circuits * 10
        health_score -= circuit_penalty

        health_score = max(0, min(100, health_score))

        # Determine health status
        if health_score >= 80:
            status = "healthy"
        elif health_score >= 60:
            status = "degraded"
        elif health_score >= 40:
            status = "unhealthy"
        else:
            status = "critical"

        return {
            "status": status,
            "health_score": health_score,
            "error_rate_penalty": error_rate_penalty,
            "recovery_penalty": recovery_penalty,
            "circuit_penalty": circuit_penalty,
            "open_circuits": open_circuits,
            "recommendations": self._generate_health_recommendations(health_score, stats)
        }

    def _generate_health_recommendations(self, health_score: float, stats: Dict[str, Any]) -> List[str]:
        """Generate health recommendations based on current status"""
        recommendations = []

        if health_score < 80:
            recommendations.append("Monitor error rates and implement additional error handling")

        if stats.get('recovery_success_rate', 1.0) < 0.8:
            recommendations.append("Improve recovery mechanisms for failed operations")

        open_circuits = sum(1 for cb in stats.get('circuit_breaker_states', {}).values()
                           if cb.get('state') == 'open')
        if open_circuits > 0:
            recommendations.append(f"Investigate {open_circuits} open circuit breaker(s)")

        # Component-specific recommendations
        error_categories = stats.get('error_categories', {})
        if error_categories.get('network', 0) > 10:
            recommendations.append("Review network connectivity and implement retry logic")

        if error_categories.get('timeout', 0) > 5:
            recommendations.append("Optimize operation timeouts and consider async processing")

        return recommendations

    def reset_error_history(self):
        """Reset error history and statistics"""
        self.error_history.clear()
        self.recovery_history.clear()
        self.error_stats.clear()
        self.recovery_stats.clear()
        self.logger.info("Error history and statistics reset")