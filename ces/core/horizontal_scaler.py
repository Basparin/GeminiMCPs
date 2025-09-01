"""
Horizontal Scaler for CES Phase 4.

This module provides horizontal scaling capabilities for supporting 1000+ concurrent users,
including Kubernetes orchestration, distributed memory systems, and intelligent load balancing.
"""

import logging
import time
import threading
import json
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from enum import Enum
import asyncio
import statistics
import psutil
import socket
import uuid

logger = logging.getLogger(__name__)


class ScalingStrategy(Enum):
    """Scaling strategies."""
    CPU_BASED = "cpu_based"
    MEMORY_BASED = "memory_based"
    REQUEST_BASED = "request_based"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"


class NodeStatus(Enum):
    """Node status states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    DRAINING = "draining"


@dataclass
class ClusterNode:
    """Represents a node in the cluster."""
    node_id: str
    host: str
    port: int
    status: NodeStatus = NodeStatus.HEALTHY
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    active_connections: int = 0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ScalingDecision:
    """Scaling decision."""
    decision_id: str
    action: str  # "scale_up", "scale_down", "redistribute"
    target_nodes: int
    reason: str
    confidence: float
    expected_impact: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LoadDistribution:
    """Load distribution across nodes."""
    node_id: str
    current_load: float
    capacity: float
    utilization: float
    recommended_load: float


class KubernetesManager:
    """Manages Kubernetes cluster operations."""

    def __init__(self, namespace: str = "ces", deployment_name: str = "ces-deployment"):
        self.namespace = namespace
        self.deployment_name = deployment_name
        self.k8s_client = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Kubernetes cluster."""
        try:
            from kubernetes import client, config
            config.load_incluster_config()  # Try in-cluster config first
            self.k8s_client = client.AppsV1Api()
            self._connected = True
            logger.info("Connected to Kubernetes cluster")
            return True
        except ImportError:
            logger.warning("Kubernetes client not available")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Kubernetes: {e}")
            return False

    def scale_deployment(self, replicas: int) -> bool:
        """Scale the CES deployment."""
        if not self._connected or not self.k8s_client:
            return False

        try:
            # Get current deployment
            deployment = self.k8s_client.read_namespaced_deployment(
                self.deployment_name, self.namespace
            )

            # Update replicas
            deployment.spec.replicas = replicas

            # Apply the update
            self.k8s_client.patch_namespaced_deployment(
                self.deployment_name, self.namespace, deployment
            )

            logger.info(f"Scaled deployment to {replicas} replicas")
            return True

        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            return False

    def get_current_replicas(self) -> int:
        """Get current number of replicas."""
        if not self._connected or not self.k8s_client:
            return 1

        try:
            deployment = self.k8s_client.read_namespaced_deployment(
                self.deployment_name, self.namespace
            )
            return deployment.spec.replicas or 1
        except Exception as e:
            logger.error(f"Failed to get current replicas: {e}")
            return 1


class RedisClusterManager:
    """Manages Redis Cluster for distributed caching."""

    def __init__(self, startup_nodes: List[Dict[str, Any]] = None):
        self.startup_nodes = startup_nodes or [{"host": "localhost", "port": 6379}]
        self.redis_cluster = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to Redis Cluster."""
        try:
            from rediscluster import RedisCluster
            self.redis_cluster = RedisCluster(
                startup_nodes=self.startup_nodes,
                decode_responses=True
            )
            self.redis_cluster.ping()
            self._connected = True
            logger.info("Connected to Redis Cluster")
            return True
        except ImportError:
            logger.warning("Redis Cluster client not available")
            return False
        except Exception as e:
            logger.error(f"Failed to connect to Redis Cluster: {e}")
            return False

    def get(self, key: str) -> Optional[str]:
        """Get value from Redis Cluster."""
        if not self._connected or not self.redis_cluster:
            return None

        try:
            return self.redis_cluster.get(key)
        except Exception as e:
            logger.error(f"Redis Cluster get error: {e}")
            return None

    def set(self, key: str, value: str, ttl: Optional[int] = None) -> bool:
        """Set value in Redis Cluster."""
        if not self._connected or not self.redis_cluster:
            return False

        try:
            return self.redis_cluster.set(key, value, ex=ttl)
        except Exception as e:
            logger.error(f"Redis Cluster set error: {e}")
            return False

    def get_cluster_info(self) -> Dict[str, Any]:
        """Get Redis Cluster information."""
        if not self._connected or not self.redis_cluster:
            return {}

        try:
            info = self.redis_cluster.cluster_info()
            nodes = self.redis_cluster.cluster_nodes()

            return {
                "cluster_state": info.get("cluster_state", "unknown"),
                "known_nodes": len(nodes) if nodes else 0,
                "connected": True
            }
        except Exception as e:
            logger.error(f"Failed to get cluster info: {e}")
            return {"connected": False, "error": str(e)}


class LoadBalancer:
    """Intelligent load balancer for distributing requests across nodes."""

    def __init__(self):
        self.nodes: Dict[str, ClusterNode] = {}
        self.node_loads: Dict[str, float] = {}
        self.request_queue: asyncio.Queue = asyncio.Queue()
        self._balancing_active = False

    def add_node(self, node: ClusterNode):
        """Add a node to the load balancer."""
        self.nodes[node.node_id] = node
        self.node_loads[node.node_id] = 0.0
        logger.info(f"Added node {node.node_id} to load balancer")

    def remove_node(self, node_id: str):
        """Remove a node from the load balancer."""
        if node_id in self.nodes:
            del self.nodes[node_id]
            del self.node_loads[node_id]
            logger.info(f"Removed node {node_id} from load balancer")

    def get_optimal_node(self, request_data: Dict[str, Any]) -> Optional[str]:
        """Get the optimal node for a request."""
        if not self.nodes:
            return None

        # Simple load balancing - least loaded node
        healthy_nodes = [
            node_id for node_id, node in self.nodes.items()
            if node.status == NodeStatus.HEALTHY
        ]

        if not healthy_nodes:
            return None

        # Find node with lowest load
        optimal_node = min(healthy_nodes, key=lambda x: self.node_loads.get(x, 0))
        return optimal_node

    def update_node_load(self, node_id: str, load: float):
        """Update load for a node."""
        if node_id in self.node_loads:
            self.node_loads[node_id] = load

    def get_load_distribution(self) -> List[LoadDistribution]:
        """Get current load distribution across nodes."""
        distributions = []

        for node_id, node in self.nodes.items():
            current_load = self.node_loads.get(node_id, 0.0)
            capacity = 100.0  # Assume 100% capacity
            utilization = (current_load / capacity) * 100 if capacity > 0 else 0

            distributions.append(LoadDistribution(
                node_id=node_id,
                current_load=current_load,
                capacity=capacity,
                utilization=utilization,
                recommended_load=min(current_load * 1.1, capacity)  # Slight increase
            ))

        return distributions

    async def balance_load(self):
        """Balance load across nodes."""
        while self._balancing_active:
            try:
                # Get current load distribution
                distributions = self.get_load_distribution()

                # Find overloaded and underloaded nodes
                overloaded = [d for d in distributions if d.utilization > 80]
                underloaded = [d for d in distributions if d.utilization < 40]

                # Balance load (simplified)
                for over in overloaded:
                    for under in underloaded:
                        if over.current_load > under.current_load + 20:
                            # Transfer load from over to under
                            transfer_amount = (over.current_load - under.current_load) / 2
                            self.node_loads[over.node_id] -= transfer_amount
                            self.node_loads[under.node_id] += transfer_amount
                            logger.info(f"Balanced load: {transfer_amount} from {over.node_id} to {under.node_id}")

                await asyncio.sleep(30)  # Balance every 30 seconds

            except Exception as e:
                logger.error(f"Load balancing error: {e}")
                await asyncio.sleep(10)

    def start_load_balancing(self):
        """Start load balancing."""
        if self._balancing_active:
            return

        self._balancing_active = True
        asyncio.create_task(self.balance_load())
        logger.info("Load balancing started")

    def stop_load_balancing(self):
        """Stop load balancing."""
        self._balancing_active = False
        logger.info("Load balancing stopped")


class HorizontalScaler:
    """Main horizontal scaling system for CES Phase 4."""

    def __init__(self, min_replicas: int = 1, max_replicas: int = 50,
                 scaling_interval_seconds: int = 60):
        self.min_replicas = min_replicas
        self.max_replicas = max_replicas
        self.scaling_interval_seconds = scaling_interval_seconds

        # Scaling components
        self.kubernetes_manager = KubernetesManager()
        self.redis_cluster = RedisClusterManager()
        self.load_balancer = LoadBalancer()

        # Scaling metrics
        self.scaling_history: List[ScalingDecision] = []
        self.current_replicas = 1
        self.target_replicas = 1

        # Performance metrics
        self.cpu_threshold_high = 70.0
        self.cpu_threshold_low = 30.0
        self.memory_threshold_high = 80.0
        self.memory_threshold_low = 40.0
        self.request_threshold_high = 1000  # requests per minute
        self.request_threshold_low = 200

        # Scaling state
        self.scaling_strategy = ScalingStrategy.HYBRID
        self._scaling_active = False
        self._scaling_thread: Optional[threading.Thread] = None

        # Node management
        self.cluster_nodes: Dict[str, ClusterNode] = {}
        self.node_heartbeats: Dict[str, datetime] = {}

        # Start scaling
        self._start_scaling()

    def _start_scaling(self):
        """Start the horizontal scaling system."""
        if self._scaling_active:
            return

        self._scaling_active = True

        # Connect to infrastructure
        self.kubernetes_manager.connect()
        self.redis_cluster.connect()

        # Start load balancing
        self.load_balancer.start_load_balancing()

        # Start scaling thread
        self._scaling_thread = threading.Thread(
            target=self._scaling_loop,
            daemon=True,
            name="HorizontalScaler"
        )
        self._scaling_thread.start()

        # Register current node
        self._register_current_node()

        logger.info("Horizontal scaling system started")

    def _register_current_node(self):
        """Register the current node in the cluster."""
        node_id = str(uuid.uuid4())
        host = socket.gethostname()
        port = 8000  # Default port

        node = ClusterNode(
            node_id=node_id,
            host=host,
            port=port,
            metadata={"type": "ces_instance", "version": "4.0"}
        )

        self.cluster_nodes[node_id] = node
        self.load_balancer.add_node(node)
        self.node_heartbeats[node_id] = datetime.now()

        logger.info(f"Registered current node: {node_id} at {host}:{port}")

    def _scaling_loop(self):
        """Main scaling loop."""
        while self._scaling_active:
            try:
                # Collect metrics
                metrics = self._collect_scaling_metrics()

                # Make scaling decision
                decision = self._make_scaling_decision(metrics)

                if decision:
                    # Apply scaling decision
                    self._apply_scaling_decision(decision)

                    # Record decision
                    self.scaling_history.append(decision)

                # Update node health
                self._update_node_health()

                time.sleep(self.scaling_interval_seconds)

            except Exception as e:
                logger.error(f"Scaling loop error: {e}")
                time.sleep(10)

    def _collect_scaling_metrics(self) -> Dict[str, Any]:
        """Collect metrics for scaling decisions."""
        metrics = {
            "timestamp": datetime.now(),
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage_percent": psutil.disk_usage('/').percent
            },
            "application": {
                "active_connections": len(self.load_balancer.nodes),
                "total_requests_per_minute": self._get_request_rate(),
                "avg_response_time_ms": self._get_avg_response_time()
            },
            "cluster": {
                "healthy_nodes": len([n for n in self.cluster_nodes.values() if n.status == NodeStatus.HEALTHY]),
                "total_nodes": len(self.cluster_nodes),
                "avg_node_load": statistics.mean(self.load_balancer.node_loads.values()) if self.load_balancer.node_loads else 0
            }
        }

        return metrics

    def _get_request_rate(self) -> float:
        """Get current request rate (requests per minute)."""
        # Placeholder - would integrate with actual request monitoring
        return 500.0  # Mock value

    def _get_avg_response_time(self) -> float:
        """Get average response time in milliseconds."""
        # Placeholder - would integrate with actual response time monitoring
        return 150.0  # Mock value

    def _make_scaling_decision(self, metrics: Dict[str, Any]) -> Optional[ScalingDecision]:
        """Make a scaling decision based on metrics."""
        system = metrics["system"]
        application = metrics["application"]
        cluster = metrics["cluster"]

        decision_id = str(uuid.uuid4())
        current_replicas = self.kubernetes_manager.get_current_replicas()

        # CPU-based scaling
        if system["cpu_percent"] > self.cpu_threshold_high:
            if current_replicas < self.max_replicas:
                return ScalingDecision(
                    decision_id=decision_id,
                    action="scale_up",
                    target_nodes=current_replicas + 1,
                    reason=f"High CPU usage: {system['cpu_percent']:.1f}%",
                    confidence=0.8,
                    expected_impact={
                        "cpu_reduction": 15,
                        "cost_increase": 20
                    }
                )

        # Memory-based scaling
        if system["memory_percent"] > self.memory_threshold_high:
            if current_replicas < self.max_replicas:
                return ScalingDecision(
                    decision_id=decision_id,
                    action="scale_up",
                    target_nodes=current_replicas + 1,
                    reason=f"High memory usage: {system['memory_percent']:.1f}%",
                    confidence=0.9,
                    expected_impact={
                        "memory_reduction": 20,
                        "cost_increase": 25
                    }
                )

        # Request-based scaling
        if application["total_requests_per_minute"] > self.request_threshold_high:
            if current_replicas < self.max_replicas:
                return ScalingDecision(
                    decision_id=decision_id,
                    action="scale_up",
                    target_nodes=current_replicas + 1,
                    reason=f"High request rate: {application['total_requests_per_minute']:.0f} req/min",
                    confidence=0.7,
                    expected_impact={
                        "response_time_improvement": 30,
                        "cost_increase": 15
                    }
                )

        # Scale down conditions
        if (system["cpu_percent"] < self.cpu_threshold_low and
            system["memory_percent"] < self.memory_threshold_low and
            application["total_requests_per_minute"] < self.request_threshold_low):
            if current_replicas > self.min_replicas:
                return ScalingDecision(
                    decision_id=decision_id,
                    action="scale_down",
                    target_nodes=current_replicas - 1,
                    reason="Low resource utilization",
                    confidence=0.6,
                    expected_impact={
                        "cost_savings": 20,
                        "performance_impact": 5
                    }
                )

        return None

    def _apply_scaling_decision(self, decision: ScalingDecision):
        """Apply a scaling decision."""
        try:
            if decision.action in ["scale_up", "scale_down"]:
                success = self.kubernetes_manager.scale_deployment(decision.target_nodes)
                if success:
                    self.current_replicas = decision.target_nodes
                    logger.info(f"Successfully {decision.action} to {decision.target_nodes} replicas")
                else:
                    logger.error(f"Failed to {decision.action}")

        except Exception as e:
            logger.error(f"Error applying scaling decision: {e}")

    def _update_node_health(self):
        """Update health status of cluster nodes."""
        current_time = datetime.now()

        for node_id, node in self.cluster_nodes.items():
            # Check heartbeat
            last_heartbeat = self.node_heartbeats.get(node_id, current_time)
            heartbeat_age = (current_time - last_heartbeat).total_seconds()

            if heartbeat_age > 60:  # No heartbeat for 60 seconds
                if node.status != NodeStatus.UNHEALTHY:
                    node.status = NodeStatus.UNHEALTHY
                    logger.warning(f"Node {node_id} marked as unhealthy (no heartbeat)")
            elif heartbeat_age > 30:  # No heartbeat for 30 seconds
                if node.status != NodeStatus.DEGRADED:
                    node.status = NodeStatus.DEGRADED
                    logger.warning(f"Node {node_id} marked as degraded")
            else:
                if node.status != NodeStatus.HEALTHY:
                    node.status = NodeStatus.HEALTHY
                    logger.info(f"Node {node_id} recovered to healthy status")

    def send_heartbeat(self, node_id: str):
        """Send heartbeat for a node."""
        self.node_heartbeats[node_id] = datetime.now()

    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status."""
        return {
            "current_replicas": self.current_replicas,
            "target_replicas": self.target_replicas,
            "scaling_strategy": self.scaling_strategy.value,
            "cluster_nodes": len(self.cluster_nodes),
            "healthy_nodes": len([n for n in self.cluster_nodes.values() if n.status == NodeStatus.HEALTHY]),
            "kubernetes_connected": self.kubernetes_manager._connected,
            "redis_connected": self.redis_cluster._connected,
            "load_balancer_active": self.load_balancer._balancing_active,
            "recent_decisions": [
                {
                    "action": d.action,
                    "target_nodes": d.target_nodes,
                    "reason": d.reason,
                    "confidence": d.confidence,
                    "timestamp": d.timestamp.isoformat()
                }
                for d in self.scaling_history[-5:]
            ]
        }

    def manual_scale(self, target_replicas: int, reason: str = "Manual scaling") -> bool:
        """Manually scale to target number of replicas."""
        if target_replicas < self.min_replicas or target_replicas > self.max_replicas:
            logger.error(f"Target replicas {target_replicas} out of range [{self.min_replicas}, {self.max_replicas}]")
            return False

        decision = ScalingDecision(
            decision_id=str(uuid.uuid4()),
            action="scale_up" if target_replicas > self.current_replicas else "scale_down",
            target_nodes=target_replicas,
            reason=reason,
            confidence=1.0,
            expected_impact={}
        )

        success = self._apply_scaling_decision(decision)
        if success:
            self.scaling_history.append(decision)

        return success

    def get_cluster_topology(self) -> Dict[str, Any]:
        """Get cluster topology information."""
        return {
            "nodes": [
                {
                    "node_id": node.node_id,
                    "host": node.host,
                    "port": node.port,
                    "status": node.status.value,
                    "cpu_usage": node.cpu_usage,
                    "memory_usage": node.memory_usage,
                    "active_connections": node.active_connections,
                    "last_heartbeat": self.node_heartbeats.get(node.node_id, datetime.now()).isoformat()
                }
                for node in self.cluster_nodes.values()
            ],
            "load_distribution": [
                {
                    "node_id": dist.node_id,
                    "current_load": dist.current_load,
                    "capacity": dist.capacity,
                    "utilization": dist.utilization,
                    "recommended_load": dist.recommended_load
                }
                for dist in self.load_balancer.get_load_distribution()
            ],
            "redis_cluster": self.redis_cluster.get_cluster_info()
        }

    def stop_scaling(self):
        """Stop the horizontal scaling system."""
        self._scaling_active = False
        self.load_balancer.stop_load_balancing()

        if self._scaling_thread:
            self._scaling_thread.join(timeout=5.0)

        logger.info("Horizontal scaling system stopped")


# Global instance
_horizontal_scaler_instance: Optional[HorizontalScaler] = None
_scaler_lock = threading.Lock()


def get_horizontal_scaler(min_replicas: int = 1, max_replicas: int = 50) -> HorizontalScaler:
    """Get the global horizontal scaler instance."""
    global _horizontal_scaler_instance

    if _horizontal_scaler_instance is None:
        with _scaler_lock:
            if _horizontal_scaler_instance is None:
                _horizontal_scaler_instance = HorizontalScaler(
                    min_replicas=min_replicas,
                    max_replicas=max_replicas
                )

    return _horizontal_scaler_instance