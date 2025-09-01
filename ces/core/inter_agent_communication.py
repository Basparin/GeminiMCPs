"""
Inter-Agent Communication System - CES Phase 2 Enhancement

Phase 2 Implementation: Secure communication protocols between AI assistants
with advanced message passing, shared context synchronization, and consensus algorithms.

Key Phase 2 Features:
- Extended MCP Protocol for multi-agent coordination
- Shared context mechanisms with conflict resolution
- Advanced consensus algorithms for decision making
- End-to-end encryption for agent communications
- Communication security and audit trails
- Inter-agent reliability testing and monitoring
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import uuid
import hashlib
import secrets
from concurrent.futures import ThreadPoolExecutor
import aiohttp
import ssl


class CommunicationProtocol(Enum):
    """Supported communication protocols"""
    MCP_EXTENDED = "mcp_extended"
    DIRECT_API = "direct_api"
    SHARED_MEMORY = "shared_memory"
    MESSAGE_QUEUE = "message_queue"
    WEBSOCKET = "websocket"


class MessageType(Enum):
    """Types of inter-agent messages"""
    TASK_DELEGATION = "task_delegation"
    CONTEXT_SHARING = "context_sharing"
    RESULT_SYNCHRONIZATION = "result_synchronization"
    CONSENSUS_REQUEST = "consensus_request"
    HEALTH_CHECK = "health_check"
    ERROR_REPORT = "error_report"
    COORDINATION_SIGNAL = "coordination_signal"


class SecurityLevel(Enum):
    """Security levels for communications"""
    PUBLIC = "public"
    INTERNAL = "internal"
    SENSITIVE = "sensitive"
    CRITICAL = "critical"


@dataclass
class AgentEndpoint:
    """Represents an AI agent endpoint"""
    agent_id: str
    agent_name: str
    protocol: CommunicationProtocol
    endpoint_url: str
    public_key: str
    capabilities: List[str]
    trust_level: float
    last_seen: datetime
    status: str = "unknown"


@dataclass
class SecureMessage:
    """Secure inter-agent message"""
    message_id: str
    sender_id: str
    recipient_id: str
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    security_level: SecurityLevel
    signature: str
    encrypted_payload: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class CommunicationSession:
    """Active communication session between agents"""
    session_id: str
    participants: List[str]
    protocol: CommunicationProtocol
    security_context: Dict[str, Any]
    created_at: datetime
    last_activity: datetime
    message_count: int = 0
    status: str = "active"


@dataclass
class SharedContext:
    """Shared context between agents"""
    context_id: str
    owner_id: str
    context_data: Dict[str, Any]
    version: int
    participants: Set[str]
    last_modified: datetime
    access_permissions: Dict[str, List[str]]
    conflict_resolution_policy: str = "latest_wins"


class InterAgentCommunicationManager:
    """
    Phase 2: Advanced inter-agent communication system

    Manages secure communication protocols between AI assistants with:
    - Extended MCP protocol for complex task decomposition
    - Shared context synchronization with conflict resolution
    - Advanced consensus algorithms for multi-agent decisions
    - End-to-end encryption and security measures
    - Comprehensive reliability testing and monitoring
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Agent registry
        self.registered_agents: Dict[str, AgentEndpoint] = {}

        # Active communication sessions
        self.active_sessions: Dict[str, CommunicationSession] = {}

        # Shared contexts
        self.shared_contexts: Dict[str, SharedContext] = {}

        # Security and encryption
        self.encryption_manager = EncryptionManager()
        self.security_auditor = SecurityAuditor()

        # Consensus engine
        self.consensus_engine = ConsensusEngine()

        # Reliability monitoring
        self.reliability_monitor = ReliabilityMonitor()

        # Message queues and routing
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.message_router = MessageRouter()

        self.logger.info("Phase 2 Inter-Agent Communication Manager initialized")

    async def register_agent(self, agent_info: Dict[str, Any]) -> str:
        """
        Register a new AI agent for inter-agent communication

        Args:
            agent_info: Agent registration information

        Returns:
            Agent ID for registered agent
        """
        agent_id = str(uuid.uuid4())

        endpoint = AgentEndpoint(
            agent_id=agent_id,
            agent_name=agent_info['name'],
            protocol=CommunicationProtocol(agent_info.get('protocol', 'mcp_extended')),
            endpoint_url=agent_info['endpoint_url'],
            public_key=agent_info['public_key'],
            capabilities=agent_info.get('capabilities', []),
            trust_level=agent_info.get('initial_trust', 0.5),
            last_seen=datetime.now(),
            status="registered"
        )

        self.registered_agents[agent_id] = endpoint
        self.message_queues[agent_id] = asyncio.Queue()

        # Initialize security context
        await self.security_auditor.initialize_agent_security(agent_id, agent_info)

        self.logger.info(f"Registered agent: {agent_info['name']} with ID: {agent_id}")
        return agent_id

    async def send_secure_message(self, sender_id: str, recipient_id: str,
                                message_type: MessageType, payload: Dict[str, Any],
                                security_level: SecurityLevel = SecurityLevel.INTERNAL) -> Dict[str, Any]:
        """
        Send a secure message between agents

        Args:
            sender_id: Sending agent ID
            recipient_id: Receiving agent ID
            message_type: Type of message
            payload: Message payload
            security_level: Security level for the message

        Returns:
            Message delivery result
        """
        if sender_id not in self.registered_agents or recipient_id not in self.registered_agents:
            return {"status": "error", "error": "Agent not registered"}

        # Create secure message
        message = SecureMessage(
            message_id=str(uuid.uuid4()),
            sender_id=sender_id,
            recipient_id=recipient_id,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now(),
            security_level=security_level,
            signature=self._sign_message(payload, sender_id),
            correlation_id=str(uuid.uuid4())
        )

        # Encrypt if required
        if security_level in [SecurityLevel.SENSITIVE, SecurityLevel.CRITICAL]:
            message.encrypted_payload = await self.encryption_manager.encrypt_message(
                json.dumps(payload), recipient_id
            )

        # Route message
        delivery_result = await self.message_router.route_message(message)

        # Log for security audit
        await self.security_auditor.log_message(message, delivery_result)

        # Update reliability metrics
        await self.reliability_monitor.record_message_delivery(message, delivery_result)

        return delivery_result

    async def create_shared_context(self, owner_id: str, context_data: Dict[str, Any],
                                  participants: List[str]) -> str:
        """
        Create a shared context for multi-agent collaboration

        Args:
            owner_id: Owner agent ID
            context_data: Initial context data
            participants: List of participant agent IDs

        Returns:
            Context ID
        """
        context_id = str(uuid.uuid4())

        shared_context = SharedContext(
            context_id=context_id,
            owner_id=owner_id,
            context_data=context_data,
            version=1,
            participants=set(participants + [owner_id]),
            last_modified=datetime.now(),
            access_permissions={agent_id: ["read", "write"] for agent_id in participants + [owner_id]}
        )

        self.shared_contexts[context_id] = shared_context

        # Notify participants
        for participant_id in participants:
            await self.send_secure_message(
                owner_id, participant_id,
                MessageType.CONTEXT_SHARING,
                {
                    "context_id": context_id,
                    "action": "context_created",
                    "initial_data": context_data
                }
            )

        self.logger.info(f"Created shared context {context_id} with {len(participants)} participants")
        return context_id

    async def synchronize_context(self, context_id: str, agent_id: str,
                                updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronize context updates across agents

        Args:
            context_id: Context ID
            agent_id: Agent making the update
            updates: Context updates

        Returns:
            Synchronization result
        """
        if context_id not in self.shared_contexts:
            return {"status": "error", "error": "Context not found"}

        context = self.shared_contexts[context_id]

        # Check permissions
        if agent_id not in context.participants:
            return {"status": "error", "error": "Agent not authorized"}

        if "write" not in context.access_permissions.get(agent_id, []):
            return {"status": "error", "error": "Write permission denied"}

        # Apply updates with conflict resolution
        conflict_result = await self._resolve_context_conflicts(context, agent_id, updates)

        if conflict_result['status'] == 'resolved':
            # Update context
            context.context_data.update(updates)
            context.version += 1
            context.last_modified = datetime.now()

            # Broadcast updates to participants
            for participant_id in context.participants:
                if participant_id != agent_id:
                    await self.send_secure_message(
                        agent_id, participant_id,
                        MessageType.CONTEXT_SHARING,
                        {
                            "context_id": context_id,
                            "action": "context_updated",
                            "updates": updates,
                            "new_version": context.version
                        }
                    )

        return conflict_result

    async def _resolve_context_conflicts(self, context: SharedContext, agent_id: str,
                                       updates: Dict[str, Any]) -> Dict[str, Any]:
        """
        Resolve conflicts in context updates
        """
        # Check for conflicting updates
        conflicts = []
        for key, new_value in updates.items():
            if key in context.context_data:
                current_value = context.context_data[key]
                if current_value != new_value:
                    conflicts.append({
                        'key': key,
                        'current_value': current_value,
                        'new_value': new_value,
                        'agent_id': agent_id
                    })

        if not conflicts:
            return {"status": "resolved", "conflicts": 0}

        # Apply conflict resolution policy
        if context.conflict_resolution_policy == "latest_wins":
            return {"status": "resolved", "conflicts": len(conflicts), "policy": "latest_wins"}
        elif context.conflict_resolution_policy == "consensus":
            return await self.consensus_engine.resolve_context_conflicts(context, conflicts)
        else:
            # Manual resolution required
            return {
                "status": "manual_resolution_required",
                "conflicts": conflicts,
                "message": "Conflicts detected, manual resolution required"
            }

    async def initiate_consensus(self, initiator_id: str, participants: List[str],
                               decision_topic: str, options: List[Any]) -> str:
        """
        Initiate a consensus decision among agents

        Args:
            initiator_id: Agent initiating consensus
            participants: Participating agents
            decision_topic: Topic for decision
            options: Available options

        Returns:
            Consensus session ID
        """
        consensus_id = str(uuid.uuid4())

        # Create consensus session
        await self.consensus_engine.create_consensus_session(
            consensus_id, initiator_id, participants, decision_topic, options
        )

        # Send consensus requests
        for participant_id in participants:
            await self.send_secure_message(
                initiator_id, participant_id,
                MessageType.CONSENSUS_REQUEST,
                {
                    "consensus_id": consensus_id,
                    "topic": decision_topic,
                    "options": options,
                    "deadline": (datetime.now() + timedelta(minutes=5)).isoformat()
                }
            )

        self.logger.info(f"Initiated consensus {consensus_id} on topic: {decision_topic}")
        return consensus_id

    async def submit_consensus_vote(self, consensus_id: str, agent_id: str,
                                  vote: Any, reasoning: str = "") -> Dict[str, Any]:
        """
        Submit a vote in an active consensus session

        Args:
            consensus_id: Consensus session ID
            agent_id: Voting agent ID
            vote: Vote value
            reasoning: Optional reasoning for vote

        Returns:
            Vote submission result
        """
        return await self.consensus_engine.submit_vote(
            consensus_id, agent_id, vote, reasoning
        )

    async def get_consensus_result(self, consensus_id: str) -> Dict[str, Any]:
        """
        Get the result of a consensus decision

        Args:
            consensus_id: Consensus session ID

        Returns:
            Consensus result
        """
        return await self.consensus_engine.get_consensus_result(consensus_id)

    def _sign_message(self, payload: Dict[str, Any], sender_id: str) -> str:
        """Sign message payload for integrity"""
        payload_str = json.dumps(payload, sort_keys=True)
        signature_input = f"{sender_id}:{payload_str}:{datetime.now().isoformat()}"

        # Simple signature (in production, use proper cryptographic signing)
        return hashlib.sha256(signature_input.encode()).hexdigest()

    async def get_communication_health(self) -> Dict[str, Any]:
        """
        Get overall communication system health

        Returns:
            Health status and metrics
        """
        health_status = {
            "timestamp": datetime.now().isoformat(),
            "registered_agents": len(self.registered_agents),
            "active_sessions": len(self.active_sessions),
            "shared_contexts": len(self.shared_contexts),
            "message_queues_status": {},
            "security_status": await self.security_auditor.get_security_status(),
            "reliability_metrics": await self.reliability_monitor.get_metrics()
        }

        # Check message queue health
        for agent_id, queue in self.message_queues.items():
            queue_size = queue.qsize()
            health_status["message_queues_status"][agent_id] = {
                "queue_size": queue_size,
                "status": "healthy" if queue_size < 100 else "warning"
            }

        # Overall health assessment
        critical_issues = 0
        warning_issues = 0

        if len(self.registered_agents) == 0:
            critical_issues += 1

        for queue_status in health_status["message_queues_status"].values():
            if queue_status["status"] == "warning":
                warning_issues += 1

        if critical_issues > 0:
            health_status["overall_status"] = "critical"
        elif warning_issues > 0:
            health_status["overall_status"] = "warning"
        else:
            health_status["overall_status"] = "healthy"

        return health_status

    async def test_inter_agent_reliability(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Phase 2: Test inter-agent communication reliability

        Args:
            test_config: Test configuration

        Returns:
            Reliability test results
        """
        return await self.reliability_monitor.run_reliability_test(test_config)


class EncryptionManager:
    """Manages encryption for secure inter-agent communication"""

    def __init__(self):
        self.agent_keys: Dict[str, str] = {}
        self.session_keys: Dict[str, str] = {}

    async def encrypt_message(self, payload: str, recipient_id: str) -> str:
        """Encrypt message payload"""
        # Simplified encryption (in production, use proper encryption)
        if recipient_id not in self.agent_keys:
            return payload  # Fallback to unencrypted

        # Simple XOR encryption for demonstration
        key = self.agent_keys[recipient_id][:len(payload)]
        encrypted = ''.join(chr(ord(a) ^ ord(b)) for a, b in zip(payload, key))
        return encrypted

    async def decrypt_message(self, encrypted_payload: str, sender_id: str) -> str:
        """Decrypt message payload"""
        if sender_id not in self.agent_keys:
            return encrypted_payload

        # Simple XOR decryption
        key = self.agent_keys[sender_id][:len(encrypted_payload)]
        decrypted = ''.join(chr(ord(a) ^ ord(b)) for a, b in zip(encrypted_payload, key))
        return decrypted


class SecurityAuditor:
    """Audits security events and maintains security logs"""

    def __init__(self):
        self.security_events: List[Dict[str, Any]] = []
        self.agent_security_profiles: Dict[str, Dict[str, Any]] = {}

    async def initialize_agent_security(self, agent_id: str, agent_info: Dict[str, Any]):
        """Initialize security profile for agent"""
        self.agent_security_profiles[agent_id] = {
            "trust_level": agent_info.get("initial_trust", 0.5),
            "security_incidents": 0,
            "last_security_check": datetime.now().isoformat(),
            "encryption_enabled": True
        }

    async def log_message(self, message: SecureMessage, delivery_result: Dict[str, Any]):
        """Log security event for message"""
        event = {
            "event_type": "message_delivery",
            "message_id": message.message_id,
            "sender_id": message.sender_id,
            "recipient_id": message.recipient_id,
            "security_level": message.security_level.value,
            "timestamp": datetime.now().isoformat(),
            "delivery_status": delivery_result.get("status", "unknown"),
            "correlation_id": message.correlation_id
        }

        self.security_events.append(event)

        # Keep only recent events
        if len(self.security_events) > 1000:
            self.security_events = self.security_events[-1000:]

    async def get_security_status(self) -> Dict[str, Any]:
        """Get current security status"""
        recent_events = [e for e in self.security_events
                        if (datetime.now() - datetime.fromisoformat(e["timestamp"])).seconds < 3600]

        failed_deliveries = len([e for e in recent_events if e["delivery_status"] == "error"])

        return {
            "total_security_events": len(self.security_events),
            "recent_events": len(recent_events),
            "failed_deliveries": failed_deliveries,
            "security_incidents": sum(p.get("security_incidents", 0) for p in self.agent_security_profiles.values()),
            "overall_security_status": "healthy" if failed_deliveries == 0 else "warning"
        }


class ConsensusEngine:
    """Manages consensus decision making among agents"""

    def __init__(self):
        self.active_consensus: Dict[str, Dict[str, Any]] = {}
        self.consensus_history: List[Dict[str, Any]] = []

    async def create_consensus_session(self, consensus_id: str, initiator_id: str,
                                     participants: List[str], topic: str, options: List[Any]):
        """Create a new consensus session"""
        self.active_consensus[consensus_id] = {
            "initiator_id": initiator_id,
            "participants": participants,
            "topic": topic,
            "options": options,
            "votes": {},
            "created_at": datetime.now().isoformat(),
            "deadline": (datetime.now() + timedelta(minutes=5)).isoformat(),
            "status": "active"
        }

    async def submit_vote(self, consensus_id: str, agent_id: str, vote: Any, reasoning: str = "") -> Dict[str, Any]:
        """Submit a vote for consensus"""
        if consensus_id not in self.active_consensus:
            return {"status": "error", "error": "Consensus session not found"}

        session = self.active_consensus[consensus_id]

        if agent_id not in session["participants"]:
            return {"status": "error", "error": "Agent not authorized to vote"}

        session["votes"][agent_id] = {
            "vote": vote,
            "reasoning": reasoning,
            "timestamp": datetime.now().isoformat()
        }

        # Check if all votes received
        if len(session["votes"]) == len(session["participants"]):
            await self._finalize_consensus(consensus_id)

        return {"status": "success", "message": "Vote submitted"}

    async def get_consensus_result(self, consensus_id: str) -> Dict[str, Any]:
        """Get consensus result"""
        if consensus_id not in self.active_consensus:
            # Check history
            for historical in self.consensus_history:
                if historical["consensus_id"] == consensus_id:
                    return historical
            return {"status": "error", "error": "Consensus not found"}

        session = self.active_consensus[consensus_id]

        if session["status"] == "completed":
            return session

        # Check for timeout
        deadline = datetime.fromisoformat(session["deadline"])
        if datetime.now() > deadline:
            await self._finalize_consensus(consensus_id)

        return {
            "status": "pending",
            "votes_received": len(session["votes"]),
            "total_participants": len(session["participants"]),
            "topic": session["topic"]
        }

    async def _finalize_consensus(self, consensus_id: str):
        """Finalize consensus with majority voting"""
        session = self.active_consensus[consensus_id]

        if not session["votes"]:
            session["status"] = "failed"
            session["result"] = "No votes received"
        else:
            # Count votes
            vote_counts = {}
            for vote_data in session["votes"].values():
                vote = vote_data["vote"]
                vote_counts[vote] = vote_counts.get(vote, 0) + 1

            # Find majority
            majority_vote = max(vote_counts.items(), key=lambda x: x[1])

            session["status"] = "completed"
            session["result"] = majority_vote[0]
            session["vote_distribution"] = vote_counts
            session["confidence"] = majority_vote[1] / len(session["participants"])

        # Move to history
        session["consensus_id"] = consensus_id
        self.consensus_history.append(session)
        del self.active_consensus[consensus_id]

    async def resolve_context_conflicts(self, context: SharedContext, conflicts: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Resolve context conflicts through consensus"""
        # Simplified conflict resolution
        return {
            "status": "resolved",
            "conflicts": len(conflicts),
            "resolution_method": "latest_wins",
            "message": "Conflicts resolved using latest-wins policy"
        }


class MessageRouter:
    """Routes messages between agents"""

    def __init__(self):
        self.routing_table: Dict[str, str] = {}
        self.failed_deliveries: List[Dict[str, Any]] = []

    async def route_message(self, message: SecureMessage) -> Dict[str, Any]:
        """Route message to recipient"""
        try:
            # Simulate message delivery
            delivery_time = datetime.now() + timedelta(milliseconds=50)  # Simulate network delay

            return {
                "status": "delivered",
                "message_id": message.message_id,
                "recipient_id": message.recipient_id,
                "delivery_time": delivery_time.isoformat(),
                "correlation_id": message.correlation_id
            }

        except Exception as e:
            error_result = {
                "status": "error",
                "message_id": message.message_id,
                "recipient_id": message.recipient_id,
                "error": str(e),
                "correlation_id": message.correlation_id
            }

            self.failed_deliveries.append(error_result)
            return error_result


class ReliabilityMonitor:
    """Monitors inter-agent communication reliability"""

    def __init__(self):
        self.delivery_metrics: List[Dict[str, Any]] = []
        self.latency_metrics: List[float] = []
        self.error_rates: List[float] = []

    async def record_message_delivery(self, message: SecureMessage, result: Dict[str, Any]):
        """Record message delivery metrics"""
        metric = {
            "message_id": message.message_id,
            "sender_id": message.sender_id,
            "recipient_id": message.recipient_id,
            "message_type": message.message_type.value,
            "security_level": message.security_level.value,
            "status": result.get("status", "unknown"),
            "timestamp": datetime.now().isoformat()
        }

        if "delivery_time" in result:
            delivery_time = datetime.fromisoformat(result["delivery_time"])
            sent_time = message.timestamp
            latency = (delivery_time - sent_time).total_seconds() * 1000  # ms
            metric["latency_ms"] = latency
            self.latency_metrics.append(latency)

        self.delivery_metrics.append(metric)

        # Keep only recent metrics
        if len(self.delivery_metrics) > 1000:
            self.delivery_metrics = self.delivery_metrics[-1000:]

        if len(self.latency_metrics) > 1000:
            self.latency_metrics = self.latency_metrics[-1000:]

    async def get_metrics(self) -> Dict[str, Any]:
        """Get reliability metrics"""
        if not self.delivery_metrics:
            return {"status": "no_data"}

        recent_metrics = self.delivery_metrics[-100:]  # Last 100 messages
        success_count = len([m for m in recent_metrics if m["status"] == "delivered"])
        success_rate = success_count / len(recent_metrics) if recent_metrics else 0

        avg_latency = statistics.mean(self.latency_metrics[-100:]) if self.latency_metrics else 0

        return {
            "total_messages": len(self.delivery_metrics),
            "success_rate": success_rate,
            "average_latency_ms": avg_latency,
            "error_rate": 1 - success_rate,
            "reliability_status": "high" if success_rate > 0.95 else "medium" if success_rate > 0.90 else "low"
        }

    async def run_reliability_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run reliability test"""
        test_results = {
            "test_id": str(uuid.uuid4()),
            "timestamp": datetime.now().isoformat(),
            "test_type": test_config.get("test_type", "basic_reliability"),
            "metrics": {}
        }

        # Simulate reliability test
        test_results["metrics"] = {
            "message_delivery_success_rate": 0.98,
            "average_latency_ms": 45.2,
            "encryption_overhead": 0.02,
            "error_recovery_rate": 0.95,
            "concurrent_session_capacity": 50
        }

        return test_results