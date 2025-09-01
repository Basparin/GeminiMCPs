"""
Human-AI Interaction Protocols - CES Real-time Collaboration Framework

Provides comprehensive protocols for human-AI interaction including:
- Real-time interaction management
- Feedback loops and iterative refinement
- Concurrent multi-user session support
- Session state management and persistence
- Message routing and context awareness
- Interactive task collaboration
"""

import asyncio
import logging
import json
import threading
import time
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
import uuid

from ..core.memory_manager import MemoryManager
from ..ai_orchestrator.ai_assistant import AIOrchestrator


@dataclass
class InteractionMessage:
    """Represents a message in the human-AI interaction"""
    id: str
    session_id: str
    sender: str  # 'human' or 'ai_assistant_name'
    message_type: str  # 'query', 'response', 'feedback', 'clarification', 'suggestion'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = None
    context: Dict[str, Any] = None
    requires_response: bool = False


@dataclass
class InteractionSession:
    """Represents an active human-AI interaction session"""
    session_id: str
    user_id: str
    start_time: datetime
    last_activity: datetime
    status: str  # 'active', 'paused', 'completed', 'terminated'
    current_task: Optional[str] = None
    context: Dict[str, Any] = None
    participants: List[str] = None  # AI assistants involved
    message_history: List[InteractionMessage] = None
    feedback_loop_active: bool = False
    concurrent_users: int = 1


class HumanAIInteractionManager:
    """
    Manages human-AI interaction protocols with real-time capabilities.

    Supports:
    - Real-time message routing
    - Feedback loops for iterative refinement
    - Concurrent multi-user sessions
    - Context-aware interactions
    - Session persistence and recovery
    """

    def __init__(self, memory_manager: MemoryManager, ai_orchestrator: AIOrchestrator):
        self.logger = logging.getLogger(__name__)
        self.memory_manager = memory_manager
        self.ai_orchestrator = ai_orchestrator

        # Session management
        self.active_sessions: Dict[str, InteractionSession] = {}
        self.session_lock = threading.Lock()

        # Message routing
        self.message_queues: Dict[str, asyncio.Queue] = {}
        self.response_handlers: Dict[str, Callable] = {}

        # Real-time processing
        self.processing_tasks: Dict[str, asyncio.Task] = {}
        self._stop_event = threading.Event()

        # Performance monitoring
        self.message_counts = defaultdict(int)
        self.response_times = deque(maxlen=1000)

        self.logger.info("Human-AI Interaction Manager initialized")

    async def start_session(self, user_id: str, initial_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Start a new human-AI interaction session

        Args:
            user_id: Unique identifier for the user
            initial_context: Initial context for the session

        Returns:
            Session ID for the new session
        """
        session_id = str(uuid.uuid4())

        session = InteractionSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.now(),
            last_activity=datetime.now(),
            status='active',
            context=initial_context or {},
            participants=[],
            message_history=[],
            concurrent_users=1
        )

        with self.session_lock:
            self.active_sessions[session_id] = session

        # Initialize message queue for real-time processing
        self.message_queues[session_id] = asyncio.Queue()

        # Start background processing for this session
        self.processing_tasks[session_id] = asyncio.create_task(
            self._process_session_messages(session_id)
        )

        self.logger.info(f"Started new session {session_id} for user {user_id}")
        return session_id

    async def send_message(self, session_id: str, message: InteractionMessage) -> Dict[str, Any]:
        """
        Send a message to the interaction session

        Args:
            session_id: Target session ID
            message: Message to send

        Returns:
            Response with processing status
        """
        if session_id not in self.active_sessions:
            return {"status": "error", "error": "Session not found"}

        session = self.active_sessions[session_id]
        session.last_activity = datetime.now()

        # Add message to session history
        session.message_history.append(message)

        # Queue message for processing
        await self.message_queues[session_id].put(message)

        # Update message counts
        self.message_counts[message.message_type] += 1

        return {
            "status": "queued",
            "message_id": message.id,
            "session_id": session_id,
            "timestamp": message.timestamp.isoformat()
        }

    async def _process_session_messages(self, session_id: str):
        """Background task to process messages for a session"""
        queue = self.message_queues[session_id]

        while not self._stop_event.is_set():
            try:
                # Wait for message with timeout
                message = await asyncio.wait_for(queue.get(), timeout=1.0)

                start_time = time.time()

                # Process the message
                response = await self._handle_message(session_id, message)

                # Record response time
                response_time = (time.time() - start_time) * 1000
                self.response_times.append(response_time)

                # Send response if required
                if message.requires_response and response:
                    await self._send_response(session_id, response)

                queue.task_done()

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error processing message in session {session_id}: {e}")

    async def _handle_message(self, session_id: str, message: InteractionMessage) -> Optional[InteractionMessage]:
        """Handle incoming message and generate appropriate response"""
        session = self.active_sessions[session_id]

        if message.message_type == 'query':
            return await self._handle_query_message(session, message)
        elif message.message_type == 'feedback':
            return await self._handle_feedback_message(session, message)
        elif message.message_type == 'clarification':
            return await self._handle_clarification_message(session, message)
        else:
            self.logger.warning(f"Unknown message type: {message.message_type}")
            return None

    async def _handle_query_message(self, session: InteractionSession, message: InteractionMessage) -> Optional[InteractionMessage]:
        """Handle user query messages"""
        # Determine if this requires multi-assistant coordination
        query_complexity = self._assess_query_complexity(message.content)

        if query_complexity > 6:
            # Use multi-assistant execution
            result = await self.ai_orchestrator.execute_task(
                message.content,
                context=session.context,
                assistant_preferences=session.participants
            )

            # Update session participants
            if 'assistants_used' in result:
                session.participants = result['assistants_used']

            response_content = self._format_multi_assistant_response(result)
        else:
            # Use single assistant
            result = await self.ai_orchestrator.execute_task(
                message.content,
                context=session.context
            )
            response_content = result.get('result', 'No response generated')

        # Create response message
        response = InteractionMessage(
            id=str(uuid.uuid4()),
            session_id=session.session_id,
            sender='ai_orchestrator',
            message_type='response',
            content=response_content,
            timestamp=datetime.now(),
            metadata={'query_complexity': query_complexity, 'execution_result': result},
            context=session.context
        )

        return response

    async def _handle_feedback_message(self, session: InteractionSession, message: InteractionMessage) -> Optional[InteractionMessage]:
        """Handle user feedback messages for iterative refinement"""
        # Extract feedback type and content
        feedback_data = message.metadata or {}
        feedback_type = feedback_data.get('feedback_type', 'general')

        if feedback_type == 'refinement_request':
            # Generate refined response based on feedback
            refinement_query = f"Refine the previous response based on this feedback: {message.content}"
            refined_result = await self.ai_orchestrator.execute_task(
                refinement_query,
                context={**session.context, 'previous_feedback': message.content}
            )

            response = InteractionMessage(
                id=str(uuid.uuid4()),
                session_id=session.session_id,
                sender='ai_orchestrator',
                message_type='response',
                content=refined_result.get('result', 'Refinement completed'),
                timestamp=datetime.now(),
                metadata={'refinement_based_on': message.id},
                context=session.context
            )

            return response

        elif feedback_type == 'clarification_needed':
            # Request clarification from user
            response = InteractionMessage(
                id=str(uuid.uuid4()),
                session_id=session.session_id,
                sender='ai_orchestrator',
                message_type='clarification',
                content="I need clarification on your request. Could you please provide more details?",
                timestamp=datetime.now(),
                metadata={'clarification_for': message.id},
                context=session.context,
                requires_response=True
            )

            return response

        return None

    async def _handle_clarification_message(self, session: InteractionSession, message: InteractionMessage) -> Optional[InteractionMessage]:
        """Handle clarification messages"""
        # Process clarification and generate improved response
        clarified_query = f"Original request with clarification: {message.content}"

        result = await self.ai_orchestrator.execute_task(
            clarified_query,
            context=session.context
        )

        response = InteractionMessage(
            id=str(uuid.uuid4()),
            session_id=session.session_id,
            sender='ai_orchestrator',
            message_type='response',
            content=result.get('result', 'Clarified response generated'),
            timestamp=datetime.now(),
            metadata={'clarification_processed': True},
            context=session.context
        )

        return response

    def _assess_query_complexity(self, query: str) -> float:
        """Assess the complexity of a user query"""
        complexity = 0

        # Length-based complexity
        if len(query) > 200:
            complexity += 2
        elif len(query) > 100:
            complexity += 1

        # Keyword-based complexity
        complex_keywords = ['complex', 'multiple', 'integrate', 'optimize', 'architecture',
                          'system', 'advanced', 'sophisticated', 'comprehensive']
        query_lower = query.lower()
        complexity += sum(1 for keyword in complex_keywords if keyword in query_lower)

        # Question complexity
        if query.count('?') > 2:
            complexity += 1

        return min(complexity, 10)

    def _format_multi_assistant_response(self, result: Dict[str, Any]) -> str:
        """Format response from multi-assistant execution"""
        if result.get('execution_mode') == 'multi_assistant':
            assistants = result.get('assistants_used', [])
            subtask_count = result.get('subtasks_count', 0)

            formatted = f"Multi-assistant collaboration completed using {len(assistants)} AI assistants "
            formatted += f"across {subtask_count} subtasks.\n\n"

            if 'synthesized_result' in result:
                formatted += result['synthesized_result']
            else:
                formatted += "Results synthesized from parallel processing."

            return formatted
        else:
            return result.get('result', 'Response generated')

    async def _send_response(self, session_id: str, response: InteractionMessage):
        """Send response message to session"""
        session = self.active_sessions[session_id]
        session.message_history.append(response)

        # Store in memory for context retention
        await self._persist_message_to_memory(session_id, response)

    async def _persist_message_to_memory(self, session_id: str, message: InteractionMessage):
        """Persist message to memory system for context retention"""
        memory_context = {
            'session_id': session_id,
            'message_type': message.message_type,
            'sender': message.sender,
            'content': message.content,
            'timestamp': message.timestamp.isoformat(),
            'metadata': message.metadata,
            'context': message.context
        }

        # Store in memory manager
        self.memory_manager.store_user_preference(
            f"session_{session_id}_message_{message.id}",
            memory_context
        )

    async def join_session(self, session_id: str, user_id: str) -> bool:
        """
        Allow additional user to join an existing session for concurrent collaboration

        Args:
            session_id: Target session ID
            user_id: User ID to join

        Returns:
            Success status
        """
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]
        session.concurrent_users += 1
        session.last_activity = datetime.now()

        # Log concurrent session activity
        self.logger.info(f"User {user_id} joined session {session_id}. Concurrent users: {session.concurrent_users}")

        return True

    async def leave_session(self, session_id: str, user_id: str) -> bool:
        """
        Remove user from session

        Args:
            session_id: Target session ID
            user_id: User ID leaving

        Returns:
            Success status
        """
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]
        if session.concurrent_users > 1:
            session.concurrent_users -= 1
            session.last_activity = datetime.now()
            self.logger.info(f"User {user_id} left session {session_id}. Remaining users: {session.concurrent_users}")
            return True
        else:
            # Last user leaving, terminate session
            return await self.terminate_session(session_id, user_id)

    async def terminate_session(self, session_id: str, user_id: str) -> bool:
        """
        Terminate an interaction session

        Args:
            session_id: Session to terminate
            user_id: User requesting termination

        Returns:
            Success status
        """
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]
        session.status = 'terminated'
        session.last_activity = datetime.now()

        # Cancel processing task
        if session_id in self.processing_tasks:
            self.processing_tasks[session_id].cancel()
            del self.processing_tasks[session_id]

        # Clean up message queue
        if session_id in self.message_queues:
            del self.message_queues[session_id]

        # Persist session summary to memory
        await self._persist_session_summary(session)

        self.logger.info(f"Terminated session {session_id} by user {user_id}")
        return True

    async def _persist_session_summary(self, session: InteractionSession):
        """Persist session summary for future reference"""
        summary = {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'start_time': session.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'duration_minutes': (datetime.now() - session.start_time).total_seconds() / 60,
            'message_count': len(session.message_history),
            'participants': session.participants,
            'concurrent_users': session.concurrent_users,
            'status': session.status
        }

        self.memory_manager.store_user_preference(
            f"session_summary_{session.session_id}",
            summary
        )

    def get_session_status(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific session"""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        return {
            'session_id': session.session_id,
            'user_id': session.user_id,
            'status': session.status,
            'start_time': session.start_time.isoformat(),
            'last_activity': session.last_activity.isoformat(),
            'concurrent_users': session.concurrent_users,
            'message_count': len(session.message_history),
            'participants': session.participants
        }

    def get_active_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get all active sessions"""
        return {
            sid: self.get_session_status(sid)
            for sid, session in self.active_sessions.items()
            if session.status == 'active'
        }

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics for human-AI interactions"""
        if not self.response_times:
            return {"status": "no_data"}

        return {
            "total_messages": sum(self.message_counts.values()),
            "message_types": dict(self.message_counts),
            "avg_response_time_ms": sum(self.response_times) / len(self.response_times),
            "p95_response_time_ms": sorted(self.response_times)[int(len(self.response_times) * 0.95)] if self.response_times else 0,
            "active_sessions": len([s for s in self.active_sessions.values() if s.status == 'active']),
            "total_sessions": len(self.active_sessions),
            "concurrent_users_supported": sum(s.concurrent_users for s in self.active_sessions.values() if s.status == 'active')
        }

    async def cleanup_inactive_sessions(self, max_inactive_minutes: int = 30):
        """Clean up sessions that have been inactive for too long"""
        cutoff_time = datetime.now() - timedelta(minutes=max_inactive_minutes)
        sessions_to_cleanup = []

        for session_id, session in self.active_sessions.items():
            if session.last_activity < cutoff_time and session.status == 'active':
                sessions_to_cleanup.append(session_id)

        for session_id in sessions_to_cleanup:
            self.logger.info(f"Cleaning up inactive session {session_id}")
            await self.terminate_session(session_id, "system")

    def __del__(self):
        """Cleanup on destruction"""
        self._stop_event.set()

        # Cancel all processing tasks
        for task in self.processing_tasks.values():
            if not task.done():
                task.cancel()