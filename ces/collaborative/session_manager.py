"""
CES Collaborative Session Manager

Manages collaborative sessions, shared tasks, and real-time collaboration
features for multi-user workflows in the Cognitive Enhancement System.
"""

import asyncio
import json
import logging
import uuid
from typing import Dict, List, Optional, Set, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import aiofiles


@dataclass
class User:
    """Represents a user in the collaborative system"""
    id: str
    name: str
    email: Optional[str] = None
    role: str = "participant"  # participant, moderator, observer
    joined_at: datetime = None
    last_active: datetime = None
    status: str = "active"  # active, inactive, disconnected

    def __post_init__(self):
        if self.joined_at is None:
            self.joined_at = datetime.now()
        if self.last_active is None:
            self.last_active = datetime.now()


@dataclass
class CollaborativeTask:
    """Represents a shared task in a collaborative session"""
    id: str
    title: str
    description: str
    status: str = "pending"  # pending, in_progress, completed, cancelled
    priority: str = "medium"  # low, medium, high, critical
    assignee: Optional[str] = None  # user_id
    created_by: str = None
    created_at: datetime = None
    updated_at: datetime = None
    due_date: Optional[datetime] = None
    tags: List[str] = None
    dependencies: List[str] = None  # task_ids
    progress: float = 0.0  # 0.0 to 1.0
    comments: List[Dict[str, Any]] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.tags is None:
            self.tags = []
        if self.dependencies is None:
            self.dependencies = []
        if self.comments is None:
            self.comments = []


@dataclass
class SessionMessage:
    """Represents a message in the collaborative session"""
    id: str
    user_id: str
    username: str
    message_type: str  # chat, system, task_update, user_join, user_leave
    content: str
    timestamp: datetime = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CollaborativeSession:
    """Represents a collaborative work session"""
    id: str
    name: str
    description: Optional[str] = None
    created_by: str = None
    created_at: datetime = None
    status: str = "active"  # active, paused, completed, archived
    max_participants: int = 10
    participants: Dict[str, User] = None
    tasks: Dict[str, CollaborativeTask] = None
    messages: List[SessionMessage] = None
    settings: Dict[str, Any] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.participants is None:
            self.participants = {}
        if self.tasks is None:
            self.tasks = {}
        if self.messages is None:
            self.messages = []
        if self.settings is None:
            self.settings = {
                "allow_guests": False,
                "require_approval": False,
                "auto_save": True,
                "real_time_updates": True
            }


class CollaborativeSessionManager:
    """
    Manages collaborative sessions and multi-user interactions
    """

    def __init__(self, storage_path: str = "./data/collaborative"):
        self.storage_path = storage_path
        self.logger = logging.getLogger(__name__)

        # Active sessions
        self.active_sessions: Dict[str, CollaborativeSession] = {}

        # WebSocket connections for real-time updates
        self.session_connections: Dict[str, Set[Any]] = {}  # session_id -> set of connections

        # User session mapping
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id

        # Initialize storage
        self._initialized = False

    async def ensure_initialized(self):
        """Ensure the manager is initialized"""
        if not self._initialized:
            await self._initialize_storage()
            self._initialized = True

    async def _initialize_storage(self):
        """Initialize storage directory and load existing sessions"""
        import os
        os.makedirs(self.storage_path, exist_ok=True)

        # Load existing sessions
        sessions_file = f"{self.storage_path}/sessions.json"
        if os.path.exists(sessions_file):
            try:
                async with aiofiles.open(sessions_file, 'r') as f:
                    data = json.loads(await f.read())
                    for session_data in data.get('sessions', []):
                        session = CollaborativeSession(**session_data)
                        # Convert datetime strings back to datetime objects
                        session.created_at = datetime.fromisoformat(session.created_at)
                        for user in session.participants.values():
                            user.joined_at = datetime.fromisoformat(user.joined_at)
                            user.last_active = datetime.fromisoformat(user.last_active)
                        for task in session.tasks.values():
                            task.created_at = datetime.fromisoformat(task.created_at)
                            task.updated_at = datetime.fromisoformat(task.updated_at)
                            if task.due_date:
                                task.due_date = datetime.fromisoformat(task.due_date)
                        for message in session.messages:
                            message.timestamp = datetime.fromisoformat(message.timestamp)

                        if session.status == "active":
                            self.active_sessions[session.id] = session
                            self.session_connections[session.id] = set()

                self.logger.info(f"Loaded {len(self.active_sessions)} active sessions")
            except Exception as e:
                self.logger.error(f"Error loading sessions: {e}")

    async def create_session(self, name: str, created_by: str,
                           description: Optional[str] = None,
                           max_participants: int = 10) -> CollaborativeSession:
        """
        Create a new collaborative session

        Args:
            name: Session name
            created_by: User ID of session creator
            description: Optional session description
            max_participants: Maximum number of participants

        Returns:
            Created session object
        """
        session_id = str(uuid.uuid4())
        session = CollaborativeSession(
            id=session_id,
            name=name,
            description=description,
            created_by=created_by,
            max_participants=max_participants
        )

        # Add creator as first participant
        creator = User(id=created_by, name=f"User_{created_by}", role="moderator")
        session.participants[created_by] = creator

        # Initialize connections set
        self.session_connections[session_id] = set()

        # Store session
        self.active_sessions[session_id] = session
        self.user_sessions[created_by] = session_id

        # Add system message
        await self._add_system_message(session_id, f"Session '{name}' created by {creator.name}")

        # Save to disk
        await self._save_sessions()

        self.logger.info(f"Created collaborative session: {session_id}")
        return session

    async def join_session(self, session_id: str, user: User) -> bool:
        """
        Join an existing collaborative session

        Args:
            session_id: Session ID to join
            user: User joining the session

        Returns:
            True if joined successfully, False otherwise
        """
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        # Check participant limit
        if len(session.participants) >= session.max_participants:
            return False

        # Check if user is already in session
        if user.id in session.participants:
            # Update user status
            session.participants[user.id].status = "active"
            session.participants[user.id].last_active = datetime.now()
        else:
            # Add new participant
            session.participants[user.id] = user

        # Update user session mapping
        self.user_sessions[user.id] = session_id

        # Add system message
        await self._add_system_message(session_id, f"{user.name} joined the session")

        # Notify other participants
        await self._broadcast_to_session(session_id, {
            "type": "user_joined",
            "user": asdict(user),
            "timestamp": datetime.now().isoformat()
        })

        # Save changes
        await self._save_sessions()

        self.logger.info(f"User {user.id} joined session {session_id}")
        return True

    async def leave_session(self, session_id: str, user_id: str) -> bool:
        """
        Leave a collaborative session

        Args:
            session_id: Session ID to leave
            user_id: User ID leaving the session

        Returns:
            True if left successfully, False otherwise
        """
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        if user_id not in session.participants:
            return False

        # Mark user as inactive (don't remove completely to preserve history)
        session.participants[user_id].status = "inactive"

        # Remove from user session mapping
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]

        # Add system message
        username = session.participants[user_id].name
        await self._add_system_message(session_id, f"{username} left the session")

        # Notify other participants
        await self._broadcast_to_session(session_id, {
            "type": "user_left",
            "user_id": user_id,
            "username": username,
            "timestamp": datetime.now().isoformat()
        })

        # Save changes
        await self._save_sessions()

        self.logger.info(f"User {user_id} left session {session_id}")
        return True

    async def create_task(self, session_id: str, task: CollaborativeTask) -> Optional[str]:
        """
        Create a new task in a collaborative session

        Args:
            session_id: Session ID
            task: Task to create

        Returns:
            Task ID if created successfully, None otherwise
        """
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        # Generate task ID
        task_id = str(uuid.uuid4())
        task.id = task_id

        # Add task to session
        session.tasks[task_id] = task

        # Add system message
        await self._add_system_message(session_id, f"Task '{task.title}' created by {task.created_by}")

        # Notify participants
        await self._broadcast_to_session(session_id, {
            "type": "task_created",
            "task": asdict(task),
            "timestamp": datetime.now().isoformat()
        })

        # Save changes
        await self._save_sessions()

        self.logger.info(f"Created task {task_id} in session {session_id}")
        return task_id

    async def update_task(self, session_id: str, task_id: str,
                         updates: Dict[str, Any], user_id: str) -> bool:
        """
        Update an existing task

        Args:
            session_id: Session ID
            task_id: Task ID to update
            updates: Fields to update
            user_id: User making the update

        Returns:
            True if updated successfully, False otherwise
        """
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        if task_id not in session.tasks:
            return False

        task = session.tasks[task_id]

        # Apply updates
        for key, value in updates.items():
            if hasattr(task, key):
                setattr(task, key, value)

        task.updated_at = datetime.now()

        # Add system message
        username = session.participants.get(user_id, User(id=user_id, name=f"User_{user_id}")).name
        await self._add_system_message(session_id, f"Task '{task.title}' updated by {username}")

        # Notify participants
        await self._broadcast_to_session(session_id, {
            "type": "task_updated",
            "task_id": task_id,
            "updates": updates,
            "updated_by": user_id,
            "timestamp": datetime.now().isoformat()
        })

        # Save changes
        await self._save_sessions()

        self.logger.info(f"Updated task {task_id} in session {session_id}")
        return True

    async def send_message(self, session_id: str, message: SessionMessage) -> bool:
        """
        Send a message to a collaborative session

        Args:
            session_id: Session ID
            message: Message to send

        Returns:
            True if sent successfully, False otherwise
        """
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        # Add message to session
        session.messages.append(message)

        # Keep only last 1000 messages to prevent memory issues
        if len(session.messages) > 1000:
            session.messages = session.messages[-1000:]

        # Notify participants
        await self._broadcast_to_session(session_id, {
            "type": "message",
            "message": asdict(message),
            "timestamp": datetime.now().isoformat()
        })

        # Save changes (messages are saved periodically)
        await self._save_sessions()

        return True

    async def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a collaborative session

        Args:
            session_id: Session ID

        Returns:
            Session information or None if not found
        """
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]

        return {
            "id": session.id,
            "name": session.name,
            "description": session.description,
            "created_by": session.created_by,
            "created_at": session.created_at.isoformat(),
            "status": session.status,
            "max_participants": session.max_participants,
            "participant_count": len([p for p in session.participants.values() if p.status == "active"]),
            "task_count": len(session.tasks),
            "message_count": len(session.messages),
            "participants": [asdict(user) for user in session.participants.values()],
            "tasks": [asdict(task) for task in session.tasks.values()],
            "recent_messages": [asdict(msg) for msg in session.messages[-50:]]
        }

    async def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List all active collaborative sessions

        Args:
            user_id: Optional user ID to filter sessions

        Returns:
            List of session information
        """
        sessions = []

        for session in self.active_sessions.values():
            if user_id and user_id not in session.participants:
                continue

            sessions.append({
                "id": session.id,
                "name": session.name,
                "description": session.description,
                "created_by": session.created_by,
                "created_at": session.created_at.isoformat(),
                "participant_count": len([p for p in session.participants.values() if p.status == "active"]),
                "task_count": len(session.tasks)
            })

        return sessions

    async def register_connection(self, session_id: str, connection: Any):
        """
        Register a WebSocket connection for real-time updates

        Args:
            session_id: Session ID
            connection: WebSocket connection object
        """
        if session_id not in self.session_connections:
            self.session_connections[session_id] = set()

        self.session_connections[session_id].add(connection)
        self.logger.debug(f"Registered connection for session {session_id}")

    async def unregister_connection(self, session_id: str, connection: Any):
        """
        Unregister a WebSocket connection

        Args:
            session_id: Session ID
            connection: WebSocket connection object
        """
        if session_id in self.session_connections:
            self.session_connections[session_id].discard(connection)
            self.logger.debug(f"Unregistered connection for session {session_id}")

    async def _broadcast_to_session(self, session_id: str, data: Dict[str, Any]):
        """
        Broadcast data to all connections in a session

        Args:
            session_id: Session ID
            data: Data to broadcast
        """
        if session_id not in self.session_connections:
            return

        disconnected = set()
        for connection in self.session_connections[session_id]:
            try:
                await connection.send_json(data)
            except Exception as e:
                self.logger.error(f"Error broadcasting to connection: {e}")
                disconnected.add(connection)

        # Remove disconnected connections
        for conn in disconnected:
            self.session_connections[session_id].discard(conn)

    async def _add_system_message(self, session_id: str, content: str):
        """
        Add a system message to a session

        Args:
            session_id: Session ID
            content: Message content
        """
        if session_id not in self.active_sessions:
            return

        message = SessionMessage(
            id=str(uuid.uuid4()),
            user_id="system",
            username="System",
            message_type="system",
            content=content
        )

        self.active_sessions[session_id].messages.append(message)

    async def _save_sessions(self):
        """
        Save active sessions to disk
        """
        await self.ensure_initialized()
        try:
            # Convert sessions to serializable format
            sessions_data = []
            for session in self.active_sessions.values():
                session_dict = asdict(session)
                # Convert datetime objects to ISO strings
                session_dict['created_at'] = session.created_at.isoformat()
                for user in session_dict['participants'].values():
                    user['joined_at'] = user['joined_at'].isoformat()
                    user['last_active'] = user['last_active'].isoformat()
                for task in session_dict['tasks'].values():
                    task['created_at'] = task['created_at'].isoformat()
                    task['updated_at'] = task['updated_at'].isoformat()
                    if task['due_date']:
                        task['due_date'] = task['due_date'].isoformat()
                for message in session_dict['messages']:
                    message['timestamp'] = message['timestamp'].isoformat()

                sessions_data.append(session_dict)

            data = {"sessions": sessions_data, "last_updated": datetime.now().isoformat()}

            async with aiofiles.open(f"{self.storage_path}/sessions.json", 'w') as f:
                await f.write(json.dumps(data, indent=2))

        except Exception as e:
            self.logger.error(f"Error saving sessions: {e}")

    async def cleanup_inactive_sessions(self):
        """
        Clean up inactive sessions and connections
        """
        # Mark inactive users (no activity for 30 minutes)
        cutoff_time = datetime.now() - timedelta(minutes=30)

        for session in self.active_sessions.values():
            for user in session.participants.values():
                if user.last_active < cutoff_time and user.status == "active":
                    user.status = "inactive"
                    await self._add_system_message(session.id, f"{user.name} became inactive")

        # Archive old sessions (older than 7 days)
        archive_cutoff = datetime.now() - timedelta(days=7)
        sessions_to_archive = []

        for session_id, session in self.active_sessions.items():
            if session.created_at < archive_cutoff and session.status == "active":
                session.status = "archived"
                sessions_to_archive.append(session_id)

        for session_id in sessions_to_archive:
            await self._add_system_message(session_id, "Session archived due to inactivity")

        if sessions_to_archive:
            await self._save_sessions()
            self.logger.info(f"Archived {len(sessions_to_archive)} inactive sessions")


# Global session manager instance
session_manager = CollaborativeSessionManager()