"""CES Session Manager.

Manages collaborative sessions, multi-user workflows, and real-time collaboration
for the Cognitive Enhancement System.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict

from ..core.logging_config import get_logger

logger = get_logger(__name__)

class SessionManager:
    """Manages collaborative sessions and multi-user workflows."""

    def __init__(self):
        self.active_sessions = {}
        self.session_tasks = defaultdict(list)
        self.user_sessions = defaultdict(list)
        self.session_history = defaultdict(list)
        self.user_permissions = defaultdict(dict)  # user_id -> session_id -> permissions
        self.session_roles = defaultdict(dict)  # session_id -> user_id -> role
        self.online_users = set()  # Track online users
        self.user_profiles = {}  # user_id -> profile data

    def is_healthy(self) -> bool:
        """Check if session manager is healthy."""
        return True

    async def create_session(self, session_data: Dict[str, Any]) -> str:
        """Create a new collaborative session."""
        session_id = session_data.get("id", str(uuid.uuid4()))

        session = {
            "id": session_id,
            "name": session_data.get("name", "Untitled Session"),
            "description": session_data.get("description", ""),
            "owner_id": session_data.get("owner_id", "anonymous"),
            "collaborators": session_data.get("collaborators", []),
            "participants": session_data.get("participants", [session_data.get("owner_id", "anonymous")]),
            "created_at": session_data.get("created_at", datetime.now().isoformat()),
            "status": session_data.get("status", "active"),
            "last_activity": datetime.now().isoformat(),
            "metadata": session_data.get("metadata", {}),
            "tasks": [],
            "messages": [],
            "shared_context": {}
        }

        self.active_sessions[session_id] = session

        # Track user's sessions
        owner_id = session["owner_id"]
        self.user_sessions[owner_id].append(session_id)

        logger.info(f"Created session {session_id} for user {owner_id}")
        return session_id

    async def join_session(self, session_id: str, user_id: str) -> bool:
        """Join an existing session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found")
            return False

        session = self.active_sessions[session_id]

        if user_id not in session["participants"]:
            session["participants"].append(user_id)
            session["last_activity"] = datetime.now().isoformat()

            # Track user's sessions
            if session_id not in self.user_sessions[user_id]:
                self.user_sessions[user_id].append(session_id)

            logger.info(f"User {user_id} joined session {session_id}")
            return True

        return True

    async def leave_session(self, session_id: str, user_id: str) -> bool:
        """Leave a session."""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]

        if user_id in session["participants"]:
            session["participants"].remove(user_id)
            session["last_activity"] = datetime.now().isoformat()

            # Remove from user's sessions if they're not the owner
            if user_id != session["owner_id"] and session_id in self.user_sessions[user_id]:
                self.user_sessions[user_id].remove(session_id)

            # If no participants left and not owned by anyone, mark as inactive
            if not session["participants"]:
                session["status"] = "inactive"
                session["ended_at"] = datetime.now().isoformat()

            logger.info(f"User {user_id} left session {session_id}")
            return True

        return False

    async def store_task(self, task_data: Dict[str, Any]) -> str:
        """Store a task in a session."""
        task_id = task_data.get("id", str(uuid.uuid4()))
        session_id = task_data.get("session_id")

        if session_id and session_id in self.active_sessions:
            task = {
                "id": task_id,
                "description": task_data.get("description", ""),
                "status": task_data.get("status", "pending"),
                "priority": task_data.get("priority", "medium"),
                "assigned_to": task_data.get("user_id", "anonymous"),
                "created_at": task_data.get("created_at", datetime.now().isoformat()),
                "updated_at": datetime.now().isoformat(),
                "tags": task_data.get("tags", []),
                "metadata": task_data.get("metadata", {}),
                "analysis": task_data.get("analysis", {})
            }

            self.active_sessions[session_id]["tasks"].append(task)
            self.session_tasks[session_id].append(task_id)

            # Update session activity
            self.active_sessions[session_id]["last_activity"] = datetime.now().isoformat()

            logger.info(f"Stored task {task_id} in session {session_id}")
            return task_id

        # Store in general task list if no session specified
        task = {
            "id": task_id,
            **task_data,
            "created_at": task_data.get("created_at", datetime.now().isoformat()),
            "updated_at": datetime.now().isoformat()
        }
        self.session_tasks["general"].append(task)
        return task_id

    async def get_task(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get a task by ID."""
        # Search in all sessions
        for session_id, session in self.active_sessions.items():
            for task in session.get("tasks", []):
                if task["id"] == task_id:
                    return task

        # Search in general tasks
        for task in self.session_tasks.get("general", []):
            if task["id"] == task_id:
                return task

        return None

    async def update_task(self, task_id: str, updates: Dict[str, Any]) -> bool:
        """Update a task."""
        # Search in all sessions
        for session_id, session in self.active_sessions.items():
            for task in session.get("tasks", []):
                if task["id"] == task_id:
                    task.update(updates)
                    task["updated_at"] = datetime.now().isoformat()
                    session["last_activity"] = datetime.now().isoformat()
                    logger.info(f"Updated task {task_id} in session {session_id}")
                    return True

        # Search in general tasks
        for task in self.session_tasks.get("general", []):
            if task["id"] == task_id:
                task.update(updates)
                task["updated_at"] = datetime.now().isoformat()
                return True

        return False

    async def add_message(self, session_id: str, message_data: Dict[str, Any]) -> str:
        """Add a message to a session."""
        if session_id not in self.active_sessions:
            logger.warning(f"Session {session_id} not found for message")
            return ""

        message_id = str(uuid.uuid4())
        message = {
            "id": message_id,
            "content": message_data.get("content", ""),
            "user_id": message_data.get("user_id", "anonymous"),
            "timestamp": datetime.now().isoformat(),
            "message_type": message_data.get("message_type", "text"),
            "metadata": message_data.get("metadata", {})
        }

        self.active_sessions[session_id]["messages"].append(message)
        self.active_sessions[session_id]["last_activity"] = datetime.now().isoformat()

        # Keep only last 100 messages per session
        if len(self.active_sessions[session_id]["messages"]) > 100:
            self.active_sessions[session_id]["messages"] = self.active_sessions[session_id]["messages"][-100:]

        logger.info(f"Added message {message_id} to session {session_id}")
        return message_id

    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """Get all active sessions."""
        return [
            {
                "id": session_id,
                "name": session.get("name", "Untitled"),
                "description": session.get("description", ""),
                "owner_id": session.get("owner_id", "anonymous"),
                "participants": session.get("participants", []),
                "participant_count": len(session.get("participants", [])),
                "task_count": len(session.get("tasks", [])),
                "created_at": session.get("created_at", ""),
                "last_activity": session.get("last_activity", ""),
                "status": session.get("status", "active")
            }
            for session_id, session in self.active_sessions.items()
            if session.get("status") == "active"
        ]

    def get_session_details(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a session."""
        if session_id not in self.active_sessions:
            return None

        session = self.active_sessions[session_id]
        return {
            **session,
            "task_count": len(session.get("tasks", [])),
            "message_count": len(session.get("messages", [])),
            "duration_minutes": self._calculate_session_duration(session)
        }

    def get_user_sessions(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all sessions for a user."""
        user_session_ids = self.user_sessions.get(user_id, [])
        return [
            self.get_session_details(session_id)
            for session_id in user_session_ids
            if session_id in self.active_sessions
        ]

    async def update_session_context(self, session_id: str, context_key: str, context_value: Any):
        """Update shared context for a session."""
        if session_id in self.active_sessions:
            self.active_sessions[session_id]["shared_context"][context_key] = context_value
            self.active_sessions[session_id]["last_activity"] = datetime.now().isoformat()
            logger.info(f"Updated context {context_key} for session {session_id}")

    def get_session_context(self, session_id: str) -> Dict[str, Any]:
        """Get shared context for a session."""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id].get("shared_context", {})
        return {}

    async def archive_session(self, session_id: str) -> bool:
        """Archive an inactive session."""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]
        if session.get("status") == "inactive":
            # Move to history
            self.session_history[session_id] = session
            del self.active_sessions[session_id]

            # Clean up user session tracking
            owner_id = session.get("owner_id")
            if owner_id and session_id in self.user_sessions[owner_id]:
                self.user_sessions[owner_id].remove(session_id)

            logger.info(f"Archived session {session_id}")
            return True

        return False

    def get_session_stats(self) -> Dict[str, Any]:
        """Get overall session statistics."""
        total_sessions = len(self.active_sessions)
        total_participants = sum(
            len(session.get("participants", []))
            for session in self.active_sessions.values()
        )
        total_tasks = sum(
            len(session.get("tasks", []))
            for session in self.active_sessions.values()
        )

        # Calculate average session duration
        durations = []
        for session in self.active_sessions.values():
            duration = self._calculate_session_duration(session)
            if duration > 0:
                durations.append(duration)

        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_active_sessions": total_sessions,
            "total_participants": total_participants,
            "total_tasks": total_tasks,
            "average_session_duration_minutes": round(avg_duration, 2),
            "sessions_by_owner": self._get_sessions_by_owner(),
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_session_duration(self, session: Dict[str, Any]) -> float:
        """Calculate session duration in minutes."""
        try:
            created_at = datetime.fromisoformat(session.get("created_at", ""))
            last_activity = datetime.fromisoformat(session.get("last_activity", ""))
            duration = last_activity - created_at
            return duration.total_seconds() / 60
        except (ValueError, TypeError):
            return 0

    def _get_sessions_by_owner(self) -> Dict[str, int]:
        """Get count of sessions by owner."""
        owner_counts = defaultdict(int)
        for session in self.active_sessions.values():
            owner_id = session.get("owner_id", "anonymous")
            owner_counts[owner_id] += 1
        return dict(owner_counts)

    async def cleanup_inactive_sessions(self, max_inactive_hours: int = 24):
        """Clean up sessions that have been inactive for too long."""
        cutoff_time = datetime.now() - timedelta(hours=max_inactive_hours)
        sessions_to_archive = []

        for session_id, session in self.active_sessions.items():
            try:
                last_activity = datetime.fromisoformat(session.get("last_activity", ""))
                if last_activity < cutoff_time:
                    sessions_to_archive.append(session_id)
            except (ValueError, TypeError):
                # If we can't parse the timestamp, assume it's old
                sessions_to_archive.append(session_id)

        for session_id in sessions_to_archive:
            await self.archive_session(session_id)

        if sessions_to_archive:
            logger.info(f"Archived {len(sessions_to_archive)} inactive sessions")

        return len(sessions_to_archive)

    # Enhanced multi-user features
    async def register_user(self, user_id: str, user_profile: Dict[str, Any]) -> bool:
        """Register a new user with profile information."""
        try:
            self.user_profiles[user_id] = {
                **user_profile,
                "registered_at": datetime.now().isoformat(),
                "last_seen": datetime.now().isoformat(),
                "status": "offline"
            }
            logger.info(f"Registered user: {user_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to register user {user_id}: {e}")
            return False

    async def update_user_status(self, user_id: str, status: str) -> bool:
        """Update user online status."""
        if user_id in self.user_profiles:
            self.user_profiles[user_id]["status"] = status
            self.user_profiles[user_id]["last_seen"] = datetime.now().isoformat()

            if status == "online":
                self.online_users.add(user_id)
            else:
                self.online_users.discard(user_id)

            logger.info(f"Updated user {user_id} status to {status}")
            return True
        return False

    def get_online_users(self) -> List[Dict[str, Any]]:
        """Get list of online users."""
        return [
            {
                "user_id": user_id,
                "profile": self.user_profiles.get(user_id, {}),
                "sessions": self.user_sessions.get(user_id, [])
            }
            for user_id in self.online_users
        ]

    def get_user_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user profile information."""
        return self.user_profiles.get(user_id)

    async def assign_user_role(self, session_id: str, user_id: str, role: str) -> bool:
        """Assign a role to a user in a session."""
        if session_id not in self.active_sessions:
            return False

        self.session_roles[session_id][user_id] = role
        logger.info(f"Assigned role '{role}' to user {user_id} in session {session_id}")
        return True

    def get_user_role(self, session_id: str, user_id: str) -> str:
        """Get user's role in a session."""
        return self.session_roles[session_id].get(user_id, "participant")

    async def set_user_permissions(self, session_id: str, user_id: str, permissions: List[str]) -> bool:
        """Set permissions for a user in a session."""
        if session_id not in self.active_sessions:
            return False

        self.user_permissions[user_id][session_id] = permissions
        logger.info(f"Set permissions for user {user_id} in session {session_id}: {permissions}")
        return True

    def get_user_permissions(self, session_id: str, user_id: str) -> List[str]:
        """Get user's permissions in a session."""
        return self.user_permissions[user_id].get(session_id, self._get_default_permissions(user_id, session_id))

    def _get_default_permissions(self, user_id: str, session_id: str) -> List[str]:
        """Get default permissions based on user role."""
        if session_id not in self.active_sessions:
            return []

        session = self.active_sessions[session_id]
        role = self.get_user_role(session_id, user_id)

        # Owner gets all permissions
        if user_id == session.get("owner_id"):
            return ["read", "write", "delete", "manage_users", "manage_settings"]

        # Role-based permissions
        role_permissions = {
            "admin": ["read", "write", "delete", "manage_users"],
            "moderator": ["read", "write", "delete"],
            "contributor": ["read", "write"],
            "viewer": ["read"],
            "participant": ["read", "write"]
        }

        return role_permissions.get(role, ["read"])

    def check_permission(self, session_id: str, user_id: str, permission: str) -> bool:
        """Check if user has a specific permission in a session."""
        permissions = self.get_user_permissions(session_id, user_id)
        return permission in permissions

    async def invite_user_to_session(self, session_id: str, inviter_id: str, invitee_id: str) -> bool:
        """Invite a user to join a session."""
        if session_id not in self.active_sessions:
            return False

        # Check if inviter has permission to invite users
        if not self.check_permission(session_id, inviter_id, "manage_users"):
            logger.warning(f"User {inviter_id} lacks permission to invite users to session {session_id}")
            return False

        session = self.active_sessions[session_id]

        # Add to pending invitations
        if "pending_invitations" not in session:
            session["pending_invitations"] = []

        if invitee_id not in session["pending_invitations"]:
            session["pending_invitations"].append({
                "user_id": invitee_id,
                "invited_by": inviter_id,
                "invited_at": datetime.now().isoformat(),
                "status": "pending"
            })

            logger.info(f"User {inviter_id} invited {invitee_id} to session {session_id}")
            return True

        return False

    async def accept_session_invitation(self, session_id: str, user_id: str) -> bool:
        """Accept a session invitation."""
        if session_id not in self.active_sessions:
            return False

        session = self.active_sessions[session_id]
        pending_invitations = session.get("pending_invitations", [])

        for invitation in pending_invitations:
            if invitation["user_id"] == user_id and invitation["status"] == "pending":
                invitation["status"] = "accepted"
                invitation["accepted_at"] = datetime.now().isoformat()

                # Add user to session
                await self.join_session(session_id, user_id)

                logger.info(f"User {user_id} accepted invitation to session {session_id}")
                return True

        return False

    def get_session_invitations(self, session_id: str) -> List[Dict[str, Any]]:
        """Get pending invitations for a session."""
        if session_id not in self.active_sessions:
            return []

        return self.active_sessions[session_id].get("pending_invitations", [])

    async def kick_user_from_session(self, session_id: str, kicker_id: str, target_user_id: str) -> bool:
        """Remove a user from a session."""
        if session_id not in self.active_sessions:
            return False

        # Check if kicker has permission
        if not self.check_permission(session_id, kicker_id, "manage_users"):
            logger.warning(f"User {kicker_id} lacks permission to kick users from session {session_id}")
            return False

        session = self.active_sessions[session_id]

        # Cannot kick the owner
        if target_user_id == session.get("owner_id"):
            logger.warning(f"Cannot kick session owner {target_user_id}")
            return False

        # Remove user from session
        if target_user_id in session.get("participants", []):
            session["participants"].remove(target_user_id)
            session["last_activity"] = datetime.now().isoformat()

            # Remove from user's sessions
            if session_id in self.user_sessions[target_user_id]:
                self.user_sessions[target_user_id].remove(session_id)

            # Clear permissions and roles
            if session_id in self.user_permissions[target_user_id]:
                del self.user_permissions[target_user_id][session_id]
            if session_id in self.session_roles[session_id]:
                del self.session_roles[session_id][target_user_id]

            logger.info(f"User {kicker_id} kicked {target_user_id} from session {session_id}")
            return True

        return False

    def get_session_participants_detailed(self, session_id: str) -> List[Dict[str, Any]]:
        """Get detailed information about session participants."""
        if session_id not in self.active_sessions:
            return []

        session = self.active_sessions[session_id]
        participants = []

        for user_id in session.get("participants", []):
            user_profile = self.user_profiles.get(user_id, {})
            role = self.get_user_role(session_id, user_id)
            permissions = self.get_user_permissions(session_id, user_id)

            participants.append({
                "user_id": user_id,
                "profile": user_profile,
                "role": role,
                "permissions": permissions,
                "joined_at": session.get("created_at", ""),
                "last_activity": user_profile.get("last_seen", ""),
                "status": user_profile.get("status", "offline")
            })

        return participants

    def get_user_activity_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive activity summary for a user."""
        sessions = self.user_sessions.get(user_id, [])
        total_sessions = len(sessions)
        active_sessions = len([s for s in sessions if s in self.active_sessions])

        # Calculate total time spent
        total_time_minutes = 0
        for session_id in sessions:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                duration = self._calculate_session_duration(session)
                total_time_minutes += duration

        # Get task statistics
        total_tasks = 0
        completed_tasks = 0
        for session_id in sessions:
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                tasks = session.get("tasks", [])
                total_tasks += len(tasks)
                completed_tasks += len([t for t in tasks if t.get("status") == "completed"])

        return {
            "user_id": user_id,
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "total_time_minutes": round(total_time_minutes, 2),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate": round(completed_tasks / total_tasks * 100, 2) if total_tasks > 0 else 0,
            "current_status": self.user_profiles.get(user_id, {}).get("status", "offline"),
            "last_seen": self.user_profiles.get(user_id, {}).get("last_seen", ""),
            "profile": self.user_profiles.get(user_id, {})
        }