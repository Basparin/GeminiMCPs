"""CES Web Application.

Main FastAPI application for the CES web interface, providing:
- Real-time dashboard with system monitoring
- AI assistant interaction interface
- Collaborative workspace management
- User session management
- Analytics and insights
"""

import asyncio
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..core.logging_config import setup_logging, get_logger
from ..codesage_integration import CodeSageIntegration
from ..ai_orchestrator.ai_assistant import AIAssistantManager
from ..ai_orchestrator.specialization import AISpecializationManager
from ..analytics.analytics_manager import AnalyticsManager
from ..analytics.advanced_analytics import AdvancedAnalyticsEngine
from ..collaborative.session_manager import SessionManager
from ..collaborative.workflow_manager import WorkflowManager
from ..plugins.plugin_manager import PluginManager
from ..onboarding.onboarding_manager import OnboardingManager
from ..core.error_recovery import ErrorRecoveryManager, ErrorCategory, ErrorSeverity, FailureRecord, FailureType
from ..config.config_manager import ConfigManager

# Setup logging
setup_logging(level="INFO", log_file="logs/ces_web.log", json_format=True)
logger = get_logger(__name__)

# Initialize components
codesage_integration = CodeSageIntegration()
ai_manager = AIAssistantManager()
specialization_manager = AISpecializationManager()
analytics_manager = AnalyticsManager()
advanced_analytics = AdvancedAnalyticsEngine()
session_manager = SessionManager()
workflow_manager = WorkflowManager()
plugin_manager = PluginManager()
onboarding_manager = OnboardingManager()
error_recovery_manager = ErrorRecoveryManager()
config_manager = ConfigManager()

# Create FastAPI app
app = FastAPI(
    title="CES Web Interface",
    description="Cognitive Enhancement System Web Dashboard",
    version="0.4.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup static files and templates
static_path = Path(__file__).parent / "static"
templates_path = Path(__file__).parent / "templates"

if static_path.exists():
    app.mount("/static", StaticFiles(directory=str(static_path)), name="static")

if templates_path.exists():
    templates = Jinja2Templates(directory=str(templates_path))

# WebSocket connection manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        self.monitoring_subscribers: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, user_id: str, session_id: str):
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_sessions[user_id] = {
            "session_id": session_id,
            "connected_at": datetime.now(),
            "last_activity": datetime.now()
        }
        logger.info(f"User {user_id} connected to session {session_id}")

    def disconnect(self, user_id: str):
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        # Also remove from monitoring subscribers
        if user_id in self.monitoring_subscribers:
            del self.monitoring_subscribers[user_id]
        logger.info(f"User {user_id} disconnected")

    async def send_personal_message(self, message: str, user_id: str):
        if user_id in self.active_connections:
            await self.active_connections[user_id].send_text(message)

    async def broadcast(self, message: str, exclude_user: Optional[str] = None):
        for user_id, connection in self.active_connections.items():
            if user_id != exclude_user:
                try:
                    await connection.send_text(message)
                except Exception as e:
                    logger.error(f"Failed to send message to {user_id}: {e}")

    async def subscribe_to_monitoring(self, user_id: str, websocket: WebSocket):
        """Subscribe user to monitoring updates."""
        self.monitoring_subscribers[user_id] = websocket
        logger.info(f"User {user_id} subscribed to monitoring")

    async def unsubscribe_from_monitoring(self, user_id: str):
        """Unsubscribe user from monitoring updates."""
        if user_id in self.monitoring_subscribers:
            del self.monitoring_subscribers[user_id]
            logger.info(f"User {user_id} unsubscribed from monitoring")

    async def broadcast_monitoring_update(self, data: Dict[str, Any]):
        """Broadcast monitoring update to all subscribers."""
        for user_id, websocket in self.monitoring_subscribers.items():
            try:
                await websocket.send_text(json.dumps({
                    "type": "monitoring_update",
                    "data": data,
                    "timestamp": datetime.now().isoformat()
                }))
            except Exception as e:
                logger.error(f"Failed to send monitoring update to {user_id}: {e}")
                # Remove failed connection
                await self.unsubscribe_from_monitoring(user_id)

    def get_active_users(self) -> List[Dict[str, Any]]:
        return [
            {
                "user_id": user_id,
                "session_id": session["session_id"],
                "connected_at": session["connected_at"].isoformat(),
                "last_activity": session["last_activity"].isoformat()
            }
            for user_id, session in self.user_sessions.items()
        ]

manager = ConnectionManager()

# Pydantic models
class TaskRequest(BaseModel):
    description: str
    priority: Optional[str] = "medium"
    tags: Optional[List[str]] = []
    user_id: Optional[str] = "anonymous"

class MessageRequest(BaseModel):
    content: str
    user_id: Optional[str] = "anonymous"
    session_id: Optional[str] = None

class SessionCreateRequest(BaseModel):
    name: str
    description: Optional[str] = ""
    user_id: str
    collaborators: Optional[List[str]] = []

# Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Main dashboard page."""
    try:
        # Get system status
        system_status = await get_system_status()

        # Get recent activity
        recent_activity = await get_recent_activity()

        # Get active sessions
        active_sessions = session_manager.get_active_sessions()

        return templates.TemplateResponse(
            "dashboard.html",
            {
                "request": request,
                "system_status": system_status,
                "recent_activity": recent_activity,
                "active_sessions": active_sessions,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Dashboard error: {e}")
        return templates.TemplateResponse(
            "error.html",
            {"request": request, "error": str(e)}
        )

@app.get("/api/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "0.4.0",
        "services": {
            "codesage": codesage_integration.is_healthy(),
            "ai_manager": ai_manager.is_healthy(),
            "specialization": specialization_manager.is_healthy(),
            "analytics": analytics_manager.is_healthy(),
            "advanced_analytics": advanced_analytics.is_healthy(),
            "plugins": plugin_manager.is_healthy(),
            "sessions": session_manager.is_healthy(),
            "workflows": workflow_manager.is_healthy(),
            "error_recovery": error_recovery_manager.is_healthy()
        }
    }

@app.get("/api/system/status")
async def get_system_status():
    """Get comprehensive system status."""
    try:
        # Get performance metrics from CodeSage
        performance_data = await codesage_integration.get_performance_metrics()

        # Get AI assistant status
        ai_status = ai_manager.get_status()

        # Get analytics summary
        analytics_summary = analytics_manager.get_summary()

        return {
            "performance": performance_data,
            "ai_assistants": ai_status,
            "analytics": analytics_summary,
            "active_users": len(manager.active_connections),
            "active_sessions": len(session_manager.get_active_sessions()),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"System status error: {e}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

@app.get("/api/analytics/overview")
async def get_analytics_overview():
    """Get analytics overview data."""
    try:
        return await analytics_manager.get_overview()
    except Exception as e:
        logger.error(f"Analytics overview error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasks")
async def create_task(task: TaskRequest):
    """Create a new task for AI processing."""
    try:
        task_id = str(uuid.uuid4())

        # Analyze task with cognitive agent (placeholder for now)
        analysis = {
            "complexity_score": 0.5,
            "required_skills": ["general"],
            "estimated_duration": 30,
            "recommended_assistants": ["general_ai"]
        }

        # Create task record
        task_data = {
            "id": task_id,
            "description": task.description,
            "priority": task.priority,
            "tags": task.tags,
            "user_id": task.user_id,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "analysis": analysis
        }

        # Store task
        await session_manager.store_task(task_data)

        # Broadcast to WebSocket clients
        await manager.broadcast(json.dumps({
            "type": "task_created",
            "data": task_data
        }))

        return {"task_id": task_id, "status": "created", "analysis": analysis}

    except Exception as e:
        logger.error(f"Task creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task details."""
    try:
        task = await session_manager.get_task(task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        return task
    except Exception as e:
        logger.error(f"Get task error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions")
async def create_session(session: SessionCreateRequest):
    """Create a new collaborative session."""
    try:
        session_id = str(uuid.uuid4())

        session_data = {
            "id": session_id,
            "name": session.name,
            "description": session.description,
            "owner_id": session.user_id,
            "collaborators": session.collaborators,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "participants": [session.user_id]
        }

        await session_manager.create_session(session_data)

        # Broadcast session creation
        await manager.broadcast(json.dumps({
            "type": "session_created",
            "data": session_data
        }))

        return {"session_id": session_id, "status": "created"}

    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions")
async def get_sessions():
    """Get all active sessions."""
    try:
        return session_manager.get_active_sessions()
    except Exception as e:
        logger.error(f"Get sessions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Enhanced Multi-User Session Management Endpoints
@app.post("/api/users/register")
async def register_user(user_data: Dict[str, Any]):
    """Register a new user."""
    try:
        user_id = user_data.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")

        success = await session_manager.register_user(user_id, user_data)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register user")

        return {"status": "registered", "user_id": user_id}
    except Exception as e:
        logger.error(f"User registration error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/users/{user_id}/status")
async def update_user_status(user_id: str, status_data: Dict[str, Any]):
    """Update user online status."""
    try:
        status = status_data.get("status", "offline")
        success = await session_manager.update_user_status(user_id, status)
        if not success:
            raise HTTPException(status_code=404, detail="User not found")

        return {"status": "updated", "user_id": user_id, "online_status": status}
    except Exception as e:
        logger.error(f"User status update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/online")
async def get_online_users():
    """Get list of online users."""
    try:
        return session_manager.get_online_users()
    except Exception as e:
        logger.error(f"Get online users error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/profile")
async def get_user_profile(user_id: str):
    """Get user profile information."""
    try:
        profile = session_manager.get_user_profile(user_id)
        if not profile:
            raise HTTPException(status_code=404, detail="User not found")
        return profile
    except Exception as e:
        logger.error(f"Get user profile error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/activity")
async def get_user_activity(user_id: str):
    """Get user activity summary."""
    try:
        activity = session_manager.get_user_activity_summary(user_id)
        return activity
    except Exception as e:
        logger.error(f"Get user activity error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions/{session_id}/invite")
async def invite_user_to_session(session_id: str, invitation: Dict[str, Any], inviter_id: str = "anonymous"):
    """Invite a user to a session."""
    try:
        invitee_id = invitation.get("invitee_id")
        if not invitee_id:
            raise HTTPException(status_code=400, detail="Invitee ID required")

        success = await session_manager.invite_user_to_session(session_id, inviter_id, invitee_id)
        if not success:
            raise HTTPException(status_code=403, detail="Not authorized to invite users")

        return {"status": "invited", "session_id": session_id, "invitee_id": invitee_id}
    except Exception as e:
        logger.error(f"Session invitation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions/{session_id}/invitations/{user_id}/accept")
async def accept_session_invitation(session_id: str, user_id: str):
    """Accept a session invitation."""
    try:
        success = await session_manager.accept_session_invitation(session_id, user_id)
        if not success:
            raise HTTPException(status_code=404, detail="Invitation not found")

        return {"status": "accepted", "session_id": session_id, "user_id": user_id}
    except Exception as e:
        logger.error(f"Accept invitation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/invitations")
async def get_session_invitations(session_id: str):
    """Get pending invitations for a session."""
    try:
        invitations = session_manager.get_session_invitations(session_id)
        return {"invitations": invitations}
    except Exception as e:
        logger.error(f"Get session invitations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions/{session_id}/users/{user_id}/kick")
async def kick_user_from_session(session_id: str, user_id: str, kicker_id: str = "anonymous"):
    """Remove a user from a session."""
    try:
        success = await session_manager.kick_user_from_session(session_id, kicker_id, user_id)
        if not success:
            raise HTTPException(status_code=403, detail="Not authorized to kick user")

        return {"status": "kicked", "session_id": session_id, "user_id": user_id}
    except Exception as e:
        logger.error(f"Kick user error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions/{session_id}/users/{user_id}/role")
async def assign_user_role(session_id: str, user_id: str, role_data: Dict[str, Any], assigner_id: str = "anonymous"):
    """Assign a role to a user in a session."""
    try:
        role = role_data.get("role")
        if not role:
            raise HTTPException(status_code=400, detail="Role required")

        success = await session_manager.assign_user_role(session_id, user_id, role)
        if not success:
            raise HTTPException(status_code=404, detail="Session not found")

        return {"status": "assigned", "session_id": session_id, "user_id": user_id, "role": role}
    except Exception as e:
        logger.error(f"Assign role error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/users/{user_id}/permissions")
async def get_user_permissions(session_id: str, user_id: str):
    """Get user's permissions in a session."""
    try:
        permissions = session_manager.get_user_permissions(session_id, user_id)
        return {"permissions": permissions, "session_id": session_id, "user_id": user_id}
    except Exception as e:
        logger.error(f"Get user permissions error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/participants")
async def get_session_participants(session_id: str):
    """Get detailed information about session participants."""
    try:
        participants = session_manager.get_session_participants_detailed(session_id)
        return {"participants": participants, "session_id": session_id}
    except Exception as e:
        logger.error(f"Get session participants error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/sessions/{session_id}/context")
async def get_session_context(session_id: str):
    """Get shared context for a session."""
    try:
        context = session_manager.get_session_context(session_id)
        return {"context": context, "session_id": session_id}
    except Exception as e:
        logger.error(f"Get session context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/sessions/{session_id}/context")
async def update_session_context(session_id: str, context_data: Dict[str, Any], user_id: str = "anonymous"):
    """Update shared context for a session."""
    try:
        context_key = context_data.get("key")
        context_value = context_data.get("value")

        if not context_key:
            raise HTTPException(status_code=400, detail="Context key required")

        await session_manager.update_session_context(session_id, context_key, context_value)
        return {"status": "updated", "session_id": session_id, "key": context_key}
    except Exception as e:
        logger.error(f"Update session context error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# User Onboarding Endpoints
@app.post("/api/onboarding/start")
async def start_user_onboarding(user_data: Dict[str, Any]):
    """Start onboarding process for a user."""
    try:
        user_id = user_data.get("user_id")
        user_profile = user_data.get("profile", {})

        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")

        result = await onboarding_manager.start_user_onboarding(user_id, user_profile)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except Exception as e:
        logger.error(f"Start onboarding error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/onboarding/{user_id}/progress")
async def get_user_onboarding_progress(user_id: str):
    """Get user's onboarding progress."""
    try:
        progress = await onboarding_manager.get_user_progress(user_id)
        if not progress:
            raise HTTPException(status_code=404, detail="User onboarding not found")
        return progress
    except Exception as e:
        logger.error(f"Get onboarding progress error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/onboarding/{user_id}/next-step")
async def get_next_onboarding_step(user_id: str):
    """Get next onboarding step for user."""
    try:
        next_step = await onboarding_manager.get_next_step(user_id)
        if not next_step:
            return {"message": "Onboarding completed", "completed": True}
        return {"step": next_step, "completed": False}
    except Exception as e:
        logger.error(f"Get next step error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/onboarding/{user_id}/complete-step")
async def complete_onboarding_step(user_id: str, completion_data: Dict[str, Any]):
    """Mark an onboarding step as completed."""
    try:
        step_id = completion_data.get("step_id")
        step_data = completion_data.get("data", {})

        if not step_id:
            raise HTTPException(status_code=400, detail="Step ID required")

        result = await onboarding_manager.complete_step(user_id, step_id, step_data)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except Exception as e:
        logger.error(f"Complete step error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/onboarding/tutorials")
async def get_available_tutorials(user_id: Optional[str] = None):
    """Get list of available tutorials."""
    try:
        tutorials = await onboarding_manager.get_available_tutorials(user_id)
        return {"tutorials": tutorials}
    except Exception as e:
        logger.error(f"Get tutorials error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/onboarding/tutorials/{tutorial_id}/start")
async def start_tutorial(tutorial_id: str, start_data: Dict[str, Any]):
    """Start a specific tutorial."""
    try:
        user_id = start_data.get("user_id")
        if not user_id:
            raise HTTPException(status_code=400, detail="User ID required")

        result = await onboarding_manager.start_tutorial(user_id, tutorial_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])

        return result
    except Exception as e:
        logger.error(f"Start tutorial error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/onboarding/tutorials/{tutorial_id}/progress")
async def get_tutorial_progress(tutorial_id: str, user_id: str):
    """Get progress for a specific tutorial."""
    try:
        progress = await onboarding_manager.get_tutorial_progress(user_id, tutorial_id)
        if "error" in progress:
            raise HTTPException(status_code=400, detail=progress["error"])
        return progress
    except Exception as e:
        logger.error(f"Get tutorial progress error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/onboarding/{user_id}/achievements")
async def get_user_achievements(user_id: str):
    """Get user's achievements."""
    try:
        progress = await onboarding_manager.get_user_progress(user_id)
        if not progress:
            return {"achievements": []}

        # Get achievement details
        achievements = []
        for achievement_id in progress.get("achievements", []):
            # This would typically come from the achievement system
            achievement_detail = {
                "id": achievement_id,
                "name": f"Achievement {achievement_id}",
                "description": f"Completed {achievement_id}",
                "unlocked_at": progress.get("last_activity", datetime.now().isoformat())
            }
            achievements.append(achievement_detail)

        return {"achievements": achievements}
    except Exception as e:
        logger.error(f"Get achievements error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/onboarding/{user_id}/recommendations")
async def get_personalized_recommendations(user_id: str):
    """Get personalized learning recommendations."""
    try:
        progress = await onboarding_manager.get_user_progress(user_id)
        tutorials = await onboarding_manager.get_available_tutorials(user_id)

        recommendations = []
        for tutorial in tutorials:
            if tutorial.get("recommended", False):
                recommendations.append({
                    "type": "tutorial",
                    "id": tutorial["id"],
                    "title": tutorial["title"],
                    "description": tutorial["description"],
                    "reason": f"Recommended for your {progress.get('skill_assessment', {}).get('level', 'beginner')} level"
                })

        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Get recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Error Recovery and Self-Healing Endpoints
@app.post("/api/errors/record")
async def record_system_error(error_data: Dict[str, Any]):
    """Record a system error for recovery analysis."""
    try:
        failure_id = str(uuid.uuid4())
        failure = FailureRecord(
            id=failure_id,
            timestamp=datetime.now().isoformat(),
            failure_type=FailureType(error_data.get("failure_type", "internal_error")),
            component=error_data.get("component", "unknown"),
            error_message=error_data.get("error_message", ""),
            context=error_data.get("context", {}),
            severity=error_data.get("severity", "medium")
        )

        recorded_id = error_recovery_manager.record_failure(failure)
        return {"failure_id": recorded_id, "status": "recorded"}
    except Exception as e:
        logger.error(f"Record error failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/errors/statistics")
async def get_error_statistics():
    """Get comprehensive error statistics."""
    try:
        stats = error_recovery_manager.get_failure_statistics()
        return stats
    except Exception as e:
        logger.error(f"Get error statistics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/errors/{component}/recommendations")
async def get_error_recovery_recommendations(component: str):
    """Get recovery recommendations for a component."""
    try:
        recommendations = error_recovery_manager.get_recovery_recommendations(component)
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Get recovery recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/errors/{component}/recover")
async def perform_automatic_recovery(component: str):
    """Perform automatic recovery for a component."""
    try:
        result = await error_recovery_manager.perform_automatic_recovery(component)
        return result
    except Exception as e:
        logger.error(f"Automatic recovery error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/errors/{failure_id}/resolve")
async def resolve_failure(failure_id: str, resolution_data: Dict[str, Any]):
    """Mark a failure as resolved."""
    try:
        notes = resolution_data.get("notes", "")
        success = await error_recovery_manager.resolve_failure(failure_id, notes)
        if not success:
            raise HTTPException(status_code=404, detail="Failure not found")
        return {"status": "resolved", "failure_id": failure_id}
    except Exception as e:
        logger.error(f"Resolve failure error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/errors/circuit-breakers")
async def get_circuit_breaker_status():
    """Get status of all circuit breakers."""
    try:
        cb_status = {}
        for component, cb in error_recovery_manager.circuit_breakers.items():
            cb_status[component] = {
                "state": cb.state,
                "failure_count": cb.failure_count,
                "success_count": cb.success_count,
                "total_requests": cb.total_requests,
                "last_failure_time": cb.last_failure_time,
                "next_retry_time": cb.next_retry_time
            }
        return {"circuit_breakers": cb_status}
    except Exception as e:
        logger.error(f"Get circuit breaker status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/errors/{component}/isolate")
async def isolate_component(component: str, isolation_data: Dict[str, Any]):
    """Isolate a component to prevent cascading failures."""
    try:
        zone = isolation_data.get("zone", "default")
        error_recovery_manager.isolate_component(component, zone)
        return {"status": "isolated", "component": component, "zone": zone}
    except Exception as e:
        logger.error(f"Isolate component error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/errors/{component}/remove-isolation")
async def remove_component_isolation(component: str, isolation_data: Dict[str, Any]):
    """Remove isolation for a component."""
    try:
        zone = isolation_data.get("zone", "default")
        error_recovery_manager.remove_isolation(component, zone)
        return {"status": "isolation_removed", "component": component, "zone": zone}
    except Exception as e:
        logger.error(f"Remove isolation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/errors/health-checks")
async def run_health_checks():
    """Run all configured health checks."""
    try:
        results = await error_recovery_manager.run_health_checks()
        return {"health_checks": results}
    except Exception as e:
        logger.error(f"Run health checks error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/errors/isolation/status")
async def get_isolation_status():
    """Get current isolation status."""
    try:
        isolation_status = {}
        for zone, components in error_recovery_manager.isolation_zones.items():
            isolation_status[zone] = list(components)
        return {"isolation_zones": isolation_status}
    except Exception as e:
        logger.error(f"Get isolation status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/errors/{component}/success")
async def record_component_success(component: str):
    """Record a successful operation for circuit breaker recovery."""
    try:
        error_recovery_manager.record_success(component)
        return {"status": "success_recorded", "component": component}
    except Exception as e:
        logger.error(f"Record success error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/errors/{component}/check-circuit-breaker")
async def check_circuit_breaker(component: str):
    """Check if a component's circuit breaker allows requests."""
    try:
        allowed = await error_recovery_manager.check_circuit_breaker(component)
        return {"component": component, "requests_allowed": allowed}
    except Exception as e:
        logger.error(f"Check circuit breaker error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time communication."""
    session_id = str(uuid.uuid4())
    await manager.connect(websocket, user_id, session_id)

    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)

            # Handle different message types
            if message_data.get("type") == "ping":
                await websocket.send_text(json.dumps({"type": "pong"}))
            elif message_data.get("type") == "task_update":
                # Broadcast task updates
                await manager.broadcast(data, exclude_user=user_id)
            elif message_data.get("type") == "session_join":
                # Handle session joining
                session_id = message_data.get("session_id")
                if session_id:
                    await session_manager.join_session(session_id, user_id)
                    await manager.broadcast(json.dumps({
                        "type": "user_joined",
                        "data": {"user_id": user_id, "session_id": session_id}
                    }))
            elif message_data.get("type") == "subscribe_monitoring":
                # Subscribe to real-time monitoring updates
                await manager.subscribe_to_monitoring(user_id, websocket)
            elif message_data.get("type") == "unsubscribe_monitoring":
                # Unsubscribe from monitoring updates
                await manager.unsubscribe_from_monitoring(user_id)

            # Update last activity
            if user_id in manager.user_sessions:
                manager.user_sessions[user_id]["last_activity"] = datetime.now()

    except WebSocketDisconnect:
        manager.disconnect(user_id)
        await manager.unsubscribe_from_monitoring(user_id)
        await manager.broadcast(json.dumps({
            "type": "user_disconnected",
            "data": {"user_id": user_id}
        }))

# Real-time monitoring endpoints
@app.websocket("/ws/monitoring")
async def monitoring_websocket(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring data."""
    await websocket.accept()

    try:
        # Send initial system status
        system_status = await get_system_status()
        await websocket.send_text(json.dumps({
            "type": "system_status",
            "data": system_status
        }))

        while True:
            # Send periodic updates every 5 seconds
            await asyncio.sleep(5)

            # Get updated system status
            system_status = await get_system_status()
            await websocket.send_text(json.dumps({
                "type": "system_status_update",
                "data": system_status,
                "timestamp": datetime.now().isoformat()
            }))

    except WebSocketDisconnect:
        pass

@app.get("/api/monitoring/realtime/metrics")
async def get_realtime_metrics():
    """Get real-time system metrics."""
    try:
        # Get current performance metrics
        current_metrics = performance_monitor.get_current_metrics()

        # Get system resource usage
        import psutil
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "network_connections": len(psutil.net_connections())
        }

        # Get active connections count
        active_connections = len(manager.active_connections)

        return {
            "performance": current_metrics,
            "system": system_metrics,
            "connections": {
                "active_websocket_connections": active_connections,
                "active_sessions": len(session_manager.get_active_sessions())
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Real-time metrics error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/alerts")
async def get_active_alerts():
    """Get active system alerts."""
    try:
        # Get performance report for alerts
        report = performance_monitor.get_performance_report()

        alerts = []
        if report:
            recent_alerts = report.get("recent_alerts", [])

            for alert in recent_alerts:
                alerts.append({
                    "id": str(uuid.uuid4()),
                    "type": alert.get("type", "unknown"),
                    "severity": alert.get("severity", "info"),
                    "message": alert.get("message", ""),
                    "timestamp": alert.get("timestamp", datetime.now().isoformat()),
                    "resolved": False
                })

        # Add custom alerts based on system status
        system_status = await get_system_status()

        # Memory usage alert
        memory_usage = system_status.get("performance", {}).get("memory_usage_percent", {}).get("value", 0)
        if memory_usage and memory_usage > 85:
            alerts.append({
                "id": str(uuid.uuid4()),
                "type": "resource",
                "severity": "high",
                "message": f"High memory usage: {memory_usage:.1f}%",
                "timestamp": datetime.now().isoformat(),
                "resolved": False
            })

        # CPU usage alert
        cpu_usage = system_status.get("performance", {}).get("cpu_usage_percent", {}).get("value", 0)
        if cpu_usage and cpu_usage > 90:
            alerts.append({
                "id": str(uuid.uuid4()),
                "type": "resource",
                "severity": "critical",
                "message": f"Critical CPU usage: {cpu_usage:.1f}%",
                "timestamp": datetime.now().isoformat(),
                "resolved": False
            })

        return {"alerts": alerts, "total": len(alerts)}
    except Exception as e:
        logger.error(f"Active alerts error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/monitoring/alerts/{alert_id}/resolve")
async def resolve_alert(alert_id: str):
    """Mark an alert as resolved."""
    try:
        # In a real implementation, this would update the alert status in a database
        return {"status": "resolved", "alert_id": alert_id}
    except Exception as e:
        logger.error(f"Resolve alert error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/logs/realtime")
async def get_realtime_logs(lines: int = 50):
    """Get recent log entries in real-time."""
    try:
        # Read recent log entries
        log_file = Path("logs/ces.log")
        if log_file.exists():
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                recent_lines = all_lines[-lines:] if len(all_lines) > lines else all_lines

            log_entries = []
            for line in recent_lines:
                try:
                    # Try to parse JSON log entries
                    log_entry = json.loads(line.strip())
                    log_entries.append(log_entry)
                except json.JSONDecodeError:
                    # Handle plain text log entries
                    log_entries.append({
                        "timestamp": datetime.now().isoformat(),
                        "level": "INFO",
                        "message": line.strip(),
                        "logger": "unknown"
                    })

            return {"logs": log_entries, "total": len(log_entries)}
        else:
            return {"logs": [], "total": 0}

    except Exception as e:
        logger.error(f"Real-time logs error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/monitoring/performance/history")
async def get_performance_history(hours: int = 1):
    """Get historical performance data."""
    try:
        # Get performance metrics for the specified time range
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # Collect metrics from analytics
        response_times = []
        cpu_usage = []
        memory_usage = []

        for metric in analytics_manager.performance_data.get("response_time_ms", []):
            if datetime.fromisoformat(metric["timestamp"]) > cutoff_time:
                response_times.append({
                    "timestamp": metric["timestamp"],
                    "value": metric["value"]
                })

        for metric in analytics_manager.performance_data.get("cpu_usage_percent", []):
            if datetime.fromisoformat(metric["timestamp"]) > cutoff_time:
                cpu_usage.append({
                    "timestamp": metric["timestamp"],
                    "value": metric["value"]
                })

        for metric in analytics_manager.performance_data.get("memory_usage_percent", []):
            if datetime.fromisoformat(metric["timestamp"]) > cutoff_time:
                memory_usage.append({
                    "timestamp": metric["timestamp"],
                    "value": metric["value"]
                })

        return {
            "response_times": response_times,
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "time_range_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Performance history error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/users/active")
async def get_active_users():
    """Get list of active users."""
    return manager.get_active_users()

@app.get("/api/ai/specialization/status")
async def get_ai_specialization_status():
    """Get AI specialization status and recommendations."""
    try:
        status = specialization_manager.get_specialization_report()
        return status
    except Exception as e:
        logger.error(f"AI specialization status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/specialization/analyze")
async def analyze_task_for_specialization(task: TaskRequest):
    """Analyze a task and recommend specialized AI assistants."""
    try:
        # Analyze task requirements
        task_profile = await specialization_manager.analyze_task_requirements(task.description)

        # Get assistant recommendations
        recommendations = await specialization_manager.recommend_assistants(task_profile)

        return {
            "task_profile": {
                "domain": task_profile.domain,
                "complexity": task_profile.complexity,
                "required_skills": task_profile.required_skills,
                "estimated_duration": task_profile.estimated_duration,
                "priority": task_profile.priority
            },
            "recommendations": [
                {
                    "assistant": rec[0],
                    "confidence": rec[1],
                    "factors": rec[2]
                } for rec in recommendations
            ]
        }
    except Exception as e:
        logger.error(f"Task analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/ai/assistants/{assistant_name}/status")
async def get_assistant_status(assistant_name: str):
    """Get detailed status for a specific AI assistant."""
    try:
        status = specialization_manager.get_assistant_status(assistant_name)
        if not status:
            raise HTTPException(status_code=404, detail="Assistant not found")
        return status
    except Exception as e:
        logger.error(f"Assistant status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ai/specialization/update")
async def update_assistant_performance(assistant_name: str, task_result: Dict[str, Any]):
    """Update assistant performance metrics."""
    try:
        await specialization_manager.update_performance_metrics(assistant_name, task_result)
        return {"status": "updated", "assistant": assistant_name}
    except Exception as e:
        logger.error(f"Performance update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Collaborative Workflow Endpoints
@app.post("/api/workflows")
async def create_workflow(workflow: Dict[str, Any]):
    """Create a new collaborative workflow."""
    try:
        workflow_id = await workflow_manager.create_workflow(workflow)
        return {"workflow_id": workflow_id, "status": "created"}
    except Exception as e:
        logger.error(f"Workflow creation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/workflows/{workflow_id}/tasks")
async def add_workflow_task(workflow_id: str, task: Dict[str, Any]):
    """Add a task to a workflow."""
    try:
        task_id = await workflow_manager.add_task_to_workflow(workflow_id, task)
        return {"task_id": task_id, "status": "added"}
    except Exception as e:
        logger.error(f"Add task error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/workflows/{workflow_id}/tasks/{task_id}/status")
async def update_task_status(workflow_id: str, task_id: str, status_update: Dict[str, Any], user_id: str = "anonymous"):
    """Update task status in a workflow."""
    try:
        success = await workflow_manager.update_task_status(workflow_id, task_id, status_update["status"], user_id)
        if not success:
            raise HTTPException(status_code=400, detail="Invalid status update")
        return {"status": "updated"}
    except Exception as e:
        logger.error(f"Task status update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/api/workflows/{workflow_id}/tasks/{task_id}/assign")
async def assign_workflow_task(workflow_id: str, task_id: str, assignment: Dict[str, Any]):
    """Assign a task to a user."""
    try:
        success = await workflow_manager.assign_task(workflow_id, task_id, assignment["user_id"])
        if not success:
            raise HTTPException(status_code=404, detail="Workflow or task not found")
        return {"status": "assigned"}
    except Exception as e:
        logger.error(f"Task assignment error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflows/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get comprehensive workflow status."""
    try:
        status = workflow_manager.get_workflow_status(workflow_id)
        if not status:
            raise HTTPException(status_code=404, detail="Workflow not found")
        return status
    except Exception as e:
        logger.error(f"Workflow status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflows")
async def get_workflows():
    """Get all workflows."""
    try:
        workflows = []
        for workflow_id in workflow_manager.workflows:
            status = workflow_manager.get_workflow_status(workflow_id)
            if status:
                workflows.append(status)
        return workflows
    except Exception as e:
        logger.error(f"Get workflows error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/users/{user_id}/workload")
async def get_user_workload(user_id: str):
    """Get user workload information."""
    try:
        workload = workflow_manager.get_user_workload(user_id)
        return workload
    except Exception as e:
        logger.error(f"User workload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/workflows/{workflow_id}/recommendations")
async def get_workflow_recommendations(workflow_id: str):
    """Get workflow optimization recommendations."""
    try:
        recommendations = await workflow_manager.get_workflow_recommendations(workflow_id)
        return {"recommendations": recommendations}
    except Exception as e:
        logger.error(f"Workflow recommendations error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Analytics Endpoints
@app.get("/api/analytics/advanced/user/{user_id}/behavior")
async def get_user_behavior_analysis(user_id: str):
    """Get advanced user behavior analysis."""
    try:
        # Get user's events from analytics manager
        user_events = []
        for event_list in analytics_manager.usage_data.values():
            user_events.extend([
                event for event in event_list
                if event.get("user_id") == user_id
            ])

        analysis = await advanced_analytics.analyze_user_behavior_patterns(user_id, user_events)
        return analysis
    except Exception as e:
        logger.error(f"User behavior analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/advanced/predict/{metric_name}")
async def predict_performance_trends(metric_name: str, hours: int = 24):
    """Predict performance trends for a metric."""
    try:
        historical_data = analytics_manager.performance_data.get(metric_name, [])
        prediction = await advanced_analytics.predict_performance_trends(
            metric_name, historical_data, hours
        )
        return prediction
    except Exception as e:
        logger.error(f"Performance prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/advanced/bottlenecks")
async def analyze_system_bottlenecks():
    """Analyze system for performance bottlenecks."""
    try:
        # Gather current performance data
        performance_data = {
            "response_times": [m["value"] for m in analytics_manager.performance_data.get("response_time_ms", [])],
            "resource_usage": {
                "cpu_percent": 45.0,  # Mock data - would come from system monitoring
                "memory_percent": 60.0
            },
            "throughput": {
                "current_rps": 25.0,
                "max_capacity_rps": 100.0
            }
        }

        analysis = await advanced_analytics.analyze_system_bottlenecks(performance_data)
        return analysis
    except Exception as e:
        logger.error(f"System bottleneck analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/advanced/segments")
async def get_user_segments():
    """Get user segmentation analysis."""
    try:
        # Get all user data
        user_data = []
        user_ids = set()

        # Collect user IDs from various data sources
        for event_list in analytics_manager.usage_data.values():
            for event in event_list:
                user_ids.add(event.get("user_id", "anonymous"))

        for user_id in user_ids:
            user_insights = await analytics_manager.get_user_insights(user_id)
            if "error" not in user_insights:
                user_data.append(user_insights)

        segments = await advanced_analytics.generate_user_segments(user_data)
        return segments
    except Exception as e:
        logger.error(f"User segmentation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/advanced/anomalies/{metric_name}")
async def detect_anomalies(metric_name: str, sensitivity: float = 0.8):
    """Detect anomalies in a metric."""
    try:
        data_stream = analytics_manager.performance_data.get(metric_name, [])
        anomalies = await advanced_analytics.detect_anomalies(data_stream, sensitivity)
        return anomalies
    except Exception as e:
        logger.error(f"Anomaly detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics/advanced/insights")
async def get_advanced_insights():
    """Get comprehensive advanced analytics insights."""
    try:
        insights = {
            "system_health": await analyze_system_bottlenecks(),
            "user_segments": await get_user_segments(),
            "performance_predictions": {},
            "anomaly_summary": {}
        }

        # Add predictions for key metrics
        key_metrics = ["response_time_ms", "throughput_rps", "error_rate_percent"]
        for metric in key_metrics:
            prediction = await predict_performance_trends(metric, 12)  # 12 hour prediction
            if "error" not in prediction:
                insights["performance_predictions"][metric] = prediction

        # Add anomaly detection for key metrics
        for metric in key_metrics:
            anomalies = await detect_anomalies(metric)
            if "error" not in anomalies:
                insights["anomaly_summary"][metric] = {
                    "anomaly_count": anomalies.get("anomalies_detected", 0),
                    "anomaly_rate": anomalies.get("anomaly_rate_percent", 0)
                }

        return insights
    except Exception as e:
        logger.error(f"Advanced insights error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Plugin Management Endpoints
@app.get("/api/plugins/discover")
async def discover_plugins():
    """Discover available plugins."""
    try:
        plugins = await plugin_manager.discover_plugins()
        return {"plugins": plugins, "count": len(plugins)}
    except Exception as e:
        logger.error(f"Plugin discovery error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plugins/{plugin_name}/load")
async def load_plugin(plugin_name: str):
    """Load a plugin."""
    try:
        success = await plugin_manager.load_plugin(plugin_name)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to load plugin")
        return {"status": "loaded", "plugin": plugin_name}
    except Exception as e:
        logger.error(f"Plugin load error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plugins/{plugin_name}/unload")
async def unload_plugin(plugin_name: str):
    """Unload a plugin."""
    try:
        success = await plugin_manager.unload_plugin(plugin_name)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to unload plugin")
        return {"status": "unloaded", "plugin": plugin_name}
    except Exception as e:
        logger.error(f"Plugin unload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plugins/{plugin_name}/reload")
async def reload_plugin(plugin_name: str):
    """Reload a plugin."""
    try:
        success = await plugin_manager.reload_plugin(plugin_name)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to reload plugin")
        return {"status": "reloaded", "plugin": plugin_name}
    except Exception as e:
        logger.error(f"Plugin reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/plugins/{plugin_name}/status")
async def get_plugin_status(plugin_name: str):
    """Get status of a specific plugin."""
    try:
        status = plugin_manager.get_plugin_status(plugin_name)
        if not status:
            raise HTTPException(status_code=404, detail="Plugin not found")
        return status
    except Exception as e:
        logger.error(f"Plugin status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/plugins")
async def get_all_plugins_status():
    """Get status of all plugins."""
    try:
        return plugin_manager.get_all_plugins_status()
    except Exception as e:
        logger.error(f"Get plugins status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/plugins/capability/{capability}")
async def get_plugins_by_capability(capability: str):
    """Get plugins that provide a specific capability."""
    try:
        plugins = plugin_manager.get_plugins_by_capability(capability)
        return {"capability": capability, "plugins": plugins}
    except Exception as e:
        logger.error(f"Get plugins by capability error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plugins/{plugin_name}/execute")
async def execute_plugin_action(plugin_name: str, action_request: Dict[str, Any]):
    """Execute an action on a plugin."""
    try:
        action = action_request.get("action")
        parameters = action_request.get("parameters", {})

        if not action:
            raise HTTPException(status_code=400, detail="Action parameter required")

        result = await plugin_manager.execute_plugin_action(plugin_name, action, parameters)
        return {"result": result}
    except Exception as e:
        logger.error(f"Plugin action execution error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plugins/{plugin_name}/install")
async def install_plugin(plugin_name: str, install_request: Dict[str, Any]):
    """Install a plugin from archive."""
    try:
        archive_path = install_request.get("archive_path")
        if not archive_path:
            raise HTTPException(status_code=400, detail="Archive path required")

        success = await plugin_manager.install_plugin(archive_path)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to install plugin")
        return {"status": "installed", "plugin": plugin_name}
    except Exception as e:
        logger.error(f"Plugin installation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/plugins/{plugin_name}/uninstall")
async def uninstall_plugin(plugin_name: str):
    """Uninstall a plugin."""
    try:
        success = await plugin_manager.uninstall_plugin(plugin_name)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to uninstall plugin")
        return {"status": "uninstalled", "plugin": plugin_name}
    except Exception as e:
        logger.error(f"Plugin uninstallation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def get_recent_activity() -> List[Dict[str, Any]]:
    """Get recent system activity."""
    try:
        # This would typically come from a database or log analysis
        return [
            {
                "id": "1",
                "type": "task_completed",
                "description": "Code analysis task completed",
                "timestamp": (datetime.now() - timedelta(minutes=5)).isoformat(),
                "user": "user_123"
            },
            {
                "id": "2",
                "type": "ai_response",
                "description": "AI assistant provided code suggestions",
                "timestamp": (datetime.now() - timedelta(minutes=10)).isoformat(),
                "user": "user_456"
            }
        ]
    except Exception as e:
        logger.error(f"Recent activity error: {e}")
        return []

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "timestamp": datetime.now().isoformat()}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "ces.web.app:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )