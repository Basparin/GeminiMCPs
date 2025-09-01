"""
CES Web Dashboard Server

FastAPI-based web server providing real-time monitoring dashboard,
collaborative features, and analytics for the Cognitive Enhancement System.
"""

import logging
import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import psutil
import aiofiles

from ..codesage_integration import CodeSageIntegration
from ..core.cognitive_agent import CognitiveAgent
from ..ai_orchestrator.ai_assistant import AIAssistantManager
from ..config.ces_config import CESConfig
from ..utils.helpers import get_system_info
from ..collaborative.session_manager import session_manager, User, CollaborativeTask, SessionMessage
from ..analytics.analytics_engine import analytics_engine
from ..feedback.feedback_manager import feedback_manager
from ..onboarding.onboarding_manager import onboarding_manager


class CESDashboard:
    """
    CES Web Dashboard with real-time monitoring and collaborative features
    """

    def __init__(self):
        self.app = FastAPI(title="CES Dashboard", version="0.4.0")
        self.logger = logging.getLogger(__name__)

        # Core components
        self.config = CESConfig()
        self.codesage = CodeSageIntegration()
        self.cognitive_agent = CognitiveAgent()
        self.ai_manager = AIAssistantManager()

        # Dashboard state
        self.active_connections: List[WebSocket] = []
        self.dashboard_data = {
            "system_status": {},
            "performance_metrics": {},
            "ai_assistants": {},
            "active_tasks": [],
            "recent_activity": [],
            "user_feedback": []
        }

        # Setup routes and middleware
        self._setup_middleware()
        self._setup_routes()
        self._setup_websocket()

        # Start background monitoring
        self.monitoring_task = None

    def _setup_middleware(self):
        """Setup CORS and other middleware"""
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],  # Configure for production
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    def _setup_routes(self):
        """Setup API routes"""
        @self.app.get("/", response_class=HTMLResponse)
        async def dashboard_home():
            """Serve main dashboard page"""
            return await self._serve_dashboard_html()

        @self.app.get("/api/status")
        async def get_system_status():
            """Get current system status"""
            return await self._get_system_status()

        @self.app.get("/api/performance")
        async def get_performance_metrics():
            """Get performance metrics"""
            return await self._get_performance_metrics()

        @self.app.get("/api/tasks")
        async def get_active_tasks():
            """Get active tasks"""
            return {"tasks": self.dashboard_data["active_tasks"]}

        @self.app.get("/api/ai-assistants")
        async def get_ai_assistants():
            """Get AI assistant status"""
            return await self._get_ai_assistants_status()

        @self.app.get("/api/analytics")
        async def get_analytics():
            """Get usage analytics"""
            return await self._get_analytics_data()

        @self.app.post("/api/tasks")
        async def create_task(request: Request):
            """Create new task"""
            data = await request.json()
            return await self._create_task(data)

        @self.app.post("/api/feedback")
        async def submit_feedback(request: Request):
            """Submit user feedback"""
            data = await request.json()
            return await self._submit_feedback(data)

        # Collaborative session endpoints
        @self.app.post("/api/sessions")
        async def create_session(request: Request):
            """Create a new collaborative session"""
            data = await request.json()
            return await self._create_session(data)

        @self.app.get("/api/sessions")
        async def list_sessions():
            """List all collaborative sessions"""
            return await self._list_sessions()

        @self.app.post("/api/sessions/{session_id}/join")
        async def join_session(session_id: str, request: Request):
            """Join a collaborative session"""
            data = await request.json()
            return await self._join_session(session_id, data)

        @self.app.post("/api/sessions/{session_id}/tasks")
        async def create_collaborative_task(session_id: str, request: Request):
            """Create a task in a collaborative session"""
            data = await request.json()
            return await self._create_collaborative_task(session_id, data)

        @self.app.get("/api/sessions/{session_id}")
        async def get_session_info(session_id: str):
            """Get collaborative session information"""
            return await self._get_session_info(session_id)

        @self.app.websocket("/ws/session/{session_id}")
        async def session_websocket(websocket: WebSocket, session_id: str):
            """WebSocket for collaborative session real-time updates"""
            await self._handle_session_websocket(websocket, session_id)

        # Analytics endpoints
        @self.app.get("/api/analytics/realtime")
        async def get_realtime_analytics():
            """Get real-time analytics data"""
            return analytics_engine.get_real_time_metrics()

        @self.app.get("/api/analytics/usage")
        async def get_usage_analytics(days: int = 7):
            """Get usage analytics report"""
            return analytics_engine.generate_usage_report(days)

        @self.app.get("/api/analytics/tasks")
        async def get_task_analytics():
            """Get task analytics report"""
            return analytics_engine.generate_task_analytics_report()

        @self.app.get("/api/analytics/user/{user_id}")
        async def get_user_analytics(user_id: str):
            """Get analytics for a specific user"""
            return analytics_engine.get_user_analytics(user_id)

        # Feedback endpoints
        @self.app.get("/api/feedback/summary")
        async def get_feedback_summary(days: int = 7):
            """Get feedback summary and analysis"""
            return feedback_manager.get_feedback_summary(days)

        @self.app.get("/api/feedback/entries")
        async def get_feedback_entries(status: Optional[str] = None,
                                     feedback_type: Optional[str] = None,
                                     limit: int = 20):
            """Get feedback entries with optional filtering"""
            return feedback_manager.get_feedback_entries(
                status=status,
                feedback_type=feedback_type,
                limit=limit
            )

        @self.app.post("/api/feedback/{feedback_id}/status")
        async def update_feedback_status(feedback_id: str, request: Request):
            """Update feedback status"""
            data = await request.json()
            success = feedback_manager.update_feedback_status(
                feedback_id=feedback_id,
                status=data.get("status", "reviewed"),
                reviewed_by=data.get("reviewed_by", "system"),
                review_notes=data.get("review_notes"),
                resolution=data.get("resolution")
            )
            return {"status": "success" if success else "error"}

        # Onboarding endpoints
        @self.app.get("/api/onboarding/tutorials")
        async def get_available_tutorials(user_id: str = "default_user",
                                        category: Optional[str] = None):
            """Get available tutorials"""
            return onboarding_manager.get_available_tutorials(user_id, category)

        @self.app.get("/api/onboarding/tutorial/{tutorial_id}")
        async def get_tutorial_details(tutorial_id: str):
            """Get tutorial details"""
            return onboarding_manager.get_tutorial_details(tutorial_id)

        @self.app.post("/api/onboarding/tutorial/{tutorial_id}/start")
        async def start_tutorial(tutorial_id: str, user_id: str = "default_user"):
            """Start a tutorial"""
            tutorial = onboarding_manager.start_tutorial(user_id, tutorial_id)
            return tutorial if tutorial else {"error": "Tutorial not found"}

        @self.app.post("/api/onboarding/tutorial/{tutorial_id}/step/{step_index}/complete")
        async def complete_tutorial_step(tutorial_id: str, step_index: int,
                                       user_id: str = "default_user"):
            """Complete a tutorial step"""
            result = onboarding_manager.complete_tutorial_step(
                user_id, tutorial_id, step_index
            )
            return result

        @self.app.get("/api/onboarding/progress")
        async def get_user_progress(user_id: str = "default_user"):
            """Get user tutorial progress"""
            return onboarding_manager.get_user_progress(user_id)

        @self.app.get("/api/onboarding/status")
        async def get_onboarding_status(user_id: str = "default_user"):
            """Get overall onboarding status"""
            return onboarding_manager.get_onboarding_status(user_id)

        @self.app.post("/api/onboarding/tutorial/{tutorial_id}/reset")
        async def reset_tutorial_progress(tutorial_id: str, user_id: str = "default_user"):
            """Reset tutorial progress"""
            success = onboarding_manager.reset_tutorial_progress(user_id, tutorial_id)
            return {"status": "success" if success else "error"}

        @self.app.get("/api/health")
        async def health_check():
            """Health check endpoint"""
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}

    def _setup_websocket(self):
        """Setup WebSocket for real-time updates"""
        @self.app.websocket("/ws/dashboard")
        async def dashboard_websocket(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)

            try:
                while True:
                    # Send periodic updates
                    await asyncio.sleep(5)  # Update every 5 seconds
                    data = await self._get_realtime_data()
                    await websocket.send_json(data)
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
            except Exception as e:
                self.logger.error(f"WebSocket error: {e}")
                if websocket in self.active_connections:
                    self.active_connections.remove(websocket)

    async def _serve_dashboard_html(self) -> str:
        """Serve the main dashboard HTML page"""
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CES Dashboard - Cognitive Enhancement System</title>
            <style>
                {await self._get_dashboard_css()}
            </style>
        </head>
        <body>
            <div class="dashboard-container">
                <header class="dashboard-header">
                    <h1>🧠 CES Dashboard</h1>
                    <div class="status-indicator" id="system-status">Loading...</div>
                </header>

                <div class="dashboard-grid">
                    <div class="metric-card">
                        <h3>System Performance</h3>
                        <div id="performance-metrics">Loading metrics...</div>
                    </div>

                    <div class="metric-card">
                        <h3>AI Assistants</h3>
                        <div id="ai-assistants">Loading assistants...</div>
                    </div>

                    <div class="metric-card">
                        <h3>Active Tasks</h3>
                        <div id="active-tasks">No active tasks</div>
                    </div>

                    <div class="metric-card">
                        <h3>Recent Activity</h3>
                        <div id="recent-activity">Loading activity...</div>
                    </div>
                </div>

                <div class="task-creation">
                    <h3>Create New Task</h3>
                    <form id="task-form">
                        <input type="text" id="task-description" placeholder="Describe your task..." required>
                        <button type="submit">Create Task</button>
                    </form>
                </div>

                <div class="collaborative-section">
                    <h3>Collaborative Sessions</h3>
                    <div class="session-controls">
                        <button id="create-session-btn">Create Session</button>
                        <button id="join-session-btn">Join Session</button>
                    </div>
                    <div id="sessions-list">Loading sessions...</div>
                    <div id="current-session" style="display: none;">
                        <h4 id="session-title">Current Session</h4>
                        <div id="session-participants">Participants: Loading...</div>
                        <div id="session-tasks">Tasks: Loading...</div>
                        <div id="session-chat">
                            <div id="chat-messages"></div>
                            <form id="chat-form">
                                <input type="text" id="chat-input" placeholder="Type a message..." required>
                                <button type="submit">Send</button>
                            </form>
                        </div>
                    </div>
                </div>

                <div class="analytics-section">
                    <h3>Analytics & Insights</h3>
                    <div class="analytics-controls">
                        <button id="refresh-analytics-btn">Refresh Analytics</button>
                        <select id="analytics-period">
                            <option value="1">Last 24 hours</option>
                            <option value="7" selected>Last 7 days</option>
                            <option value="30">Last 30 days</option>
                        </select>
                    </div>
                    <div id="analytics-dashboard">
                        <div class="analytics-metric">
                            <h4>Real-time Metrics</h4>
                            <div id="realtime-metrics">Loading...</div>
                        </div>
                        <div class="analytics-metric">
                            <h4>Usage Summary</h4>
                            <div id="usage-summary">Loading...</div>
                        </div>
                        <div class="analytics-metric">
                            <h4>Task Analytics</h4>
                            <div id="task-analytics">Loading...</div>
                        </div>
                        <div class="analytics-metric">
                            <h4>System Insights</h4>
                            <div id="system-insights">Loading...</div>
                        </div>
                    </div>
                </div>

                <div class="feedback-section">
                    <h3>User Feedback</h3>
                    <div class="feedback-controls">
                        <button id="load-feedback-btn">Load Recent Feedback</button>
                        <select id="feedback-filter">
                            <option value="">All Types</option>
                            <option value="bug">Bug Reports</option>
                            <option value="feature">Feature Requests</option>
                            <option value="improvement">Improvements</option>
                            <option value="general">General</option>
                        </select>
                    </div>
                    <div id="feedback-summary">Loading feedback summary...</div>
                    <div id="feedback-list">Loading feedback...</div>

                    <div class="feedback-form-section">
                        <h4>Submit New Feedback</h4>
                        <form id="feedback-form">
                            <select id="feedback-type" required>
                                <option value="">Select feedback type...</option>
                                <option value="bug">Bug Report</option>
                                <option value="feature">Feature Request</option>
                                <option value="improvement">Improvement Suggestion</option>
                                <option value="general">General Feedback</option>
                            </select>
                            <input type="text" id="feedback-title" placeholder="Brief title..." required>
                            <textarea id="feedback-message" placeholder="Your feedback..." required></textarea>
                            <div class="rating-section">
                                <label>Rating (optional):</label>
                                <select id="feedback-rating">
                                    <option value="">No rating</option>
                                    <option value="5">⭐⭐⭐⭐⭐ (5/5)</option>
                                    <option value="4">⭐⭐⭐⭐ (4/5)</option>
                                    <option value="3">⭐⭐⭐ (3/5)</option>
                                    <option value="2">⭐⭐ (2/5)</option>
                                    <option value="1">⭐ (1/5)</option>
                                </select>
                            </div>
                            <button type="submit">Submit Feedback</button>
                        </form>
                    </div>
                </div>
    
                <!-- Onboarding Section -->
                <div class="onboarding-section">
                    <h3>🎓 Learning Center</h3>
                    <div class="onboarding-controls">
                        <button id="load-tutorials-btn">Load Tutorials</button>
                        <select id="tutorial-category">
                            <option value="">All Categories</option>
                            <option value="beginner">Beginner</option>
                            <option value="intermediate">Intermediate</option>
                            <option value="advanced">Advanced</option>
                        </select>
                    </div>
                    <div id="onboarding-status">Loading onboarding status...</div>
                    <div id="tutorials-list">Loading tutorials...</div>
                    <div id="tutorial-view" style="display: none;">
                        <div id="tutorial-header"></div>
                        <div id="tutorial-progress"></div>
                        <div id="tutorial-content"></div>
                        <div id="tutorial-navigation"></div>
                    </div>
                </div>
            </div>

            <script>
                {await self._get_dashboard_js()}
            </script>
        </body>
        </html>
        """
        return html_content

    async def _get_dashboard_css(self) -> str:
        """Get dashboard CSS styles"""
        return """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            color: #333;
        }

        .dashboard-container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }

        .dashboard-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .dashboard-header h1 {
            color: #2c3e50;
        }

        .status-indicator {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
        }

        .status-healthy { background: #27ae60; color: white; }
        .status-warning { background: #f39c12; color: white; }
        .status-error { background: #e74c3c; color: white; }

        .dashboard-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .metric-card {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }

        .metric-card h3 {
            color: #2c3e50;
            margin-bottom: 15px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 5px;
        }

        .task-creation, .feedback-section {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .task-creation h3, .feedback-section h3 {
            color: #2c3e50;
            margin-bottom: 15px;
        }

        form {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        input, textarea, select {
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }

        textarea {
            min-height: 80px;
            resize: vertical;
        }

        button {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
            transition: background 0.3s;
        }

        button:hover {
            background: #2980b9;
        }

        .metric-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }

        .metric-value {
            font-weight: bold;
            color: #27ae60;
        }

        .activity-item {
            padding: 8px 0;
            border-bottom: 1px solid #eee;
            font-size: 14px;
        }

        .activity-timestamp {
            color: #7f8c8d;
            font-size: 12px;
        }

        .collaborative-section {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .session-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
        }

        .session-controls button {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }

        .session-controls button:hover {
            background: #2980b9;
        }

        #sessions-list ul {
            list-style: none;
            padding: 0;
        }

        #sessions-list li {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 10px;
            border-bottom: 1px solid #eee;
        }

        #current-session {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #3498db;
        }

        #session-chat {
            margin-top: 15px;
        }

        #chat-messages {
            height: 200px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 10px;
            background: #f9f9f9;
        }

        .chat-message {
            margin-bottom: 5px;
        }

        .system-message {
            color: #7f8c8d;
            font-style: italic;
        }

        #chat-form {
            display: flex;
            gap: 10px;
        }

        #chat-form input {
            flex: 1;
        }

        .participant, .session-task {
            padding: 5px 0;
            border-bottom: 1px solid #eee;
        }

        .analytics-section {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .analytics-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            align-items: center;
        }

        .analytics-controls button {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }

        .analytics-controls button:hover {
            background: #2980b9;
        }

        .analytics-controls select {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }

        #analytics-dashboard {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
        }

        .analytics-metric {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }

        .analytics-metric h4 {
            color: #2c3e50;
            margin-bottom: 10px;
            font-size: 16px;
        }

        .analytics-item {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #eee;
            font-size: 14px;
        }

        .analytics-value {
            font-weight: bold;
            color: #27ae60;
        }

        .insights-list {
            max-height: 150px;
            overflow-y: auto;
        }

        .insight-item {
            padding: 8px;
            margin: 5px 0;
            background: #e8f4fd;
            border-left: 4px solid #3498db;
            border-radius: 4px;
            font-size: 13px;
        }

        .feedback-section {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .feedback-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            align-items: center;
        }

        .feedback-controls button {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }

        .feedback-controls button:hover {
            background: #2980b9;
        }

        .feedback-controls select {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }

        .feedback-summary {
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
        }

        .feedback-item {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }

        .feedback-item.bug { border-left-color: #e74c3c; }
        .feedback-item.feature { border-left-color: #27ae60; }
        .feedback-item.improvement { border-left-color: #f39c12; }
        .feedback-item.general { border-left-color: #95a5a6; }

        .feedback-meta {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #7f8c8d;
            margin-bottom: 8px;
        }

        .feedback-rating {
            color: #f39c12;
            font-size: 14px;
        }

        .feedback-form-section {
            margin-top: 20px;
            padding-top: 20px;
            border-top: 2px solid #ecf0f1;
        }

        .rating-section {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 10px;
        }

        .rating-section label {
            font-weight: bold;
            color: #2c3e50;
        }

        .rating-section select {
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }

        .onboarding-section {
            background: rgba(255, 255, 255, 0.95);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .onboarding-section h3 {
            color: #2c3e50;
            margin-bottom: 15px;
        }

        .onboarding-controls {
            display: flex;
            gap: 10px;
            margin-bottom: 15px;
            align-items: center;
        }

        .onboarding-controls button {
            background: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }

        .onboarding-controls button:hover {
            background: #2980b9;
        }

        .onboarding-controls select {
            padding: 8px;
            border: 1px solid #ddd;
            border-radius: 5px;
            font-size: 14px;
        }

        .onboarding-status {
            background: #e8f4fd;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 15px;
            border-left: 4px solid #3498db;
        }

        .tutorial-item {
            background: #f8f9fa;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
            border-left: 4px solid #3498db;
            cursor: pointer;
            transition: background 0.3s;
        }

        .tutorial-item:hover {
            background: #e9ecef;
        }

        .tutorial-item.completed {
            border-left-color: #27ae60;
            background: #d4edda;
        }

        .tutorial-item.in-progress {
            border-left-color: #f39c12;
            background: #fff3cd;
        }

        .tutorial-meta {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: #7f8c8d;
            margin-top: 8px;
        }

        .tutorial-progress-bar {
            width: 100%;
            height: 8px;
            background: #e9ecef;
            border-radius: 4px;
            margin: 5px 0;
            overflow: hidden;
        }

        .tutorial-progress-fill {
            height: 100%;
            background: #3498db;
            border-radius: 4px;
            transition: width 0.3s;
        }

        .tutorial-view {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            margin-top: 15px;
        }

        .tutorial-step {
            margin-bottom: 20px;
        }

        .tutorial-step h4 {
            color: #2c3e50;
            margin-bottom: 10px;
        }

        .tutorial-navigation {
            display: flex;
            justify-content: space-between;
            margin-top: 20px;
        }

        .tutorial-navigation button {
            background: #3498db;
            color: white;
            border: none;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 14px;
        }

        .tutorial-navigation button:hover {
            background: #2980b9;
        }

        .tutorial-navigation button:disabled {
            background: #bdc3c7;
            cursor: not-allowed;
        }
        """

    async def _get_dashboard_js(self) -> str:
        """Get dashboard JavaScript"""
        return """
        let websocket;
        let reconnectAttempts = 0;
        const maxReconnectAttempts = 5;

        function connectWebSocket() {
            websocket = new WebSocket('ws://' + window.location.host + '/ws/dashboard');

            websocket.onopen = function(event) {
                console.log('WebSocket connected');
                reconnectAttempts = 0;
                updateConnectionStatus('Connected', 'healthy');
            };

            websocket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                updateDashboard(data);
            };

            websocket.onclose = function(event) {
                console.log('WebSocket disconnected');
                updateConnectionStatus('Disconnected', 'error');

                if (reconnectAttempts < maxReconnectAttempts) {
                    reconnectAttempts++;
                    setTimeout(connectWebSocket, 2000);
                }
            };

            websocket.onerror = function(error) {
                console.error('WebSocket error:', error);
                updateConnectionStatus('Connection Error', 'error');
            };
        }

        function updateConnectionStatus(status, statusClass) {
            const indicator = document.getElementById('system-status');
            indicator.textContent = status;
            indicator.className = 'status-indicator status-' + statusClass;
        }

        function updateDashboard(data) {
            // Update system status
            if (data.system_status) {
                updateSystemStatus(data.system_status);
            }

            // Update performance metrics
            if (data.performance_metrics) {
                updatePerformanceMetrics(data.performance_metrics);
            }

            // Update AI assistants
            if (data.ai_assistants) {
                updateAIAssistants(data.ai_assistants);
            }

            // Update active tasks
            if (data.active_tasks) {
                updateActiveTasks(data.active_tasks);
            }

            // Update recent activity
            if (data.recent_activity) {
                updateRecentActivity(data.recent_activity);
            }
        }

        function updateSystemStatus(status) {
            const indicator = document.getElementById('system-status');
            const statusClass = status.overall_status === 'healthy' ? 'healthy' :
                              status.overall_status === 'warning' ? 'warning' : 'error';
            indicator.textContent = status.overall_status.toUpperCase();
            indicator.className = 'status-indicator status-' + statusClass;
        }

        function updatePerformanceMetrics(metrics) {
            const container = document.getElementById('performance-metrics');
            let html = '';

            for (const [key, value] of Object.entries(metrics)) {
                const displayKey = key.replace(/_/g, ' ').replace(/\\b\\w/g, l => l.toUpperCase());
                html += `<div class="metric-item">
                    <span>${displayKey}</span>
                    <span class="metric-value">${value}</span>
                </div>`;
            }

            container.innerHTML = html;
        }

        function updateAIAssistants(assistants) {
            const container = document.getElementById('ai-assistants');
            let html = '';

            for (const [name, status] of Object.entries(assistants)) {
                const statusClass = status.status === 'healthy' ? 'healthy' :
                                  status.status === 'warning' ? 'warning' : 'error';
                html += `<div class="metric-item">
                    <span>${name}</span>
                    <span class="metric-value" style="color: ${statusClass === 'healthy' ? '#27ae60' : statusClass === 'warning' ? '#f39c12' : '#e74c3c'}">${status.status}</span>
                </div>`;
            }

            container.innerHTML = html;
        }

        function updateActiveTasks(tasks) {
            const container = document.getElementById('active-tasks');
            if (tasks.length === 0) {
                container.innerHTML = 'No active tasks';
                return;
            }

            let html = '';
            tasks.forEach(task => {
                html += `<div class="activity-item">
                    <strong>${task.description}</strong><br>
                    <span class="activity-timestamp">Started: ${new Date(task.created_at).toLocaleString()}</span>
                </div>`;
            });

            container.innerHTML = html;
        }

        function updateRecentActivity(activities) {
            const container = document.getElementById('recent-activity');
            if (activities.length === 0) {
                container.innerHTML = 'No recent activity';
                return;
            }

            let html = '';
            activities.slice(0, 10).forEach(activity => {
                html += `<div class="activity-item">
                    ${activity.description}<br>
                    <span class="activity-timestamp">${new Date(activity.timestamp).toLocaleString()}</span>
                </div>`;
            });

            container.innerHTML = html;
        }

        // Form handlers
        document.getElementById('task-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const description = document.getElementById('task-description').value;

            try {
                const response = await fetch('/api/tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ description })
                });

                if (response.ok) {
                    document.getElementById('task-description').value = '';
                    alert('Task created successfully!');
                } else {
                    alert('Failed to create task');
                }
            } catch (error) {
                console.error('Error creating task:', error);
                alert('Error creating task');
            }
        });

        // Feedback functionality
        async function loadFeedback() {
            try {
                const filter = document.getElementById('feedback-filter').value;

                // Load feedback summary
                const summaryResponse = await fetch('/api/feedback/summary');
                if (summaryResponse.ok) {
                    const summary = await summaryResponse.json();
                    displayFeedbackSummary(summary);
                }

                // Load feedback entries
                const entriesResponse = await fetch(`/api/feedback/entries${filter ? `?feedback_type=${filter}` : ''}`);
                if (entriesResponse.ok) {
                    const entries = await entriesResponse.json();
                    displayFeedbackEntries(entries);
                }

            } catch (error) {
                console.error('Error loading feedback:', error);
            }
        }

        function displayFeedbackSummary(summary) {
            const container = document.getElementById('feedback-summary');
            let html = '<div class="feedback-summary">';

            html += `<div class="analytics-item">
                <span>Total Feedback</span>
                <span class="analytics-value">${summary.total_feedback}</span>
            </div>`;

            if (summary.average_rating > 0) {
                html += `<div class="analytics-item">
                    <span>Average Rating</span>
                    <span class="analytics-value">${summary.average_rating.toFixed(1)}/5</span>
                </div>`;
            }

            // Show feedback type distribution
            for (const [type, count] of Object.entries(summary.feedback_types)) {
                html += `<div class="analytics-item">
                    <span>${type.charAt(0).toUpperCase() + type.slice(1)} Reports</span>
                    <span class="analytics-value">${count}</span>
                </div>`;
            }

            html += '</div>';
            container.innerHTML = html;
        }

        function displayFeedbackEntries(entries) {
            const container = document.getElementById('feedback-list');
            if (entries.length === 0) {
                container.innerHTML = 'No feedback entries found.';
                return;
            }

            let html = '';
            entries.forEach(entry => {
                const rating = entry.rating ? `⭐`.repeat(entry.rating) : '';
                html += `<div class="feedback-item ${entry.feedback_type}">
                    <div class="feedback-meta">
                        <span><strong>${entry.title}</strong></span>
                        <span>${entry.feedback_type.toUpperCase()} • ${new Date(entry.created_at).toLocaleDateString()}</span>
                    </div>
                    <p>${entry.message}</p>
                    ${rating ? `<div class="feedback-rating">${rating} (${entry.rating}/5)</div>` : ''}
                    <div style="font-size: 12px; color: #7f8c8d; margin-top: 5px;">
                        Status: ${entry.status} • Priority: ${entry.priority}
                    </div>
                </div>`;
            });

            container.innerHTML = html;
        }

        document.getElementById('load-feedback-btn').addEventListener('click', loadFeedback);
        document.getElementById('feedback-filter').addEventListener('change', loadFeedback);

        document.getElementById('feedback-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            const type = document.getElementById('feedback-type').value;
            const title = document.getElementById('feedback-title').value;
            const message = document.getElementById('feedback-message').value;
            const rating = document.getElementById('feedback-rating').value;

            try {
                const response = await fetch('/api/feedback', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        type,
                        title,
                        message,
                        rating: rating ? parseInt(rating) : null,
                        user_id: 'user_' + Date.now()
                    })
                });

                if (response.ok) {
                    document.getElementById('feedback-type').value = '';
                    document.getElementById('feedback-title').value = '';
                    document.getElementById('feedback-message').value = '';
                    document.getElementById('feedback-rating').value = '';
                    alert('Feedback submitted successfully!');
                    loadFeedback(); // Refresh feedback list
                } else {
                    alert('Failed to submit feedback');
                }
            } catch (error) {
                console.error('Error submitting feedback:', error);
                alert('Error submitting feedback');
            }
        });

        // Collaborative session handlers
        let currentSessionId = null;
        let sessionWebSocket = null;

        document.getElementById('create-session-btn').addEventListener('click', async function() {
            const sessionName = prompt('Enter session name:');
            if (!sessionName) return;

            try {
                const response = await fetch('/api/sessions', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        name: sessionName,
                        created_by: 'user_' + Date.now(),
                        description: 'Collaborative coding session'
                    })
                });

                if (response.ok) {
                    const result = await response.json();
                    joinSession(result.session.id);
                } else {
                    alert('Failed to create session');
                }
            } catch (error) {
                console.error('Error creating session:', error);
                alert('Error creating session');
            }
        });

        document.getElementById('join-session-btn').addEventListener('click', async function() {
            const sessionId = prompt('Enter session ID:');
            if (!sessionId) return;

            joinSession(sessionId);
        });

        document.getElementById('chat-form').addEventListener('submit', async function(e) {
            e.preventDefault();
            if (!currentSessionId || !sessionWebSocket) return;

            const message = document.getElementById('chat-input').value;
            const messageData = {
                type: 'message',
                user_id: 'user_' + Date.now(),
                username: 'Anonymous',
                content: message
            };

            try {
                sessionWebSocket.send(JSON.stringify(messageData));
                document.getElementById('chat-input').value = '';
            } catch (error) {
                console.error('Error sending message:', error);
            }
        });

        async function joinSession(sessionId) {
            try {
                const response = await fetch(`/api/sessions/${sessionId}/join`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        user_id: 'user_' + Date.now(),
                        user_name: 'Anonymous User',
                        user_role: 'participant'
                    })
                });

                if (response.ok) {
                    currentSessionId = sessionId;
                    connectSessionWebSocket(sessionId);
                    loadSessionInfo(sessionId);
                    document.getElementById('current-session').style.display = 'block';
                } else {
                    alert('Failed to join session');
                }
            } catch (error) {
                console.error('Error joining session:', error);
                alert('Error joining session');
            }
        }

        function connectSessionWebSocket(sessionId) {
            if (sessionWebSocket) {
                sessionWebSocket.close();
            }

            sessionWebSocket = new WebSocket(`ws://${window.location.host}/ws/session/${sessionId}`);

            sessionWebSocket.onopen = function(event) {
                console.log('Session WebSocket connected');
            };

            sessionWebSocket.onmessage = function(event) {
                const data = JSON.parse(event.data);
                handleSessionMessage(data);
            };

            sessionWebSocket.onclose = function(event) {
                console.log('Session WebSocket disconnected');
            };

            sessionWebSocket.onerror = function(error) {
                console.error('Session WebSocket error:', error);
            };
        }

        function handleSessionMessage(data) {
            if (data.type === 'message') {
                addChatMessage(data.message);
            } else if (data.type === 'user_joined') {
                addSystemMessage(`${data.user.name} joined the session`);
                updateParticipants(data.user);
            } else if (data.type === 'user_left') {
                addSystemMessage(`${data.username} left the session`);
            } else if (data.type === 'task_created') {
                addTaskToUI(data.task);
            } else if (data.type === 'task_updated') {
                updateTaskInUI(data.task_id, data.updates);
            }
        }

        function addChatMessage(message) {
            const chatMessages = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'chat-message';
            messageDiv.innerHTML = `<strong>${message.username}:</strong> ${message.content}`;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function addSystemMessage(content) {
            const chatMessages = document.getElementById('chat-messages');
            const messageDiv = document.createElement('div');
            messageDiv.className = 'system-message';
            messageDiv.innerHTML = `<em>${content}</em>`;
            chatMessages.appendChild(messageDiv);
            chatMessages.scrollTop = chatMessages.scrollHeight;
        }

        function updateParticipants(user) {
            const participantsDiv = document.getElementById('session-participants');
            const participantDiv = document.createElement('div');
            participantDiv.className = 'participant';
            participantDiv.innerHTML = `${user.name} (${user.role})`;
            participantsDiv.appendChild(participantDiv);
        }

        function addTaskToUI(task) {
            const tasksDiv = document.getElementById('session-tasks');
            const taskDiv = document.createElement('div');
            taskDiv.className = 'session-task';
            taskDiv.innerHTML = `<strong>${task.title}</strong><br>${task.description}`;
            tasksDiv.appendChild(taskDiv);
        }

        function updateTaskInUI(taskId, updates) {
            // Update task in UI - implementation depends on task display
            console.log('Task updated:', taskId, updates);
        }

        async function loadSessionInfo(sessionId) {
            try {
                const response = await fetch(`/api/sessions/${sessionId}`);
                if (response.ok) {
                    const result = await response.json();
                    const session = result.session;

                    document.getElementById('session-title').textContent = session.name;
                    // Update participants and tasks
                    updateSessionUI(session);
                }
            } catch (error) {
                console.error('Error loading session info:', error);
            }
        }

        function updateSessionUI(session) {
            // Update session UI with loaded data
            const participantsDiv = document.getElementById('session-participants');
            participantsDiv.innerHTML = 'Participants: ' + session.participants.map(p => p.name).join(', ');

            const tasksDiv = document.getElementById('session-tasks');
            tasksDiv.innerHTML = 'Tasks: ' + session.tasks.length;
        }

        // Load available sessions on page load
        async function loadSessions() {
            try {
                const response = await fetch('/api/sessions');
                if (response.ok) {
                    const result = await response.json();
                    displaySessions(result.sessions);
                }
            } catch (error) {
                console.error('Error loading sessions:', error);
            }
        }

        function displaySessions(sessions) {
            const sessionsDiv = document.getElementById('sessions-list');
            if (sessions.length === 0) {
                sessionsDiv.innerHTML = 'No active sessions';
                return;
            }

            let html = '<ul>';
            sessions.forEach(session => {
                html += `<li>
                    <strong>${session.name}</strong>
                    <button onclick="joinSession('${session.id}')">Join</button>
                </li>`;
            });
            html += '</ul>';
            sessionsDiv.innerHTML = html;
        }

        // Analytics functions
        async function loadAnalytics() {
            try {
                const period = document.getElementById('analytics-period').value;

                // Load real-time metrics
                const realtimeResponse = await fetch('/api/analytics/realtime');
                if (realtimeResponse.ok) {
                    const realtimeData = await realtimeResponse.json();
                    displayRealtimeMetrics(realtimeData);
                }

                // Load usage analytics
                const usageResponse = await fetch(`/api/analytics/usage?days=${period}`);
                if (usageResponse.ok) {
                    const usageData = await usageResponse.json();
                    displayUsageAnalytics(usageData);
                }

                // Load task analytics
                const taskResponse = await fetch('/api/analytics/tasks');
                if (taskResponse.ok) {
                    const taskData = await taskResponse.json();
                    displayTaskAnalytics(taskData);
                }

            } catch (error) {
                console.error('Error loading analytics:', error);
            }
        }

        function displayRealtimeMetrics(data) {
            const container = document.getElementById('realtime-metrics');
            let html = '';

            html += `<div class="analytics-item">
                <span>Active Users</span>
                <span class="analytics-value">${data.active_users}</span>
            </div>`;

            html += `<div class="analytics-item">
                <span>Current Tasks</span>
                <span class="analytics-value">${data.current_tasks}</span>
            </div>`;

            html += `<div class="analytics-item">
                <span>System Load</span>
                <span class="analytics-value">${(data.system_load * 100).toFixed(1)}%</span>
            </div>`;

            html += `<div class="analytics-item">
                <span>Events Today</span>
                <span class="analytics-value">${data.total_events_today}</span>
            </div>`;

            container.innerHTML = html;
        }

        function displayUsageAnalytics(data) {
            const container = document.getElementById('usage-summary');
            const summary = data.summary;
            let html = '';

            html += `<div class="analytics-item">
                <span>Total Events</span>
                <span class="analytics-value">${summary.total_events}</span>
            </div>`;

            html += `<div class="analytics-item">
                <span>Unique Users</span>
                <span class="analytics-value">${summary.unique_users}</span>
            </div>`;

            html += `<div class="analytics-item">
                <span>Avg Events/User</span>
                <span class="analytics-value">${summary.average_events_per_user.toFixed(1)}</span>
            </div>`;

            html += `<div class="analytics-item">
                <span>Task Success Rate</span>
                <span class="analytics-value">${(summary.task_success_rate * 100).toFixed(1)}%</span>
            </div>`;

            container.innerHTML = html;

            // Display insights
            displaySystemInsights(data.insights);
        }

        function displayTaskAnalytics(data) {
            const container = document.getElementById('task-analytics');
            let html = '';

            // Show task types
            for (const [taskType, stats] of Object.entries(data.task_types)) {
                html += `<div class="analytics-item">
                    <span>${taskType.replace('_', ' ').toUpperCase()}</span>
                    <span class="analytics-value">${stats.total_tasks} tasks</span>
                </div>`;
                html += `<div class="analytics-item">
                    <span>Success Rate</span>
                    <span class="analytics-value">${(stats.success_rate * 100).toFixed(1)}%</span>
                </div>`;
            }

            // Show assistant performance
            html += '<h5>Assistant Performance</h5>';
            for (const [assistant, perf] of Object.entries(data.assistant_performance)) {
                html += `<div class="analytics-item">
                    <span>${assistant.toUpperCase()}</span>
                    <span class="analytics-value">${(perf.success_rate * 100).toFixed(1)}%</span>
                </div>`;
            }

            container.innerHTML = html;
        }

        function displaySystemInsights(insights) {
            const container = document.getElementById('system-insights');
            let html = '<div class="insights-list">';

            insights.forEach(insight => {
                html += `<div class="insight-item">${insight}</div>`;
            });

            html += '</div>';
            container.innerHTML = html;
        }

        // Analytics event handlers
        document.getElementById('refresh-analytics-btn').addEventListener('click', loadAnalytics);
        document.getElementById('analytics-period').addEventListener('change', loadAnalytics);

        // Onboarding functionality
        let currentTutorial = null;
        let currentStepIndex = 0;

        async function loadTutorials() {
            try {
                const category = document.getElementById('tutorial-category').value;
                const response = await fetch(`/api/onboarding/tutorials${category ? `?category=${category}` : ''}`);
                if (response.ok) {
                    const tutorials = await response.json();
                    displayTutorials(tutorials);
                }

                // Load onboarding status
                const statusResponse = await fetch('/api/onboarding/status');
                if (statusResponse.ok) {
                    const status = await statusResponse.json();
                    displayOnboardingStatus(status);
                }

            } catch (error) {
                console.error('Error loading tutorials:', error);
            }
        }

        function displayOnboardingStatus(status) {
            const container = document.getElementById('onboarding-status');
            let html = `<div class="onboarding-status">
                <strong>Level:</strong> ${status.level.charAt(0).toUpperCase() + status.level.slice(1)} |
                <strong>Progress:</strong> ${(status.overall_progress * 100).toFixed(1)}% |
                <strong>Completed:</strong> ${status.completed_tutorials}/${status.total_tutorials} tutorials
                <div class="tutorial-progress-bar">
                    <div class="tutorial-progress-fill" style="width: ${status.overall_progress * 100}%"></div>
                </div>
                <div style="margin-top: 10px;">
                    <strong>Next Steps:</strong>
                    <ul>${status.next_steps.map(step => `<li>${step}</li>`).join('')}</ul>
                </div>
            </div>`;
            container.innerHTML = html;
        }

        function displayTutorials(tutorials) {
            const container = document.getElementById('tutorials-list');
            if (tutorials.length === 0) {
                container.innerHTML = 'No tutorials found.';
                return;
            }

            let html = '';
            tutorials.forEach(tutorial => {
                const statusClass = tutorial.completed ? 'completed' :
                                  tutorial.in_progress ? 'in-progress' : '';
                const statusText = tutorial.completed ? 'Completed' :
                                 tutorial.in_progress ? 'In Progress' : 'Not Started';
                const progressPercent = (tutorial.progress * 100).toFixed(1);

                html += `<div class="tutorial-item ${statusClass}" onclick="startTutorial('${tutorial.id}')">
                    <h4>${tutorial.title}</h4>
                    <p>${tutorial.description}</p>
                    <div class="tutorial-meta">
                        <span>Category: ${tutorial.category} | Duration: ${tutorial.estimated_duration}min</span>
                        <span>Status: ${statusText} (${progressPercent}%)</span>
                    </div>
                    <div class="tutorial-progress-bar">
                        <div class="tutorial-progress-fill" style="width: ${progressPercent}%"></div>
                    </div>
                </div>`;
            });

            container.innerHTML = html;
        }

        async function startTutorial(tutorialId) {
            try {
                const response = await fetch(`/api/onboarding/tutorial/${tutorialId}/start`, {
                    method: 'POST'
                });

                if (response.ok) {
                    currentTutorial = await response.json();
                    currentStepIndex = currentTutorial.current_step;
                    showTutorialView();
                    displayCurrentStep();
                } else {
                    alert('Failed to start tutorial');
                }
            } catch (error) {
                console.error('Error starting tutorial:', error);
                alert('Error starting tutorial');
            }
        }

        function showTutorialView() {
            document.getElementById('tutorials-list').style.display = 'none';
            document.getElementById('tutorial-view').style.display = 'block';
        }

        function hideTutorialView() {
            document.getElementById('tutorial-view').style.display = 'none';
            document.getElementById('tutorials-list').style.display = 'block';
            currentTutorial = null;
            currentStepIndex = 0;
        }

        function displayCurrentStep() {
            if (!currentTutorial || !currentTutorial.steps) return;

            const step = currentTutorial.steps[currentStepIndex];
            if (!step) return;

            // Update header
            document.getElementById('tutorial-header').innerHTML = `
                <h3>${currentTutorial.title}</h3>
                <p>Step ${currentStepIndex + 1} of ${currentTutorial.steps.length}</p>
            `;

            // Update progress
            const progressPercent = ((currentStepIndex + 1) / currentTutorial.steps.length * 100);
            document.getElementById('tutorial-progress').innerHTML = `
                <div class="tutorial-progress-bar">
                    <div class="tutorial-progress-fill" style="width: ${progressPercent}%"></div>
                </div>
            `;

            // Update content
            document.getElementById('tutorial-content').innerHTML = `
                <div class="tutorial-step">
                    <h4>${step.title}</h4>
                    <p>${step.description}</p>
                    <div style="background: #f8f9fa; padding: 15px; border-radius: 8px; margin: 10px 0;">
                        ${step.content}
                    </div>
                    ${step.hints.length > 0 ? `
                        <div style="margin-top: 10px;">
                            <strong>Hints:</strong>
                            <ul>${step.hints.map(hint => `<li>${hint}</li>`).join('')}</ul>
                        </div>
                    ` : ''}
                </div>
            `;

            // Update navigation
            const prevDisabled = currentStepIndex === 0;
            const nextText = currentStepIndex === currentTutorial.steps.length - 1 ? 'Complete Tutorial' : 'Next Step';
            const nextDisabled = false; // Allow user to proceed

            document.getElementById('tutorial-navigation').innerHTML = `
                <button onclick="hideTutorialView()" style="background: #95a5a6;">Exit Tutorial</button>
                <div>
                    <button onclick="previousStep()" ${prevDisabled ? 'disabled' : ''} style="margin-right: 10px;">Previous</button>
                    <button onclick="nextStep()" ${nextDisabled ? 'disabled' : ''}>${nextText}</button>
                </div>
            `;
        }

        async function nextStep() {
            if (!currentTutorial) return;

            if (currentStepIndex < currentTutorial.steps.length - 1) {
                // Mark current step as completed
                await completeCurrentStep();

                // Move to next step
                currentStepIndex++;
                displayCurrentStep();
            } else {
                // Complete tutorial
                await completeCurrentStep();
                alert('Tutorial completed! 🎉');
                hideTutorialView();
                loadTutorials(); // Refresh tutorial list
            }
        }

        function previousStep() {
            if (currentStepIndex > 0) {
                currentStepIndex--;
                displayCurrentStep();
            }
        }

        async function completeCurrentStep() {
            if (!currentTutorial) return;

            try {
                const response = await fetch(`/api/onboarding/tutorial/${currentTutorial.id}/step/${currentStepIndex}/complete`, {
                    method: 'POST'
                });

                if (!response.ok) {
                    console.error('Failed to complete step');
                }
            } catch (error) {
                console.error('Error completing step:', error);
            }
        }

        // Event listeners
        document.getElementById('load-tutorials-btn').addEventListener('click', loadTutorials);
        document.getElementById('tutorial-category').addEventListener('change', loadTutorials);

        // Initialize WebSocket connection
        connectWebSocket();
        loadSessions();
        loadAnalytics();
        loadTutorials();
        """

    async def _get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        try:
            # Get system information
            system_info = get_system_info()

            # Get CodeSage status
            codesage_status = await self.codesage.health_check()

            # Get AI assistants status
            ai_status = await self._get_ai_assistants_status()

            # Determine overall status
            all_healthy = (
                codesage_status.get("overall_status") == "healthy" and
                all(assistant.get("status") == "healthy" for assistant in ai_status.values())
            )

            status = {
                "overall_status": "healthy" if all_healthy else "warning",
                "timestamp": datetime.now().isoformat(),
                "components": {
                    "system": system_info,
                    "codesage": codesage_status,
                    "ai_assistants": ai_status
                }
            }

            self.dashboard_data["system_status"] = status
            return status

        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {
                "overall_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        try:
            # CPU and memory usage
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Network I/O
            net_io = psutil.net_io_counters()

            metrics = {
                "cpu_usage": f"{cpu_percent:.1f}%",
                "memory_usage": f"{memory.percent:.1f}%",
                "memory_used": f"{memory.used / 1024 / 1024:.0f}MB",
                "memory_total": f"{memory.total / 1024 / 1024:.0f}MB",
                "disk_usage": f"{disk.percent:.1f}%",
                "disk_used": f"{disk.used / 1024 / 1024 / 1024:.1f}GB",
                "disk_total": f"{disk.total / 1024 / 1024 / 1024:.1f}GB",
                "network_sent": f"{net_io.bytes_sent / 1024 / 1024:.1f}MB",
                "network_recv": f"{net_io.bytes_recv / 1024 / 1024:.1f}MB",
                "timestamp": datetime.now().isoformat()
            }

            self.dashboard_data["performance_metrics"] = metrics
            return metrics

        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

    async def _get_ai_assistants_status(self) -> Dict[str, Any]:
        """Get AI assistants status"""
        try:
            # This would integrate with actual AI assistant monitoring
            # For now, return mock data
            assistants = {
                "grok": {"status": "healthy", "last_active": datetime.now().isoformat()},
                "qwen": {"status": "healthy", "last_active": datetime.now().isoformat()},
                "gemini": {"status": "healthy", "last_active": datetime.now().isoformat()}
            }

            self.dashboard_data["ai_assistants"] = assistants
            return assistants

        except Exception as e:
            self.logger.error(f"Error getting AI assistants status: {e}")
            return {"error": str(e)}

    async def _get_analytics_data(self) -> Dict[str, Any]:
        """Get analytics data"""
        try:
            # Mock analytics data - would be replaced with real analytics
            analytics = {
                "total_tasks": len(self.dashboard_data["active_tasks"]),
                "completed_tasks": 0,  # Would track completed tasks
                "user_sessions": 1,  # Would track user sessions
                "avg_response_time": "500ms",  # Would calculate from actual data
                "feedback_count": len(self.dashboard_data["user_feedback"]),
                "timestamp": datetime.now().isoformat()
            }

            return analytics

        except Exception as e:
            self.logger.error(f"Error getting analytics data: {e}")
            return {"error": str(e)}

    async def _get_realtime_data(self) -> Dict[str, Any]:
        """Get real-time dashboard data for WebSocket updates"""
        try:
            # Update all dashboard data
            await self._get_system_status()
            await self._get_performance_metrics()
            await self._get_ai_assistants_status()

            return self.dashboard_data

        except Exception as e:
            self.logger.error(f"Error getting realtime data: {e}")
            return {"error": str(e)}

    async def _create_task(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new task"""
        try:
            task = {
                "id": f"task_{int(datetime.now().timestamp())}",
                "description": task_data["description"],
                "status": "pending",
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }

            self.dashboard_data["active_tasks"].append(task)

            # Add to recent activity
            self.dashboard_data["recent_activity"].insert(0, {
                "description": f"New task created: {task['description']}",
                "timestamp": task["created_at"],
                "type": "task_created"
            })

            # Broadcast update to all connected clients
            await self._broadcast_update()

            return {"status": "success", "task": task}

        except Exception as e:
            self.logger.error(f"Error creating task: {e}")
            return {"status": "error", "error": str(e)}

    async def _submit_feedback(self, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Submit user feedback using the feedback manager"""
        try:
            user_id = feedback_data.get("user_id", "anonymous")
            feedback_type = feedback_data.get("type", "general")
            title = feedback_data.get("title", f"{feedback_type.title()} Feedback")
            message = feedback_data.get("message", "")
            rating = feedback_data.get("rating")

            # Submit feedback using the feedback manager
            feedback_id = feedback_manager.submit_feedback(
                user_id=user_id,
                feedback_type=feedback_type,
                title=title,
                message=message,
                rating=rating
            )

            # Also track as usage event for analytics
            analytics_engine.track_usage_event(
                "feedback_submitted",
                user_id=user_id,
                metadata={
                    "feedback_type": feedback_type,
                    "feedback_id": feedback_id,
                    "rating": rating
                }
            )

            # Add to recent activity
            self.dashboard_data["recent_activity"].insert(0, {
                "description": f"New {feedback_type} feedback submitted",
                "timestamp": datetime.now().isoformat(),
                "type": "feedback_submitted"
            })

            # Broadcast update to all connected clients
            await self._broadcast_update()

            return {
                "status": "success",
                "feedback_id": feedback_id,
                "message": "Feedback submitted successfully"
            }

        except Exception as e:
            self.logger.error(f"Error submitting feedback: {e}")
            return {"status": "error", "error": str(e)}

    async def _broadcast_update(self):
        """Broadcast dashboard update to all connected WebSocket clients"""
        if not self.active_connections:
            return

        try:
            data = await self._get_realtime_data()
            for connection in self.active_connections:
                try:
                    await connection.send_json(data)
                except Exception as e:
                    self.logger.error(f"Error broadcasting to client: {e}")
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)
        except Exception as e:
            self.logger.error(f"Error broadcasting update: {e}")

    # Collaborative session handlers
    async def _create_session(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new collaborative session"""
        try:
            name = data.get("name", "New Session")
            created_by = data.get("created_by", "anonymous")
            description = data.get("description")
            max_participants = data.get("max_participants", 10)

            session = await session_manager.create_session(
                name=name,
                created_by=created_by,
                description=description,
                max_participants=max_participants
            )

            return {
                "status": "success",
                "session": {
                    "id": session.id,
                    "name": session.name,
                    "description": session.description,
                    "created_by": session.created_by,
                    "created_at": session.created_at.isoformat()
                }
            }
        except Exception as e:
            self.logger.error(f"Error creating session: {e}")
            return {"status": "error", "error": str(e)}

    async def _list_sessions(self) -> Dict[str, Any]:
        """List all collaborative sessions"""
        try:
            sessions = await session_manager.list_sessions()
            return {"status": "success", "sessions": sessions}
        except Exception as e:
            self.logger.error(f"Error listing sessions: {e}")
            return {"status": "error", "error": str(e)}

    async def _join_session(self, session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Join a collaborative session"""
        try:
            user_id = data.get("user_id", "anonymous")
            user_name = data.get("user_name", f"User_{user_id}")
            user_email = data.get("user_email")
            user_role = data.get("user_role", "participant")

            user = User(
                id=user_id,
                name=user_name,
                email=user_email,
                role=user_role
            )

            success = await session_manager.join_session(session_id, user)

            if success:
                return {"status": "success", "message": "Joined session successfully"}
            else:
                return {"status": "error", "error": "Failed to join session"}
        except Exception as e:
            self.logger.error(f"Error joining session: {e}")
            return {"status": "error", "error": str(e)}

    async def _create_collaborative_task(self, session_id: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a task in a collaborative session"""
        try:
            task = CollaborativeTask(
                id="",  # Will be generated
                title=data.get("title", "New Task"),
                description=data.get("description", ""),
                priority=data.get("priority", "medium"),
                assignee=data.get("assignee"),
                created_by=data.get("created_by", "anonymous"),
                tags=data.get("tags", []),
                dependencies=data.get("dependencies", [])
            )

            task_id = await session_manager.create_task(session_id, task)

            if task_id:
                return {"status": "success", "task_id": task_id}
            else:
                return {"status": "error", "error": "Failed to create task"}
        except Exception as e:
            self.logger.error(f"Error creating collaborative task: {e}")
            return {"status": "error", "error": str(e)}

    async def _get_session_info(self, session_id: str) -> Dict[str, Any]:
        """Get collaborative session information"""
        try:
            session_info = await session_manager.get_session_info(session_id)

            if session_info:
                return {"status": "success", "session": session_info}
            else:
                return {"status": "error", "error": "Session not found"}
        except Exception as e:
            self.logger.error(f"Error getting session info: {e}")
            return {"status": "error", "error": str(e)}

    async def _handle_session_websocket(self, websocket: WebSocket, session_id: str):
        """Handle WebSocket connections for collaborative sessions"""
        await websocket.accept()

        # Register connection with session manager
        await session_manager.register_connection(session_id, websocket)

        try:
            while True:
                # Receive message from client
                data = await websocket.receive_json()

                message_type = data.get("type")

                if message_type == "join":
                    # User joining via WebSocket
                    user_data = data.get("user", {})
                    user = User(**user_data)
                    await session_manager.join_session(session_id, user)

                elif message_type == "leave":
                    # User leaving
                    user_id = data.get("user_id")
                    await session_manager.leave_session(session_id, user_id)

                elif message_type == "message":
                    # Chat message
                    message = SessionMessage(
                        id="",
                        user_id=data.get("user_id", "anonymous"),
                        username=data.get("username", "Anonymous"),
                        message_type="chat",
                        content=data.get("content", "")
                    )
                    await session_manager.send_message(session_id, message)

                elif message_type == "task_update":
                    # Task update
                    task_id = data.get("task_id")
                    updates = data.get("updates", {})
                    user_id = data.get("user_id", "anonymous")
                    await session_manager.update_task(session_id, task_id, updates, user_id)

        except Exception as e:
            self.logger.error(f"Session WebSocket error: {e}")
        finally:
            # Unregister connection
            await session_manager.unregister_connection(session_id, websocket)

    async def start_monitoring(self):
        """Start background monitoring task"""
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())

    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                # Update dashboard data periodically
                await self._get_system_status()
                await self._get_performance_metrics()

                # Broadcast updates to connected clients
                await self._broadcast_update()

                # Wait before next update
                await asyncio.sleep(30)  # Update every 30 seconds

            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying

    async def run_async(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the dashboard server asynchronously"""
        self.logger.info(f"Starting CES Dashboard on {host}:{port}")

        # Start monitoring task
        await self.start_monitoring()

        # Configure uvicorn
        config = uvicorn.Config(
            self.app,
            host=host,
            port=port,
            log_level="info"
        )

        server = uvicorn.Server(config)

        # Run the server
        await server.serve()

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Run the dashboard server"""
        asyncio.run(self.run_async(host, port))


# Global dashboard instance
dashboard = CESDashboard()


if __name__ == "__main__":
    dashboard.run()