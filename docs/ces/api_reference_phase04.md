# CES API Reference - Phase 0.4 Enhanced Features

## Overview

This document provides comprehensive API reference for CES Phase 0.4 enhanced features including web dashboard, collaborative workflows, analytics, feedback system, and plugin architecture.

## Dashboard API

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, no authentication is required for Phase 0.4. Authentication will be added in future phases.

### Endpoints

#### GET /api/status
Get comprehensive system status information.

**Response:**
```json
{
  "overall_status": "healthy",
  "timestamp": "2025-09-01T14:00:00.000Z",
  "components": {
    "system": {
      "cpu_usage": "15.2%",
      "memory_usage": "45.8%",
      "disk_usage": "23.1%"
    },
    "codesage": {
      "status": "healthy",
      "tools_count": 60,
      "last_check": "2025-09-01T14:00:00.000Z"
    },
    "ai_assistants": {
      "grok": {"status": "healthy"},
      "qwen": {"status": "healthy"},
      "gemini": {"status": "healthy"}
    }
  }
}
```

#### GET /api/performance
Get detailed performance metrics.

**Response:**
```json
{
  "cpu_usage": "15.2%",
  "memory_usage": "45.8%",
  "memory_used": "756MB",
  "memory_total": "16384MB",
  "disk_usage": "23.1%",
  "network_sent": "1.2MB",
  "network_recv": "2.8MB",
  "timestamp": "2025-09-01T14:00:00.000Z"
}
```

#### GET /api/tasks
Get list of active tasks.

**Response:**
```json
{
  "tasks": [
    {
      "id": "task_1234567890",
      "description": "Implement user authentication",
      "status": "running",
      "created_at": "2025-09-01T13:45:00.000Z",
      "updated_at": "2025-09-01T13:50:00.000Z"
    }
  ]
}
```

#### POST /api/tasks
Create a new task.

**Request:**
```json
{
  "description": "Implement user authentication system"
}
```

**Response:**
```json
{
  "status": "success",
  "task": {
    "id": "task_1234567890",
    "description": "Implement user authentication system",
    "status": "pending",
    "created_at": "2025-09-01T14:00:00.000Z"
  }
}
```

#### GET /api/ai-assistants
Get AI assistant status information.

**Response:**
```json
{
  "grok": {
    "status": "healthy",
    "last_active": "2025-09-01T14:00:00.000Z",
    "performance_score": 0.95
  },
  "qwen": {
    "status": "healthy",
    "last_active": "2025-09-01T13:58:00.000Z",
    "performance_score": 0.92
  },
  "gemini": {
    "status": "healthy",
    "last_active": "2025-09-01T13:55:00.000Z",
    "performance_score": 0.88
  }
}
```

## Analytics API

### GET /api/analytics/realtime
Get real-time analytics data.

**Response:**
```json
{
  "active_users": 3,
  "current_tasks": 2,
  "system_load": 0.45,
  "total_events_today": 156,
  "timestamp": "2025-09-01T14:00:00.000Z"
}
```

### GET /api/analytics/usage
Get usage analytics report.

**Parameters:**
- `days` (optional): Number of days to analyze (default: 7)

**Response:**
```json
{
  "report_period_days": 7,
  "generated_at": "2025-09-01T14:00:00.000Z",
  "summary": {
    "total_events": 1247,
    "unique_users": 8,
    "average_events_per_user": 155.9,
    "task_success_rate": 0.94
  },
  "event_distribution": {
    "task_execution": 456,
    "user_login": 89,
    "feedback_submitted": 23
  },
  "performance_metrics": {
    "response_time": {
      "average": 0.45,
      "min": 0.12,
      "max": 2.34,
      "count": 456
    }
  },
  "insights": [
    "System operating within normal parameters",
    "Task success rate indicates good assistant matching",
    "Consider optimizing for high-traffic periods"
  ]
}
```

### GET /api/analytics/tasks
Get task analytics report.

**Response:**
```json
{
  "generated_at": "2025-09-01T14:00:00.000Z",
  "task_types": {
    "code_generation": {
      "total_tasks": 234,
      "success_rate": 0.96,
      "average_execution_time": 2.34,
      "assistants_used": ["qwen", "gemini"]
    },
    "analysis": {
      "total_tasks": 156,
      "success_rate": 0.98,
      "average_execution_time": 1.12,
      "assistants_used": ["grok", "gemini"]
    }
  },
  "assistant_performance": {
    "grok": {
      "total_tasks": 145,
      "success_rate": 0.95,
      "average_execution_time": 1.23
    },
    "qwen": {
      "total_tasks": 189,
      "success_rate": 0.97,
      "average_execution_time": 2.45
    }
  }
}
```

### GET /api/analytics/user/{user_id}
Get analytics for a specific user.

**Response:**
```json
{
  "user_id": "user123",
  "total_events": 89,
  "total_tasks": 23,
  "task_success_rate": 0.91,
  "event_distribution": {
    "task_execution": 23,
    "feedback_submitted": 5,
    "user_login": 12
  },
  "first_activity": "2025-08-15T10:30:00.000Z",
  "last_activity": "2025-09-01T14:00:00.000Z",
  "days_active": 17
}
```

## Feedback API

### GET /api/feedback/summary
Get feedback summary and analysis.

**Parameters:**
- `days` (optional): Number of days to analyze (default: 7)

**Response:**
```json
{
  "period_days": 7,
  "total_feedback": 23,
  "average_rating": 4.2,
  "feedback_types": {
    "bug": 8,
    "feature": 12,
    "improvement": 3
  },
  "categories": {
    "ai_assistant": 10,
    "user_interface": 8,
    "performance": 5
  },
  "sentiment": {
    "positive": 0.65,
    "negative": 0.15,
    "neutral": 0.20
  },
  "urgent_issues": [
    "High priority bug: Authentication fails intermittently",
    "Feature request: Add dark mode support"
  ]
}
```

### GET /api/feedback/entries
Get feedback entries with optional filtering.

**Parameters:**
- `status` (optional): Filter by status (new, reviewed, addressed, closed)
- `feedback_type` (optional): Filter by type (bug, feature, improvement, general)
- `limit` (optional): Maximum entries to return (default: 20)

**Response:**
```json
{
  "entries": [
    {
      "id": "feedback_1234567890",
      "user_id": "user123",
      "feedback_type": "feature",
      "title": "Add dark mode support",
      "message": "Please add dark mode support to the dashboard",
      "rating": 4,
      "category": "user_interface",
      "status": "new",
      "priority": "medium",
      "created_at": "2025-09-01T13:45:00.000Z",
      "updated_at": "2025-09-01T13:45:00.000Z"
    }
  ]
}
```

### POST /api/feedback
Submit new user feedback.

**Request:**
```json
{
  "user_id": "user123",
  "type": "feature",
  "title": "Add dark mode support",
  "message": "Please add dark mode support to improve user experience",
  "rating": 4
}
```

**Response:**
```json
{
  "status": "success",
  "feedback_id": "feedback_1234567890",
  "message": "Feedback submitted successfully"
}
```

### POST /api/feedback/{feedback_id}/status
Update feedback status.

**Request:**
```json
{
  "status": "reviewed",
  "reviewed_by": "admin",
  "review_notes": "Reviewed and added to roadmap",
  "resolution": "Planned for next release"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Feedback status updated"
}
```

## WebSocket API

### /ws/dashboard
Real-time dashboard updates.

**Message Format:**
```json
{
  "system_status": {
    "overall_status": "healthy",
    "components": {...}
  },
  "performance_metrics": {
    "cpu_usage": "15.2%",
    "memory_usage": "45.8%",
    ...
  },
  "ai_assistants": {
    "grok": {"status": "healthy"},
    ...
  },
  "active_tasks": [...],
  "recent_activity": [...]
}
```

### /ws/session/{session_id}
Collaborative session real-time updates.

**Message Format:**
```json
{
  "session_id": "session_123",
  "event_type": "task_completed",
  "data": {
    "task_id": "task_456",
    "user_id": "user123",
    "result": {...}
  },
  "timestamp": "2025-09-01T14:00:00.000Z"
}
```

## Plugin API

### Python API

#### Loading Plugins
```python
from ces.plugins.manager import plugin_manager

# Load a plugin
success = plugin_manager.load_plugin("sample_plugin")

# Enable the plugin
success = plugin_manager.enable_plugin("sample_plugin")

# Get plugin information
info = plugin_manager.get_plugin_info("sample_plugin")
```

#### Creating Plugins
```python
from ces.plugins.base import CESPlugin, PluginInfo, TaskExecutionHook

class MyPlugin(CESPlugin):
    def get_info(self) -> PluginInfo:
        return PluginInfo(
            name="My Plugin",
            version="1.0.0",
            description="Custom plugin functionality",
            author="Developer",
            tags=["custom"]
        )

    def initialize(self, context):
        super().initialize(context)

        # Register hooks
        hook = TaskExecutionHook()
        hook.register(self.on_task_execution)
        self.register_hook(hook)

        return True

    def on_task_execution(self, task_description, assistant, result):
        # Custom logic for task execution
        print(f"Task executed: {task_description}")
```

#### Plugin Hooks
```python
# Available hook types
from ces.plugins.base import (
    TaskExecutionHook,
    AIInteractionHook,
    FeedbackSubmissionHook,
    AnalyticsEventHook,
    SystemHealthHook
)

# Register hook handlers
hook = TaskExecutionHook()
hook.register(my_handler_function)
plugin.register_hook(hook)
```

## CLI API

### Task Management
```bash
# Execute task
ces task "Implement authentication" --verbose

# Execute with specific assistant
ces task "Code review" --assistant gemini
```

### AI Assistant Management
```bash
# Analyze task
ces ai analyze "Implement REST API"

# Get AI status
ces ai status

# Get AI performance
ces ai performance
```

### Analytics
```bash
# Usage analytics
ces analytics usage --days 7

# Task analytics
ces analytics tasks

# Real-time metrics
ces analytics realtime

# User analytics
ces analytics user username
```

### Feedback Management
```bash
# Submit feedback
ces feedback submit --type feature --title "Dark mode" --message "Add dark mode"

# List feedback
ces feedback list --type bug --limit 10

# Feedback summary
ces feedback summary --days 30

# Update feedback
ces feedback update feedback_123 --status reviewed --notes "Reviewed"
```

### Plugin Management
```bash
# List plugins
ces plugin list

# Discover plugins
ces plugin discover

# Load plugin
ces plugin load sample_plugin

# Enable plugin
ces plugin enable sample_plugin

# Plugin info
ces plugin info sample_plugin
```

## Error Handling

### HTTP Status Codes
- `200`: Success
- `400`: Bad Request (invalid parameters)
- `404`: Not Found (resource doesn't exist)
- `500`: Internal Server Error

### Error Response Format
```json
{
  "status": "error",
  "error": "Detailed error message",
  "timestamp": "2025-09-01T14:00:00.000Z"
}
```

### WebSocket Error Handling
WebSocket connections automatically reconnect on disconnection with exponential backoff.

## Rate Limiting

### API Limits
- Dashboard API: 100 requests per minute per IP
- Analytics API: 50 requests per minute per IP
- Feedback API: 20 submissions per hour per user

### WebSocket Limits
- Maximum 10 concurrent connections per IP
- Message rate limit: 100 messages per minute per connection

## Data Formats

### Timestamps
All timestamps use ISO 8601 format:
```
2025-09-01T14:00:00.000Z
```

### Pagination
For list endpoints that support pagination:
```json
{
  "items": [...],
  "total": 150,
  "page": 1,
  "per_page": 20,
  "total_pages": 8
}
```

## Security Considerations

### Phase 0.4 Security Features
- Input validation on all endpoints
- CORS configuration for web dashboard
- Basic rate limiting
- No authentication (to be added in future phases)

### Future Security Enhancements
- JWT-based authentication
- API key management
- Role-based access control
- Audit logging
- Data encryption

## Versioning

### API Versioning
- Current API version: v1
- Version specified in URL path: `/api/v1/...`
- Backward compatibility maintained within major versions

### Breaking Changes
- Major version increments for breaking changes
- Deprecation notices provided 2 versions in advance
- Migration guides provided for major updates

## Support and Documentation

### Additional Resources
- **User Guide**: `docs/ces/user_guide.md`
- **Development Guide**: `docs/ces/development_guide.md`
- **Troubleshooting**: `docs/troubleshooting_guide.md`

### Getting Help
- **GitHub Issues**: For bug reports and feature requests
- **API Documentation**: This document
- **Community Support**: GitHub Discussions

---

*This API reference covers CES Phase 0.4 enhanced features. For legacy API endpoints, see the main API reference.*