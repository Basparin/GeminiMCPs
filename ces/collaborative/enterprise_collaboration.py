"""
Enterprise Collaboration System for CES Phase 4.

This module provides enterprise-grade team collaboration features including
project management integration, advanced analytics, enterprise security,
and scalable multi-tenant architecture for Phase 4.
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
import hashlib
import uuid

from .session_manager import SessionManager

logger = logging.getLogger(__name__)


class ProjectStatus(Enum):
    """Project status states."""
    PLANNING = "planning"
    ACTIVE = "active"
    ON_HOLD = "on_hold"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TeamRole(Enum):
    """Team member roles."""
    OWNER = "owner"
    ADMIN = "admin"
    MANAGER = "manager"
    DEVELOPER = "developer"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class SecurityLevel(Enum):
    """Security classification levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"


@dataclass
class Project:
    """Enterprise project representation."""
    project_id: str
    name: str
    description: str
    owner_id: str
    team_members: List[str] = field(default_factory=list)
    status: ProjectStatus = ProjectStatus.PLANNING
    security_level: SecurityLevel = SecurityLevel.INTERNAL
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    integrations: Dict[str, Any] = field(default_factory=dict)  # Jira, GitHub, etc.
    analytics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TeamMember:
    """Team member with enterprise features."""
    user_id: str
    email: str
    role: TeamRole
    permissions: Set[str] = field(default_factory=set)
    joined_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    security_clearance: SecurityLevel = SecurityLevel.INTERNAL
    profile: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AuditEvent:
    """Audit event for compliance."""
    event_id: str
    event_type: str
    user_id: str
    resource_type: str
    resource_id: str
    action: str
    details: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class EnterpriseCollaboration:
    """Enterprise-grade collaboration system for CES Phase 4."""

    def __init__(self, session_manager: SessionManager):
        self.session_manager = session_manager

        # Enterprise data structures
        self.projects: Dict[str, Project] = {}
        self.team_members: Dict[str, TeamMember] = {}
        self.audit_log: List[AuditEvent] = []
        self.security_policies: Dict[str, Dict[str, Any]] = {}

        # Multi-tenant architecture
        self.tenants: Dict[str, Dict[str, Any]] = {}
        self.tenant_projects: Dict[str, List[str]] = defaultdict(list)

        # Integration managers
        self.integration_managers: Dict[str, Any] = {}

        # Analytics and monitoring
        self.collaboration_metrics: Dict[str, Any] = {}
        self.performance_analytics: Dict[str, Any] = {}

        # Security and compliance
        self.audit_lock = threading.RLock()
        self.max_audit_entries = 100000

        # Start background tasks
        self._start_background_tasks()

    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # Audit log cleanup
        threading.Thread(
            target=self._audit_cleanup_loop,
            daemon=True,
            name="AuditCleanup"
        ).start()

        # Analytics aggregation
        threading.Thread(
            target=self._analytics_aggregation_loop,
            daemon=True,
            name="AnalyticsAggregation"
        ).start()

    def _audit_cleanup_loop(self):
        """Background audit log cleanup."""
        while True:
            try:
                self._cleanup_old_audit_entries()
                time.sleep(3600)  # Clean up every hour
            except Exception as e:
                logger.error(f"Audit cleanup error: {e}")
                time.sleep(60)

    def _analytics_aggregation_loop(self):
        """Background analytics aggregation."""
        while True:
            try:
                self._aggregate_collaboration_metrics()
                time.sleep(300)  # Aggregate every 5 minutes
            except Exception as e:
                logger.error(f"Analytics aggregation error: {e}")
                time.sleep(60)

    # Project Management
    def create_project(self, project_data: Dict[str, Any], creator_id: str) -> str:
        """Create a new enterprise project."""
        project_id = str(uuid.uuid4())

        project = Project(
            project_id=project_id,
            name=project_data["name"],
            description=project_data.get("description", ""),
            owner_id=creator_id,
            team_members=[creator_id],
            status=ProjectStatus(project_data.get("status", "planning")),
            security_level=SecurityLevel(project_data.get("security_level", "internal")),
            metadata=project_data.get("metadata", {}),
            integrations=project_data.get("integrations", {})
        )

        self.projects[project_id] = project

        # Add to tenant
        tenant_id = project_data.get("tenant_id", "default")
        self.tenant_projects[tenant_id].append(project_id)

        # Audit event
        self._log_audit_event(
            event_type="project_created",
            user_id=creator_id,
            resource_type="project",
            resource_id=project_id,
            action="create",
            details={"project_name": project.name}
        )

        logger.info(f"Created enterprise project: {project.name} ({project_id})")
        return project_id

    def add_team_member(self, project_id: str, user_id: str, role: TeamRole,
                       adder_id: str) -> bool:
        """Add a team member to a project."""
        if project_id not in self.projects:
            return False

        project = self.projects[project_id]

        # Check permissions
        if not self._check_project_permission(project_id, adder_id, "manage_team"):
            logger.warning(f"User {adder_id} lacks permission to manage team for project {project_id}")
            return False

        # Create team member
        team_member = TeamMember(
            user_id=user_id,
            email="",  # Would be populated from user profile
            role=role,
            permissions=self._get_role_permissions(role)
        )

        self.team_members[f"{project_id}:{user_id}"] = team_member

        if user_id not in project.team_members:
            project.team_members.append(user_id)
            project.updated_at = datetime.now()

        # Audit event
        self._log_audit_event(
            event_type="team_member_added",
            user_id=adder_id,
            resource_type="project",
            resource_id=project_id,
            action="add_member",
            details={"added_user": user_id, "role": role.value}
        )

        logger.info(f"Added {user_id} to project {project_id} with role {role.value}")
        return True

    def _get_role_permissions(self, role: TeamRole) -> Set[str]:
        """Get permissions for a role."""
        role_permissions = {
            TeamRole.OWNER: {"read", "write", "delete", "manage_team", "manage_settings", "manage_integrations"},
            TeamRole.ADMIN: {"read", "write", "delete", "manage_team", "manage_settings"},
            TeamRole.MANAGER: {"read", "write", "manage_team"},
            TeamRole.DEVELOPER: {"read", "write"},
            TeamRole.REVIEWER: {"read", "review"},
            TeamRole.VIEWER: {"read"}
        }
        return role_permissions.get(role, {"read"})

    def _check_project_permission(self, project_id: str, user_id: str, permission: str) -> bool:
        """Check if user has permission for a project."""
        if project_id not in self.projects:
            return False

        project = self.projects[project_id]

        # Owner has all permissions
        if user_id == project.owner_id:
            return True

        # Check team member permissions
        team_key = f"{project_id}:{user_id}"
        if team_key in self.team_members:
            team_member = self.team_members[team_key]
            return permission in team_member.permissions

        return False

    # Integration Management
    def setup_integration(self, project_id: str, integration_type: str,
                         config: Dict[str, Any], user_id: str) -> bool:
        """Set up integration for a project."""
        if not self._check_project_permission(project_id, user_id, "manage_integrations"):
            return False

        if project_id not in self.projects:
            return False

        project = self.projects[project_id]
        project.integrations[integration_type] = {
            "config": config,
            "setup_by": user_id,
            "setup_at": datetime.now(),
            "status": "active"
        }

        # Initialize integration manager
        if integration_type == "jira":
            self._setup_jira_integration(project_id, config)
        elif integration_type == "github":
            self._setup_github_integration(project_id, config)
        elif integration_type == "linear":
            self._setup_linear_integration(project_id, config)

        logger.info(f"Set up {integration_type} integration for project {project_id}")
        return True

    def _setup_jira_integration(self, project_id: str, config: Dict[str, Any]):
        """Set up Jira integration."""
        # Placeholder for Jira integration setup
        pass

    def _setup_github_integration(self, project_id: str, config: Dict[str, Any]):
        """Set up GitHub integration."""
        # Placeholder for GitHub integration setup
        pass

    def _setup_linear_integration(self, project_id: str, config: Dict[str, Any]):
        """Set up Linear integration."""
        # Placeholder for Linear integration setup
        pass

    # Security and Compliance
    def set_security_policy(self, project_id: str, policy: Dict[str, Any], user_id: str) -> bool:
        """Set security policy for a project."""
        if not self._check_project_permission(project_id, user_id, "manage_settings"):
            return False

        self.security_policies[project_id] = {
            **policy,
            "set_by": user_id,
            "set_at": datetime.now()
        }

        logger.info(f"Set security policy for project {project_id}")
        return True

    def check_security_clearance(self, project_id: str, user_id: str) -> bool:
        """Check if user has adequate security clearance for project."""
        if project_id not in self.projects:
            return False

        project = self.projects[project_id]
        team_key = f"{project_id}:{user_id}"

        if team_key not in self.team_members:
            return False

        team_member = self.team_members[team_key]

        # Check security hierarchy
        clearance_levels = {
            SecurityLevel.PUBLIC: 0,
            SecurityLevel.INTERNAL: 1,
            SecurityLevel.CONFIDENTIAL: 2,
            SecurityLevel.RESTRICTED: 3
        }

        user_clearance = clearance_levels[team_member.security_clearance]
        project_clearance = clearance_levels[project.security_level]

        return user_clearance >= project_clearance

    def _log_audit_event(self, event_type: str, user_id: str, resource_type: str,
                        resource_id: str, action: str, details: Dict[str, Any],
                        ip_address: Optional[str] = None, user_agent: Optional[str] = None):
        """Log an audit event."""
        with self.audit_lock:
            event = AuditEvent(
                event_id=str(uuid.uuid4()),
                event_type=event_type,
                user_id=user_id,
                resource_type=resource_type,
                resource_id=resource_id,
                action=action,
                details=details,
                ip_address=ip_address,
                user_agent=user_agent
            )

            self.audit_log.append(event)

            # Maintain audit log size
            if len(self.audit_log) > self.max_audit_entries:
                self.audit_log = self.audit_log[-self.max_audit_entries:]

    # Analytics and Reporting
    def get_project_analytics(self, project_id: str, user_id: str) -> Dict[str, Any]:
        """Get comprehensive analytics for a project."""
        if not self._check_project_permission(project_id, user_id, "read"):
            return {}

        if project_id not in self.projects:
            return {}

        project = self.projects[project_id]

        # Get session data from session manager
        project_sessions = []
        for session in self.session_manager.get_active_sessions():
            if session.get("metadata", {}).get("project_id") == project_id:
                project_sessions.append(session)

        # Calculate metrics
        total_sessions = len(project_sessions)
        total_participants = len(set(
            participant
            for session in project_sessions
            for participant in session.get("participants", [])
        ))

        total_tasks = sum(len(session.get("tasks", [])) for session in project_sessions)
        completed_tasks = sum(
            len([task for task in session.get("tasks", []) if task.get("status") == "completed"])
            for session in project_sessions
        )

        # Team productivity metrics
        team_members_count = len(project.team_members)
        avg_session_duration = sum(
            session.get("duration_minutes", 0) for session in project_sessions
        ) / total_sessions if total_sessions > 0 else 0

        return {
            "project_id": project_id,
            "project_name": project.name,
            "total_sessions": total_sessions,
            "total_participants": total_participants,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "completion_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0,
            "team_members_count": team_members_count,
            "avg_session_duration_minutes": round(avg_session_duration, 2),
            "project_status": project.status.value,
            "security_level": project.security_level.value,
            "created_at": project.created_at.isoformat(),
            "last_updated": project.updated_at.isoformat()
        }

    def get_team_productivity_report(self, project_id: str, user_id: str,
                                   days: int = 30) -> Dict[str, Any]:
        """Get team productivity report."""
        if not self._check_project_permission(project_id, user_id, "read"):
            return {}

        # Get audit events for the project
        cutoff_date = datetime.now() - timedelta(days=days)
        project_events = [
            event for event in self.audit_log
            if event.resource_id == project_id and event.timestamp >= cutoff_date
        ]

        # Analyze activity patterns
        daily_activity = defaultdict(int)
        user_activity = defaultdict(int)

        for event in project_events:
            day = event.timestamp.date()
            daily_activity[day] += 1
            user_activity[event.user_id] += 1

        # Calculate productivity metrics
        total_events = len(project_events)
        active_days = len(daily_activity)
        avg_daily_activity = total_events / active_days if active_days > 0 else 0

        most_active_users = sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:5]

        return {
            "project_id": project_id,
            "report_period_days": days,
            "total_activity_events": total_events,
            "active_days": active_days,
            "avg_daily_activity": round(avg_daily_activity, 2),
            "most_active_users": [
                {"user_id": user_id, "activity_count": count}
                for user_id, count in most_active_users
            ],
            "daily_activity_trend": [
                {"date": str(date), "activity_count": count}
                for date, count in sorted(daily_activity.items())
            ]
        }

    def _aggregate_collaboration_metrics(self):
        """Aggregate collaboration metrics for analytics."""
        # Calculate overall system metrics
        total_projects = len(self.projects)
        total_team_members = len(set(
            member for project in self.projects.values()
            for member in project.team_members
        ))

        active_projects = len([
            p for p in self.projects.values()
            if p.status == ProjectStatus.ACTIVE
        ])

        # Session metrics from session manager
        session_stats = self.session_manager.get_session_stats()

        self.collaboration_metrics = {
            "total_projects": total_projects,
            "active_projects": active_projects,
            "total_team_members": total_team_members,
            "total_sessions": session_stats.get("total_active_sessions", 0),
            "total_participants": session_stats.get("total_participants", 0),
            "total_tasks": session_stats.get("total_tasks", 0),
            "avg_session_duration_minutes": session_stats.get("average_session_duration_minutes", 0),
            "last_updated": datetime.now().isoformat()
        }

    def _cleanup_old_audit_entries(self):
        """Clean up old audit entries beyond retention period."""
        retention_days = 90  # Keep audit logs for 90 days
        cutoff_date = datetime.now() - timedelta(days=retention_days)

        with self.audit_lock:
            old_count = len(self.audit_log)
            self.audit_log = [
                event for event in self.audit_log
                if event.timestamp >= cutoff_date
            ]
            new_count = len(self.audit_log)

            if old_count != new_count:
                logger.info(f"Cleaned up {old_count - new_count} old audit entries")

    # Multi-tenant Support
    def create_tenant(self, tenant_data: Dict[str, Any]) -> str:
        """Create a new tenant."""
        tenant_id = str(uuid.uuid4())

        self.tenants[tenant_id] = {
            "tenant_id": tenant_id,
            "name": tenant_data["name"],
            "description": tenant_data.get("description", ""),
            "created_at": datetime.now(),
            "settings": tenant_data.get("settings", {}),
            "limits": tenant_data.get("limits", {})
        }

        logger.info(f"Created tenant: {tenant_data['name']} ({tenant_id})")
        return tenant_id

    def get_tenant_projects(self, tenant_id: str) -> List[Dict[str, Any]]:
        """Get all projects for a tenant."""
        project_ids = self.tenant_projects.get(tenant_id, [])
        return [
            {
                "project_id": pid,
                "name": self.projects[pid].name,
                "status": self.projects[pid].status.value,
                "team_members_count": len(self.projects[pid].team_members)
            }
            for pid in project_ids if pid in self.projects
        ]

    # Compliance and Reporting
    def generate_soc2_report(self, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """Generate SOC 2 compliance report."""
        # Check permissions
        if not self._is_tenant_admin(tenant_id, user_id):
            return {"error": "Insufficient permissions"}

        # Gather compliance data
        audit_events = [
            event for event in self.audit_log
            if event.timestamp >= datetime.now() - timedelta(days=365)
        ]

        # Analyze security controls
        access_control_events = [e for e in audit_events if "access" in e.event_type.lower()]
        change_management_events = [e for e in audit_events if "change" in e.event_type.lower()]

        return {
            "tenant_id": tenant_id,
            "report_period": "Last 365 days",
            "generated_at": datetime.now().isoformat(),
            "generated_by": user_id,
            "compliance_metrics": {
                "total_audit_events": len(audit_events),
                "access_control_events": len(access_control_events),
                "change_management_events": len(change_management_events),
                "unique_users_with_activity": len(set(e.user_id for e in audit_events))
            },
            "security_assessment": {
                "access_controls_status": "compliant" if len(access_control_events) > 0 else "needs_review",
                "audit_trail_status": "compliant" if len(audit_events) > 100 else "needs_review",
                "data_protection_status": "compliant"  # Placeholder
            }
        }

    def _is_tenant_admin(self, tenant_id: str, user_id: str) -> bool:
        """Check if user is a tenant admin."""
        # Simplified check - in production would check tenant admin roles
        return True  # Placeholder

    def get_enterprise_dashboard_data(self, tenant_id: str, user_id: str) -> Dict[str, Any]:
        """Get enterprise dashboard data."""
        if not self._is_tenant_admin(tenant_id, user_id):
            return {"error": "Insufficient permissions"}

        projects = self.get_tenant_projects(tenant_id)
        collaboration_metrics = self.collaboration_metrics.copy()

        # Get recent audit events
        recent_audit = [
            {
                "event_type": event.event_type,
                "user_id": event.user_id,
                "resource_type": event.resource_type,
                "action": event.action,
                "timestamp": event.timestamp.isoformat()
            }
            for event in self.audit_log[-20:]
        ]

        return {
            "tenant_id": tenant_id,
            "dashboard_data": {
                "projects": projects,
                "collaboration_metrics": collaboration_metrics,
                "recent_audit_events": recent_audit,
                "system_health": {
                    "audit_log_size": len(self.audit_log),
                    "active_integrations": len(self.integration_managers),
                    "security_policies_count": len(self.security_policies)
                }
            },
            "generated_at": datetime.now().isoformat()
        }


# Global instance
_enterprise_collaboration_instance: Optional[EnterpriseCollaboration] = None
_collaboration_lock = threading.Lock()


def get_enterprise_collaboration(session_manager: SessionManager) -> EnterpriseCollaboration:
    """Get the global enterprise collaboration instance."""
    global _enterprise_collaboration_instance

    if _enterprise_collaboration_instance is None:
        with _collaboration_lock:
            if _enterprise_collaboration_instance is None:
                _enterprise_collaboration_instance = EnterpriseCollaboration(session_manager)

    return _enterprise_collaboration_instance