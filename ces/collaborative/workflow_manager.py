"""CES Collaborative Workflow Manager.

Advanced collaborative workflow capabilities including task dependencies,
workflow orchestration, progress tracking, and team coordination features.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from collections import defaultdict, deque
from dataclasses import dataclass, asdict

from ..core.logging_config import get_logger

logger = get_logger(__name__)

@dataclass
class WorkflowTask:
    """Represents a task in a collaborative workflow."""
    id: str
    title: str
    description: str
    status: str  # pending, in_progress, completed, blocked, cancelled
    priority: str  # low, medium, high, critical
    assigned_to: Optional[str]
    dependencies: List[str]  # IDs of tasks this depends on
    dependents: List[str]  # IDs of tasks that depend on this
    estimated_duration: int  # minutes
    actual_duration: Optional[int]
    tags: List[str]
    metadata: Dict[str, Any]
    created_at: str
    updated_at: str
    completed_at: Optional[str]

@dataclass
class Workflow:
    """Represents a collaborative workflow."""
    id: str
    name: str
    description: str
    session_id: str
    status: str  # planning, active, paused, completed, cancelled
    tasks: Dict[str, WorkflowTask]
    participants: List[str]
    owner: str
    progress: float  # 0.0 to 1.0
    priority_matrix: Dict[str, int]  # task_id -> priority score
    created_at: str
    updated_at: str
    completed_at: Optional[str]

class WorkflowManager:
    """Manages collaborative workflows with advanced features."""

    def __init__(self):
        self.workflows = {}
        self.task_dependencies = defaultdict(set)
        self.task_dependents = defaultdict(set)
        self.workflow_participants = defaultdict(set)

    def is_healthy(self) -> bool:
        """Check if workflow manager is healthy."""
        return True

    async def create_workflow(self, workflow_data: Dict[str, Any]) -> str:
        """Create a new collaborative workflow."""
        workflow_id = workflow_data.get("id", str(uuid.uuid4()))

        workflow = Workflow(
            id=workflow_id,
            name=workflow_data.get("name", "Untitled Workflow"),
            description=workflow_data.get("description", ""),
            session_id=workflow_data.get("session_id", ""),
            status=workflow_data.get("status", "planning"),
            tasks={},
            participants=workflow_data.get("participants", []),
            owner=workflow_data.get("owner", "anonymous"),
            progress=0.0,
            priority_matrix={},
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            completed_at=None
        )

        self.workflows[workflow_id] = workflow

        # Track participants
        for participant in workflow.participants:
            self.workflow_participants[participant].add(workflow_id)

        logger.info(f"Created workflow {workflow_id} for session {workflow.session_id}")
        return workflow_id

    async def add_task_to_workflow(self, workflow_id: str, task_data: Dict[str, Any]) -> str:
        """Add a task to an existing workflow."""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]
        task_id = task_data.get("id", str(uuid.uuid4()))

        task = WorkflowTask(
            id=task_id,
            title=task_data.get("title", "Untitled Task"),
            description=task_data.get("description", ""),
            status=task_data.get("status", "pending"),
            priority=task_data.get("priority", "medium"),
            assigned_to=task_data.get("assigned_to"),
            dependencies=task_data.get("dependencies", []),
            dependents=[],  # Will be populated when dependencies are processed
            estimated_duration=task_data.get("estimated_duration", 60),
            actual_duration=None,
            tags=task_data.get("tags", []),
            metadata=task_data.get("metadata", {}),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            completed_at=None
        )

        workflow.tasks[task_id] = task
        workflow.updated_at = datetime.now().isoformat()

        # Update dependency relationships
        await self._update_task_dependencies(workflow_id, task_id, task.dependencies)

        # Recalculate workflow progress
        await self._recalculate_workflow_progress(workflow_id)

        logger.info(f"Added task {task_id} to workflow {workflow_id}")
        return task_id

    async def update_task_status(self, workflow_id: str, task_id: str, new_status: str,
                                user_id: str) -> bool:
        """Update the status of a task in a workflow."""
        if workflow_id not in self.workflows:
            return False

        workflow = self.workflows[workflow_id]
        if task_id not in workflow.tasks:
            return False

        task = workflow.tasks[task_id]
        old_status = task.status

        # Validate status transition
        if not self._is_valid_status_transition(old_status, new_status):
            logger.warning(f"Invalid status transition from {old_status} to {new_status}")
            return False

        task.status = new_status
        task.updated_at = datetime.now().isoformat()

        if new_status == "completed" and not task.completed_at:
            task.completed_at = datetime.now().isoformat()
            if task.assigned_to:
                task.actual_duration = self._calculate_task_duration(task)

        # Check if this affects other tasks
        await self._handle_task_completion(workflow_id, task_id, new_status)

        # Recalculate workflow progress
        await self._recalculate_workflow_progress(workflow_id)

        # Update workflow status if needed
        await self._update_workflow_status(workflow_id)

        logger.info(f"Updated task {task_id} status from {old_status} to {new_status}")
        return True

    async def assign_task(self, workflow_id: str, task_id: str, user_id: str) -> bool:
        """Assign a task to a user."""
        if workflow_id not in self.workflows:
            return False

        workflow = self.workflows[workflow_id]
        if task_id not in workflow.tasks:
            return False

        task = workflow.tasks[task_id]
        task.assigned_to = user_id
        task.updated_at = datetime.now().isoformat()

        # Ensure user is in workflow participants
        if user_id not in workflow.participants:
            workflow.participants.append(user_id)
            self.workflow_participants[user_id].add(workflow_id)

        logger.info(f"Assigned task {task_id} to user {user_id}")
        return True

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive workflow status."""
        if workflow_id not in self.workflows:
            return None

        workflow = self.workflows[workflow_id]

        # Calculate task statistics
        task_stats = self._calculate_task_statistics(workflow)

        # Get critical path
        critical_path = self._calculate_critical_path(workflow_id)

        # Get workflow bottlenecks
        bottlenecks = self._identify_bottlenecks(workflow)

        return {
            "id": workflow.id,
            "name": workflow.name,
            "description": workflow.description,
            "session_id": workflow.session_id,
            "status": workflow.status,
            "progress": workflow.progress,
            "participants": workflow.participants,
            "owner": workflow.owner,
            "task_stats": task_stats,
            "critical_path": critical_path,
            "bottlenecks": bottlenecks,
            "created_at": workflow.created_at,
            "updated_at": workflow.updated_at,
            "estimated_completion": self._estimate_completion_time(workflow_id)
        }

    def get_user_workload(self, user_id: str) -> Dict[str, Any]:
        """Get workload information for a user across all workflows."""
        user_workflows = self.workflow_participants.get(user_id, set())
        user_tasks = []

        for workflow_id in user_workflows:
            if workflow_id in self.workflows:
                workflow = self.workflows[workflow_id]
                for task in workflow.tasks.values():
                    if task.assigned_to == user_id:
                        user_tasks.append({
                            "task_id": task.id,
                            "workflow_id": workflow_id,
                            "workflow_name": workflow.name,
                            "title": task.title,
                            "status": task.status,
                            "priority": task.priority,
                            "estimated_duration": task.estimated_duration,
                            "due_date": task.metadata.get("due_date")
                        })

        # Calculate workload metrics
        workload_metrics = self._calculate_workload_metrics(user_tasks)

        return {
            "user_id": user_id,
            "total_tasks": len(user_tasks),
            "active_tasks": len([t for t in user_tasks if t["status"] == "in_progress"]),
            "pending_tasks": len([t for t in user_tasks if t["status"] == "pending"]),
            "completed_tasks": len([t for t in user_tasks if t["status"] == "completed"]),
            "overdue_tasks": len([t for t in user_tasks if self._is_task_overdue(t)]),
            "workload_metrics": workload_metrics,
            "tasks": user_tasks
        }

    async def get_workflow_recommendations(self, workflow_id: str) -> List[Dict[str, Any]]:
        """Get recommendations for workflow optimization."""
        if workflow_id not in self.workflows:
            return []

        workflow = self.workflows[workflow_id]
        recommendations = []

        # Check for resource conflicts
        resource_conflicts = self._identify_resource_conflicts(workflow)
        if resource_conflicts:
            recommendations.append({
                "type": "resource_conflict",
                "priority": "high",
                "message": f"Resource conflicts detected: {len(resource_conflicts)} tasks assigned to same person simultaneously",
                "actions": ["Reassign tasks", "Adjust deadlines", "Add more participants"]
            })

        # Check for blocked tasks
        blocked_tasks = [t for t in workflow.tasks.values() if t.status == "blocked"]
        if blocked_tasks:
            recommendations.append({
                "type": "blocked_tasks",
                "priority": "high",
                "message": f"{len(blocked_tasks)} tasks are blocked by dependencies",
                "actions": ["Review dependencies", "Parallelize tasks", "Remove unnecessary dependencies"]
            })

        # Check for overdue tasks
        overdue_tasks = [t for t in workflow.tasks.values()
                        if t.status != "completed" and self._is_task_overdue_from_workflow(t)]
        if overdue_tasks:
            recommendations.append({
                "type": "overdue_tasks",
                "priority": "medium",
                "message": f"{len(overdue_tasks)} tasks are overdue",
                "actions": ["Reassign tasks", "Extend deadlines", "Break down tasks"]
            })

        # Check for workload imbalance
        workload_imbalance = self._check_workload_balance(workflow)
        if workload_imbalance:
            recommendations.append({
                "type": "workload_imbalance",
                "priority": "medium",
                "message": "Workload is imbalanced among participants",
                "actions": ["Rebalance assignments", "Add more participants", "Redistribute tasks"]
            })

        return recommendations

    async def _update_task_dependencies(self, workflow_id: str, task_id: str, dependencies: List[str]):
        """Update task dependency relationships."""
        workflow = self.workflows[workflow_id]
        task = workflow.tasks[task_id]

        # Clear existing relationships
        for dep_id in task.dependencies:
            if dep_id in self.task_dependents:
                self.task_dependents[dep_id].discard(task_id)

        for dep_id in task.dependents:
            if dep_id in self.task_dependencies:
                self.task_dependencies[dep_id].discard(task_id)

        # Set new relationships
        task.dependencies = dependencies
        task.dependents = []

        for dep_id in dependencies:
            if dep_id in workflow.tasks:
                self.task_dependents[dep_id].add(task_id)
                self.task_dependencies[task_id].add(dep_id)

                # Update dependent task's dependents list
                if dep_id in workflow.tasks:
                    workflow.tasks[dep_id].dependents.append(task_id)

    async def _handle_task_completion(self, workflow_id: str, task_id: str, new_status: str):
        """Handle the completion of a task and update dependent tasks."""
        if new_status != "completed":
            return

        workflow = self.workflows[workflow_id]

        # Check if any dependent tasks can now be started
        for dependent_id in self.task_dependents.get(task_id, set()):
            if dependent_id in workflow.tasks:
                dependent_task = workflow.tasks[dependent_id]

                # Check if all dependencies are completed
                all_deps_completed = all(
                    workflow.tasks.get(dep_id, WorkflowTask("", "", "", "", "", None, [], [], 0, None, [], {}, "", "", None)).status == "completed"
                    for dep_id in dependent_task.dependencies
                )

                if all_deps_completed and dependent_task.status == "pending":
                    dependent_task.status = "ready"
                    dependent_task.updated_at = datetime.now().isoformat()
                    logger.info(f"Task {dependent_id} is now ready (dependencies completed)")

    async def _recalculate_workflow_progress(self, workflow_id: str):
        """Recalculate overall workflow progress."""
        workflow = self.workflows[workflow_id]
        tasks = list(workflow.tasks.values())

        if not tasks:
            workflow.progress = 0.0
            return

        # Calculate weighted progress
        total_weight = len(tasks)
        completed_weight = sum(1 for task in tasks if task.status == "completed")

        workflow.progress = completed_weight / total_weight if total_weight > 0 else 0.0

    async def _update_workflow_status(self, workflow_id: str):
        """Update workflow status based on task completion."""
        workflow = self.workflows[workflow_id]
        tasks = list(workflow.tasks.values())

        if not tasks:
            return

        completed_tasks = sum(1 for task in tasks if task.status == "completed")
        total_tasks = len(tasks)

        if completed_tasks == total_tasks:
            workflow.status = "completed"
            workflow.completed_at = datetime.now().isoformat()
        elif workflow.status == "planning" and any(t.status in ["in_progress", "ready"] for t in tasks):
            workflow.status = "active"

        workflow.updated_at = datetime.now().isoformat()

    def _calculate_task_statistics(self, workflow: Workflow) -> Dict[str, Any]:
        """Calculate statistics for workflow tasks."""
        tasks = list(workflow.tasks.values())

        return {
            "total": len(tasks),
            "completed": len([t for t in tasks if t.status == "completed"]),
            "in_progress": len([t for t in tasks if t.status == "in_progress"]),
            "pending": len([t for t in tasks if t.status == "pending"]),
            "blocked": len([t for t in tasks if t.status == "blocked"]),
            "ready": len([t for t in tasks if t.status == "ready"]),
            "cancelled": len([t for t in tasks if t.status == "cancelled"])
        }

    def _calculate_critical_path(self, workflow_id: str) -> List[str]:
        """Calculate the critical path for the workflow."""
        if workflow_id not in self.workflows:
            return []

        workflow = self.workflows[workflow_id]

        # Simple critical path calculation (longest path)
        # In a real implementation, this would consider task durations and dependencies

        # For now, return tasks with no dependencies (starting points)
        critical_tasks = []
        for task in workflow.tasks.values():
            if not task.dependencies and task.status != "completed":
                critical_tasks.append(task.id)

        return critical_tasks

    def _identify_bottlenecks(self, workflow: Workflow) -> List[Dict[str, Any]]:
        """Identify workflow bottlenecks."""
        bottlenecks = []

        # Check for tasks that are blocking many others
        for task_id, task in workflow.tasks.items():
            if task.status != "completed" and len(task.dependents) > 2:
                bottlenecks.append({
                    "task_id": task_id,
                    "type": "blocking_many",
                    "description": f"Task '{task.title}' is blocking {len(task.dependents)} other tasks",
                    "severity": "high" if len(task.dependents) > 5 else "medium"
                })

        # Check for overdue tasks
        for task_id, task in workflow.tasks.items():
            if task.status != "completed" and self._is_task_overdue_from_workflow(task):
                bottlenecks.append({
                    "task_id": task_id,
                    "type": "overdue",
                    "description": f"Task '{task.title}' is overdue",
                    "severity": "high"
                })

        return bottlenecks

    def _calculate_workload_metrics(self, user_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate workload metrics for a user."""
        if not user_tasks:
            return {"capacity": 1.0, "stress_level": "low", "recommendations": []}

        active_tasks = len([t for t in user_tasks if t["status"] == "in_progress"])
        high_priority_tasks = len([t for t in user_tasks if t["priority"] in ["high", "critical"]])

        # Simple capacity calculation
        capacity = max(0.1, 1.0 - (active_tasks * 0.2) - (high_priority_tasks * 0.1))

        # Determine stress level
        if capacity > 0.7:
            stress_level = "low"
        elif capacity > 0.4:
            stress_level = "medium"
        else:
            stress_level = "high"

        recommendations = []
        if stress_level == "high":
            recommendations.append("Consider delegating some tasks")
        elif stress_level == "medium":
            recommendations.append("Monitor workload closely")

        return {
            "capacity": capacity,
            "stress_level": stress_level,
            "active_tasks": active_tasks,
            "high_priority_tasks": high_priority_tasks,
            "recommendations": recommendations
        }

    def _is_valid_status_transition(self, old_status: str, new_status: str) -> bool:
        """Validate task status transitions."""
        valid_transitions = {
            "pending": ["ready", "in_progress", "cancelled"],
            "ready": ["in_progress", "cancelled"],
            "in_progress": ["completed", "blocked", "pending", "cancelled"],
            "blocked": ["in_progress", "cancelled"],
            "completed": [],  # Terminal state
            "cancelled": []   # Terminal state
        }

        return new_status in valid_transitions.get(old_status, [])

    def _calculate_task_duration(self, task: WorkflowTask) -> Optional[int]:
        """Calculate actual task duration."""
        if not task.completed_at:
            return None

        try:
            created = datetime.fromisoformat(task.created_at)
            completed = datetime.fromisoformat(task.completed_at)
            duration = completed - created
            return int(duration.total_seconds() / 60)
        except (ValueError, TypeError):
            return None

    def _estimate_completion_time(self, workflow_id: str) -> Optional[str]:
        """Estimate workflow completion time."""
        if workflow_id not in self.workflows:
            return None

        workflow = self.workflows[workflow_id]

        # Simple estimation based on remaining tasks
        remaining_tasks = [t for t in workflow.tasks.values() if t.status != "completed"]
        if not remaining_tasks:
            return workflow.completed_at

        total_remaining_duration = sum(t.estimated_duration for t in remaining_tasks)
        estimated_completion = datetime.now() + timedelta(minutes=total_remaining_duration)

        return estimated_completion.isoformat()

    def _identify_resource_conflicts(self, workflow: Workflow) -> List[str]:
        """Identify resource conflicts in the workflow."""
        conflicts = []
        user_assignments = defaultdict(list)

        for task in workflow.tasks.values():
            if task.assigned_to:
                user_assignments[task.assigned_to].append(task)

        # Check for overlapping tasks per user
        for user_id, tasks in user_assignments.items():
            active_tasks = [t for t in tasks if t.status == "in_progress"]
            if len(active_tasks) > 1:
                conflicts.append(user_id)

        return conflicts

    def _check_workload_balance(self, workflow: Workflow) -> bool:
        """Check if workload is balanced among participants."""
        if len(workflow.participants) < 2:
            return False

        user_task_counts = defaultdict(int)
        for task in workflow.tasks.values():
            if task.assigned_to:
                user_task_counts[task.assigned_to] += 1

        if not user_task_counts:
            return False

        task_counts = list(user_task_counts.values())
        avg_tasks = sum(task_counts) / len(task_counts)

        # Check if any user has significantly more tasks than average
        imbalance_threshold = avg_tasks * 1.5
        return any(count > imbalance_threshold for count in task_counts)

    def _is_task_overdue(self, task: Dict[str, Any]) -> bool:
        """Check if a task is overdue."""
        due_date = task.get("due_date")
        if not due_date:
            return False

        try:
            due = datetime.fromisoformat(due_date)
            return datetime.now() > due and task["status"] != "completed"
        except (ValueError, TypeError):
            return False

    def _is_task_overdue_from_workflow(self, task: WorkflowTask) -> bool:
        """Check if a workflow task is overdue."""
        # For now, consider tasks overdue if they've been in progress for more than their estimated duration
        if task.status != "in_progress" or not task.estimated_duration:
            return False

        try:
            updated = datetime.fromisoformat(task.updated_at)
            elapsed = datetime.now() - updated
            elapsed_minutes = elapsed.total_seconds() / 60
            return elapsed_minutes > task.estimated_duration * 1.5  # 150% of estimated time
        except (ValueError, TypeError):
            return False