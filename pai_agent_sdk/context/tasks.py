"""Task management for agent sessions.

This module provides task tracking with dependencies for managing
multi-step work within agent sessions.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"


class Task(BaseModel):
    """A single task with dependencies and metadata.

    Tasks support blocking relationships where a task can block other tasks
    or be blocked by other tasks. When a blocking task is completed, the
    blocked tasks are automatically unblocked.

    Attributes:
        id: Unique task identifier (e.g., "1", "2").
        subject: Task title in imperative form (e.g., "Run tests").
        description: Detailed task description.
        active_form: Present progressive form shown during in_progress (e.g., "Running tests").
        status: Current task status.
        owner: Optional task owner/assignee.
        blocks: List of task IDs that this task blocks.
        blocked_by: List of task IDs that block this task.
        metadata: Additional task metadata.
        created_at: Task creation timestamp.
        updated_at: Last update timestamp.
    """

    id: str
    subject: str
    description: str
    active_form: str | None = None
    status: TaskStatus = TaskStatus.PENDING
    owner: str | None = None
    blocks: list[str] = Field(default_factory=list)
    blocked_by: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

    def is_blocked(self) -> bool:
        """Check if task is blocked by any incomplete tasks."""
        return len(self.blocked_by) > 0


class TaskManager(BaseModel):
    """Manager for task lifecycle and dependencies.

    Handles task creation, updates, and automatic dependency resolution.
    When a task is completed, it is automatically removed from the blocked_by
    list of tasks it was blocking.

    TaskManager is shared between parent and subagent contexts (shallow copy),
    providing a unified view of task state across the agent hierarchy.

    Example:
        manager = TaskManager()
        task1 = manager.create("Implement API", "Create REST endpoints")
        task2 = manager.create("Write tests", "Unit tests for API")
        manager.update(task2.id, add_blocked_by=[task1.id])  # task2 blocked by task1
        manager.update(task1.id, status=TaskStatus.COMPLETED)  # task2 unblocked
    """

    tasks: dict[str, Task] = Field(default_factory=dict)
    """All tasks keyed by task ID."""

    _next_id: int = 1
    """Counter for generating sequential task IDs."""

    model_config = {"arbitrary_types_allowed": True}

    def __init__(self, **data: Any) -> None:
        """Initialize TaskManager.

        Args:
            **data: Additional model fields.
        """
        super().__init__(**data)
        # Sync _next_id with existing tasks
        if self.tasks:
            max_id = max(int(task_id) for task_id in self.tasks if task_id.isdigit())
            object.__setattr__(self, "_next_id", max_id + 1)

    def _generate_id(self) -> str:
        """Generate next task ID."""
        task_id = str(self._next_id)
        object.__setattr__(self, "_next_id", self._next_id + 1)
        return task_id

    def create(
        self,
        subject: str,
        description: str,
        active_form: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task:
        """Create a new task.

        Args:
            subject: Task title in imperative form.
            description: Detailed task description.
            active_form: Present progressive form for in_progress status.
            metadata: Optional additional metadata.

        Returns:
            The created Task instance.
        """
        task_id = self._generate_id()
        now = datetime.now()
        task = Task(
            id=task_id,
            subject=subject,
            description=description,
            active_form=active_form,
            metadata=metadata or {},
            created_at=now,
            updated_at=now,
        )
        self.tasks[task_id] = task
        return task

    def get(self, task_id: str) -> Task | None:
        """Get a task by ID.

        Args:
            task_id: The task ID to look up.

        Returns:
            The Task if found, None otherwise.
        """
        return self.tasks.get(task_id)

    def _add_blocking_relationship(self, task_id: str, blocked_id: str) -> None:
        """Add a blocking relationship: task_id blocks blocked_id."""
        task = self.tasks.get(task_id)
        blocked_task = self.tasks.get(blocked_id)
        if task and blocked_id not in task.blocks:
            task.blocks.append(blocked_id)
        if blocked_task and task_id not in blocked_task.blocked_by:
            blocked_task.blocked_by.append(task_id)
            blocked_task.updated_at = datetime.now()

    def _add_blocked_by_relationship(self, task_id: str, blocker_id: str) -> None:
        """Add a blocked-by relationship: task_id is blocked by blocker_id."""
        task = self.tasks.get(task_id)
        blocker_task = self.tasks.get(blocker_id)
        if task and blocker_id not in task.blocked_by:
            task.blocked_by.append(blocker_id)
        if blocker_task and task_id not in blocker_task.blocks:
            blocker_task.blocks.append(task_id)
            blocker_task.updated_at = datetime.now()

    def _resolve_completion(self, task: Task) -> None:
        """Remove completed task from blocked_by lists of tasks it blocks."""
        for blocked_id in task.blocks:
            blocked_task = self.tasks.get(blocked_id)
            if blocked_task and task.id in blocked_task.blocked_by:
                blocked_task.blocked_by.remove(task.id)
                blocked_task.updated_at = datetime.now()

    def _update_task_fields(
        self,
        task: Task,
        status: TaskStatus | None,
        subject: str | None,
        description: str | None,
        active_form: str | None,
        owner: str | None,
        metadata: dict[str, Any] | None,
    ) -> None:
        """Update simple task fields."""
        if status is not None:
            task.status = status
        if subject is not None:
            task.subject = subject
        if description is not None:
            task.description = description
        if active_form is not None:
            task.active_form = active_form
        if owner is not None:
            task.owner = owner
        if metadata:
            task.metadata.update(metadata)

    def update(
        self,
        task_id: str,
        *,
        status: TaskStatus | None = None,
        subject: str | None = None,
        description: str | None = None,
        active_form: str | None = None,
        owner: str | None = None,
        add_blocks: list[str] | None = None,
        add_blocked_by: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Task | None:
        """Update a task's properties.

        When status changes to COMPLETED, automatically removes this task
        from the blocked_by list of all tasks it was blocking.

        Args:
            task_id: The task ID to update.
            status: New task status.
            subject: New task subject.
            description: New task description.
            active_form: New active form text.
            owner: New task owner.
            add_blocks: Task IDs to add to blocks list.
            add_blocked_by: Task IDs to add to blocked_by list.
            metadata: Metadata to merge into existing metadata.

        Returns:
            The updated Task if found, None otherwise.
        """
        task = self.tasks.get(task_id)
        if task is None:
            return None

        # Track completion for dependency resolution
        was_completed = status == TaskStatus.COMPLETED and task.status != TaskStatus.COMPLETED

        # Update simple fields
        self._update_task_fields(task, status, subject, description, active_form, owner, metadata)

        # Update relationships
        for blocked_id in add_blocks or []:
            self._add_blocking_relationship(task_id, blocked_id)
        for blocker_id in add_blocked_by or []:
            self._add_blocked_by_relationship(task_id, blocker_id)

        task.updated_at = datetime.now()

        # Handle completion: remove this task from blocked_by of tasks it blocks
        if was_completed:
            self._resolve_completion(task)

        return task

    def list_all(self) -> list[Task]:
        """Get all tasks sorted by ID.

        Returns:
            List of all tasks sorted by numeric ID.
        """
        return sorted(self.tasks.values(), key=lambda t: int(t.id) if t.id.isdigit() else 0)

    def export_tasks(self) -> dict[str, dict[str, Any]]:
        """Export tasks for serialization.

        Returns:
            Dict of task data keyed by task ID.
        """
        return {task_id: task.model_dump(mode="json") for task_id, task in self.tasks.items()}

    @classmethod
    def from_exported(cls, data: dict[str, dict[str, Any]]) -> TaskManager:
        """Restore TaskManager from exported data.

        Args:
            data: Exported task data from export_tasks().

        Returns:
            Restored TaskManager instance.
        """
        tasks = {task_id: Task.model_validate(task_data) for task_id, task_data in data.items()}
        return cls(tasks=tasks)
