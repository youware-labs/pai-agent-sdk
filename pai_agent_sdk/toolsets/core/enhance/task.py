"""Task management tools for tracking work items and dependencies.

These tools allow the agent to manage tasks with dependencies,
enabling structured work tracking with blocking relationships.
"""

from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.context import AgentContext, TaskStatus
from pai_agent_sdk.toolsets.base import Instruction
from pai_agent_sdk.toolsets.core.base import BaseTool

logger = get_logger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"


class TaskCreateParams(BaseModel):
    """Parameters for creating a new task."""

    subject: str = Field(..., description="Task title in imperative form (e.g., 'Run tests').")
    description: str = Field(..., description="Detailed task description.")
    active_form: str | None = Field(
        None, description="Present progressive form shown during in_progress (e.g., 'Running tests')."
    )
    metadata: dict[str, Any] | None = Field(None, description="Optional additional metadata.")


class TaskCreateTool(BaseTool):
    """Tool for creating new tasks."""

    name = "task_create"
    description = "Create a new task. Task status defaults to pending."
    auto_inherit = True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> Instruction | None:
        """Get instruction for this tool (shared with other task tools)."""
        instruction_file = _PROMPTS_DIR / "task_manager.md"
        if instruction_file.exists():
            return Instruction(group="task-manager", content=instruction_file.read_text())
        return None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        subject: Annotated[str, Field(description="Task title in imperative form (e.g., 'Run tests').")],
        description: Annotated[str, Field(description="Detailed task description.")],
        active_form: Annotated[
            str | None,
            Field(description="Present progressive form shown during in_progress (e.g., 'Running tests')."),
        ] = None,
        metadata: Annotated[dict[str, Any] | None, Field(description="Optional additional metadata.")] = None,
    ) -> str:
        task = ctx.deps.task_manager.create(
            subject=subject,
            description=description,
            active_form=active_form,
            metadata=metadata,
        )
        return f"Task #{task.id} created successfully: {task.subject}"


class TaskGetTool(BaseTool):
    """Tool for getting task details."""

    name = "task_get"
    description = "Get task details by ID."
    auto_inherit = True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> Instruction | None:
        """Get instruction for this tool (shared with other task tools)."""
        instruction_file = _PROMPTS_DIR / "task_manager.md"
        if instruction_file.exists():
            return Instruction(group="task-manager", content=instruction_file.read_text())
        return None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        task_id: Annotated[str, Field(description="The task ID to retrieve.")],
    ) -> str:
        task = ctx.deps.task_manager.get(task_id)
        if task is None:
            return f"Task #{task_id} not found."

        lines = [
            f"Task #{task.id}: {task.subject}",
            f"Status: {task.status.value}",
            f"Description: {task.description}",
        ]
        if task.active_form:
            lines.append(f"Active Form: {task.active_form}")
        if task.owner:
            lines.append(f"Owner: {task.owner}")
        if task.blocked_by:
            lines.append(f"Blocked By: #{', #'.join(task.blocked_by)}")
        if task.blocks:
            lines.append(f"Blocks: #{', #'.join(task.blocks)}")
        if task.metadata:
            lines.append(f"Metadata: {task.metadata}")
        return "\n".join(lines)


class TaskUpdateTool(BaseTool):
    """Tool for updating task properties."""

    name = "task_update"
    description = "Update task status, content, or dependencies."
    auto_inherit = True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> Instruction | None:
        """Get instruction for this tool (shared with other task tools)."""
        instruction_file = _PROMPTS_DIR / "task_manager.md"
        if instruction_file.exists():
            return Instruction(group="task-manager", content=instruction_file.read_text())
        return None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        task_id: Annotated[str, Field(description="The task ID to update.")],
        status: Annotated[
            Literal["pending", "in_progress", "completed"] | None,
            Field(description="New task status."),
        ] = None,
        subject: Annotated[str | None, Field(description="New task title.")] = None,
        description: Annotated[str | None, Field(description="New task description.")] = None,
        active_form: Annotated[str | None, Field(description="New present progressive form.")] = None,
        owner: Annotated[str | None, Field(description="Task owner/assignee.")] = None,
        add_blocks: Annotated[list[str] | None, Field(description="Task IDs that this task blocks.")] = None,
        add_blocked_by: Annotated[list[str] | None, Field(description="Task IDs that block this task.")] = None,
        metadata: Annotated[dict[str, Any] | None, Field(description="Metadata to merge into task.")] = None,
    ) -> str:
        # Convert status string to enum
        task_status = TaskStatus(status) if status else None

        task = ctx.deps.task_manager.update(
            task_id,
            status=task_status,
            subject=subject,
            description=description,
            active_form=active_form,
            owner=owner,
            add_blocks=add_blocks,
            add_blocked_by=add_blocked_by,
            metadata=metadata,
        )

        if task is None:
            return f"Task #{task_id} not found."

        # Build update summary
        updates = []
        if status:
            updates.append(f"status -> {status}")
        if subject:
            updates.append("subject")
        if description:
            updates.append("description")
        if active_form:
            updates.append("activeForm")
        if owner:
            updates.append("owner")
        if add_blocks:
            updates.append(f"blocks (+{len(add_blocks)})")
        if add_blocked_by:
            updates.append(f"blockedBy (+{len(add_blocked_by)})")
        if metadata:
            updates.append("metadata")

        update_text = ", ".join(updates) if updates else "no changes"
        return f"Updated task #{task_id}: {update_text}"


class TaskListTool(BaseTool):
    """Tool for listing all tasks."""

    name = "task_list"
    description = "List all tasks and their status."
    auto_inherit = True

    def get_instruction(self, ctx: RunContext[AgentContext]) -> Instruction | None:
        """Get instruction for this tool (shared with other task tools)."""
        instruction_file = _PROMPTS_DIR / "task_manager.md"
        if instruction_file.exists():
            return Instruction(group="task-manager", content=instruction_file.read_text())
        return None

    async def call(
        self,
        ctx: RunContext[AgentContext],
    ) -> str:
        tasks = ctx.deps.task_manager.list_all()

        if not tasks:
            return "No tasks found."

        lines = []
        for task in tasks:
            # Status indicator
            if task.status == TaskStatus.COMPLETED:
                status_str = "[completed]"
            elif task.status == TaskStatus.IN_PROGRESS:
                status_str = f"[in_progress: {task.active_form or task.subject}]"
            else:
                status_str = "[pending]"

            # Build line
            line = f"#{task.id} {status_str} {task.subject}"

            # Add blocking info for incomplete tasks
            if task.status != TaskStatus.COMPLETED and task.blocked_by:
                # Filter to only show incomplete blockers
                active_blockers = [
                    bid
                    for bid in task.blocked_by
                    if (blocker := ctx.deps.task_manager.get(bid)) and blocker.status != TaskStatus.COMPLETED
                ]
                if active_blockers:
                    line += f" [blocked by #{', #'.join(active_blockers)}]"

            lines.append(line)

        return "\n".join(lines)
