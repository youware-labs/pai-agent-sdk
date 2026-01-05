"""TO-DO list tools for task planning and tracking.

These tools allow the agent to manage a session-level to-do list
stored in the context's temporary directory.
"""

from typing import Annotated, Literal

import pydantic
from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.base import BaseTool


def _get_todo_file_name(run_id: str) -> str:
    """Generate TO-DO file name with run_id to distinguish subagent todos."""
    return f"TO-DO-{run_id}.json"


class TodoItem(BaseModel):
    """A single TO-DO item."""

    id: str = Field(..., description="The unique identifier for the TO-DO item.")
    content: str = Field(..., description="The content of the TO-DO item.")
    status: Literal["pending", "in_progress", "completed"] = Field(
        ...,
        description="The status of the TO-DO item.",
    )
    priority: Literal["high", "medium", "low"] = Field(..., description="The priority of the TO-DO item.")


TodoItemsTypeAdapter = pydantic.TypeAdapter(
    list[TodoItem],
    config=pydantic.ConfigDict(defer_build=True, ser_json_bytes="base64", val_json_bytes="base64"),
)


class TodoReadTool(BaseTool):
    """Tool for reading the TO-DO list."""

    name = "to_do_read"
    description = """Read the current session's to-do list to plan or track task progress.

**When to use**: Call when you need the current task list (e.g., before updating it or resuming work). Avoid unnecessary polling.

**Input**: None. Call with no arguments (do not pass a dummy key).

**Output**:
- `list[TodoItem]` on success, where each item has: `id` (str), `content` (str), `status` ("pending"|"in_progress"|"completed"), `priority` ("high"|"medium"|"low")
- String messages when empty/missing/corrupt: "No TO-DO file found", "No TO-DOs found", or "Error reading to_do file, please try again."
- If you receive a string message, treat it as "no tasks yet" and continue accordingly

**Note**: If you plan to modify tasks, read first, update locally, then write the full updated list via `to_do_write`.
"""
    instruction: str | None = None

    async def call(
        self,
        ctx: RunContext[AgentContext],
    ) -> list[TodoItem] | str:
        to_do_file_path = ctx.deps.tmp_dir / _get_todo_file_name(ctx.deps.run_id)

        try:
            if not to_do_file_path.exists():
                return "No TO-DO file found"

            file_content = to_do_file_path.read_text()
            if not file_content.strip():
                return "No TO-DOs found"

            todos = TodoItemsTypeAdapter.validate_json(file_content)
            return todos

        except Exception:
            # Remove the file if it is corrupted
            if to_do_file_path.exists():
                to_do_file_path.unlink()
            return "Error reading to_do file, please try again."


class TodoWriteTool(BaseTool):
    """Tool for writing/updating the TO-DO list."""

    name = "to_do_write"
    description = """Replace the session's to-do list with an updated list.

**When to use**: Use for multi-step or multi-item work; skip for trivial one-off actions.

**Input**:
- `to_dos`: list[TodoItem]
- TodoItem fields:
  - `id`: str (unique and stable across updates; preserve existing IDs when changing status/content)
  - `content`: str
  - `status`: "pending" | "in_progress" | "completed" (only these values are valid)
  - `priority`: "high" | "medium" | "low"

**Behavior**:
- Overwrites the entire list. Read first if unsure, then modify and resend the full list.
- Pass an empty list `[]` to clear the file (returns "TO-DO list cleared successfully.").
- It's recommended (not enforced) to keep at most one item "in_progress".

**Output**:
- Returns the saved `list[TodoItem]` on success, or a status/error message on failure.

**Tips**:
- Update statuses immediately after finishing an item
- Preserve items that didn't change; only adjust relevant fields on modified items
"""
    instruction: str | None = None

    async def call(
        self,
        ctx: RunContext[AgentContext],
        to_dos: Annotated[list[TodoItem], Field(description="The updated TO-DO list.")],
    ) -> list[TodoItem] | str:
        to_do_file_path = ctx.deps.tmp_dir / _get_todo_file_name(ctx.deps.run_id)
        try:
            if not to_dos and to_do_file_path.exists():
                to_do_file_path.unlink()
                return "TO-DO list cleared successfully."

            # Write TO-DOs to file
            to_do_json = TodoItemsTypeAdapter.dump_json(to_dos)
            to_do_file_path.write_bytes(to_do_json)
            return to_dos

        except Exception as e:
            # If there is an error, remove the file
            if to_do_file_path.exists():
                to_do_file_path.unlink()
            return f"Error writing to_do file: {e}, please try again."
