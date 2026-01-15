"""Handoff tool for context management.

This tool allows the agent to summarize current work and clear context
to start fresh while preserving essential information.

Note:
    This tool must be used together with `pai_agent_sdk.filters.handoff.process_handoff_message`.
    The tool stores the handoff summary in context, and the history processor injects it
    into the message history on the next model request.

Example::

    from pai_agent_sdk.context import AgentContext
    from pai_agent_sdk.toolsets.core.base import Toolset
    from pai_agent_sdk.toolsets.core.context.handoff import HandoffTool
    from pai_agent_sdk.filters.handoff import process_handoff_message

    async with AgentContext() as ctx:
        toolset = Toolset(tools=[HandoffTool])
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            toolsets=[toolset],
            history_processors=[process_handoff_message],  # Required for handoff to work
        )
        result = await agent.run('prompt', deps=ctx)
"""

from functools import cache
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from pydantic_ai import RunContext

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool

_PROMPTS_DIR = Path(__file__).parent / "prompts"


@cache
def _load_instruction() -> str:
    """Load handoff instruction from prompts/handoff.md."""
    prompt_file = _PROMPTS_DIR / "handoff.md"
    return prompt_file.read_text()


class HandoffMessage(BaseModel):
    """Structured summary for context handoff between conversation sessions."""

    primary_request: str = Field(
        ...,
        description="The user's main request and intent that drove this conversation session.",
    )
    key_decisions: list[str] = Field(
        default_factory=list,
        description="Important technical decisions, architectural choices, or patterns established.",
    )
    files_modified: list[str] = Field(
        default_factory=list,
        description="List of file paths that were created, modified, or are relevant to continue work.",
    )
    current_state: str = Field(
        ...,
        description="What was accomplished and the current state of the work. Include specific details.",
    )
    pending_tasks: list[str] = Field(
        default_factory=list,
        description="Tasks explicitly requested by the user that are not yet completed.",
    )
    next_step: str | None = Field(
        default=None,
        description="The immediate next action to take, if work is ongoing. Must align with user's explicit request.",
    )

    def render(self) -> str:
        """Render handoff message as structured XML for context injection."""
        from xml.etree.ElementTree import Element, SubElement, tostring

        root = Element("context-handoff")

        SubElement(root, "primary-request").text = self.primary_request
        SubElement(root, "current-state").text = self.current_state

        if self.key_decisions:
            decisions_elem = SubElement(root, "key-decisions")
            for decision in self.key_decisions:
                SubElement(decisions_elem, "decision").text = decision

        if self.files_modified:
            files_elem = SubElement(root, "files-modified")
            for file in self.files_modified:
                SubElement(files_elem, "file").text = file

        if self.pending_tasks:
            tasks_elem = SubElement(root, "pending-tasks")
            for task in self.pending_tasks:
                SubElement(tasks_elem, "task").text = task

        if self.next_step:
            SubElement(root, "next-step").text = self.next_step

        return tostring(root, encoding="unicode")


class HandoffTool(BaseTool):
    """Tool for context handoff between sessions."""

    name = "handoff"
    description = """Summarize current work and clear context to start fresh.

Use this tool when context is getting large and you need to preserve essential information
before resetting. The handoff message will be injected into the new context automatically.
"""

    def get_instruction(self, ctx: RunContext[AgentContext]) -> str:
        """Load instruction from prompts/handoff.md."""
        return _load_instruction()

    async def call(
        self,
        ctx: RunContext[AgentContext],
        message: Annotated[
            HandoffMessage,
            Field(description="Structured summary of the conversation to preserve across context reset."),
        ],
    ) -> str:
        # Store rendered message for history processor to pick up
        rendered = message.render()
        ctx.deps.handoff_message = rendered
        return f"Handoff complete. Summary:\n\n{rendered}"
