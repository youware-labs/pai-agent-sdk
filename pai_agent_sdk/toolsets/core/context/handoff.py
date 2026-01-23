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

    content: str = Field(
        ...,
        description="""Context summary to preserve across handoff. Should include:
1. User's primary request and intent
2. Current state and completed work
3. Key technical decisions
4. Pending tasks
5. Next step (if any)
""",
    )

    auto_load_files: list[str] = Field(
        default_factory=list,
        description="""File paths to auto-load after handoff.
Files will be read and injected into context on next request.
Use for: key config files, source code being edited, important references.
""",
    )

    def render(self) -> str:
        """Render handoff message as Markdown for context injection."""
        return f"# Context Handoff\n\n{self.content}"


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
        # Set auto_load_files for the auto_load_files filter to process
        ctx.deps.auto_load_files = message.auto_load_files
        return f"Handoff complete. Summary:\n\n{rendered}"
