"""
Standard Usage Example for pai-agent-sdk

This example demonstrates the recommended patterns for building a production-ready
agent application with session persistence and human-in-the-loop (HITL) support.

Usage:
    # Clone the repository and navigate to the examples directory, then copy .env.example to .env
    cd examples
    cp .env.example .env

    # To run the example:
    uv run python general.py

    # With debug logging enabled:
    PAI_AGENT_LOG_LEVEL=DEBUG uv run python general.py

Key features demonstrated:
- Session state persistence (message history + context state)
- Model configuration with capabilities
- Streaming agent responses with formatted output
- Human-in-the-loop tool integration
- Error handling and graceful shutdown

The session data is stored in .session/ directory, allowing conversation
continuity across multiple runs.
"""

from __future__ import annotations

from typing import cast

from dotenv import load_dotenv

load_dotenv()

import json
from pathlib import Path

from pydantic_ai import (
    DeferredToolRequests,
    DeferredToolResults,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelSettings,
    PartEndEvent,
    PartStartEvent,
    TextPart,
)
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelResponse,
    PartDeltaEvent,
    TextPartDelta,
    ToolCallPart,
)

from pai_agent_sdk.agents.main import create_agent, stream_agent
from pai_agent_sdk.context import ModelCapability, ModelConfig, ResumableState, RunContextMetadata, StreamEvent
from pai_agent_sdk.presets import ANTHROPIC_DEFAULT
from pai_agent_sdk.toolsets.core.base import UserInteraction
from pai_agent_sdk.toolsets.core.content import tools as content_tools
from pai_agent_sdk.toolsets.core.context import tools as context_tools
from pai_agent_sdk.toolsets.core.document import tools as document_tools
from pai_agent_sdk.toolsets.core.enhance import tools as enhance_tools
from pai_agent_sdk.toolsets.core.filesystem import tools as filesystem_tools
from pai_agent_sdk.toolsets.core.multimodal import tools as multimodal_tools
from pai_agent_sdk.toolsets.core.shell import tools as shell_tools
from pai_agent_sdk.toolsets.core.subagent import tools as subagent_tools
from pai_agent_sdk.toolsets.core.web import tools as web_tools

# =============================================================================
# Prompt Configuration
# =============================================================================

PROMPT_FILE = Path(__file__).parent / "prompts" / "general.md"


def load_system_prompt() -> str:
    """Load system prompt from markdown file."""
    if not PROMPT_FILE.exists():
        raise FileNotFoundError(f"System prompt file not found: {PROMPT_FILE}")
    return PROMPT_FILE.read_text(encoding="utf-8")


# =============================================================================
# Output Formatting Configuration
# =============================================================================

# Maximum length for tool arguments/results before truncation
MAX_TOOL_CONTENT_LENGTH = 200

# Session file paths
SESSION_DIR = Path(__file__).parent / ".session"
MESSAGE_HISTORY_FILE = SESSION_DIR / "message_history.json"
STATE_FILE = SESSION_DIR / "context_state.json"


def ensure_session_dir() -> None:
    """Ensure session directory exists."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)


def load_message_history() -> list[ModelMessage] | None:
    """Load message history from JSON file."""
    if not MESSAGE_HISTORY_FILE.exists():
        return None
    try:
        with open(MESSAGE_HISTORY_FILE) as f:
            data = json.load(f)
        return ModelMessagesTypeAdapter.validate_python(data)
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to load message history: {e}")
        return None


def save_message_history(messages_json: bytes) -> None:
    """Save message history to JSON file."""
    ensure_session_dir()
    with open(MESSAGE_HISTORY_FILE, "wb") as f:
        f.write(messages_json)
    print(f"Message history saved to {MESSAGE_HISTORY_FILE}")


def load_state() -> ResumableState | None:
    """Load context state from JSON file."""
    if not STATE_FILE.exists():
        return None
    try:
        with open(STATE_FILE) as f:
            return ResumableState.model_validate_json(f.read())
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Warning: Failed to load context state: {e}")
        return None


def save_state(state: ResumableState) -> None:
    """Save context state to JSON file."""
    ensure_session_dir()
    with open(STATE_FILE, "w") as f:
        f.write(state.model_dump_json(indent=2))
    print(f"Context state saved to {STATE_FILE}")


def get_user_input(prompt: str = "You: ") -> str:
    """Get user input from console with graceful interrupt handling."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return ""


def format_tool_call_for_approval(tool_call: ToolCallPart) -> str:
    """Format a tool call for user approval display."""
    args_str = json.dumps(tool_call.args, ensure_ascii=False, indent=2) if tool_call.args else "{}"
    return f"Tool: {tool_call.tool_name}\nArgs: {truncate(args_str, 500)}\nID: {tool_call.tool_call_id}"


def get_user_prompt_and_interactions(
    message_history: list[ModelMessage] | None,
) -> tuple[str, list[UserInteraction] | None]:
    """
    Get user prompt and handle pending tool call approvals.

    If the message history ends with a ModelResponse containing tool calls,
    prompt the user to approve/reject each tool call before continuing.

    Returns:
        Tuple of (user_prompt, user_interactions)
        - user_prompt: The user's text input (empty string if only approvals)
        - user_interactions: List of approval/rejection decisions for pending tool calls
    """
    # Check if there are pending tool calls to approve
    if message_history:
        last_message = message_history[-1]
        if isinstance(last_message, ModelResponse):
            # Extract tool calls from the response
            tool_calls = [part for part in last_message.parts if isinstance(part, ToolCallPart)]
            if tool_calls:
                print(f"\n[Pending Tool Calls: {len(tool_calls)}]")
                print("Review each tool call and approve (Y/y/yes/Enter) or provide rejection reason:\n")

                interactions: list[UserInteraction] = []
                for i, tool_call in enumerate(tool_calls, 1):
                    print(f"--- Tool Call {i}/{len(tool_calls)} ---")
                    print(format_tool_call_for_approval(tool_call))
                    print()

                    response = get_user_input("Approve? [Y/reason]: ")

                    # Approve if empty, Y, y, yes, or YES
                    if response.lower() in ("", "y", "yes"):
                        interactions.append(
                            UserInteraction(
                                tool_call_id=tool_call.tool_call_id,
                                approved=True,
                                reason=None,
                                user_input=None,
                            )
                        )
                        print("-> Approved\n")
                    else:
                        # Any other input is treated as rejection reason
                        interactions.append(
                            UserInteraction(
                                tool_call_id=tool_call.tool_call_id,
                                approved=False,
                                reason=response,
                                user_input=None,
                            )
                        )
                        print(f"-> Rejected: {response}\n")

                # After handling approvals, get the actual user prompt
                # Empty string means just process the approvals
                user_prompt = get_user_input("You (or Enter to continue): ")
                return user_prompt, interactions

    # No pending tool calls, just get user prompt
    user_prompt = get_user_input()
    return user_prompt, None


# =============================================================================
# Stream Event Formatting
# =============================================================================


def truncate(text: str, max_length: int = MAX_TOOL_CONTENT_LENGTH) -> str:
    """Truncate text to max_length, adding ellipsis if truncated."""
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def format_tool_call(event: FunctionToolCallEvent) -> str:
    """Format a tool call event for display."""
    tool_name = event.part.tool_name
    args_str = json.dumps(event.part.args, ensure_ascii=False) if event.part.args else "{}"
    return f"[ToolCall] {tool_name}({truncate(args_str)})"


def format_tool_result(event: FunctionToolResultEvent) -> str:
    """Format a tool result event for display."""
    result = event.result
    # Get tool name from the result part
    tool_name = getattr(result, "tool_name", "unknown")
    # Get content - prefer event.content, fallback to result.content
    content = event.content
    if content is None:
        content = getattr(result, "content", "")
    content_str = str(content) if content else ""
    return f"[ToolResult] {tool_name}: {truncate(content_str)}"


def print_stream_event(event: StreamEvent) -> None:
    """
    Print stream event with appropriate formatting.

    - Text deltas: printed directly without newline (streaming effect)
    - Tool calls: formatted with [ToolCall] prefix
    - Tool results: formatted with [ToolResult] prefix
    - Other events: ignored
    """
    message_event = event.event
    if isinstance(message_event, PartStartEvent) and isinstance(message_event.part, TextPart):
        # Text streaming - print directly without newline
        print(message_event.part.content, end="", flush=True)
    if isinstance(message_event, PartDeltaEvent) and isinstance(message_event.delta, TextPartDelta):
        # Text streaming - print directly without newline
        print(message_event.delta.content_delta, end="", flush=True)
    if isinstance(message_event, PartEndEvent) and isinstance(message_event.part, TextPart):
        # Text streaming - print directly without newline
        print()
    elif isinstance(message_event, FunctionToolCallEvent):
        # Tool call - print on new line
        print(format_tool_call(message_event))
        print()
    elif isinstance(message_event, FunctionToolResultEvent):
        # Tool result - print on new line
        print(format_tool_result(message_event))
        print()


async def main():
    # Load previous session state
    message_history: list[ModelMessage] | None = load_message_history()
    state: ResumableState | None = load_state()

    if message_history:
        print(f"Loaded {len(message_history)} messages from history")
    if state:
        print("Loaded previous context state")

    # Get user input and handle any pending tool call approvals
    user_prompt, user_interactions = get_user_prompt_and_interactions(message_history)

    # Exit only if no prompt AND no pending interactions to process
    if not user_prompt and not user_interactions:
        print("No input provided, exiting.")
        return
    _deferred_tool_results: DeferredToolResults | None

    # Load system prompt from file
    system_prompt = load_system_prompt()

    runtime = create_agent(
        model="anthropic:claude-4-5-sonnet-by-all",
        model_settings=cast(ModelSettings, ANTHROPIC_DEFAULT),
        system_prompt=system_prompt,
        tools=[
            *content_tools,
            *context_tools,
            *document_tools,
            *enhance_tools,
            *filesystem_tools,
            *multimodal_tools,
            *shell_tools,
            *web_tools,
            *subagent_tools,
        ],
        need_user_approve_tools=["shell"],
        model_cfg=ModelConfig(context_window=200_000, capabilities={ModelCapability.vision}),
        state=state,
        output_type=[str, DeferredToolRequests],
        include_builtin_subagents=True,
        metadata=RunContextMetadata(context_manage_tool="handoff"),
    )

    # Process pending HITL interactions if any
    if message_history and user_interactions and runtime.core_toolset:
        _deferred_tool_results = await runtime.core_toolset.process_hitl_call(
            runtime.ctx, user_interactions, message_history
        )
    else:
        _deferred_tool_results = None

    async with stream_agent(
        runtime,
        user_prompt=user_prompt,
        message_history=message_history,
        deferred_tool_results=_deferred_tool_results,
    ) as stream:
        async for event in stream:
            print_stream_event(event)
        # Ensure final newline after streaming text
        print()
        # Check for exceptions that occurred during streaming
        stream.raise_if_exception()
        run = stream.run

    if run:
        print(f"\nUsage: {run.usage()}")
        print(f"Messages so far: {len(run.all_messages())}")
        save_message_history(run.all_messages_json())
        new_state = runtime.ctx.export_state()
        save_state(new_state)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
