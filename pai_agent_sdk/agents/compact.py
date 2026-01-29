"""Compact agent for conversation summarization.

This module provides a compact agent that can summarize conversation history
and return structured results including analysis and context for continuing
the conversation.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from inspect import isawaitable
from pathlib import Path
from typing import cast
from uuid import uuid4

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelSettings, UserContent
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.tools import RunContext

from pai_agent_sdk._config import AgentSettings
from pai_agent_sdk._logger import logger
from pai_agent_sdk.agents.models import infer_model
from pai_agent_sdk.context import AgentContext, ModelConfig
from pai_agent_sdk.events import CompactCompleteEvent, CompactFailedEvent, CompactStartEvent
from pai_agent_sdk.filters import (
    create_system_prompt_filter,
    drop_extra_images,
    drop_extra_videos,
    drop_gif_images,
    fix_truncated_tool_args,
)
from pai_agent_sdk.usage import InternalUsage
from pai_agent_sdk.utils import get_latest_request_usage

# =============================================================================
# Constants
# =============================================================================

AGENT_NAME = "compact"

DEFAULT_COMPACT_INSTRUCTION = """Use `condense` to generate a summary and context of the conversation so far.
This summary covers important details of the historical conversation with the user which has been truncated.
It's crucial that you respond by ONLY asking the user what you should work on next.
You should NOT take any initiative or make any assumptions about continuing with work.
Keep this response CONCISE and wrap your analysis in `analysis` and `context` fields to organize your thoughts and ensure you've covered all necessary points.

IMPORTANT: If the message history contains any access to Skills (files in /skills/ directory, such as reading SKILL.md or using skill resources), you MUST include a reminder in the context to re-read the relevant skill documentation when resuming work."""


# =============================================================================
# Utilities
# =============================================================================


def _load_system_prompt() -> str:
    """Load system prompt from the prompts directory."""
    prompt_path = Path(__file__).parent / "prompts" / "compact.md"
    if prompt_path.exists():
        return prompt_path.read_text()
    return ""


# =============================================================================
# Models
# =============================================================================


class CondenseResult(BaseModel):
    analysis: str = Field(
        ...,
        description="""A summary of the conversation so far, capturing technical details, code patterns, and architectural decisions.""",
    )
    context: str = Field(
        ...,
        description="""The context to continue the conversation with. If applicable based on the current task, this should include:

1. Primary Request and Intent: Capture all of the user's explicit requests and intents in detail
2. Key Technical Concepts: List all important technical concepts, technologies, and frameworks discussed.
3. Files and Code Sections: Enumerate specific files and code sections examined, modified, or created. Pay special attention to the most recent messages and include full code snippets where applicable and include a summary of why this file read or edit is important.
4. Problem Solving: Document problems solved and any ongoing troubleshooting efforts.
5. Pending Tasks: Outline any pending tasks that you have explicitly been asked to work on.
6. Current Work: Describe in detail precisely what was being worked on immediately before this summary request, paying special attention to the most recent messages from both user and assistant. Include file names and code snippets where applicable.
7. Optional Next Step: List the next step that you will take that is related to the most recent work you were doing. IMPORTANT: ensure that this step is DIRECTLY in line with the user's explicit requests, and the task you were working on immediately before this summary request. If your last task was concluded, then only list next steps if they are explicitly in line with the users request. Do not start on tangential requests without confirming with the user first.
8. If there is a next step, include direct quotes from the most recent conversation showing exactly what task you were working on and where you left off. This should be verbatim to ensure there's no drift in task interpretation.
""",
    )
    original_prompt: str = Field(
        ...,
        description="The original prompt and key information from the user. "
        "Used as fallback when agent_ctx.user_prompts is not set.",
    )
    auto_load_files: list[str] = Field(
        default_factory=list,
        description="File paths to auto-load when resuming. "
        "Files will be read and injected into context on next request. "
        "Use for: files being actively edited, key references needed to continue work.",
    )


# =============================================================================
# Agent Factory
# =============================================================================


def get_compact_agent(
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
    main_model: str | Model | None = None,
    main_model_settings: ModelSettings | None = None,
) -> Agent[AgentContext, CondenseResult]:
    """Create a compact agent.

    Args:
        model: Model string or Model instance. Highest priority.
        model_settings: Optional model settings dict.
        main_model: Fallback model inherited from main agent. Lowest priority.
        main_model_settings: Fallback model settings inherited from main agent.

    Model resolution priority:
        1. model parameter (explicit configuration)
        2. PAI_AGENT_COMPACT_MODEL environment variable
        3. main_model parameter (inherited from main agent)

    Returns:
        Agent configured for compact with AgentContext as deps type.

    Raises:
        ValueError: If no model is available from any source.
    """
    effective_model: str | Model | None = model
    effective_settings: ModelSettings | None = model_settings

    # Priority: model > env var > main_model
    if effective_model is None:
        settings = AgentSettings()
        if settings.compact_model:
            effective_model = settings.compact_model
        elif main_model is not None:
            effective_model = main_model
        else:
            raise ValueError(
                "No model specified. Provide model parameter, set PAI_AGENT_COMPACT_MODEL, "
                "or pass main_model for inheritance."
            )

    # model_settings: model_settings > main_model_settings
    if effective_settings is None and main_model_settings is not None:
        effective_settings = main_model_settings

    system_prompt = _load_system_prompt()
    return Agent[AgentContext, CondenseResult](
        model=infer_model(effective_model),
        model_settings=effective_settings,
        output_type=CondenseResult,
        deps_type=AgentContext,
        system_prompt=system_prompt,
        history_processors=[
            create_system_prompt_filter(system_prompt),  # Ensure system prompt is consistent
            drop_gif_images,  # Gemini 2.5 pro can't handle gifs
            drop_extra_images,
            drop_extra_videos,
            fix_truncated_tool_args,
        ],
    )


# =============================================================================
# Utilities
# =============================================================================


def condense_result_to_markdown(result: CondenseResult) -> str:
    """Convert CondenseResult to markdown format.

    Args:
        result: The CondenseResult to convert.

    Returns:
        Markdown formatted string with analysis and context.
    """
    return f"""## Condensed conversation summary

### Analysis

{result.analysis}

### Context

{result.context}
"""


def _need_compact(ctx: AgentContext, message_history: list[ModelMessage]) -> bool:
    """Check if compaction is needed based on token usage threshold.

    Args:
        ctx: Agent context with model configuration.
        message_history: Current message history.

    Returns:
        True if compaction should be triggered.
    """
    if not message_history:
        return False

    model_cfg = ctx.model_cfg
    if model_cfg.context_window is None:
        logger.debug("Unknown context window, skipping compact check.")
        return False

    # Get current token usage from message history
    request_usage = get_latest_request_usage(message_history)
    if request_usage is None or request_usage.total_tokens is None:
        return False

    threshold_tokens = int(model_cfg.context_window * model_cfg.compact_threshold)
    current_tokens = request_usage.total_tokens

    logger.debug(f"Compact check: {current_tokens} tokens vs {threshold_tokens} threshold")
    return current_tokens >= threshold_tokens


def _build_compacted_messages(summary: str, original_prompt: str | Sequence[UserContent]) -> list[ModelMessage]:
    """Build compacted message history.

    Args:
        summary: The compacted summary content.
        system_prompt: Optional system prompt to include.

    Returns:
        List of ModelMessage representing the compacted history.
    """
    request_parts: list[SystemPromptPart | UserPromptPart] = [
        SystemPromptPart(content="Placeholder system prompt"),
        UserPromptPart(
            content="You have exceeded the maximum token limit for this conversation. "
            "Please provide a summary of the conversation so far and what you should work on next "
            "and I'll resume the conversation."
        ),
    ]
    return [
        ModelRequest(parts=request_parts),
        ModelResponse(parts=[TextPart(content=summary)]),
        ModelRequest(
            parts=[
                UserPromptPart(content=original_prompt),
                UserPromptPart(content="<compact-complete>Context compacted. Resume task.</compact-complete>"),
            ]
        ),
    ]


def create_compact_filter(
    model: str | Model | None = None,
    model_settings: ModelSettings | None = None,
    model_cfg: ModelConfig | None = None,
    main_model: str | Model | None = None,
    main_model_settings: ModelSettings | None = None,
) -> Callable[[RunContext[AgentContext], list[ModelMessage]], Awaitable[list[ModelMessage]]]:
    """Create a compact filter for automatic context compaction.

    The returned filter checks token usage and compacts the conversation history
    when usage exceeds the configured threshold (ModelConfig.compact_threshold).

    Args:
        model: Model string or Model instance for the compact agent. Highest priority.
        model_settings: Optional model settings for the compact agent.
        model_cfg: Model configuration for threshold checking.
        main_model: Fallback model inherited from main agent. Lowest priority.
        main_model_settings: Fallback model settings inherited from main agent.

    Model resolution priority:
        1. model parameter (explicit configuration)
        2. PAI_AGENT_COMPACT_MODEL environment variable
        3. main_model parameter (inherited from main agent)

    Returns:
        An async filter function compatible with pydantic-ai history_processors.

    Example::

        compact_filter = await create_compact_filter(model="openai:gpt-4o-mini")
        agent = Agent(
            'openai:gpt-4',
            deps_type=AgentContext,
            history_processors=[compact_filter],
        )
    """
    agent = get_compact_agent(
        model=model,
        model_settings=model_settings,
        main_model=main_model,
        main_model_settings=main_model_settings,
    )

    async def compact_filter(
        ctx: RunContext[AgentContext],
        message_history: list[ModelMessage],
    ) -> list[ModelMessage]:
        """Filter that compacts message history when threshold is exceeded.

        Args:
            ctx: Runtime context containing AgentContext.
            message_history: Current message history to potentially compact.

        Returns:
            Original or compacted message history.
        """
        agent_ctx = ctx.deps

        if not _need_compact(agent_ctx, message_history):
            logger.debug("No need to compact history.")
            return message_history

        logger.info("Compacting conversation history...")

        # Generate event_id to correlate start/complete events
        event_id = uuid4().hex[:8]

        # Apply model wrapper if configured
        original_model = agent.model
        if agent_ctx.model_wrapper is not None:
            wrapper_metadata = agent_ctx.get_wrapper_metadata()
            wrapped = agent_ctx.model_wrapper(cast(Model, original_model), AGENT_NAME, wrapper_metadata)
            agent.model = await wrapped if isawaitable(wrapped) else wrapped

        try:
            # Emit start event
            await agent_ctx.emit_event(CompactStartEvent(event_id=event_id, message_count=len(message_history)))

            # Run compact agent on full message history with AgentContext as deps
            result = await agent.run(
                DEFAULT_COMPACT_INSTRUCTION,
                message_history=message_history,
                deps=AgentContext(
                    model_cfg=model_cfg or ModelConfig(),
                ),
            )

            # Record usage in extra_usages

            model_id = cast(Model, agent.model).model_name
            agent_ctx.add_extra_usage(
                agent=AGENT_NAME,
                internal_usage=InternalUsage(model_id=model_id, usage=result.usage()),
                uuid=uuid4().hex,
            )

            condense_result: CondenseResult = result.output

            # Append auto_load_files for the auto_load_files filter to process
            # Use extend instead of assignment to preserve any files set by external callers
            agent_ctx.auto_load_files.extend(condense_result.auto_load_files)

            # Build summary with condense result and user prompts
            condense_markdown = condense_result_to_markdown(condense_result)

            # Build compacted messages
            # Priority: agent_ctx.user_prompts > condense_result.original_prompt
            # user_prompts is set by main agent from actual user input, while original_prompt
            # is extracted by LLM from conversation history and may be less accurate
            compacted = _build_compacted_messages(
                condense_markdown, agent_ctx.user_prompts or condense_result.original_prompt
            )

            # Emit complete event with summary
            await agent_ctx.emit_event(
                CompactCompleteEvent(
                    event_id=event_id,
                    summary_markdown=condense_markdown,
                    original_message_count=len(message_history),
                    compacted_message_count=len(compacted),
                )
            )

            logger.info(f"Compacted history from {len(message_history)} messages to {len(compacted)} messages")
            return compacted

        except Exception as e:
            logger.error(f"Failed to compact history: {e}")
            # Emit failed event so consumers know compact did not succeed
            await agent_ctx.emit_event(
                CompactFailedEvent(event_id=event_id, error=str(e), message_count=len(message_history))
            )
            # On error, return original history
            return message_history

        finally:
            # Restore original model to avoid side effects on shared agent
            agent.model = original_model

    return compact_filter
