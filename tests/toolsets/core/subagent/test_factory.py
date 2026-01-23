"""Tests for subagent tool factory."""

import inspect
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import RunUsage

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import BaseTool, Toolset
from pai_agent_sdk.toolsets.core.subagent import (
    create_subagent_call_func,
    create_subagent_tool,
)
from pai_agent_sdk.toolsets.core.subagent.factory import generate_unique_id

# Tests for create_subagent_tool with create_subagent_call_func


def test_creates_tool_class_with_agent():
    """Test that create_subagent_tool creates a BaseTool subclass from agent call func."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "search"

    call_func = create_subagent_call_func(mock_agent)
    SearchTool = create_subagent_tool(
        name="search",
        description="Search the web",
        call_func=call_func,
    )

    assert SearchTool.name == "search"
    assert SearchTool.description == "Search the web"
    assert SearchTool.__name__ == "SearchTool"


def test_pascal_case_naming():
    """Test that tool class names are converted to PascalCase."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "web_search"

    call_func = create_subagent_call_func(mock_agent)

    Tool1 = create_subagent_tool(
        name="web_search",
        description="desc",
        call_func=call_func,
    )
    assert Tool1.__name__ == "WebSearchTool"

    Tool2 = create_subagent_tool(
        name="analyze-content",
        description="desc",
        call_func=call_func,
    )
    assert Tool2.__name__ == "AnalyzeContentTool"


def test_call_signature_has_correct_parameters():
    """Test that the call method has the correct signature for pydantic-ai."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "search"

    call_func = create_subagent_call_func(mock_agent)
    SearchTool = create_subagent_tool(
        name="search",
        description="Search the web",
        call_func=call_func,
    )

    # Get the signature from the class's call method
    sig = inspect.signature(SearchTool.call)
    params = list(sig.parameters.items())
    param_names = [name for name, _ in params]

    # The signature should contain: ctx, prompt, agent_id
    assert "ctx" in param_names, "Should have ctx parameter"
    assert "prompt" in param_names, "Should have prompt parameter"
    assert "agent_id" in param_names, "Should have agent_id parameter"

    # Check annotations exist (may be string due to PEP 563)
    annotations = SearchTool.call.__annotations__
    assert "ctx" in annotations, "Should have ctx annotation"
    assert "return" in annotations, "Should have return annotation"


def test_instruction_string(agent_context: AgentContext):
    """Test static instruction string."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "search"

    call_func = create_subagent_call_func(mock_agent)
    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=call_func,
        instruction="Use this to search the web.",
    )

    tool = SearchTool()
    ctx = _create_mock_run_context(agent_context)

    assert tool.get_instruction(ctx) == "Use this to search the web."


def test_instruction_callable(agent_context: AgentContext):
    """Test dynamic instruction callable."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "search"

    def dynamic_instruction(ctx: RunContext[AgentContext]) -> str:
        return f"Search with run_id: {ctx.deps.run_id}"

    call_func = create_subagent_call_func(mock_agent)
    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=call_func,
        instruction=dynamic_instruction,
    )

    tool = SearchTool()
    ctx = _create_mock_run_context(agent_context)

    instruction = tool.get_instruction(ctx)
    assert agent_context.run_id in instruction


def test_instruction_none(agent_context: AgentContext):
    """Test that no instruction returns None."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "search"

    call_func = create_subagent_call_func(mock_agent)
    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=call_func,
    )

    tool = SearchTool()
    ctx = _create_mock_run_context(agent_context)

    assert tool.get_instruction(ctx) is None


async def test_with_toolset(agent_context: AgentContext):
    """Test that created tool works with Toolset."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "search"

    call_func = create_subagent_call_func(mock_agent)
    SearchTool = create_subagent_tool(
        name="search",
        description="Search the web",
        call_func=call_func,
    )

    toolset = Toolset(tools=[SearchTool])
    ctx = _create_mock_run_context(agent_context)

    tools = await toolset.get_tools(ctx)

    assert "search" in tools
    assert tools["search"].tool_def.name == "search"
    assert tools["search"].tool_def.description == "Search the web"


# Tests for agent_stream_queues functionality


def test_stream_queues_default_dict(agent_context: AgentContext):
    """Test that stream_queues is a defaultdict creating queues on access."""
    # Access non-existent key should create a new queue
    queue = agent_context.agent_stream_queues["test-tool-call-id"]
    assert queue is not None
    assert queue.empty()


async def test_stream_queues_put_get(agent_context: AgentContext):
    """Test putting and getting events from stream queue."""
    tool_call_id = "test-tool-call-id"
    queue = agent_context.agent_stream_queues[tool_call_id]

    # Put a custom event
    custom_event = {"event_kind": "custom", "data": "test"}
    await queue.put(custom_event)

    # Get the event
    event = await queue.get()
    assert event == custom_event


async def test_stream_queues_multiple_tools(agent_context: AgentContext):
    """Test that different tool calls have separate queues."""
    queue1 = agent_context.agent_stream_queues["tool-1"]
    queue2 = agent_context.agent_stream_queues["tool-2"]

    await queue1.put("event1")
    await queue2.put("event2")

    assert await queue1.get() == "event1"
    assert await queue2.get() == "event2"


# Tests for create_subagent_call_func


async def test_create_subagent_call_func_basic():
    """Test create_subagent_call_func creates a working call function."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "search"

    mock_run = MagicMock()
    mock_run.result = MagicMock()
    mock_run.result.output = "search result"
    mock_run.result.all_messages = MagicMock(return_value=[])
    mock_run.result.usage = MagicMock(return_value=RunUsage(requests=1, input_tokens=10, output_tokens=20))

    async def empty_async_iter():
        return
        yield

    mock_run.__aiter__ = lambda _: empty_async_iter()

    @asynccontextmanager
    async def mock_iter(prompt, deps, message_history=None, usage_limits=None):
        yield mock_run

    mock_agent.iter = mock_iter

    call_func = create_subagent_call_func(mock_agent)

    # Create context and mock tool instance
    ctx = AgentContext()
    run_ctx = _create_mock_run_context(ctx, tool_call_id="test-call-123")

    # Create a mock self (BaseTool instance)
    mock_self = MagicMock(spec=BaseTool)

    # Call the function
    output = await call_func(mock_self, run_ctx, prompt="test query")

    # Output is wrapped in XML tags
    assert "search result" in output
    assert "<id>search-" in output

    # Usage should be recorded
    assert len(ctx.extra_usages) == 1
    assert ctx.extra_usages[0].uuid == "test-call-123"
    # agent field uses agent_id (e.g., "search-xxxx")
    assert ctx.extra_usages[0].agent.startswith("search-")


async def test_create_subagent_call_func_registers_agent():
    """Test that create_subagent_call_func registers agent in agent_registry."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "analyze"

    mock_run = MagicMock()
    mock_run.result = MagicMock()
    mock_run.result.output = "analysis result"
    mock_run.result.all_messages = MagicMock(return_value=[])
    mock_run.result.usage = MagicMock(return_value=RunUsage(requests=1))

    async def empty_async_iter():
        return
        yield

    mock_run.__aiter__ = lambda _: empty_async_iter()

    @asynccontextmanager
    async def mock_iter(prompt, deps, message_history=None, usage_limits=None):
        yield mock_run

    mock_agent.iter = mock_iter

    call_func = create_subagent_call_func(mock_agent)

    ctx = AgentContext()
    run_ctx = _create_mock_run_context(ctx)
    mock_self = MagicMock(spec=BaseTool)

    await call_func(mock_self, run_ctx, prompt="test")

    # Agent should be registered
    assert len(ctx.agent_registry) == 1
    agent_id = next(iter(ctx.agent_registry.keys()))
    assert agent_id.startswith("analyze-")
    assert ctx.agent_registry[agent_id].agent_name == "analyze"
    # parent_agent_id is now the agent_id of the parent context, not run_id
    assert ctx.agent_registry[agent_id].parent_agent_id == ctx._agent_id


async def test_create_subagent_call_func_stores_history():
    """Test that create_subagent_call_func stores subagent history."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "chat"

    mock_messages = [{"role": "user", "content": "test"}, {"role": "assistant", "content": "response"}]
    mock_run = MagicMock()
    mock_run.result = MagicMock()
    mock_run.result.output = "chat response"
    mock_run.result.all_messages = MagicMock(return_value=mock_messages)
    mock_run.result.usage = MagicMock(return_value=RunUsage(requests=1))

    async def empty_async_iter():
        return
        yield

    mock_run.__aiter__ = lambda _: empty_async_iter()

    @asynccontextmanager
    async def mock_iter(prompt, deps, message_history=None, usage_limits=None):
        yield mock_run

    mock_agent.iter = mock_iter

    call_func = create_subagent_call_func(mock_agent)

    ctx = AgentContext()
    run_ctx = _create_mock_run_context(ctx)
    mock_self = MagicMock(spec=BaseTool)

    await call_func(mock_self, run_ctx, prompt="test")

    # History should be stored
    assert len(ctx.subagent_history) == 1
    agent_id = next(iter(ctx.subagent_history.keys()))
    assert ctx.subagent_history[agent_id] == mock_messages


async def test_create_subagent_call_func_with_streaming_nodes():
    """Test create_subagent_call_func handles model request and call tools nodes."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "streamer"

    mock_run = MagicMock()
    mock_run.result = MagicMock()
    mock_run.result.output = "result with streaming"
    mock_run.result.all_messages = MagicMock(return_value=[])
    mock_run.result.usage = MagicMock(return_value=RunUsage(requests=2, input_tokens=20, output_tokens=30))
    mock_run.ctx = MagicMock()

    # Create mock nodes
    mock_model_node = MagicMock()
    mock_tools_node = MagicMock()
    mock_end_node = MagicMock()

    # Create mock stream that yields events
    mock_event = {"type": "text", "content": "streaming..."}

    @asynccontextmanager
    async def mock_stream(ctx):
        async def event_gen():
            yield mock_event

        yield event_gen()

    mock_model_node.stream = mock_stream
    mock_tools_node.stream = mock_stream

    # Create async iterator that yields nodes
    async def node_iter():
        yield mock_model_node
        yield mock_tools_node
        yield mock_end_node

    mock_run.__aiter__ = lambda _: node_iter()

    @asynccontextmanager
    async def mock_iter(prompt, deps, message_history=None, usage_limits=None):
        yield mock_run

    mock_agent.iter = mock_iter

    # Configure node type detection
    def is_model_request(node):
        return node is mock_model_node

    def is_call_tools(node):
        return node is mock_tools_node

    def is_end(node):
        return node is mock_end_node

    # Patch Agent class methods
    Agent.is_user_prompt_node = staticmethod(lambda n: False)
    Agent.is_end_node = staticmethod(is_end)
    Agent.is_model_request_node = staticmethod(is_model_request)
    Agent.is_call_tools_node = staticmethod(is_call_tools)

    call_func = create_subagent_call_func(mock_agent)

    ctx = AgentContext()
    ctx._stream_queue_enabled = True
    run_ctx = _create_mock_run_context(ctx)
    mock_self = MagicMock(spec=BaseTool)

    output = await call_func(mock_self, run_ctx, prompt="test streaming")

    assert "result with streaming" in output

    # Check that events were put into subagent's stream queue
    # The subagent has agent_id like "streamer-xxxx", find it from registry
    subagent_id = next(iter(ctx.agent_registry.keys()))
    queue = ctx.agent_stream_queues[subagent_id]
    assert not queue.empty()

    # First event should be SubagentStartEvent
    from pai_agent_sdk.events import SubagentCompleteEvent, SubagentStartEvent

    start_event = await queue.get()
    assert isinstance(start_event, SubagentStartEvent)
    assert start_event.agent_name == "streamer"
    assert start_event.prompt_preview == "test streaming"
    assert start_event.agent_id.startswith("streamer-")

    # Then the streamed events
    event = await queue.get()
    assert event == mock_event

    # After all streaming, there should be SubagentCompleteEvent
    # (need to drain any remaining mock_events first)
    complete_event = None
    while not queue.empty():
        e = await queue.get()
        if isinstance(e, SubagentCompleteEvent):
            complete_event = e
            break

    assert complete_event is not None
    assert complete_event.agent_name == "streamer"
    assert complete_event.success is True
    # Start and Complete should share the same event_id (= agent_id)
    assert complete_event.event_id == start_event.event_id
    assert complete_event.event_id == start_event.agent_id


# Tests for agent_id generation


async def test_create_subagent_call_func_agent_id_with_name():
    """Test that agent_id includes agent name when agent has a name."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "search"

    mock_run = MagicMock()
    mock_run.result = MagicMock()
    mock_run.result.output = "search result"
    mock_run.result.all_messages = MagicMock(return_value=[])
    mock_run.result.usage = MagicMock(return_value=RunUsage(requests=1))

    async def empty_async_iter():
        return
        yield

    mock_run.__aiter__ = lambda _: empty_async_iter()

    @asynccontextmanager
    async def mock_iter(prompt, deps, message_history=None, usage_limits=None):
        yield mock_run

    mock_agent.iter = mock_iter

    call_func = create_subagent_call_func(mock_agent)

    ctx = AgentContext()
    run_ctx = _create_mock_run_context(ctx)
    mock_self = MagicMock(spec=BaseTool)

    output = await call_func(mock_self, run_ctx, prompt="test query")

    # Verify agent_id format: {agent.name}-{short_id}
    assert "<id>search-" in output
    import re

    match = re.search(r"<id>(search-[a-f0-9]{4})</id>", output)
    assert match is not None, f"Expected agent_id format 'search-XXXX' not found in: {output}"


async def test_create_subagent_call_func_agent_id_without_name():
    """Test that agent_id uses 'subagent' as default name when agent has no name."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = None  # No name

    mock_run = MagicMock()
    mock_run.result = MagicMock()
    mock_run.result.output = "result"
    mock_run.result.all_messages = MagicMock(return_value=[])
    mock_run.result.usage = MagicMock(return_value=RunUsage(requests=1))

    async def empty_async_iter():
        return
        yield

    mock_run.__aiter__ = lambda _: empty_async_iter()

    @asynccontextmanager
    async def mock_iter(prompt, deps, message_history=None, usage_limits=None):
        yield mock_run

    mock_agent.iter = mock_iter

    call_func = create_subagent_call_func(mock_agent)

    ctx = AgentContext()
    run_ctx = _create_mock_run_context(ctx)
    mock_self = MagicMock(spec=BaseTool)

    output = await call_func(mock_self, run_ctx, prompt="test query")

    # Verify agent_id uses default name "subagent"
    assert "<id>subagent-" in output


async def test_create_subagent_call_func_resume_with_agent_id():
    """Test that resuming with explicit agent_id works correctly."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "analyze"

    mock_run = MagicMock()
    mock_run.result = MagicMock()
    mock_run.result.output = "resumed result"
    mock_run.result.all_messages = MagicMock(return_value=[{"role": "user", "content": "prev"}])
    mock_run.result.usage = MagicMock(return_value=RunUsage(requests=1))

    async def empty_async_iter():
        return
        yield

    mock_run.__aiter__ = lambda _: empty_async_iter()

    @asynccontextmanager
    async def mock_iter(prompt, deps, message_history=None, usage_limits=None):
        yield mock_run

    mock_agent.iter = mock_iter

    call_func = create_subagent_call_func(mock_agent)

    ctx = AgentContext()
    # Pre-populate history for the agent_id
    ctx.subagent_history["analyze-abcd"] = [{"role": "user", "content": "previous"}]
    # Pre-register in agent_registry
    from pai_agent_sdk.context import AgentInfo

    ctx.agent_registry["analyze-abcd"] = AgentInfo(
        agent_id="analyze-abcd",
        agent_name="analyze",
        parent_agent_id=ctx.run_id,
    )

    run_ctx = _create_mock_run_context(ctx)
    mock_self = MagicMock(spec=BaseTool)

    # Resume with explicit agent_id
    output = await call_func(mock_self, run_ctx, prompt="continue", agent_id="analyze-abcd")

    # Should use the provided agent_id, not generate a new one
    assert "<id>analyze-abcd</id>" in output


async def test_usage_not_recorded_without_tool_call_id():
    """Test that usage is not recorded when tool_call_id is None."""
    mock_agent = MagicMock(spec=Agent)
    mock_agent.model.model_name = "test-model"
    mock_agent.name = "search"

    mock_run = MagicMock()
    mock_run.result = MagicMock()
    mock_run.result.output = "result"
    mock_run.result.all_messages = MagicMock(return_value=[])
    mock_run.result.usage = MagicMock(return_value=RunUsage(requests=1))

    async def empty_async_iter():
        return
        yield

    mock_run.__aiter__ = lambda _: empty_async_iter()

    @asynccontextmanager
    async def mock_iter(prompt, deps, message_history=None, usage_limits=None):
        yield mock_run

    mock_agent.iter = mock_iter

    call_func = create_subagent_call_func(mock_agent)

    ctx = AgentContext()
    run_ctx = _create_mock_run_context(ctx, tool_call_id=None)
    mock_self = MagicMock(spec=BaseTool)

    await call_func(mock_self, run_ctx, prompt="test")

    # No usage should be recorded
    assert len(ctx.extra_usages) == 0


# Tests for generate_unique_id function


def test_generate_unique_id_uses_run_id_suffix():
    """Test that generate_unique_id returns last 4 chars of run_id when no collision."""
    existing: set[str] = set()

    result = generate_unique_id(existing)

    assert result


def test_generate_unique_id_with_collision():
    """Test that generate_unique_id generates new ID on collision."""
    existing = {"5678"}  # Collision with run_id[-4:]

    result = generate_unique_id(existing)

    assert result != "5678"
    assert len(result) == 4


def test_generate_unique_id_retries_until_unique():
    """Test that generate_unique_id retries until finding unique ID."""
    existing = {"5678"}

    result = generate_unique_id(existing)

    assert result not in existing
    assert len(result) == 4


def test_generate_unique_id_raises_on_max_retries(monkeypatch):
    """Test that generate_unique_id raises RuntimeError after max retries."""

    class MockUUID:
        hex = "aaaa0000bbbb1111"

    monkeypatch.setattr("pai_agent_sdk.toolsets.core.subagent.factory.uuid4", lambda: MockUUID())

    existing = {"5678", "aaaa"}

    with pytest.raises(RuntimeError, match="Failed to generate unique agent_id after 10 retries"):
        generate_unique_id(existing)


# Helper functions


def _create_mock_run_context(
    agent_context: AgentContext,
    tool_call_id: str | None = "mock-tool-call-id",
) -> RunContext[AgentContext]:
    """Create a mock RunContext for testing."""
    return RunContext[AgentContext](
        deps=agent_context,
        model=None,  # type: ignore[arg-type]
        usage=RunUsage(),
        prompt="test",
        messages=[],
        run_step=0,
        tool_call_id=tool_call_id,
    )
