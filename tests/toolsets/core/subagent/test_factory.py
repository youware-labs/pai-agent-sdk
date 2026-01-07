"""Tests for subagent tool factory."""

import inspect
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.usage import RunUsage

from pai_agent_sdk.context import AgentContext
from pai_agent_sdk.toolsets.core.base import Toolset
from pai_agent_sdk.toolsets.core.subagent import (
    SubagentCallFunc,
    create_subagent_call_func,
    create_subagent_tool,
)
from pai_agent_sdk.toolsets.core.subagent.factory import generate_unique_id

# Test fixtures and mock functions


async def mock_search(
    ctx: AgentContext,  # Now receives subagent context directly
    query: str,
    max_results: int = 10,
) -> tuple[str, RunUsage]:
    """Mock search function that returns query info and usage."""
    usage = RunUsage(requests=1, input_tokens=10, output_tokens=20)
    return f"Results for '{query}' (max: {max_results})", usage


async def mock_analyze(
    ctx: AgentContext,  # Now receives subagent context directly
    content: str,
) -> tuple[dict, RunUsage]:
    """Mock analyze function that returns a dict."""
    usage = RunUsage(requests=1, input_tokens=5, output_tokens=15)
    return {"analysis": content, "score": 0.95}, usage


# Tests for create_subagent_tool function


def test_call_signature_is_correct():
    """Test that the call method has correct signature for pydantic-ai.

    The call method should have:
    - First parameter: ctx with type RunContext[AgentContext]
    - Remaining parameters: copied from call_func (without ctx)
    - Return type: str

    Note: 'self' is not in __signature__ because we assign a plain function
    to the class attribute, not a bound method. pydantic-ai Tool extracts
    parameters starting from 'ctx', which is correct.
    """
    SearchTool = create_subagent_tool(
        name="search",
        description="Search the web",
        call_func=mock_search,
    )

    # Get the signature from the class's call method
    sig = inspect.signature(SearchTool.call)
    params = list(sig.parameters.items())
    param_names = [name for name, _ in params]

    # The signature should contain: ctx, query, max_results
    # (self is handled separately by Python method binding, not in __signature__)
    assert param_names[0] == "ctx", f"First param should be ctx, got {param_names[0]}"
    assert "query" in param_names, "Should have query parameter from call_func"
    assert "max_results" in param_names, "Should have max_results parameter from call_func"

    # Check annotations - the key check is that ctx is RunContext[AgentContext], NOT AgentContext
    annotations = SearchTool.call.__annotations__
    assert annotations.get("ctx") == RunContext[AgentContext], (
        f"ctx should be RunContext[AgentContext], got {annotations.get('ctx')}"
    )
    assert annotations.get("query") is str, f"query should be str, got {annotations.get('query')}"
    assert annotations.get("max_results") is int, f"max_results should be int, got {annotations.get('max_results')}"
    assert annotations.get("return") is str, f"return should be str, got {annotations.get('return')}"

    # Check signature return annotation
    assert sig.return_annotation is str, f"Return annotation should be str, got {sig.return_annotation}"

    # Check parameter defaults
    params_dict = dict(params)
    if "max_results" in params_dict:
        max_results_param = params_dict["max_results"]
        assert max_results_param.default == 10, f"max_results default should be 10, got {max_results_param.default}"

    # Verify the signature parameter types match annotations
    ctx_param = params_dict["ctx"]
    assert ctx_param.annotation == RunContext[AgentContext], (
        f"ctx param annotation should be RunContext[AgentContext], got {ctx_param.annotation}"
    )


def test_creates_tool_class():
    """Test that create_subagent_tool returns a BaseTool subclass."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search the web",
        call_func=mock_search,
    )

    assert SearchTool.name == "search"
    assert SearchTool.description == "Search the web"
    assert SearchTool.__name__ == "SearchTool"


def test_pascal_case_naming():
    """Test that tool class names are converted to PascalCase."""
    Tool1 = create_subagent_tool(
        name="web_search",
        description="desc",
        call_func=mock_search,
    )
    assert Tool1.__name__ == "WebSearchTool"

    Tool2 = create_subagent_tool(
        name="analyze-content",
        description="desc",
        call_func=mock_search,
    )
    assert Tool2.__name__ == "AnalyzeContentTool"


async def test_call_returns_string(agent_context: AgentContext):
    """Test that tool call returns string output."""
    AnalyzeTool = create_subagent_tool(
        name="analyze",
        description="Analyze content",
        call_func=mock_analyze,
    )

    tool = AnalyzeTool(agent_context)
    ctx = _create_mock_run_context(agent_context, tool_call_id="test-call-1")

    result = await tool.call(ctx, content="test content")

    # Dict should be converted to string
    assert isinstance(result, str)
    assert "analysis" in result
    assert "test content" in result


async def test_usage_recorded(agent_context: AgentContext):
    """Test that usage is recorded in extra_usages."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=mock_search,
    )

    tool = SearchTool(agent_context)
    tool_call_id = "test-call-123"
    ctx = _create_mock_run_context(agent_context, tool_call_id=tool_call_id)

    await tool.call(ctx, query="test query")

    # Check usage was recorded
    assert len(agent_context.extra_usages) == 1
    record = agent_context.extra_usages[0]
    assert record.uuid == tool_call_id
    assert record.agent == "search"
    assert record.usage.requests == 1
    assert record.usage.input_tokens == 10
    assert record.usage.output_tokens == 20


async def test_usage_not_recorded_without_tool_call_id(agent_context: AgentContext):
    """Test that usage is not recorded when tool_call_id is None."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=mock_search,
    )

    tool = SearchTool(agent_context)
    ctx = _create_mock_run_context(agent_context, tool_call_id=None)

    await tool.call(ctx, query="test query")

    # No usage should be recorded
    assert len(agent_context.extra_usages) == 0


def test_instruction_string(agent_context: AgentContext):
    """Test static instruction string."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=mock_search,
        instruction="Use this to search the web.",
    )

    tool = SearchTool(agent_context)
    ctx = _create_mock_run_context(agent_context)

    assert tool.get_instruction(ctx) == "Use this to search the web."


def test_instruction_callable(agent_context: AgentContext):
    """Test dynamic instruction callable."""

    def dynamic_instruction(ctx: RunContext[AgentContext]) -> str:
        return f"Search with run_id: {ctx.deps.run_id}"

    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=mock_search,
        instruction=dynamic_instruction,
    )

    tool = SearchTool(agent_context)
    ctx = _create_mock_run_context(agent_context)

    instruction = tool.get_instruction(ctx)
    assert agent_context.run_id in instruction


def test_instruction_none(agent_context: AgentContext):
    """Test that no instruction returns None."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search",
        call_func=mock_search,
    )

    tool = SearchTool(agent_context)
    ctx = _create_mock_run_context(agent_context)

    assert tool.get_instruction(ctx) is None


async def test_with_toolset(agent_context: AgentContext):
    """Test that created tool works with Toolset."""
    SearchTool = create_subagent_tool(
        name="search",
        description="Search the web",
        call_func=mock_search,
    )

    toolset = Toolset(agent_context, tools=[SearchTool])
    ctx = _create_mock_run_context(agent_context)

    tools = await toolset.get_tools(ctx)

    assert "search" in tools
    assert tools["search"].tool_def.name == "search"
    assert tools["search"].tool_def.description == "Search the web"


# Tests for subagent_stream_queues functionality


def test_stream_queues_default_dict(agent_context: AgentContext):
    """Test that stream_queues is a defaultdict creating queues on access."""
    # Access non-existent key should create a new queue
    queue = agent_context.subagent_stream_queues["test-tool-call-id"]
    assert queue is not None
    assert queue.empty()


async def test_stream_queues_put_get(agent_context: AgentContext):
    """Test putting and getting events from stream queue."""
    tool_call_id = "test-tool-call-id"
    queue = agent_context.subagent_stream_queues[tool_call_id]

    # Put a custom event
    custom_event = {"event_kind": "custom", "data": "test"}
    await queue.put(custom_event)

    # Get the event
    event = await queue.get()
    assert event == custom_event


async def test_stream_queues_multiple_tools(agent_context: AgentContext):
    """Test that different tool calls have separate queues."""
    queue1 = agent_context.subagent_stream_queues["tool-1"]
    queue2 = agent_context.subagent_stream_queues["tool-2"]

    await queue1.put("event1")
    await queue2.put("event2")

    assert await queue1.get() == "event1"
    assert await queue2.get() == "event2"


# Helper functions


async def test_call_method_uses_enter_subagent(agent_context: AgentContext):
    """Test that the call method uses enter_subagent for subagent context creation."""
    parent_run_id = agent_context.run_id

    async def mock_with_parent_check(
        ctx: AgentContext,
        query: str,
    ) -> tuple[str, RunUsage]:
        """Mock function that verifies subagent context is created correctly."""
        # Verify subagent context has run_id set to tool_call_id
        assert ctx.run_id == "test-call-subagent"
        # Verify parent_run_id points to the parent's run_id
        assert ctx.parent_run_id == parent_run_id
        usage = RunUsage(requests=1, input_tokens=5, output_tokens=10)
        return f"Query: {query}", usage

    SearchTool = create_subagent_tool(
        name="search_subagent",
        description="Search with subagent",
        call_func=mock_with_parent_check,
    )

    tool = SearchTool(agent_context)
    ctx = _create_mock_run_context(agent_context, tool_call_id="test-call-subagent")

    result = await tool.call(ctx, query="test")

    assert "Query: test" in result


async def test_create_subagent_call_func():
    """Test create_subagent_call_func creates a working SubagentCallFunc."""
    # Create a mock Agent
    mock_agent = MagicMock(spec=Agent)

    # Mock the iter context manager and its return
    mock_run = MagicMock()
    mock_run.result = MagicMock()
    mock_run.result.output = "search result"
    mock_run.result.usage = MagicMock(return_value=RunUsage(requests=1, input_tokens=10, output_tokens=20))

    # Create an async iterator that yields nothing (empty iteration)
    async def empty_async_iter():
        return
        yield  # Make this an async generator

    mock_run.__aiter__ = lambda _: empty_async_iter()

    # Mock iter as async context manager
    @asynccontextmanager
    async def mock_iter(prompt, deps, message_history=None):
        yield mock_run

    mock_agent.iter = mock_iter
    mock_agent.is_user_prompt_node = MagicMock(return_value=False)
    mock_agent.is_end_node = MagicMock(return_value=False)
    mock_agent.is_model_request_node = MagicMock(return_value=False)
    mock_agent.is_call_tools_node = MagicMock(return_value=False)

    # Create the call func
    call_func = create_subagent_call_func(mock_agent)

    # Create context
    ctx = AgentContext()

    # Call the function
    output, usage = await call_func(ctx, prompt="test query")

    # Output is wrapped in XML tags
    assert "search result" in output
    assert usage.requests == 1


async def test_create_subagent_call_func_with_streaming_nodes():
    """Test create_subagent_call_func handles model request and call tools nodes."""
    # Create a mock Agent
    mock_agent = MagicMock(spec=Agent)

    # Mock result
    mock_run = MagicMock()
    mock_run.result = MagicMock()
    mock_run.result.output = "result with streaming"
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
    async def mock_iter(prompt, deps, message_history=None):
        yield mock_run

    mock_agent.iter = mock_iter

    # Configure node type detection
    def is_model_request(node):
        return node is mock_model_node

    def is_call_tools(node):
        return node is mock_tools_node

    def is_end(node):
        return node is mock_end_node

    mock_agent.is_user_prompt_node = MagicMock(return_value=False)
    mock_agent.is_end_node = is_end
    mock_agent.is_model_request_node = is_model_request
    mock_agent.is_call_tools_node = is_call_tools

    # Patch Agent class methods
    Agent.is_user_prompt_node = staticmethod(lambda n: False)
    Agent.is_end_node = staticmethod(is_end)
    Agent.is_model_request_node = staticmethod(is_model_request)
    Agent.is_call_tools_node = staticmethod(is_call_tools)

    # Create the call func
    call_func = create_subagent_call_func(mock_agent)

    # Create context
    ctx = AgentContext()

    # Call the function
    output, usage = await call_func(ctx, prompt="test streaming")

    # Output is wrapped in XML tags
    assert "result with streaming" in output
    assert usage.requests == 2

    # Check that events were put into stream queue
    queue = ctx.subagent_stream_queues[ctx.run_id]
    assert not queue.empty()
    event = await queue.get()
    assert event == mock_event


async def test_subagent_call_func_protocol():
    """Test SubagentCallFunc protocol checking."""

    # A valid SubagentCallFunc
    async def valid_func(ctx: AgentContext, query: str) -> tuple[str, RunUsage]:
        return "result", RunUsage()

    # Check protocol conformance at runtime
    assert isinstance(valid_func, SubagentCallFunc)


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


# Tests for generate_unique_id function


def test_generate_unique_id_uses_run_id_suffix():
    """Test that generate_unique_id returns last 4 chars of run_id when no collision."""
    run_id = "abcd1234efgh5678"
    existing: set[str] = set()

    result = generate_unique_id(run_id, existing)

    assert result == "5678"


def test_generate_unique_id_with_collision():
    """Test that generate_unique_id generates new ID on collision."""
    run_id = "abcd1234efgh5678"
    existing = {"5678"}  # Collision with run_id[-4:]

    result = generate_unique_id(run_id, existing)

    assert result != "5678"
    assert len(result) == 4


def test_generate_unique_id_retries_until_unique():
    """Test that generate_unique_id retries until finding unique ID."""
    run_id = "abcd1234efgh5678"
    # Create a set that will cause multiple collisions
    # The function tries run_id[-4:] first, then random UUIDs
    existing = {"5678"}

    result = generate_unique_id(run_id, existing)

    assert result not in existing
    assert len(result) == 4


def test_generate_unique_id_raises_on_max_retries(monkeypatch):
    """Test that generate_unique_id raises RuntimeError after max retries."""

    run_id = "abcd1234efgh5678"

    # Mock uuid4 to always return the same value
    class MockUUID:
        hex = "aaaa0000bbbb1111"

    monkeypatch.setattr("pai_agent_sdk.toolsets.core.subagent.factory.uuid4", lambda: MockUUID())

    # Both the run_id suffix and the mocked uuid will collide
    existing = {"5678", "aaaa"}

    with pytest.raises(RuntimeError, match="Failed to generate unique agent_id after 10 retries"):
        generate_unique_id(run_id, existing)


def test_generate_unique_id_with_custom_max_retries():
    """Test generate_unique_id with custom max_retries parameter."""
    run_id = "abcd1234efgh5678"
    existing: set[str] = set()

    result = generate_unique_id(run_id, existing, max_retries=5)

    assert result == "5678"
