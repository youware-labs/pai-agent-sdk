import os
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from pydantic_ai.models import Model
from pydantic_ai.models import infer_model as legacy_infer_model
from pydantic_ai.providers import Provider

from pai_agent_sdk.agents.models.utils import cached_async_http_client, create_async_http_client


def _request_hook(api_key: str) -> Callable[[httpx.Request], Awaitable[httpx.Request]]:
    """Request hook for the gateway provider.

    It adds the `"Authorization"` header to the request.
    """

    async def _hook(request: httpx.Request) -> httpx.Request:
        if "Authorization" not in request.headers:
            request.headers["Authorization"] = f"Bearer {api_key}"

        return request

    return _hook


def make_gateway_provider(
    gateway_name: str,
    extra_headers: dict[str, str] | None = None,
) -> Callable[[str], Provider[Any]]:
    """Create a gateway_provider function with optional extra headers.

    Args:
        extra_headers: Additional HTTP headers to include in all requests.
            Useful for sticky routing via x-session-id header.

    Returns:
        A gateway_provider function that can be passed to legacy_infer_model.

    Usage:
        # With extra headers (new client per call)
        model = infer_model("gemini:...", extra_headers={"x-session-id": session_id})

        # Without extra headers (uses cached client)
        model = infer_model("gemini:...")
    """
    gateway_prefix = gateway_name.upper()
    api_key_env_var = f"{gateway_prefix}_API_KEY"
    base_url_env_var = f"{gateway_prefix}_BASE_URL"

    def gateway_provider(provider_name: str) -> Provider[Any]:
        api_key = os.getenv(api_key_env_var)
        if not api_key:
            raise KeyError(f"API key not found, check environment variable: {api_key_env_var}.")

        base_url = os.getenv(base_url_env_var)
        if not base_url:
            raise KeyError(f"Gateway URL not found, check environment variable: {base_url_env_var}.")

        # Only gemini/bedrock need extra_headers via http_client (their providers don't support direct header injection)
        needs_extra_headers_patch = provider_name in ("google-vertex", "gemini", "bedrock", "converse")

        if extra_headers and needs_extra_headers_patch:
            http_client = create_async_http_client(extra_headers=extra_headers)
        else:
            http_client = cached_async_http_client(provider=f"{gateway_name}/{provider_name}")

        http_client.event_hooks = {"request": [_request_hook(api_key)]}

        if provider_name in (
            "openai",
            "openai-chat",
            "openai-responses",
            "chat",
            "responses",
        ):
            from pydantic_ai.providers.openai import OpenAIProvider

            return OpenAIProvider(api_key=api_key, base_url=base_url, http_client=http_client)
        elif provider_name == "groq":
            from pydantic_ai.providers.groq import GroqProvider

            return GroqProvider(api_key=api_key, base_url=base_url, http_client=http_client)
        elif provider_name == "anthropic":
            from anthropic import AsyncAnthropic  # pyright: ignore[reportMissingImports]
            from pydantic_ai.providers.anthropic import AnthropicProvider

            return AnthropicProvider(
                anthropic_client=AsyncAnthropic(auth_token=api_key, base_url=base_url, http_client=http_client)
            )
        elif provider_name in ("bedrock", "converse"):
            from pydantic_ai.providers.bedrock import BedrockProvider

            return BedrockProvider(
                api_key=api_key,
                base_url=base_url,
                region_name=gateway_name,  # Fake region name to avoid NoRegionError
            )
        elif provider_name in ("google-vertex", "gemini"):
            from pydantic_ai.providers.google import GoogleProvider

            return GoogleProvider(vertexai=True, api_key=api_key, base_url=base_url, http_client=http_client)
        else:
            raise KeyError(f"Unknown upstream provider: {provider_name}")

    return gateway_provider


def infer_model(gateway_name: str, model: str, extra_headers: dict[str, str] | None = None) -> Model:
    """Infer model from string, optionally with extra HTTP headers.

    Args:
        model: Model string in format "provider:model_name"
        extra_headers: Optional dict of extra headers to send with each request.
            When provided, a new http client is created (not cached).
            Useful for sticky routing via x-session-id header.

    Returns:
        The inferred Model instance.
    """
    provider_factory = make_gateway_provider(gateway_name, extra_headers)
    return legacy_infer_model(model, provider_factory)
