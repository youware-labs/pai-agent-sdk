from functools import cache

import httpx
from pydantic_ai.models import (
    get_user_agent,
)
from pydantic_ai.retries import AsyncTenacityTransport, RetryConfig
from tenacity import retry_if_exception_type, stop_after_attempt, wait_exponential


def create_async_http_client(
    *,
    extra_headers: dict[str, str] | None = None,
    timeout: int = 900,
    connect: int = 5,
    read: int = 300,
) -> httpx.AsyncClient:
    """Create a new httpx.AsyncClient with optional extra headers.

    Args:
        extra_headers: Additional headers to include in all requests.
            Useful for sticky routing via x-session-id header.
        timeout: Total timeout in seconds.
        connect: Connection timeout in seconds.
        read: Read timeout in seconds.

    Returns:
        A new httpx.AsyncClient instance (not cached).
    """
    headers = {"User-Agent": get_user_agent()}
    if extra_headers:
        headers.update(extra_headers)

    return httpx.AsyncClient(
        transport=AsyncTenacityTransport(
            config=RetryConfig(
                retry=retry_if_exception_type((
                    httpx.HTTPError,
                    httpx.StreamError,
                )),
                wait=wait_exponential(multiplier=1, max=10),
                stop=stop_after_attempt(10),
                reraise=True,
            )
        ),
        timeout=httpx.Timeout(timeout=timeout, connect=connect, read=read),
        headers=headers,
    )


@cache
def _cached_async_http_client(
    provider: str | None, timeout: int = 900, connect: int = 5, read: int = 300
) -> httpx.AsyncClient:
    return create_async_http_client(timeout=timeout, connect=connect, read=read)


def cached_async_http_client(
    *,
    provider: str | None = None,
    timeout: int = 900,
    connect: int = 5,
    read: int = 300,
) -> httpx.AsyncClient:
    """Cached HTTPX async client that creates a separate client for each provider.

    The client is cached based on the provider parameter. If provider is None, it's used for non-provider specific
    requests (like downloading images). Multiple agents and calls can share the same client when they use the same provider.

    There are good reasons why in production you should use a `httpx.AsyncClient` as an async context manager as
    described in [encode/httpx#2026](https://github.com/encode/httpx/pull/2026), but when experimenting or showing
    examples, it's very useful not to.
    """
    client = _cached_async_http_client(provider=provider, timeout=timeout, connect=connect, read=read)
    if client.is_closed:
        # This happens if the context manager is used, so we need to create a new client.
        _cached_async_http_client.cache_clear()
        client = _cached_async_http_client(provider=provider, timeout=timeout, connect=connect, read=read)
    return client
