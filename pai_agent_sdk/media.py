"""S3 media upload utilities for converting media to URLs.

This module provides S3-based media upload functionality for the
image_to_url_hook and video_to_url_hook in ToolConfig.

Requires optional dependency: pip install pai-agent-sdk[s3]

Example:
    from pai_agent_sdk.media import S3MediaConfig, create_s3_media_hook
    from pai_agent_sdk.context import ToolConfig

    config = S3MediaConfig(
        bucket="my-bucket",
        url_mode="cdn",
        cdn_base_url="https://cdn.example.com",
    )
    hook = create_s3_media_hook(config)

    tool_config = ToolConfig(
        image_to_url_hook=hook,
        video_to_url_hook=hook,
    )
"""

from __future__ import annotations

import hashlib
from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Protocol, runtime_checkable

import anyio
import anyio.to_thread
from pydantic import BaseModel, model_validator

from pai_agent_sdk._logger import get_logger

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client
    from pydantic_ai import RunContext

    from pai_agent_sdk.context import AgentContext

__all__ = [
    "MediaUploader",
    "S3MediaConfig",
    "S3MediaUploader",
    "create_s3_media_hook",
]

logger = get_logger(__name__)


# =============================================================================
# MediaUploader Protocol
# =============================================================================


@runtime_checkable
class MediaUploader(Protocol):
    """Protocol for media upload services.

    Implement this protocol to support different storage backends:
    - S3 (AWS, MinIO, Ceph, R2)
    - Azure Blob Storage
    - Google Cloud Storage
    - Local file server with public URL
    - imgbb, imgur, etc.

    Example::

        class LocalFileUploader:
            def __init__(self, base_dir: Path, base_url: str):
                self.base_dir = base_dir
                self.base_url = base_url

            async def upload(self, data: bytes, media_type: str) -> str:
                ext = media_type.split("/")[-1]
                filename = f"{uuid4()}.{ext}"
                (self.base_dir / filename).write_bytes(data)
                return f"{self.base_url}/{filename}"
    """

    async def upload(self, data: bytes, media_type: str) -> str:
        """Upload media data and return public URL.

        Args:
            data: Raw media bytes.
            media_type: MIME type (e.g., 'image/png', 'video/mp4').

        Returns:
            Public URL to access the uploaded media.

        Raises:
            Exception: If upload fails.
        """
        ...


# =============================================================================
# S3 Implementation
# =============================================================================


class S3MediaConfig(BaseModel):
    """Configuration for S3 media upload.

    Example:
        # Minimal config with default AWS credentials
        config = S3MediaConfig(bucket="my-bucket")

        # With CDN
        config = S3MediaConfig(
            bucket="my-bucket",
            url_mode="cdn",
            cdn_base_url="https://cdn.example.com",
        )

        # With explicit credentials and custom endpoint (MinIO, R2)
        config = S3MediaConfig(
            bucket="my-bucket",
            endpoint_url="https://s3.example.com",
            access_key_id="...",
            secret_access_key="...",
        )
    """

    bucket: str
    """S3 bucket name."""

    region: str = "us-east-1"
    """AWS region."""

    access_key_id: str | None = None
    """AWS access key ID. If None, uses default credential chain."""

    secret_access_key: str | None = None
    """AWS secret access key. If None, uses default credential chain."""

    endpoint_url: str | None = None
    """Custom S3 endpoint URL for S3-compatible services (MinIO, R2, etc.)."""

    prefix: str = ""
    """Object key prefix for uploaded files. e.g., 'uploads/' or 'uploads'"""

    # URL generation
    url_mode: Literal["cdn", "presign"] = "presign"
    """URL generation mode: 'cdn' for CDN mapping, 'presign' for presigned URLs."""

    cdn_base_url: str | None = None
    """CDN base URL (required if url_mode='cdn'). e.g., 'https://cdn.example.com'"""

    presign_expires_seconds: int = 3600
    """Presigned URL expiration time in seconds (default: 1 hour)."""

    force_path_style: bool = False
    """Use path-style URLs instead of virtual-hosted style. Required for some S3-compatible services."""

    @model_validator(mode="after")
    def _normalize_prefix(self) -> S3MediaConfig:
        """Ensure prefix ends with '/' if not empty."""
        if self.prefix and not self.prefix.endswith("/"):
            object.__setattr__(self, "prefix", self.prefix + "/")
        return self


class S3MediaUploader:
    """Upload media files to S3 and generate public URLs.

    Thread-safe and async-safe. Client is lazily created and cached.

    Example:
        config = S3MediaConfig(bucket="my-bucket")
        uploader = S3MediaUploader(config)

        url = await uploader.upload(image_bytes, "image/png")
    """

    # MIME type to extension mapping
    EXTENSION_MAP: ClassVar[dict[str, str]] = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/gif": "gif",
        "image/webp": "webp",
        "image/svg+xml": "svg",
        "video/mp4": "mp4",
        "video/webm": "webm",
        "video/quicktime": "mov",
        "audio/mpeg": "mp3",
        "audio/wav": "wav",
    }

    def __init__(self, config: S3MediaConfig) -> None:
        self.config = config
        self._client: S3Client | None = None

    def _get_client(self) -> S3Client:
        """Lazily create and cache boto3 S3 client."""
        if self._client is not None:
            return self._client

        try:
            import boto3
        except ImportError as e:
            raise ImportError("S3 media upload requires 'boto3'. Install with: pip install pai-agent-sdk[s3]") from e

        client_kwargs: dict[str, Any] = {
            "region_name": self.config.region,
        }

        if self.config.access_key_id and self.config.secret_access_key:
            client_kwargs["aws_access_key_id"] = self.config.access_key_id
            client_kwargs["aws_secret_access_key"] = self.config.secret_access_key

        if self.config.endpoint_url:
            client_kwargs["endpoint_url"] = self.config.endpoint_url

        # Use path-style addressing for S3-compatible services
        if self.config.force_path_style:
            from botocore.config import Config as BotoConfig

            client_kwargs["config"] = BotoConfig(s3={"addressing_style": "path"})

        self._client = boto3.client("s3", **client_kwargs)
        return self._client

    async def upload(self, data: bytes, media_type: str) -> str:
        """Upload media data and return public URL.

        Args:
            data: Raw media bytes.
            media_type: MIME type (e.g., 'image/png', 'video/mp4').

        Returns:
            Public URL to access the uploaded media.

        Raises:
            Exception: If upload fails (caller should handle and fallback).
        """
        # Generate object key: prefix + date + content_hash.ext
        ext = self.EXTENSION_MAP.get(media_type, "bin")
        content_hash = hashlib.sha256(data).hexdigest()[:12]
        date_prefix = datetime.now().strftime("%Y-%m-%d")
        key = f"{self.config.prefix}{date_prefix}/{content_hash}.{ext}"

        # Upload to S3 (run sync boto3 in thread)
        client = self._get_client()
        await anyio.to_thread.run_sync(
            lambda: client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=data,
                ContentType=media_type,
            )
        )

        logger.debug("Uploaded media to s3://%s/%s", self.config.bucket, key)

        # Generate URL
        return await self._generate_url(key)

    async def _generate_url(self, key: str) -> str:
        """Generate public URL for the given key."""
        if self.config.url_mode == "cdn" and self.config.cdn_base_url:
            base = self.config.cdn_base_url.rstrip("/")
            return f"{base}/{key}"

        # Presigned URL
        client = self._get_client()
        url: str = await anyio.to_thread.run_sync(
            lambda: client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.config.bucket, "Key": key},
                ExpiresIn=self.config.presign_expires_seconds,
            )
        )
        return url


def create_s3_media_hook(
    config: S3MediaConfig,
) -> Callable[[RunContext[AgentContext], bytes, str], Awaitable[str | None]]:
    """Create a media-to-URL hook for S3 upload.

    The returned hook can be used for both image_to_url_hook and video_to_url_hook
    in ToolConfig. On upload failure, returns None to fallback to default behavior.

    Example:
        config = S3MediaConfig(bucket="my-bucket")
        hook = create_s3_media_hook(config)

        tool_config = ToolConfig(
            image_to_url_hook=hook,
            video_to_url_hook=hook,
        )

    Args:
        config: S3 configuration.

    Returns:
        Async hook function compatible with ToolConfig.
    """
    uploader = S3MediaUploader(config)

    async def hook(
        ctx: RunContext[AgentContext],
        data: bytes,
        media_type: str,
    ) -> str | None:
        try:
            return await uploader.upload(data, media_type)
        except Exception:
            logger.exception("Failed to upload media to S3, falling back to binary")
            return None

    return hook
