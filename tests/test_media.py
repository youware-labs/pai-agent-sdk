"""Tests for S3 media upload utilities."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pai_agent_sdk.media import MediaUploader, S3MediaConfig, S3MediaUploader, create_s3_media_hook


def test_media_uploader_protocol():
    """Test that S3MediaUploader implements MediaUploader protocol."""
    config = S3MediaConfig(bucket="test")
    uploader = S3MediaUploader(config)
    assert isinstance(uploader, MediaUploader)


def test_s3_media_config_defaults():
    """Test default values for S3MediaConfig."""
    config = S3MediaConfig(bucket="test-bucket")

    assert config.bucket == "test-bucket"
    assert config.region == "us-east-1"
    assert config.access_key_id is None
    assert config.secret_access_key is None
    assert config.endpoint_url is None
    assert config.prefix == ""
    assert config.url_mode == "presign"
    assert config.cdn_base_url is None
    assert config.presign_expires_seconds == 3600


def test_s3_media_config_cdn_requires_base_url():
    """Test that CDN mode requires cdn_base_url."""
    with pytest.raises(ValueError, match="cdn_base_url is required when url_mode='cdn'"):
        S3MediaConfig(bucket="test", url_mode="cdn")

    with pytest.raises(ValueError, match="cdn_base_url is required when url_mode='cdn'"):
        S3MediaConfig(bucket="test", url_mode="cdn", cdn_base_url=None)

    # Should work with cdn_base_url
    config = S3MediaConfig(bucket="test", url_mode="cdn", cdn_base_url="https://cdn.example.com")
    assert config.url_mode == "cdn"
    assert config.cdn_base_url == "https://cdn.example.com"


def test_s3_media_config_prefix_normalization():
    """Test that prefix is normalized to end with /."""
    # No trailing slash - should be added
    config = S3MediaConfig(bucket="test", prefix="uploads")
    assert config.prefix == "uploads/"

    # Already has trailing slash - should be unchanged
    config = S3MediaConfig(bucket="test", prefix="uploads/")
    assert config.prefix == "uploads/"

    # Empty prefix - should remain empty
    config = S3MediaConfig(bucket="test", prefix="")
    assert config.prefix == ""

    # Nested path without trailing slash
    config = S3MediaConfig(bucket="test", prefix="a/b/c")
    assert config.prefix == "a/b/c/"


def test_s3_media_config_full():
    """Test S3MediaConfig with all fields."""
    config = S3MediaConfig(
        bucket="my-bucket",
        region="ap-northeast-1",
        access_key_id="AKID",
        secret_access_key="SECRET",  # noqa: S106
        endpoint_url="https://s3.example.com",
        prefix="media/",
        url_mode="cdn",
        cdn_base_url="https://cdn.example.com",
        presign_expires_seconds=7200,
    )

    assert config.bucket == "my-bucket"
    assert config.region == "ap-northeast-1"
    assert config.access_key_id == "AKID"
    assert config.secret_access_key == "SECRET"  # noqa: S105
    assert config.endpoint_url == "https://s3.example.com"
    assert config.prefix == "media/"
    assert config.url_mode == "cdn"
    assert config.cdn_base_url == "https://cdn.example.com"
    assert config.presign_expires_seconds == 7200


def test_s3_media_uploader_extension_map():
    """Test MIME type to extension mapping."""
    config = S3MediaConfig(bucket="test")
    uploader = S3MediaUploader(config)

    assert uploader.EXTENSION_MAP["image/png"] == "png"
    assert uploader.EXTENSION_MAP["image/jpeg"] == "jpg"
    assert uploader.EXTENSION_MAP["video/mp4"] == "mp4"
    assert uploader.EXTENSION_MAP["video/webm"] == "webm"


@pytest.mark.asyncio
async def test_s3_media_uploader_upload_cdn_mode():
    """Test upload with CDN URL mode."""
    config = S3MediaConfig(
        bucket="test-bucket",
        prefix="uploads/",
        url_mode="cdn",
        cdn_base_url="https://cdn.example.com",
    )
    uploader = S3MediaUploader(config)

    # Mock boto3 client
    mock_client = MagicMock()

    with patch.object(uploader, "_get_client", return_value=mock_client):
        url = await uploader.upload(b"test image data", "image/png")

    # Verify put_object was called
    mock_client.put_object.assert_called_once()
    call_kwargs = mock_client.put_object.call_args[1]
    assert call_kwargs["Bucket"] == "test-bucket"
    assert call_kwargs["Body"] == b"test image data"
    assert call_kwargs["ContentType"] == "image/png"
    assert call_kwargs["Key"].startswith("uploads/")
    assert call_kwargs["Key"].endswith(".png")

    # Verify CDN URL format
    assert url.startswith("https://cdn.example.com/uploads/")
    assert url.endswith(".png")


@pytest.mark.asyncio
async def test_s3_media_uploader_upload_presign_mode():
    """Test upload with presigned URL mode."""
    config = S3MediaConfig(
        bucket="test-bucket",
        prefix="media/",
        url_mode="presign",
        presign_expires_seconds=3600,
    )
    uploader = S3MediaUploader(config)

    # Mock boto3 client
    mock_client = MagicMock()
    mock_client.generate_presigned_url.return_value = "https://s3.amazonaws.com/presigned-url"

    with patch.object(uploader, "_get_client", return_value=mock_client):
        url = await uploader.upload(b"video data", "video/mp4")

    # Verify put_object was called
    mock_client.put_object.assert_called_once()

    # Verify presigned URL generation
    mock_client.generate_presigned_url.assert_called_once()
    call_args = mock_client.generate_presigned_url.call_args
    assert call_args[0][0] == "get_object"
    assert call_args[1]["Params"]["Bucket"] == "test-bucket"
    assert call_args[1]["ExpiresIn"] == 3600

    assert url == "https://s3.amazonaws.com/presigned-url"


@pytest.mark.asyncio
async def test_s3_media_uploader_content_hash():
    """Test that same content produces same key (for deduplication)."""
    config = S3MediaConfig(bucket="test", prefix="")
    uploader = S3MediaUploader(config)

    mock_client = MagicMock()
    mock_client.generate_presigned_url.return_value = "https://example.com/url"

    data = b"identical content"

    with patch.object(uploader, "_get_client", return_value=mock_client):
        await uploader.upload(data, "image/png")
        key1 = mock_client.put_object.call_args[1]["Key"]

        await uploader.upload(data, "image/png")
        key2 = mock_client.put_object.call_args[1]["Key"]

    # Same content should produce same key (content hash based)
    assert key1 == key2


@pytest.mark.asyncio
async def test_create_s3_media_hook_success():
    """Test create_s3_media_hook returns working hook."""
    config = S3MediaConfig(
        bucket="test-bucket",
        url_mode="cdn",
        cdn_base_url="https://cdn.example.com",
    )

    hook = create_s3_media_hook(config)

    # Mock the uploader inside the hook
    with patch("pai_agent_sdk.media.S3MediaUploader.upload") as mock_upload:
        mock_upload.return_value = "https://cdn.example.com/2024-01-01/abc123.png"

        # RunContext is not used, pass None
        result = await hook(None, b"image data", "image/png")  # type: ignore[arg-type]

    assert result == "https://cdn.example.com/2024-01-01/abc123.png"


@pytest.mark.asyncio
async def test_create_s3_media_hook_fallback_on_error():
    """Test that hook returns None on upload failure (fallback)."""
    config = S3MediaConfig(bucket="test-bucket")
    hook = create_s3_media_hook(config)

    # Mock upload to raise an exception
    with patch("pai_agent_sdk.media.S3MediaUploader.upload") as mock_upload:
        mock_upload.side_effect = Exception("S3 error")

        result = await hook(None, b"image data", "image/png")  # type: ignore[arg-type]

    # Should return None for fallback, not raise
    assert result is None


def test_s3_media_uploader_get_client_creates_boto3_client():
    """Test that _get_client creates boto3 client with correct params."""
    config = S3MediaConfig(
        bucket="test",
        region="eu-west-1",
        access_key_id="AKID",
        secret_access_key="SECRET",  # noqa: S106
        endpoint_url="https://minio.example.com",
    )
    uploader = S3MediaUploader(config)

    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client.return_value = MagicMock()
        uploader._get_client()

        mock_boto3_client.assert_called_once_with(
            "s3",
            region_name="eu-west-1",
            aws_access_key_id="AKID",
            aws_secret_access_key="SECRET",  # noqa: S106
            endpoint_url="https://minio.example.com",
        )


def test_s3_media_uploader_get_client_default_credentials():
    """Test that _get_client uses default credential chain when no explicit creds."""
    config = S3MediaConfig(bucket="test", region="us-west-2")
    uploader = S3MediaUploader(config)

    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client.return_value = MagicMock()
        uploader._get_client()

        mock_boto3_client.assert_called_once_with("s3", region_name="us-west-2")


def test_s3_media_uploader_get_client_caches():
    """Test that _get_client caches the client."""
    config = S3MediaConfig(bucket="test")
    uploader = S3MediaUploader(config)

    with patch("boto3.client") as mock_boto3_client:
        mock_client = MagicMock()
        mock_boto3_client.return_value = mock_client

        client1 = uploader._get_client()
        client2 = uploader._get_client()

        # Should only create once
        mock_boto3_client.assert_called_once()
        assert client1 is client2


def test_s3_media_uploader_force_path_style():
    """Test that force_path_style configures path-style addressing."""
    config = S3MediaConfig(
        bucket="test",
        endpoint_url="https://minio.example.com",
        force_path_style=True,
    )
    uploader = S3MediaUploader(config)

    with patch("boto3.client") as mock_boto3_client:
        mock_boto3_client.return_value = MagicMock()
        uploader._get_client()

        call_kwargs = mock_boto3_client.call_args[1]
        assert "config" in call_kwargs
        # Verify BotoConfig was passed with path-style addressing
        boto_config = call_kwargs["config"]
        assert boto_config.s3["addressing_style"] == "path"


def test_s3_media_config_force_path_style_default():
    """Test that force_path_style defaults to False."""
    config = S3MediaConfig(bucket="test")
    assert config.force_path_style is False
