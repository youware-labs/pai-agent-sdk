"""Dialog handling tools for browser control."""

from __future__ import annotations

import asyncio
from typing import Any

from pai_agent_sdk._logger import get_logger
from pai_agent_sdk.toolsets.browser_use._tools import get_browser_session
from pai_agent_sdk.toolsets.browser_use.tools._types import DialogResult

logger = get_logger(__name__)


async def handle_dialog(
    accept: bool = True,
    prompt_text: str | None = None,
    timeout: int = 5000,
) -> dict[str, Any]:
    """Handle JavaScript dialog (alert, confirm, prompt).

    This function waits for a dialog to appear and handles it using CDP events.

    Args:
        accept: Whether to accept (True) or dismiss (False) the dialog
        prompt_text: Text to enter for prompt dialogs (optional)
        timeout: Maximum time to wait for dialog in milliseconds (default: 5000)

    Returns:
        DialogResult dictionary
    """
    logger.info(f"Waiting for dialog (accept={accept}, timeout={timeout}ms)")
    session = get_browser_session()

    try:
        # Enable Page domain
        await session.cdp_client.send.Page.enable(session_id=session.page)

        timeout_seconds = timeout / 1000
        start_time = asyncio.get_event_loop().time()
        dialog_detected = False

        # Event handler for dialog opening
        def on_dialog_opening(event: Any, session_id: str | None) -> None:
            nonlocal dialog_detected
            dialog_detected = True
            logger.debug(f"Dialog detected: type={event.get('type')}, message={event.get('message')}")

        # Register event handler
        session.cdp_client.register.Page.javascriptDialogOpening(on_dialog_opening)

        try:
            # First, try to handle any existing dialog immediately
            try:
                await session.cdp_client.send.Page.handleJavaScriptDialog(
                    params={
                        "accept": accept,
                        "promptText": prompt_text if prompt_text else "",
                    },
                    session_id=session.page,
                )
                logger.info("Existing dialog handled successfully")
                return DialogResult(
                    status="success",
                    accepted=accept,
                    prompt_text=prompt_text,
                ).model_dump()
            except Exception:
                # No dialog present, will wait for one
                logger.debug("No existing dialog found")

            # Wait for dialog event or timeout
            while True:
                elapsed = asyncio.get_event_loop().time() - start_time
                if elapsed > timeout_seconds:
                    logger.warning(f"No dialog detected within {timeout}ms")
                    return DialogResult(
                        status="no_dialog",
                        error_message=f"No dialog detected within {timeout}ms",
                    ).model_dump()

                if dialog_detected:
                    # Dialog event fired, now handle it
                    try:
                        await session.cdp_client.send.Page.handleJavaScriptDialog(
                            params={
                                "accept": accept,
                                "promptText": prompt_text if prompt_text else "",
                            },
                            session_id=session.page,
                        )
                        logger.info(f"Dialog handled after {elapsed:.2f}s")
                        return DialogResult(
                            status="success",
                            accepted=accept,
                            prompt_text=prompt_text,
                        ).model_dump()
                    except Exception:
                        logger.warning("Failed to handle dialog")
                        # Dialog might have been closed already, treat as success
                        return DialogResult(
                            status="success",
                            accepted=accept,
                            prompt_text=prompt_text,
                        ).model_dump()

                # Brief sleep to avoid busy waiting
                await asyncio.sleep(0.05)

        finally:
            # Event handlers are cleaned up when session/client is closed
            pass

    except Exception as e:  # pragma: no cover
        logger.exception("Error handling dialog")
        return DialogResult(
            status="error",
            error_message=str(e),
        ).model_dump()


async def dismiss_dialog(timeout: int = 5000) -> dict[str, Any]:
    """Dismiss/cancel a JavaScript dialog.

    Convenience function that calls handle_dialog with accept=False.

    Args:
        timeout: Maximum time to wait for dialog in milliseconds (default: 5000)

    Returns:
        DialogResult dictionary
    """
    return await handle_dialog(accept=False, timeout=timeout)


async def accept_dialog(prompt_text: str | None = None, timeout: int = 5000) -> dict[str, Any]:
    """Accept/confirm a JavaScript dialog.

    Convenience function that calls handle_dialog with accept=True.

    Args:
        prompt_text: Text to enter for prompt dialogs (optional)
        timeout: Maximum time to wait for dialog in milliseconds (default: 5000)

    Returns:
        DialogResult dictionary
    """
    return await handle_dialog(accept=True, prompt_text=prompt_text, timeout=timeout)
