from pydantic_ai import ModelMessage, ModelResponse, RunContext, ThinkingPart

from pai_agent_sdk._logger import logger
from pai_agent_sdk.context import AgentContext


def handle_model_switch(ctx: RunContext[AgentContext], message_history: list[ModelMessage]) -> list[ModelMessage]:
    """Handle model switching by removing incompatible ThinkingParts.

    Different models produce ThinkingParts in different formats, which may be
    incompatible with each other. This filter removes all ThinkingParts when
    the model has changed since the last response.

    Args:
        ctx: Runtime context containing AgentContext with current model configuration
        message_history: List of messages to process

    Returns:
        The modified message history with ThinkingParts removed if model changed
    """
    if not message_history:
        return message_history

    # Find the last ModelResponse to get its model_name
    last_model_name = None
    for message in reversed(message_history):
        if isinstance(message, ModelResponse) and message.model_name:
            last_model_name = message.model_name
            break

    # If no previous response or model hasn't changed, keep ThinkingParts
    current_model_id = ctx.model.model_name
    if last_model_name is None or last_model_name == current_model_id:
        return message_history

    # Model changed, remove all ThinkingParts from ModelResponse messages
    # ThinkingPart only exists in ModelResponsePart, not in ModelRequestPart
    logger.info(f"Model switched from {last_model_name} to {current_model_id}, removing ThinkingParts")
    for message in message_history:
        if isinstance(message, ModelResponse):
            message.parts = [part for part in message.parts if not isinstance(part, ThinkingPart)]

    return message_history
