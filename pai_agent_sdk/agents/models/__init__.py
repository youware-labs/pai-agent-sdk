from pydantic_ai.models import (
    Model,
)
from pydantic_ai.models import infer_model as legacy_infer_model

from pai_agent_sdk.agents.models.gateway import infer_model as infer_gateway_model

__all__ = ["Model", "infer_model"]


def infer_model(model: str | Model, extra_headers: dict[str, str] | None = None) -> Model:
    """Infer model from string or return Model instance.

    Args:
        model: Model string or Model instance.
        extra_headers: Optional dict of extra HTTP headers for colorist provider.
            Useful for sticky routing via x-session-id header.
            Only applies to colorist provider (model@colorist format).

    Returns:
        The inferred Model instance.
    """
    if not isinstance(model, str):
        return legacy_infer_model(model)
    if "@" in model:
        gateway_name, model_name = model.split("@", 1)
        return infer_gateway_model(gateway_name, model_name, extra_headers=extra_headers)
    return legacy_infer_model(model)
