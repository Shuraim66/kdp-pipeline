"""Provider clients for the external APIs the pipeline depends on.

`fal` renders coloring-book images; `anthropic` writes listing metadata. Both
expose an async client class, a `get_*_provider()` factory that reads the key
from settings, and a result dataclass.
"""

from src.providers.anthropic import (
    AnthropicProvider,
    AnthropicProviderError,
    AnthropicResult,
    get_anthropic_provider,
)
from src.providers.fal import (
    FalImageResult,
    FalProvider,
    FalProviderError,
    get_fal_provider,
)

__all__ = [
    "AnthropicProvider",
    "AnthropicProviderError",
    "AnthropicResult",
    "FalImageResult",
    "FalProvider",
    "FalProviderError",
    "get_anthropic_provider",
    "get_fal_provider",
]
