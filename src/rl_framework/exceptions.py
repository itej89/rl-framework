"""Domain exception hierarchy for rl_framework.

All exceptions inherit from RLFrameworkError so callers can catch
the base class when they don't care about the specific failure type.
"""

__all__ = [
    "RLFrameworkError",
    "EnvironmentError",
    "AgentError",
    "ConfigurationError",
]


class RLFrameworkError(Exception):
    """Base class for all rl_framework exceptions."""


class EnvironmentError(RLFrameworkError):
    """Raised when an environment operation fails."""


class AgentError(RLFrameworkError):
    """Raised when an agent operation fails."""


class ConfigurationError(RLFrameworkError):
    """Raised when configuration values are invalid."""
