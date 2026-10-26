"""Configuration package re-exporting the canonical configuration module."""

from src.config import (
     PROJECT_ROOT,
    AppConfig,
    ConfigurationError,
    SecretRedactingFilter,
    setup_logging,
)

__all__ = [
    "PROJECT_ROOT",
    "AppConfig",
    "ConfigurationError",
    "SecretRedactingFilter",
    "setup_logging",
]
