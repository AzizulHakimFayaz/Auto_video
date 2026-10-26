"""Modular official social platform publishing package."""

from src.publishers.base import (
    BasePublisher,
    MissingCredentialsError,
    PublisherError,
    PublishResult,
)
from src.publishers.facebook import FacebookPublisher
from src.publishers.instagram import InstagramPublisher
from src.publishers.tiktok import TikTokPublisher
from src.publishers.youtube import YouTubePublisher

__all__ = [
    "BasePublisher",
    "MissingCredentialsError",
    "PublisherError",
    "PublishResult",
    "YouTubePublisher",
    "InstagramPublisher",
    "FacebookPublisher",
    "TikTokPublisher",
]
