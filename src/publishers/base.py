"""Base classes and shared safeguards for official social platform publishers."""

from __future__ import annotations

import abc
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests

from src.config import AppConfig
from src.metadata_generator import PlatformMetadataBundle

logger = logging.getLogger(__name__)


class PublisherError(RuntimeError):
    """Base error raised when a platform upload or API check fails."""

    def __init__(
        self,
        message: str,
        platform: str = "",
        status_code: Optional[int] = None,
        is_transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.platform = platform
        self.status_code = status_code
        self.is_transient = is_transient


class MissingCredentialsError(PublisherError):
    """Raised when an enabled publisher is missing required OAuth credentials."""


@dataclass
class PublishResult:
    """Confirmed outcome of a platform publishing attempt."""

    platform: str
    status: str  # "published", "dry_run", "skipped", "failed"
    published: bool
    platform_post_id: Optional[str] = None
    url: Optional[str] = None
    message: str = ""
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BasePublisher(abc.ABC):
    """Abstract base class for official platform publishing adapters."""

    platform_name: str = "base"

    def __init__(
        self,
        config: AppConfig,
        session: Optional[requests.Session] = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.config = config
        self.session = session or requests.Session()
        self.sleep_fn = sleep_fn

    @abc.abstractmethod
    def is_enabled(self) -> bool:
        """Return True if this platform is enabled in configuration."""

    @abc.abstractmethod
    def validate_credentials(self) -> List[str]:
        """Return list of missing environment variable names for this platform."""

    @abc.abstractmethod
    def _publish_live(
        self, video_path: Path, metadata: PlatformMetadataBundle
    ) -> PublishResult:
        """Perform the actual official API upload and status confirmation.

        Only invoked when `is_enabled()` is True, `dry_run` is False, and credentials are valid.
        """

    def publish(
        self, video_path: Path, metadata: PlatformMetadataBundle
    ) -> PublishResult:
        """Execute publishing with dry-run, credential, file, and retry safeguards."""
        if not self.is_enabled():
            logger.info("[%s] Platform is disabled; skipping.", self.platform_name)
            return PublishResult(
                platform=self.platform_name,
                status="skipped",
                published=False,
                message=f"{self.platform_name} publishing is disabled.",
            )

        if not video_path.exists() or not video_path.is_file() or video_path.stat().st_size < 1024:
            raise PublisherError(
                f"[{self.platform_name}] Video file is missing or invalid: {video_path}",
                platform=self.platform_name,
            )

        # Safeguard: NEVER publish while dry_run mode is active
        if self.config.dry_run:
            missing = self.validate_credentials()
            cred_note = (
                "credentials configured"
                if not missing
                else f"missing credentials for live mode: {', '.join(missing)}"
            )
            logger.info(
                "[%s] DRY_RUN=true: Skipping live API upload for '%s' (%s).",
                self.platform_name,
                video_path.name,
                cred_note,
            )
            return PublishResult(
                platform=self.platform_name,
                status="dry_run",
                published=False,
                message=f"Dry-run mode active ({cred_note}). No external API request was made.",
            )

        missing_vars = self.validate_credentials()
        if missing_vars:
            raise MissingCredentialsError(
                f"[{self.platform_name}] Cannot publish: missing required credentials: "
                f"{', '.join(missing_vars)}",
                platform=self.platform_name,
            )

        max_attempts = 2
        last_err: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            try:
                return self._publish_live(video_path=video_path, metadata=metadata)
            except PublisherError as exc:
                last_err = exc
                if not exc.is_transient or attempt >= max_attempts:
                    raise
                logger.warning(
                    "[%s] Transient error on attempt %d/%d (%s); retrying...",
                    self.platform_name,
                    attempt,
                    max_attempts,
                    exc,
                )
                self.sleep_fn(float(2**attempt))

        raise PublisherError(
            f"[{self.platform_name}] Upload failed after {max_attempts} attempts: {last_err}",
            platform=self.platform_name,
        )
