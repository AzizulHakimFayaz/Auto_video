"""Official YouTube Data API v3 publisher for YouTube Shorts.

Uses OAuth 2.0 refresh token exchange and the official YouTube Data API v3
resumable upload protocol (`https://www.googleapis.com/upload/youtube/v3/videos`).
Handles quota errors, upload confirmation, and unverified OAuth project restrictions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import requests

from src.metadata_generator import PlatformMetadataBundle
from src.publishers.base import BasePublisher, PublisherError, PublishResult

logger = logging.getLogger(__name__)

GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
YOUTUBE_UPLOAD_URL = (
    "https://www.googleapis.com/upload/youtube/v3/videos"
    "?part=snippet,status&uploadType=resumable"
)
YOUTUBE_VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


class YouTubePublisher(BasePublisher):
    """Publishes vertical MP4 videos to YouTube Shorts via the official YouTube Data API v3."""

    platform_name = "youtube"

    def is_enabled(self) -> bool:
        return self.config.enable_youtube

    def validate_credentials(self) -> List[str]:
        missing: List[str] = []
        if not self.config.youtube_client_id:
            missing.append("YOUTUBE_CLIENT_ID")
        if not self.config.youtube_client_secret:
            missing.append("YOUTUBE_CLIENT_SECRET")
        if not self.config.youtube_refresh_token:
            missing.append("YOUTUBE_REFRESH_TOKEN")
        return missing

    def _refresh_access_token(self) -> str:
        """Exchange OAuth2 refresh token for a short-lived access token."""
        try:
            resp = self.session.post(
                GOOGLE_TOKEN_URL,
                data={
                    "client_id": self.config.youtube_client_id,
                    "client_secret": self.config.youtube_client_secret,
                    "refresh_token": self.config.youtube_refresh_token,
                    "grant_type": "refresh_token",
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            raise PublisherError(
                f"[youtube] Network failure refreshing OAuth token: {type(exc).__name__}",
                platform="youtube",
                is_transient=True,
            ) from exc

        if resp.status_code != 200:
            raise PublisherError(
                f"[youtube] OAuth token refresh failed with HTTP {resp.status_code}. "
                "Verify YOUTUBE_CLIENT_ID, YOUTUBE_CLIENT_SECRET, and YOUTUBE_REFRESH_TOKEN.",
                platform="youtube",
                status_code=resp.status_code,
                is_transient=resp.status_code >= 500,
            )

        data = resp.json()
        token = data.get("access_token")
        if not token:
            raise PublisherError(
                "[youtube] OAuth token response did not include an access_token.",
                platform="youtube",
            )
        return str(token)

    def _publish_live(
        self, video_path: Path, metadata: PlatformMetadataBundle
    ) -> PublishResult:
        access_token = self._refresh_access_token()
        yt_meta = metadata.youtube
        file_size = video_path.stat().st_size

        body = {
            "snippet": {
                "title": yt_meta.title,
                "description": yt_meta.description,
                "tags": yt_meta.tags,
                "categoryId": yt_meta.category_id,
            },
            "status": {
                "privacyStatus": yt_meta.privacy_status,
                "selfDeclaredMadeForKids": False,
            },
        }

        init_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json; charset=UTF-8",
            "X-Upload-Content-Length": str(file_size),
            "X-Upload-Content-Type": "video/mp4",
        }

        try:
            init_resp = self.session.post(
                YOUTUBE_UPLOAD_URL,
                headers=init_headers,
                json=body,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise PublisherError(
                f"[youtube] Failed to initialize resumable upload: {type(exc).__name__}",
                platform="youtube",
                is_transient=True,
            ) from exc

        if init_resp.status_code == 403 and "quotaExceeded" in init_resp.text:
            raise PublisherError(
                "[youtube] YouTube Data API v3 daily quota exceeded (HTTP 403 quotaExceeded).",
                platform="youtube",
                status_code=403,
                is_transient=False,
            )

        if init_resp.status_code not in {200, 201}:
            raise PublisherError(
                f"[youtube] Resumable upload initialization failed with HTTP {init_resp.status_code}.",
                platform="youtube",
                status_code=init_resp.status_code,
                is_transient=init_resp.status_code >= 500,
            )

        upload_url = init_resp.headers.get("Location")
        if not upload_url:
            raise PublisherError(
                "[youtube] Resumable upload initialization did not return a Location header.",
                platform="youtube",
            )

        # Upload binary video file to the single-use resumable session URI
        with open(video_path, "rb") as vf:
            video_bytes = vf.read()

        upload_headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Length": str(file_size),
            "Content-Type": "video/mp4",
        }

        try:
            put_resp = self.session.put(
                upload_url,
                headers=upload_headers,
                data=video_bytes,
                timeout=300,
            )
        except requests.RequestException as exc:
            raise PublisherError(
                f"[youtube] Network error during binary video upload: {type(exc).__name__}",
                platform="youtube",
                is_transient=False,  # Avoid blind retry of full binary if uncertain
            ) from exc

        if put_resp.status_code not in {200, 201}:
            raise PublisherError(
                f"[youtube] Video binary upload failed with HTTP {put_resp.status_code}.",
                platform="youtube",
                status_code=put_resp.status_code,
            )

        resp_data = put_resp.json()
        video_id = resp_data.get("id")
        if not video_id:
            raise PublisherError(
                "[youtube] Upload completed but YouTube API did not return a video ID.",
                platform="youtube",
            )

        upload_status = (
            resp_data.get("status", {}).get("uploadStatus", "uploaded")
            if isinstance(resp_data.get("status"), dict)
            else "uploaded"
        )
        privacy = (
            resp_data.get("status", {}).get("privacyStatus", yt_meta.privacy_status)
            if isinstance(resp_data.get("status"), dict)
            else yt_meta.privacy_status
        )

        url = f"https://www.youtube.com/shorts/{video_id}"
        logger.info(
            "[youtube] Confirmed YouTube Shorts upload: id=%s, status=%s, privacy=%s",
            video_id,
            upload_status,
            privacy,
        )
        return PublishResult(
            platform="youtube",
            status="published",
            published=True,
            platform_post_id=str(video_id),
            url=url,
            message=f"Uploaded to YouTube Shorts (uploadStatus={upload_status}, privacy={privacy}).",
        )
