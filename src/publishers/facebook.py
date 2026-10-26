"""Official Meta Facebook Page Reels API publisher.

Implements the official 3-step Facebook Page Reels publishing workflow:
1. Initialize session (`POST /{page_id}/video_reels` with `upload_phase=start`).
2. Upload binary MP4 to `rupload.facebook.com` using `Authorization: OAuth <token>`.
3. Finish and publish (`POST /{page_id}/video_reels` with `upload_phase=finish`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import requests

from src.metadata_generator import PlatformMetadataBundle
from src.publishers.base import BasePublisher, PublisherError, PublishResult

logger = logging.getLogger(__name__)


class FacebookPublisher(BasePublisher):
    """Publishes vertical MP4 videos to Facebook Reels via the Page Graph API."""

    platform_name = "facebook"

    def is_enabled(self) -> bool:
        return self.config.enable_facebook

    def validate_credentials(self) -> List[str]:
        missing: List[str] = []
        if not self.config.facebook_page_id:
            missing.append("FACEBOOK_PAGE_ID")
        if not self.config.facebook_page_access_token:
            missing.append("FACEBOOK_PAGE_ACCESS_TOKEN")
        return missing

    def _publish_live(
        self, video_path: Path, metadata: PlatformMetadataBundle
    ) -> PublishResult:
        version = self.config.facebook_graph_version.strip("/")
        page_id = self.config.facebook_page_id.strip()
        token = self.config.facebook_page_access_token.strip()
        reels_endpoint = f"https://graph.facebook.com/{version}/{page_id}/video_reels"

        # Step 1: Initialize upload session
        try:
            start_resp = self.session.post(
                reels_endpoint,
                data={"upload_phase": "start", "access_token": token},
                timeout=30,
            )
        except requests.RequestException as exc:
            raise PublisherError(
                f"[facebook] Network error starting Reels upload session: {type(exc).__name__}",
                platform="facebook",
                is_transient=True,
            ) from exc

        if start_resp.status_code != 200:
            raise PublisherError(
                f"[facebook] upload_phase=start failed with HTTP {start_resp.status_code}. "
                "Verify FACEBOOK_PAGE_ID and pages_manage_posts permission.",
                platform="facebook",
                status_code=start_resp.status_code,
            )

        start_data = start_resp.json()
        video_id = start_data.get("video_id")
        upload_url = start_data.get("upload_url")
        if not video_id or not upload_url:
            raise PublisherError(
                "[facebook] upload_phase=start did not return video_id and upload_url.",
                platform="facebook",
            )

        # Step 2: Binary upload to rupload.facebook.com
        file_size = video_path.stat().st_size
        with open(video_path, "rb") as vf:
            binary_data = vf.read()

        rupload_headers = {
            "Authorization": f"OAuth {token}",
            "offset": "0",
            "file_size": str(file_size),
            "Content-Type": "application/octet-stream",
        }

        try:
            up_resp = self.session.post(
                upload_url,
                headers=rupload_headers,
                data=binary_data,
                timeout=300,
            )
        except requests.RequestException as exc:
            raise PublisherError(
                f"[facebook] Binary transfer to rupload.facebook.com failed: {type(exc).__name__}",
                platform="facebook",
            ) from exc

        if up_resp.status_code != 200 or not up_resp.json().get("success", False):
            raise PublisherError(
                f"[facebook] Binary upload to rupload.facebook.com did not succeed (HTTP {up_resp.status_code}).",
                platform="facebook",
                status_code=up_resp.status_code,
            )

        # Step 3: Finish and publish Reel
        try:
            finish_resp = self.session.post(
                reels_endpoint,
                data={
                    "upload_phase": "finish",
                    "video_id": video_id,
                    "video_state": "PUBLISHED",
                    "description": metadata.facebook.description,
                    "access_token": token,
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            raise PublisherError(
                f"[facebook] Network error finalizing Reel {video_id}: {type(exc).__name__}",
                platform="facebook",
            ) from exc

        if finish_resp.status_code != 200 or not finish_resp.json().get("success", False):
            raise PublisherError(
                f"[facebook] upload_phase=finish failed with HTTP {finish_resp.status_code}.",
                platform="facebook",
                status_code=finish_resp.status_code,
            )

        url = f"https://www.facebook.com/reel/{video_id}"
        logger.info("[facebook] Confirmed Facebook Page Reel publication: video_id=%s", video_id)
        return PublishResult(
            platform="facebook",
            status="published",
            published=True,
            platform_post_id=str(video_id),
            url=url,
            message=f"Published Facebook Page Reel (video_id={video_id}).",
        )
