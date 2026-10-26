"""Official Meta Instagram Graph API publisher for Instagram Reels.

Supports both:
- Direct Resumable Binary Upload (`upload_type=resumable` -> `rupload.facebook.com/ig-api-upload/...`),
  requiring NO external hosting server.
- Public URL ingestion (`video_url`) when `INSTAGRAM_VIDEO_PUBLIC_URL_BASE` is configured.

Workflow:
1. Create Reel media container (`POST /{ig_user_id}/media` with `media_type=REELS`).
2. If `upload_type=resumable`, stream local MP4 binary to `rupload.facebook.com/ig-api-upload/{version}/{container_id}`.
3. Poll container `status_code` until `FINISHED`.
4. Publish container (`POST /{ig_user_id}/media_publish`).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import requests

from src.metadata_generator import PlatformMetadataBundle
from src.publishers.base import BasePublisher, PublisherError, PublishResult

logger = logging.getLogger(__name__)


class InstagramPublisher(BasePublisher):
    """Publishes vertical MP4 videos to Instagram Reels via the Meta Graph API."""

    platform_name = "instagram"

    def is_enabled(self) -> bool:
        return self.config.enable_instagram

    def validate_credentials(self) -> List[str]:
        missing: List[str] = []
        if not self.config.instagram_account_id:
            missing.append("INSTAGRAM_ACCOUNT_ID")
        if not self.config.instagram_access_token:
            missing.append("INSTAGRAM_ACCESS_TOKEN")
        return missing

    def _publish_live(
        self, video_path: Path, metadata: PlatformMetadataBundle
    ) -> PublishResult:
        version = self.config.instagram_graph_version.strip("/")
        ig_id = self.config.instagram_account_id.strip()
        token = self.config.instagram_access_token.strip()
        public_base = self.config.instagram_video_public_url_base.rstrip("/")

        # Step 1: Create Reels media container (resumable binary upload by default, or video_url if configured)
        container_endpoint = f"https://graph.facebook.com/{version}/{ig_id}/media"
        container_payload = {
            "media_type": "REELS",
            "caption": metadata.instagram.caption,
            "share_to_feed": "true",
            "access_token": token,
        }
        use_resumable = not bool(public_base)
        if use_resumable:
            container_payload["upload_type"] = "resumable"
        else:
            container_payload["video_url"] = f"{public_base}/{video_path.name}"

        try:
            create_resp = self.session.post(
                container_endpoint,
                data=container_payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise PublisherError(
                f"[instagram] Network error creating media container: {type(exc).__name__}",
                platform="instagram",
                is_transient=True,
            ) from exc

        if create_resp.status_code != 200:
            raise PublisherError(
                f"[instagram] Media container creation failed (HTTP {create_resp.status_code}). "
                "Ensure the account is an eligible Instagram Professional account with "
                "instagram_content_publish / instagram_business_content_publish permission.",
                platform="instagram",
                status_code=create_resp.status_code,
            )

        container_id = create_resp.json().get("id")
        if not container_id:
            raise PublisherError(
                "[instagram] Graph API did not return a container ID.",
                platform="instagram",
            )

        # Step 2: If using direct resumable upload, stream binary MP4 to rupload.facebook.com
        if use_resumable:
            file_size = video_path.stat().st_size
            with open(video_path, "rb") as vf:
                binary_data = vf.read()

            rupload_url = f"https://rupload.facebook.com/ig-api-upload/{version}/{container_id}"
            rupload_headers = {
                "Authorization": f"OAuth {token}",
                "offset": "0",
                "file_size": str(file_size),
                "Content-Type": "application/octet-stream",
            }
            try:
                up_resp = self.session.post(
                    rupload_url,
                    headers=rupload_headers,
                    data=binary_data,
                    timeout=300,
                )
            except requests.RequestException as exc:
                raise PublisherError(
                    f"[instagram] Binary upload to rupload.facebook.com failed: {type(exc).__name__}",
                    platform="instagram",
                ) from exc

            if up_resp.status_code != 200:
                raise PublisherError(
                    f"[instagram] Binary upload to rupload.facebook.com failed with HTTP {up_resp.status_code}.",
                    platform="instagram",
                    status_code=up_resp.status_code,
                )

        # Step 3: Poll container status until FINISHED
        status_endpoint = f"https://graph.facebook.com/{version}/{container_id}"
        container_ready = False
        for _ in range(1, 16):
            try:
                status_resp = self.session.get(
                    status_endpoint,
                    params={"fields": "status_code,status", "access_token": token},
                    timeout=20,
                )
            except requests.RequestException as exc:
                raise PublisherError(
                    f"[instagram] Error polling container {container_id}: {type(exc).__name__}",
                    platform="instagram",
                ) from exc

            if status_resp.status_code != 200:
                raise PublisherError(
                    f"[instagram] Container status check failed with HTTP {status_resp.status_code}.",
                    platform="instagram",
                    status_code=status_resp.status_code,
                )

            s_data = status_resp.json()
            status_code = str(s_data.get("status_code", "")).upper()
            if status_code == "FINISHED":
                container_ready = True
                break
            if status_code in {"ERROR", "EXPIRED"}:
                raise PublisherError(
                    f"[instagram] Media container processing failed with status_code={status_code}: "
                    f"{s_data.get('status', '')}",
                    platform="instagram",
                )
            self.sleep_fn(4.0)

        if not container_ready:
            raise PublisherError(
                f"[instagram] Media container {container_id} did not reach FINISHED state in time.",
                platform="instagram",
            )

        # Step 4: Publish the processed container
        publish_endpoint = f"https://graph.facebook.com/{version}/{ig_id}/media_publish"
        try:
            pub_resp = self.session.post(
                publish_endpoint,
                data={"creation_id": container_id, "access_token": token},
                timeout=30,
            )
        except requests.RequestException as exc:
            raise PublisherError(
                f"[instagram] Network error publishing container {container_id}: {type(exc).__name__}",
                platform="instagram",
            ) from exc

        if pub_resp.status_code != 200:
            raise PublisherError(
                f"[instagram] media_publish failed with HTTP {pub_resp.status_code}.",
                platform="instagram",
                status_code=pub_resp.status_code,
            )

        media_id = pub_resp.json().get("id")
        if not media_id:
            raise PublisherError(
                "[instagram] media_publish response did not contain a published media ID.",
                platform="instagram",
            )

        logger.info("[instagram] Confirmed Instagram Reel publication: media_id=%s", media_id)
        return PublishResult(
            platform="instagram",
            status="published",
            published=True,
            platform_post_id=str(media_id),
            url="https://www.instagram.com/",
            message=f"Published Instagram Reel (media_id={media_id}, container_id={container_id}).",
        )
