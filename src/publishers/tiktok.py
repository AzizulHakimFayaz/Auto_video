"""Official TikTok Content Posting API v2 publisher.

Implements the official TikTok Direct Post workflow:
1. Query creator info (`POST /v2/post/publish/creator_info/query/`) to verify
   account eligibility and allowed privacy levels (unaudited apps are restricted
   to `SELF_ONLY`).
2. Initialize video file upload (`POST /v2/post/publish/video/init/`).
3. PUT video binary stream to the returned `upload_url`.
4. Poll publication status (`POST /v2/post/publish/status/fetch/`) to confirm
   `PUBLISH_COMPLETE`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import requests

from src.metadata_generator import PlatformMetadataBundle
from src.publishers.base import BasePublisher, PublisherError, PublishResult

logger = logging.getLogger(__name__)

TIKTOK_CREATOR_INFO_URL = "https://open.tiktokapis.com/v2/post/publish/creator_info/query/"
TIKTOK_VIDEO_INIT_URL = "https://open.tiktokapis.com/v2/post/publish/video/init/"
TIKTOK_STATUS_FETCH_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"


class TikTokPublisher(BasePublisher):
    """Publishes vertical MP4 videos via the official TikTok Content Posting API v2."""

    platform_name = "tiktok"

    def is_enabled(self) -> bool:
        return self.config.enable_tiktok

    def validate_credentials(self) -> List[str]:
        missing: List[str] = []
        if not self.config.tiktok_access_token:
            missing.append("TIKTOK_ACCESS_TOKEN")
        return missing

    def _publish_live(
        self, video_path: Path, metadata: PlatformMetadataBundle
    ) -> PublishResult:
        token = self.config.tiktok_access_token.strip()
        auth_headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
        }

        # Step 1: Query creator info to verify permissions & allowed privacy_level
        try:
            info_resp = self.session.post(
                TIKTOK_CREATOR_INFO_URL,
                headers=auth_headers,
                json={},
                timeout=20,
            )
        except requests.RequestException as exc:
            raise PublisherError(
                f"[tiktok] Network error querying creator_info: {type(exc).__name__}",
                platform="tiktok",
                is_transient=True,
            ) from exc

        if info_resp.status_code != 200:
            raise PublisherError(
                f"[tiktok] creator_info query failed with HTTP {info_resp.status_code}. "
                "Verify TIKTOK_ACCESS_TOKEN and `video.publish` scope.",
                platform="tiktok",
                status_code=info_resp.status_code,
            )

        info_body = info_resp.json()
        err_code = info_body.get("error", {}).get("code", "ok")
        if err_code != "ok":
            raise PublisherError(
                f"[tiktok] creator_info API error ({err_code}): "
                f"{info_body.get('error', {}).get('message', '')}",
                platform="tiktok",
            )

        creator_data = info_body.get("data", {})
        allowed_privacy = creator_data.get("privacy_level_options", ["SELF_ONLY"])
        desired_privacy = metadata.tiktok.privacy_level
        if desired_privacy not in allowed_privacy:
            if "SELF_ONLY" in allowed_privacy:
                logger.warning(
                    "[tiktok] Desired privacy '%s' is not permitted for this app/account "
                    "(unaudited TikTok apps only allow SELF_ONLY). Falling back to SELF_ONLY.",
                    desired_privacy,
                )
                desired_privacy = "SELF_ONLY"
            else:
                raise PublisherError(
                    f"[tiktok] Account does not permit privacy_level={desired_privacy!r} "
                    f"(allowed={allowed_privacy}). App audit may be required.",
                    platform="tiktok",
                )

        # Step 2: Initialize Direct Post FILE_UPLOAD
        file_size = video_path.stat().st_size
        init_payload = {
            "post_info": {
                "title": metadata.tiktok.title,
                "privacy_level": desired_privacy,
                "disable_duet": False,
                "disable_comment": False,
                "disable_stitch": False,
                "video_cover_timestamp_ms": 1000,
            },
            "source_info": {
                "source": "FILE_UPLOAD",
                "video_size": file_size,
                "chunk_size": file_size,
                "total_chunk_count": 1,
            },
        }

        try:
            init_resp = self.session.post(
                TIKTOK_VIDEO_INIT_URL,
                headers=auth_headers,
                json=init_payload,
                timeout=30,
            )
        except requests.RequestException as exc:
            raise PublisherError(
                f"[tiktok] Network error initializing video upload: {type(exc).__name__}",
                platform="tiktok",
                is_transient=True,
            ) from exc

        if init_resp.status_code != 200:
            raise PublisherError(
                f"[tiktok] video/init failed with HTTP {init_resp.status_code}.",
                platform="tiktok",
                status_code=init_resp.status_code,
            )

        init_body = init_resp.json()
        init_err = init_body.get("error", {}).get("code", "ok")
        if init_err != "ok":
            raise PublisherError(
                f"[tiktok] video/init API error ({init_err}): "
                f"{init_body.get('error', {}).get('message', '')} "
                "(Note: unaudited apps require SELF_ONLY privacy or Content Posting API audit).",
                platform="tiktok",
            )

        init_data = init_body.get("data", {})
        publish_id = init_data.get("publish_id")
        upload_url = init_data.get("upload_url")
        if not publish_id or not upload_url:
            raise PublisherError(
                "[tiktok] video/init did not return publish_id and upload_url.",
                platform="tiktok",
            )

        # Step 3: PUT binary file chunk
        with open(video_path, "rb") as vf:
            video_bytes = vf.read()

        put_headers = {
            "Content-Type": "video/mp4",
            "Content-Length": str(file_size),
            "Content-Range": f"bytes 0-{file_size - 1}/{file_size}",
        }
        try:
            put_resp = self.session.put(
                upload_url,
                headers=put_headers,
                data=video_bytes,
                timeout=300,
            )
        except requests.RequestException as exc:
            raise PublisherError(
                f"[tiktok] Binary PUT upload failed: {type(exc).__name__}",
                platform="tiktok",
            ) from exc

        if put_resp.status_code not in {200, 201, 206}:
            raise PublisherError(
                f"[tiktok] Binary PUT to upload_url returned HTTP {put_resp.status_code}.",
                platform="tiktok",
                status_code=put_resp.status_code,
            )

        # Step 4: Poll status/fetch to confirm PUBLISH_COMPLETE
        for _ in range(12):
            try:
                st_resp = self.session.post(
                    TIKTOK_STATUS_FETCH_URL,
                    headers=auth_headers,
                    json={"publish_id": publish_id},
                    timeout=20,
                )
            except requests.RequestException as exc:
                raise PublisherError(
                    f"[tiktok] Error polling publish status for {publish_id}: {type(exc).__name__}",
                    platform="tiktok",
                ) from exc

            if st_resp.status_code != 200:
                raise PublisherError(
                    f"[tiktok] status/fetch returned HTTP {st_resp.status_code}.",
                    platform="tiktok",
                    status_code=st_resp.status_code,
                )

            st_body = st_resp.json()
            st_data = st_body.get("data", {})
            status_val = str(st_data.get("status", "")).upper()
            if status_val in {"PUBLISH_COMPLETE", "SEND_TO_USER_INBOX"}:
                logger.info(
                    "[tiktok] Confirmed TikTok publication: publish_id=%s, status=%s, privacy=%s",
                    publish_id,
                    status_val,
                    desired_privacy,
                )
                return PublishResult(
                    platform="tiktok",
                    status="published",
                    published=True,
                    platform_post_id=str(publish_id),
                    url="https://www.tiktok.com/",
                    message=f"Published to TikTok (publish_id={publish_id}, status={status_val}, privacy={desired_privacy}).",
                )
            if status_val == "FAILED":
                fail_reason = st_data.get("fail_reason", "unknown")
                raise PublisherError(
                    f"[tiktok] TikTok server reported publication FAILED (reason={fail_reason}).",
                    platform="tiktok",
                )
            self.sleep_fn(4.0)

        raise PublisherError(
            f"[tiktok] TikTok publication {publish_id} did not reach PUBLISH_COMPLETE within timeout.",
            platform="tiktok",
        )
