"""Unit tests for platform publishers, dry-run safeguards, missing credentials, and isolated failures."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pytest

from src.config import AppConfig
from src.main import DailyVideoPipeline
from src.metadata_generator import (
    FacebookMetadata,
    InstagramMetadata,
    PlatformMetadataBundle,
    TikTokMetadata,
    YouTubeMetadata,
)
from src.publishers import (
    FacebookPublisher,
    InstagramPublisher,
    MissingCredentialsError,
    PublisherError,
    TikTokPublisher,
    YouTubePublisher,
)


def _dummy_video(tmp_path: Path) -> Path:
    vid = tmp_path / "video.mp4"
    vid.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4096)
    return vid


def _sample_bundle() -> PlatformMetadataBundle:
    return PlatformMetadataBundle(
        production_date="2026-10-26",
        run_id="run_2026-10-26_test",
        story_title="The Last Umbrella",
        language="English",
        mood="hopeful",
        youtube=YouTubeMetadata(
            title="The Last Umbrella #Shorts",
            description="Emotional story #Shorts",
            tags=["Shorts", "story"],
        ),
        instagram=InstagramMetadata(caption="The Last Umbrella #reels"),
        facebook=FacebookMetadata(
            title="The Last Umbrella", description="The Last Umbrella #Reels"
        ),
        tiktok=TikTokMetadata(
            title="The Last Umbrella #storytime", privacy_level="PUBLIC_TO_EVERYONE"
        ),
        attributions=["Visual: CC0"],
    )


class FakePublisherResponse:
    def __init__(
        self,
        status_code: int = 200,
        json_data: Dict[str, Any] | None = None,
        headers: Dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json = json_data or {}
        self.headers = headers or {}
        self.text = text

    def json(self) -> Dict[str, Any]:
        return self._json


class FakeSession:
    def __init__(self, post_fn=None, put_fn=None, get_fn=None) -> None:
        self._post_fn = post_fn
        self._put_fn = put_fn
        self._get_fn = get_fn
        self.calls: list[tuple[str, str]] = []

    def post(self, url: str, **kwargs: Any) -> FakePublisherResponse:
        self.calls.append(("POST", url))
        return self._post_fn(url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> FakePublisherResponse:
        self.calls.append(("PUT", url))
        return self._put_fn(url, **kwargs)

    def get(self, url: str, **kwargs: Any) -> FakePublisherResponse:
        self.calls.append(("GET", url))
        return self._get_fn(url, **kwargs)


def test_dry_run_never_makes_live_network_calls(tmp_path: Path) -> None:
    vid = _dummy_video(tmp_path)
    bundle = _sample_bundle()
    cfg = AppConfig(
        dry_run=True,
        enable_youtube=True,
        enable_instagram=True,
        enable_facebook=True,
        enable_tiktok=True,
    )
    session = FakeSession()

    for pub_cls in (YouTubePublisher, InstagramPublisher, FacebookPublisher, TikTokPublisher):
        pub = pub_cls(config=cfg, session=session)
        res = pub.publish(vid, bundle)
        assert res.status == "dry_run"
        assert res.published is False

    assert len(session.calls) == 0


def test_missing_credentials_raises_clear_error_when_live(tmp_path: Path) -> None:
    vid = _dummy_video(tmp_path)
    bundle = _sample_bundle()
    cfg = AppConfig(
        dry_run=False,
        enable_youtube=True,
        youtube_client_id="",  # Missing!
    )
    yt = YouTubePublisher(config=cfg)
    with pytest.raises(MissingCredentialsError, match="YOUTUBE_CLIENT_ID"):
        yt.publish(vid, bundle)


def test_youtube_and_tiktok_live_upload_and_isolation_on_failure(tmp_path: Path) -> None:
    vid = _dummy_video(tmp_path)
    bundle = _sample_bundle()
    cfg = AppConfig(
        dry_run=False,
        enable_youtube=True,
        enable_tiktok=True,
        enable_instagram=False,
        enable_facebook=False,
        youtube_client_id="cid",
        youtube_client_secret="csecret",
        youtube_refresh_token="rtoken",
        tiktok_access_token="tt_token",
        history_file=tmp_path / "history.json",
    )

    # YouTube succeeds; TikTok fails with HTTP 403
    def yt_post(url: str, **kwargs: Any) -> FakePublisherResponse:
        if "oauth2.googleapis.com" in url:
            return FakePublisherResponse(200, {"access_token": "ya29.token"})
        return FakePublisherResponse(
            200, {}, headers={"Location": "https://upload.youtube.com/resumable/123"}
        )

    def yt_put(url: str, **kwargs: Any) -> FakePublisherResponse:
        return FakePublisherResponse(
            200,
            {"id": "yt_short_abc", "status": {"uploadStatus": "uploaded", "privacyStatus": "private"}},
        )

    def tt_post(url: str, **kwargs: Any) -> FakePublisherResponse:
        return FakePublisherResponse(403, {"error": {"code": "access_token_invalid", "message": "Forbidden"}})

    yt_pub = YouTubePublisher(config=cfg, session=FakeSession(post_fn=yt_post, put_fn=yt_put))
    tt_pub = TikTokPublisher(config=cfg, session=FakeSession(post_fn=tt_post))
    ig_pub = InstagramPublisher(config=cfg)
    fb_pub = FacebookPublisher(config=cfg)

    pipeline = DailyVideoPipeline(
        config=cfg,
        publishers=[yt_pub, ig_pub, fb_pub, tt_pub],
    )
    # Seed a run record in history
    from src.history_manager import ProductionRunRecord

    pipeline.history.save_or_update_run(
        ProductionRunRecord(
            run_id=bundle.run_id,
            production_date=bundle.production_date,
            title=bundle.story_title,
            story_hash="h1",
            story_text="text",
            mood="hopeful",
            language="English",
        )
    )

    outcomes = pipeline.run_publishing_stage(
        run_id=bundle.run_id,
        production_date=bundle.production_date,
        video_path=vid,
        metadata_bundle=bundle,
    )

    assert outcomes["youtube"].status == "published"
    assert outcomes["youtube"].platform_post_id == "yt_short_abc"
    assert outcomes["tiktok"].status == "failed"
    assert outcomes["instagram"].status == "skipped"
    assert outcomes["facebook"].status == "skipped"


def test_all_four_platforms_publish_simultaneously(tmp_path: Path) -> None:
    """Verify that YouTube, Instagram, Facebook, and TikTok all publish in one run when enabled."""
    vid = _dummy_video(tmp_path)
    bundle = _sample_bundle()
    cfg = AppConfig(
        dry_run=False,
        enable_youtube=True,
        enable_instagram=True,
        enable_facebook=True,
        enable_tiktok=True,
        youtube_client_id="cid",
        youtube_client_secret="csecret",
        youtube_refresh_token="rtoken",
        instagram_account_id="ig_999",
        instagram_access_token="ig_tok",
        facebook_page_id="fb_888",
        facebook_page_access_token="fb_tok",
        tiktok_access_token="tt_tok",
        history_file=tmp_path / "history.json",
    )

    def yt_post(url: str, **kwargs: Any) -> FakePublisherResponse:
        if "oauth2.googleapis.com" in url:
            return FakePublisherResponse(200, {"access_token": "ya29.token"})
        return FakePublisherResponse(
            200, {}, headers={"Location": "https://upload.youtube.com/resumable/123"}
        )

    def yt_put(url: str, **kwargs: Any) -> FakePublisherResponse:
        return FakePublisherResponse(
            200, {"id": "yt_101", "status": {"uploadStatus": "uploaded", "privacyStatus": "public"}}
        )

    def ig_post(url: str, **kwargs: Any) -> FakePublisherResponse:
        if "rupload.facebook.com" in url:
            return FakePublisherResponse(200, {"success": True})
        if url.endswith("/media"):
            return FakePublisherResponse(200, {"id": "ig_container_1"})
        if url.endswith("/media_publish"):
            return FakePublisherResponse(200, {"id": "ig_reel_202"})
        return FakePublisherResponse(200, {})

    def ig_get(url: str, **kwargs: Any) -> FakePublisherResponse:
        return FakePublisherResponse(200, {"status_code": "FINISHED"})

    def fb_post(url: str, **kwargs: Any) -> FakePublisherResponse:
        if "rupload.facebook.com" in url:
            return FakePublisherResponse(200, {"success": True})
        data = kwargs.get("data", {})
        if data.get("upload_phase") == "start":
            return FakePublisherResponse(
                200,
                {
                    "video_id": "fb_reel_303",
                    "upload_url": "https://rupload.facebook.com/video-upload/v22.0/fb_reel_303",
                },
            )
        return FakePublisherResponse(200, {"success": True})

    def tt_post(url: str, **kwargs: Any) -> FakePublisherResponse:
        if "creator_info" in url:
            return FakePublisherResponse(
                200,
                {
                    "error": {"code": "ok"},
                    "data": {"privacy_level_options": ["PUBLIC_TO_EVERYONE", "SELF_ONLY"]},
                },
            )
        if "video/init" in url:
            return FakePublisherResponse(
                200,
                {
                    "error": {"code": "ok"},
                    "data": {
                        "publish_id": "tt_pub_404",
                        "upload_url": "https://open-upload.tiktokapis.com/video/404",
                    },
                },
            )
        return FakePublisherResponse(
            200,
            {"error": {"code": "ok"}, "data": {"status": "PUBLISH_COMPLETE"}},
        )

    def tt_put(url: str, **kwargs: Any) -> FakePublisherResponse:
        return FakePublisherResponse(201, {})

    pipeline = DailyVideoPipeline(
        config=cfg,
        publishers=[
            YouTubePublisher(config=cfg, session=FakeSession(post_fn=yt_post, put_fn=yt_put)),
            InstagramPublisher(config=cfg, session=FakeSession(post_fn=ig_post, get_fn=ig_get)),
            FacebookPublisher(config=cfg, session=FakeSession(post_fn=fb_post)),
            TikTokPublisher(config=cfg, session=FakeSession(post_fn=tt_post, put_fn=tt_put)),
        ],
    )

    from src.history_manager import ProductionRunRecord

    pipeline.history.save_or_update_run(
        ProductionRunRecord(
            run_id=bundle.run_id,
            production_date=bundle.production_date,
            title=bundle.story_title,
            story_hash="h_all",
            story_text="text",
            mood="hopeful",
            language="English",
        )
    )

    outcomes = pipeline.run_publishing_stage(
        run_id=bundle.run_id,
        production_date=bundle.production_date,
        video_path=vid,
        metadata_bundle=bundle,
    )

    assert outcomes["youtube"].status == "published"
    assert outcomes["instagram"].status == "published"
    assert outcomes["facebook"].status == "published"
    assert outcomes["tiktok"].status == "published"
