"""Main orchestration entry point for the Free Daily AI Story Video Automation System.

Executes the 16-stage daily production pipeline in strict sequence, supporting
dry-run execution, offline story JSON input for local verification, recovery/retry
of failed platform uploads without regenerating stories or duplicating posts,
and isolated per-platform error handling.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.asset_bootstrap import bootstrap_local_assets
from src.config import AppConfig, ConfigurationError, setup_logging
from src.history_manager import HistoryManager, ProductionRunRecord
from src.image_selector import ImageMetadata, ImageSelectionError, ImageSelector
from src.metadata_generator import (
    FacebookMetadata,
    InstagramMetadata,
    MetadataGenerator,
    PlatformMetadataBundle,
    TikTokMetadata,
    YouTubeMetadata,
)
from src.music_downloader import MusicDownloader
from src.music_selector import MusicSelector, MusicTrackMetadata
from src.publishers import (
    BasePublisher,
    FacebookPublisher,
    InstagramPublisher,
    PublisherError,
    PublishResult,
    TikTokPublisher,
    YouTubePublisher,
)
from src.story_generator import (
    StoryGenerationError,
    StoryGenerator,
    StoryPayload,
    validate_story_payload,
)
from src.subtitle_generator import SubtitleGenerator
from src.video_renderer import VideoRenderer, VideoRenderError
from src.video_validator import VideoValidationError, VideoValidator
from src.voice_generator import VoiceGenerationError, VoiceGenerator

logger = logging.getLogger(__name__)

EXIT_SUCCESS = 0
EXIT_CONFIG_ERROR = 2
EXIT_STORY_ERROR = 3
EXIT_ASSET_OR_TTS_ERROR = 4
EXIT_RENDER_OR_VALIDATION_ERROR = 5
EXIT_ALL_PUBLISHERS_FAILED = 6


def _load_bundle_from_dict(data: Dict[str, Any]) -> PlatformMetadataBundle:
    yt = data.get("youtube", {})
    ig = data.get("instagram", {})
    fb = data.get("facebook", {})
    tt = data.get("tiktok", {})
    return PlatformMetadataBundle(
        production_date=str(data.get("production_date", "")),
        run_id=str(data.get("run_id", "")),
        story_title=str(data.get("story_title", "")),
        language=str(data.get("language", "English")),
        mood=str(data.get("mood", "")),
        youtube=YouTubeMetadata(**yt),
        instagram=InstagramMetadata(**ig),
        facebook=FacebookMetadata(**fb),
        tiktok=TikTokMetadata(**tt),
        attributions=list(data.get("attributions", [])),
    )


class DailyVideoPipeline:
    """Coordinates all 16 stages of daily story video creation and publishing."""

    def __init__(
        self,
        config: AppConfig,
        story_generator: Optional[StoryGenerator] = None,
        image_selector: Optional[ImageSelector] = None,
        voice_generator: Optional[VoiceGenerator] = None,
        music_selector: Optional[MusicSelector] = None,
        subtitle_generator: Optional[SubtitleGenerator] = None,
        video_renderer: Optional[VideoRenderer] = None,
        video_validator: Optional[VideoValidator] = None,
        metadata_generator: Optional[MetadataGenerator] = None,
        publishers: Optional[List[BasePublisher]] = None,
    ) -> None:
        self.config = config
        self.history = HistoryManager(
            history_file=config.history_file,
            similarity_threshold=config.story_similarity_threshold,
        )
        self.story_generator = story_generator or StoryGenerator(
            config=config, history_manager=self.history
        )
        self.image_selector = image_selector or ImageSelector(config=config)
        self.voice_generator = voice_generator or VoiceGenerator(config=config)
        self.music_downloader = MusicDownloader(config=config)
        self.music_selector = music_selector or MusicSelector(
            config=config, downloader=self.music_downloader
        )
        self.subtitle_generator = subtitle_generator or SubtitleGenerator()
        self.video_renderer = video_renderer or VideoRenderer(config=config)
        self.video_validator = video_validator or VideoValidator(config=config)
        self.metadata_generator = metadata_generator or MetadataGenerator(
            youtube_category_id=config.youtube_category_id,
            youtube_privacy_status=config.youtube_privacy_status,
            tiktok_privacy_level=config.tiktok_privacy_level,
        )
        self.publishers = publishers or [
            YouTubePublisher(config=config),
            InstagramPublisher(config=config),
            FacebookPublisher(config=config),
            TikTokPublisher(config=config),
        ]

    def run_publishing_stage(
        self,
        run_id: str,
        production_date: str,
        video_path: Path,
        metadata_bundle: PlatformMetadataBundle,
    ) -> Dict[str, PublishResult]:
        """Execute Stage 13 & 14: Publish independently to each enabled platform."""
        outcomes: Dict[str, PublishResult] = {}

        for pub in self.publishers:
            platform = pub.platform_name

            # Prevent accidental duplicate uploads if already confirmed published
            if self.history.is_platform_already_published(production_date, platform):
                logger.info(
                    "[%s] Video for %s is already confirmed published in history; skipping duplicate upload.",
                    platform,
                    production_date,
                )
                outcomes[platform] = PublishResult(
                    platform=platform,
                    status="published",
                    published=True,
                    message="Already published in previous run; skipped duplicate upload.",
                )
                continue

            try:
                result = pub.publish(video_path=video_path, metadata=metadata_bundle)
                outcomes[platform] = result
                self.history.record_platform_outcome(
                    run_id=run_id,
                    platform=platform,
                    status=result.status,
                    platform_post_id=result.platform_post_id,
                    url=result.url,
                    error_message=result.error,
                )
            except PublisherError as exc:
                logger.error(
                    "[%s] Publishing failed (isolated; continuing other platforms): %s",
                    platform,
                    exc,
                )
                fail_res = PublishResult(
                    platform=platform,
                    status="failed",
                    published=False,
                    error=str(exc),
                    message=f"{platform} upload failed: {exc}",
                )
                outcomes[platform] = fail_res
                self.history.record_platform_outcome(
                    run_id=run_id,
                    platform=platform,
                    status="failed",
                    error_message=str(exc),
                )
            except Exception as exc:
                logger.error(
                    "[%s] Unexpected error during publishing (isolated): %s",
                    platform,
                    exc,
                )
                fail_res = PublishResult(
                    platform=platform,
                    status="failed",
                    published=False,
                    error=str(exc),
                    message=f"Unexpected error: {exc}",
                )
                outcomes[platform] = fail_res
                self.history.record_platform_outcome(
                    run_id=run_id,
                    platform=platform,
                    status="failed",
                    error_message=str(exc),
                )

        return outcomes

    def execute(
        self,
        production_date: Optional[str] = None,
        story_file: Optional[Path] = None,
        retry_publish_only: bool = False,
        force_regenerate: bool = False,
    ) -> int:
        """Execute the complete 16-stage daily workflow and return a meaningful exit code."""
        # Stage 1: Validate configuration
        try:
            require_api_key = (story_file is None) and (not retry_publish_only)
            self.config.validate_core_settings(require_api_key=require_api_key)
        except ConfigurationError as exc:
            logger.error("Stage 1 [Configuration] failed: %s", exc)
            return EXIT_CONFIG_ERROR

        # Ensure local fallback directories and CC0 assets exist
        bootstrap_local_assets(self.config.project_root)

        # Stage 2: Determine production date
        prod_date = production_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        day_output_dir = self.config.output_directory / prod_date
        day_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "=== Starting Daily Story Video Pipeline | Date: %s | Dry-Run: %s ===",
            prod_date,
            self.config.dry_run,
        )

        # Stage 3: Check current run history and support retry/recovery without duplicate generation
        existing_run = self.history.get_run_by_date(prod_date)
        if existing_run and not force_regenerate:
            existing_video = Path(existing_run.get("video_path") or "")
            existing_meta = Path(existing_run.get("metadata_path") or "")
            run_id = str(existing_run.get("run_id"))

            if (
                retry_publish_only
                or (
                    existing_run.get("stage_status", {}).get("video_validation") == "success"
                    and existing_video.exists()
                    and existing_meta.exists()
                )
            ):
                if not existing_video.exists() or not existing_meta.exists():
                    logger.error(
                        "Cannot retry publishing for %s: preserved video or metadata file not found.",
                        prod_date,
                    )
                    return EXIT_RENDER_OR_VALIDATION_ERROR

                logger.info(
                    "Stage 3 [History]: Found validated video for %s (run_id=%s). "
                    "Resuming at publishing stage without regenerating story.",
                    prod_date,
                    run_id,
                )
                meta_dict = json.loads(existing_meta.read_text(encoding="utf-8"))
                bundle = _load_bundle_from_dict(meta_dict)
                outcomes = self.run_publishing_stage(
                    run_id=run_id,
                    production_date=prod_date,
                    video_path=existing_video,
                    metadata_bundle=bundle,
                )
                return self._summarize_and_exit(
                    run_id=run_id,
                    prod_date=prod_date,
                    video_path=existing_video,
                    outcomes=outcomes,
                )

        # Stage 4: Generate (or load) the story
        try:
            if story_file is not None:
                logger.info("Stage 4 [Story]: Loading story from file %s", story_file)
                raw_story = json.loads(story_file.read_text(encoding="utf-8"))
                story = validate_story_payload(
                    raw_story,
                    min_words=self.config.story_min_words,
                    max_words=self.config.story_max_words,
                    min_scenes=self.config.story_min_scenes,
                    max_scenes=self.config.story_max_scenes,
                    expected_language=self.config.story_language,
                    model_used="local-story-file",
                )
            else:
                story = self.story_generator.generate_story()
        except (StoryGenerationError, OSError, json.JSONDecodeError) as exc:
            logger.error("Stage 4 [Story Generation] failed: %s", exc)
            return EXIT_STORY_ERROR

        # Stage 5: Validate and save story metadata
        run_id = self.history.compute_run_id(prod_date, story.title, story.story)
        story_hash = self.history.compute_story_hash(story.story)
        story_json_path = day_output_dir / f"{run_id}_story.json"
        story_json_path.write_text(
            json.dumps(story.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
        )

        run_record = ProductionRunRecord(
            run_id=run_id,
            production_date=prod_date,
            title=story.title,
            story_hash=story_hash,
            story_text=story.story,
            mood=story.mood,
            language=story.language,
            stage_status={"story_generation": "success"},
        )
        self.history.save_or_update_run(run_record)
        logger.info("Stage 5 [Story Saved]: %s (%s)", story.title, story_json_path)

        # Stage 6: Select suitable image for each scene
        try:
            recent_imgs = self.history.get_recently_used_images(limit_runs=10)
            selected_images: List[ImageMetadata] = (
                self.image_selector.select_images_for_story(
                    scenes=story.scenes, recently_used_ids=recent_imgs
                )
            )
            run_record.selected_images = [img.to_dict() for img in selected_images]
            run_record.stage_status["image_selection"] = "success"
            self.history.save_or_update_run(run_record)
        except ImageSelectionError as exc:
            run_record.stage_status["image_selection"] = f"failed: {exc}"
            self.history.save_or_update_run(run_record)
            logger.error("Stage 6 [Image Selection] failed: %s", exc)
            return EXIT_ASSET_OR_TTS_ERROR

        temp_audio_dir = Path(tempfile.mkdtemp(prefix=f"auto_video_audio_{prod_date}_"))
        try:
            # Stage 7: Generate narration audio via Piper TTS (respecting AI dramatic pauses)
            inter_pauses = [
                getattr(s, "pause_after", self.config.piper_sentence_silence)
                for s in story.scenes[:-1]
            ]
            try:
                audio_segments, master_wav, total_duration = (
                    self.voice_generator.generate_scene_narrations(
                        scenes=story.scenes,
                        work_dir=temp_audio_dir,
                        inter_scene_pause_seconds=self.config.piper_sentence_silence,
                    )
                )
                run_record.stage_status["voice_generation"] = "success"
                self.history.save_or_update_run(run_record)
            except VoiceGenerationError as exc:
                run_record.stage_status["voice_generation"] = f"failed: {exc}"
                self.history.save_or_update_run(run_record)
                logger.error("Stage 7 [Voice Generation] failed: %s", exc)
                return EXIT_ASSET_OR_TTS_ERROR

            # Stage 8: Select and validate background music (auto-downloading if needed)
            try:
                music_style = getattr(story, "music_style", "")
                music_track: Optional[MusicTrackMetadata] = (
                    self.music_selector.select_music(
                        story_mood=story.mood, music_style=music_style
                    )
                )
                run_record.music_metadata = (
                    music_track.to_dict() if music_track is not None else None
                )
                run_record.stage_status["music_selection"] = (
                    "success" if music_track else "narration_only_fallback"
                )
                self.history.save_or_update_run(run_record)
            except Exception as exc:
                run_record.stage_status["music_selection"] = f"failed: {exc}"
                self.history.save_or_update_run(run_record)
                logger.error("Stage 8 [Music Selection] failed: %s", exc)
                return EXIT_ASSET_OR_TTS_ERROR

            # Stage 9: Generate synchronized SRT & ASS subtitles
            try:
                cues = self.subtitle_generator.build_cues(
                    audio_segments=audio_segments,
                    inter_scene_pauses=inter_pauses,
                    inter_scene_pause_seconds=self.config.piper_sentence_silence,
                    max_video_duration=total_duration,
                )
                srt_path = day_output_dir / f"{run_id}.srt"
                ass_path = day_output_dir / f"{run_id}.ass"
                self.subtitle_generator.write_srt(cues, srt_path)
                self.subtitle_generator.write_ass(
                    cues,
                    ass_path,
                    video_width=self.config.video_width,
                    video_height=self.config.video_height,
                    font_name=self.config.subtitle_font_name,
                )
                run_record.stage_status["subtitle_generation"] = "success"
                self.history.save_or_update_run(run_record)
            except Exception as exc:
                run_record.stage_status["subtitle_generation"] = f"failed: {exc}"
                self.history.save_or_update_run(run_record)
                logger.error("Stage 9 [Subtitle Generation] failed: %s", exc)
                return EXIT_RENDER_OR_VALIDATION_ERROR

            # Stage 10: Render final MP4
            output_mp4 = day_output_dir / f"{run_id}.mp4"
            try:
                self.video_renderer.render(
                    images=selected_images,
                    audio_segments=audio_segments,
                    master_narration_wav=master_wav,
                    subtitle_file=srt_path,
                    output_mp4=output_mp4,
                    story_title=story.title,
                    total_duration_seconds=total_duration,
                    music_track=music_track,
                    inter_scene_pauses=inter_pauses,
                    inter_scene_pause_seconds=self.config.piper_sentence_silence,
                    diagnostics_dir=day_output_dir / "diagnostics",
                )
                run_record.video_path = str(output_mp4.resolve())
                run_record.stage_status["video_rendering"] = "success"
                self.history.save_or_update_run(run_record)
            except VideoRenderError as exc:
                run_record.stage_status["video_rendering"] = f"failed: {exc}"
                self.history.save_or_update_run(run_record)
                logger.error("Stage 10 [Video Rendering] failed: %s", exc)
                return EXIT_RENDER_OR_VALIDATION_ERROR

            # Stage 11: Validate finished video using FFprobe
            try:
                validation_report = self.video_validator.validate(
                    video_path=output_mp4,
                    expected_duration_seconds=total_duration,
                    duration_tolerance_seconds=2.5,
                )
                (day_output_dir / f"{run_id}_validation.json").write_text(
                    json.dumps(validation_report.to_dict(), indent=2), encoding="utf-8"
                )
                run_record.stage_status["video_validation"] = "success"
                self.history.save_or_update_run(run_record)
            except VideoValidationError as exc:
                run_record.stage_status["video_validation"] = f"failed: {exc}"
                self.history.save_or_update_run(run_record)
                logger.error("Stage 11 [Video Validation] failed: %s", exc)
                return EXIT_RENDER_OR_VALIDATION_ERROR

            # Stage 12: Generate platform-specific metadata
            voice_license = (
                audio_segments[0].license_status if audio_segments else "Verified Piper TTS"
            )
            metadata_bundle = self.metadata_generator.generate(
                story=story,
                production_date=prod_date,
                run_id=run_id,
                images=selected_images,
                music_track=music_track,
                voice_license=voice_license,
            )
            metadata_path = day_output_dir / f"{run_id}_metadata.json"
            self.metadata_generator.save_metadata(metadata_bundle, metadata_path)
            run_record.metadata_path = str(metadata_path.resolve())
            run_record.stage_status["metadata_generation"] = "success"
            self.history.save_or_update_run(run_record)

            # Stages 13 & 14: Publish to enabled platforms & record outcomes independently
            outcomes = self.run_publishing_stage(
                run_id=run_id,
                production_date=prod_date,
                video_path=output_mp4,
                metadata_bundle=metadata_bundle,
            )

            # Stages 15 & 16: Preserve final video/logs & exit with meaningful status code
            return self._summarize_and_exit(
                run_id=run_id,
                prod_date=prod_date,
                video_path=output_mp4,
                outcomes=outcomes,
            )

        finally:
            if not self.config.keep_temp_files and temp_audio_dir.exists():
                shutil.rmtree(temp_audio_dir, ignore_errors=True)

    def _summarize_and_exit(
        self,
        run_id: str,
        prod_date: str,
        video_path: Path,
        outcomes: Dict[str, PublishResult],
    ) -> int:
        summary = {
            "run_id": run_id,
            "production_date": prod_date,
            "dry_run": self.config.dry_run,
            "video_path": str(video_path.resolve()),
            "platforms": {k: v.to_dict() for k, v in outcomes.items()},
        }
        summary_path = video_path.parent / f"{run_id}_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        logger.info("Saved run summary to %s", summary_path)

        enabled_platforms = [p.platform_name for p in self.publishers if p.is_enabled()]
        if enabled_platforms and not self.config.dry_run:
            failed_enabled = [
                plat
                for plat in enabled_platforms
                if outcomes.get(plat) and outcomes[plat].status == "failed"
            ]
            if len(failed_enabled) == len(enabled_platforms):
                logger.error(
                    "All enabled platforms (%s) failed to publish. Video preserved at %s",
                    ", ".join(failed_enabled),
                    video_path,
                )
                return EXIT_ALL_PUBLISHERS_FAILED
            if failed_enabled:
                logger.warning(
                    "Some platforms failed (%s), while others succeeded. Video preserved at %s",
                    ", ".join(failed_enabled),
                    video_path,
                )

        logger.info(
            "=== Pipeline completed successfully | Video preserved: %s ===",
            video_path,
        )
        return EXIT_SUCCESS


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Free Daily AI Story Video Automation System"
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Production date in YYYY-MM-DD format (defaults to today in UTC).",
    )
    parser.add_argument(
        "--story-file",
        type=Path,
        default=None,
        help="Optional path to a pre-generated story JSON file (for offline testing).",
    )
    parser.add_argument(
        "--retry-publish",
        action="store_true",
        help="Rerun only the publishing stage for an existing validated video on --date.",
    )
    parser.add_argument(
        "--force-regenerate",
        action="store_true",
        help="Force generating a new story and video even if one exists for --date.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
        help="Optional path to a custom .env configuration file.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    config = AppConfig.from_env(env_file=args.env_file)
    prod_date = args.date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
    log_path = config.output_directory / prod_date / "pipeline.log"
    setup_logging(config=config, log_file=log_path)

    pipeline = DailyVideoPipeline(config=config)
    return pipeline.execute(
        production_date=prod_date,
        story_file=args.story_file,
        retry_publish_only=args.retry_publish,
        force_regenerate=args.force_regenerate,
    )


if __name__ == "__main__":
    sys.exit(main())
