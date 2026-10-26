"""Unit tests for FFmpeg command construction, video rendering, and FFprobe validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import AppConfig
from src.image_selector import ImageMetadata
from src.music_selector import MusicTrackMetadata
from src.video_renderer import VideoRenderer, VideoRenderError
from src.video_validator import VideoValidationError, VideoValidator
from src.voice_generator import AudioSegmentResult


def _sample_images(tmp_path: Path) -> list[ImageMetadata]:
    return [
        ImageMetadata(
            id="local:sadness/img1.jpg",
            local_path=str(tmp_path / "img1.jpg"),
            provider="local",
            creator="Tester",
            source_url="local://img1",
            license_name="CC0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            attribution_text="img1 CC0",
            retrieved_at="2026-10-26T00:00:00Z",
            width=1080,
            height=1920,
            mood="sadness",
            scene_index=0,
        ),
        ImageMetadata(
            id="local:hope/img2.jpg",
            local_path=str(tmp_path / "img2.jpg"),
            provider="local",
            creator="Tester",
            source_url="local://img2",
            license_name="CC0",
            license_url="https://creativecommons.org/publicdomain/zero/1.0/",
            attribution_text="img2 CC0",
            retrieved_at="2026-10-26T00:00:00Z",
            width=1080,
            height=1920,
            mood="hope",
            scene_index=1,
        ),
    ]


def _sample_audio_segments(tmp_path: Path) -> list[AudioSegmentResult]:
    return [
        AudioSegmentResult(
            scene_index=0,
            narration_text="Scene one narration.",
            audio_path=str(tmp_path / "s1.wav"),
            duration_seconds=4.0,
            sample_rate=22050,
            voice_used="en_US-ljspeech-medium",
            license_status="Public Domain",
        ),
        AudioSegmentResult(
            scene_index=1,
            narration_text="Scene two narration.",
            audio_path=str(tmp_path / "s2.wav"),
            duration_seconds=4.5,
            sample_rate=22050,
            voice_used="en_US-ljspeech-medium",
            license_status="Public Domain",
        ),
    ]


def test_ffmpeg_command_generation_with_music_and_ducking(tmp_path: Path) -> None:
    cfg = AppConfig()
    renderer = VideoRenderer(config=cfg)
    images = _sample_images(tmp_path)
    segments = _sample_audio_segments(tmp_path)
    music = MusicTrackMetadata(
        track_id="music:sad.wav",
        file_path=str(tmp_path / "sad.wav"),
        title="Sad Track",
        creator="Composer",
        source_url="local://sad.wav",
        license_name="CC0",
        license_url="",
        attribution_instructions="CC0",
        commercial_use_allowed=True,
        mood_tags=["sad"],
    )

    cmd = renderer.build_ffmpeg_command(
        images=images,
        audio_segments=segments,
        master_narration_wav=tmp_path / "master.wav",
        subtitle_file=tmp_path / "subs.srt",
        output_mp4=tmp_path / "out.mp4",
        story_title="The Last Letter",
        total_duration_seconds=8.75,
        music_track=music,
        inter_scene_pause_seconds=0.25,
    )

    joined = " ".join(cmd)
    assert "libx264" in cmd
    assert "aac" in cmd
    assert "yuv420p" in cmd
    assert "+faststart" in cmd
    assert "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920" in joined
    assert "zoompan=" in joined
    assert "xfade=transition=fade" in joined
    assert "sidechaincompress=" in joined
    assert "subtitles=" in joined


def test_ffmpeg_command_generation_narration_only_fallback(tmp_path: Path) -> None:
    cfg = AppConfig()
    renderer = VideoRenderer(config=cfg)
    cmd = renderer.build_ffmpeg_command(
        images=_sample_images(tmp_path),
        audio_segments=_sample_audio_segments(tmp_path),
        master_narration_wav=tmp_path / "master.wav",
        subtitle_file=tmp_path / "subs.ass",
        output_mp4=tmp_path / "out.mp4",
        story_title="Narration Only Story",
        total_duration_seconds=8.75,
        music_track=None,
    )
    joined = " ".join(cmd)
    assert "sidechaincompress=" not in joined
    assert "ass=" in joined


def test_video_renderer_rejects_mismatched_scene_counts(tmp_path: Path) -> None:
    renderer = VideoRenderer(config=AppConfig())
    with pytest.raises(VideoRenderError, match="Scene count mismatch"):
        renderer.build_ffmpeg_command(
            images=_sample_images(tmp_path)[:1],
            audio_segments=_sample_audio_segments(tmp_path),
            master_narration_wav=tmp_path / "master.wav",
            subtitle_file=tmp_path / "subs.srt",
            output_mp4=tmp_path / "out.mp4",
            story_title="Mismatch",
            total_duration_seconds=5.0,
        )


def test_video_validator_rejects_missing_or_corrupt_mp4(tmp_path: Path) -> None:
    validator = VideoValidator(config=AppConfig())
    with pytest.raises(VideoValidationError, match="does not exist"):
        validator.validate(tmp_path / "missing.mp4")

    tiny_file = tmp_path / "tiny.mp4"
    tiny_file.write_bytes(b"0000")
    with pytest.raises(VideoValidationError, match="suspiciously small"):
        validator.validate(tiny_file)


def test_end_to_end_short_ffmpeg_render_and_ffprobe_validation(tmp_path: Path) -> None:
    """Render an actual 2-scene 1080x1920 30fps MP4 with FFmpeg and validate with FFprobe."""
    from PIL import Image
    from src.story_generator import SceneSegment
    from src.subtitle_generator import SubtitleGenerator
    from src.voice_generator import VoiceGenerator

    img1 = tmp_path / "s1.jpg"
    img2 = tmp_path / "s2.jpg"
    Image.new("RGB", (1080, 1920), color=(25, 45, 75)).save(img1, format="JPEG")
    Image.new("RGB", (1080, 1920), color=(65, 95, 135)).save(img2, format="JPEG")

    cfg = AppConfig(
        output_directory=tmp_path / "out",
        allow_Offline_tts_fallback=True,
        ffmpeg_preset="ultrafast",
    )
    scenes = [
        SceneSegment("A short opening sentence.", "night sky", "sadness", 2.0),
        SceneSegment("A hopeful closing sentence.", "sunrise sky", "hope", 2.0),
    ]
    vg = VoiceGenerator(config=cfg)
    segments, master_wav, total_dur = vg.generate_scene_narrations(
        scenes=scenes, work_dir=tmp_path / "audio", inter_scene_pause_seconds=0.2
    )

    sub_gen = SubtitleGenerator()
    cues = sub_gen.build_cues(
        audio_segments=segments,
        inter_scene_pause_seconds=0.2,
        max_video_duration=total_dur,
    )
    srt_path = sub_gen.write_srt(cues, tmp_path / "subs.srt")

    images = _sample_images(tmp_path)
    images[0].local_path = str(img1)
    images[1].local_path = str(img2)

    output_mp4 = tmp_path / "rendered.mp4"
    renderer = VideoRenderer(config=cfg)
    renderer.render(
        images=images,
        audio_segments=segments,
        master_narration_wav=master_wav,
        subtitle_file=srt_path,
        output_mp4=output_mp4,
        story_title="Integration Test Video",
        total_duration_seconds=total_dur,
        music_track=None,
        inter_scene_pause_seconds=0.2,
    )

    validator = VideoValidator(config=cfg)
    report = validator.validate(
        output_mp4, expected_duration_seconds=total_dur, duration_tolerance_seconds=1.5
    )
    assert report.is_valid is True
    assert report.video_codec == "h264"
    assert report.audio_codec == "aac"
    assert report.width == 1080
    assert report.height == 1920
    assert abs(report.fps - 30.0) <= 1.0
