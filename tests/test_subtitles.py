"""Unit tests for subtitle synchronization, timestamp monotonicity, and SRT/ASS generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.subtitle_generator import (
    SubtitleGenerationError,
    SubtitleGenerator,
    format_ass_timestamp,
    format_srt_timestamp,
    split_text_into_mobile_phrases,
)
from src.voice_generator import AudioSegmentResult


def test_split_text_into_mobile_phrases_preserves_punctuation() -> None:
    text = (
        "Every evening at six, retired stationmaster Aris stood beneath the leaking roof "
        "of Platform Four, holding a bright yellow umbrella he never opened for himself."
    )
    phrases = split_text_into_mobile_phrases(text, max_chars_per_cue=50, max_words_per_cue=9)
    assert len(phrases) >= 3
    for phrase in phrases:
        assert len(phrase.split()) <= 9
    assert phrases[-1].endswith("himself.")


def test_subtitle_cues_match_measured_audio_durations_without_overlap(
    tmp_path: Path,
) -> None:
    segments = [
        AudioSegmentResult(
            scene_index=0,
            narration_text="Tonight, the downpour was merciless. A student sat trembling on a wooden bench.",
            audio_path="scene_01.wav",
            duration_seconds=5.4,
            sample_rate=22050,
            voice_used="en_US-ljspeech-medium",
            license_status="Public Domain",
        ),
        AudioSegmentResult(
            scene_index=1,
            narration_text="Aris walked over, opened the yellow umbrella, and smiled gently.",
            audio_path="scene_02.wav",
            duration_seconds=4.2,
            sample_rate=22050,
            voice_used="en_US-ljspeech-medium",
            license_status="Public Domain",
        ),
    ]
    pause = 0.25
    total_video_duration = 5.4 + pause + 4.2

    gen = SubtitleGenerator(max_chars_per_cue=48, max_words_per_cue=8, max_line_chars=28)
    cues = gen.build_cues(
        audio_segments=segments,
        inter_scene_pause_seconds=pause,
        max_video_duration=total_video_duration,
    )

    assert len(cues) >= 3
    prev_end = 0.0
    for cue in cues:
        assert cue.start_seconds >= 0.0
        assert cue.end_seconds > cue.start_seconds
        assert cue.start_seconds >= prev_end - 1e-4
        assert cue.end_seconds <= total_video_duration + 1e-4
        prev_end = cue.end_seconds

    srt_file = gen.write_srt(cues, tmp_path / "test.srt")
    ass_file = gen.write_ass(cues, tmp_path / "test.ass")
    assert srt_file.exists()
    assert "-->" in srt_file.read_text(encoding="utf-8")
    assert ass_file.exists()
    assert "Dialogue:" in ass_file.read_text(encoding="utf-8")


def test_timestamp_formatting_helpers() -> None:
    assert format_srt_timestamp(0.0) == "00:00:00,000"
    assert format_srt_timestamp(65.432) == "00:01:05,432"
    assert format_ass_timestamp(65.432) == "0:01:05.43"

    gen = SubtitleGenerator()
    with pytest.raises(SubtitleGenerationError):
        gen.build_cues([])
