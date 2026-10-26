"""Tests for dramatic pauses, speech synthesis pause splitting, and music downloader."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.config import AppConfig
from src.music_downloader import MusicDownloader
from src.music_selector import MusicSelector, classify_story_mood
from src.story_generator import (
    count_words,
    strip_pause_markers,
    validate_story_payload,
)
from src.subtitle_generator import SubtitleGenerator, split_text_into_mobile_phrases
from src.voice_generator import (
    AudioSegmentResult,
    SpokenChunk,
    create_silence_wav,
    split_narration_pauses,
    validate_wav_file,
)


def test_split_narration_pauses() -> None:
    text = "She picked up his phone. [pause: 1.2s] A strange notification appeared. [pause] She froze."
    chunks = split_narration_pauses(text)
    assert len(chunks) == 3
    assert chunks[0] == ("She picked up his phone.", 1.2)
    assert chunks[1] == ("A strange notification appeared.", 1.0)
    assert chunks[2] == ("She froze.", 0.0)


def test_strip_pause_markers() -> None:
    text = "Her hands shook. [pause: 0.8s] The truth was finally out."
    cleaned = strip_pause_markers(text)
    assert cleaned == "Her hands shook. The truth was finally out."
    assert "[pause" not in cleaned
    assert count_words(text) == 8


def test_create_silence_wav(tmp_path: Path) -> None:
    silence_wav = tmp_path / "silence.wav"
    create_silence_wav(silence_wav, duration_seconds=1.0, sample_rate=22050)
    dur, sr = validate_wav_file(silence_wav)
    assert sr == 22050
    assert 0.95 <= dur <= 1.05


def test_subtitle_cues_with_spoken_chunks() -> None:
    seg = AudioSegmentResult(
        scene_index=0,
        narration_text="He said he was at work. But his car was parked outside her apartment.",
        audio_path="dummy.wav",
        duration_seconds=7.0,
        sample_rate=22050,
        voice_used="piper-voice",
        license_status="Public Domain",
        spoken_chunks=[
            SpokenChunk(text="He said he was at work.", duration_seconds=2.5, pause_after_seconds=1.0),
            SpokenChunk(text="But his car was parked outside her apartment.", duration_seconds=3.5, pause_after_seconds=0.0),
        ],
    )
    gen = SubtitleGenerator()
    cues = gen.build_cues([seg], max_video_duration=10.0)
    assert len(cues) >= 2
    # Ensure cue 1 ends before pause, and cue 2 starts after pause
    assert cues[0].end_seconds <= 2.55
    assert cues[-1].start_seconds >= 3.45  # 2.5 + 1.0 pause


def test_classify_story_mood_betrayal() -> None:
    assert classify_story_mood("cheating and painful betrayal") == "betrayal"
    assert classify_story_mood("heartbreak and tears") == "heartbreak"
    assert classify_story_mood("quiet revenge") == "betrayal"


def test_music_downloader_mocked_archive_success(tmp_path: Path) -> None:
    cfg = AppConfig(
        music_directory=tmp_path / "music",
        license_directory=tmp_path / "licenses",
        internet_archive_music_enabled=True,
    )
    downloader = MusicDownloader(config=cfg)

    mock_search = MagicMock()
    mock_search.status_code = 200
    mock_search.json.return_value = {
        "response": {
            "docs": [
                {
                    "identifier": "test-dramatic-track",
                    "title": "Dramatic Betrayal Score",
                    "creator": "Open Artist",
                    "licenseurl": "https://creativecommons.org/publicdomain/zero/1.0/",
                }
            ]
        }
    }

    mock_files = MagicMock()
    mock_files.status_code = 200
    mock_files.json.return_value = {
        "files": [
            {"name": "score.mp3", "format": "VBR MP3", "length": 75.0}
        ]
    }

    mock_download = MagicMock()
    mock_download.status_code = 200
    mock_download.iter_content.return_value = [b"ID3" + b"\x00" * 15000]

    with patch("requests.get", side_effect=[mock_search, mock_files, mock_download]):
        result = downloader.search_and_download("dark cello betrayal", "betrayal")

    assert result is not None
    assert result.exists()
    assert result.stat().st_size > 10240

    manifest_path = cfg.license_directory / "music_manifest.json"
    assert manifest_path.exists()
    manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert result.name in manifest_data["tracks"]
    assert manifest_data["tracks"][result.name]["title"] == "Dramatic Betrayal Score"
