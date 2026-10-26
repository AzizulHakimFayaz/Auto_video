"""Unit tests for mood-based music selection, license checks, and Piper TTS validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import AppConfig
from src.music_selector import MusicSelectionError, MusicSelector
from src.story_generator import SceneSegment
from src.voice_generator import (
    VoiceGenerationError,
    VoiceGenerator,
    validate_wav_file,
)


def test_music_selector_matches_mood_with_verified_license(tmp_path: Path) -> None:
    music_dir = tmp_path / "music"
    lic_dir = tmp_path / "licenses"
    music_dir.mkdir(parents=True)
    lic_dir.mkdir(parents=True)

    sad_track = music_dir / "sad_piano.wav"
    sad_track.write_bytes(b"RIFF" + b"\x00" * 1024)
    hopeful_track = music_dir / "hopeful_sunrise.wav"
    hopeful_track.write_bytes(b"RIFF" + b"\x00" * 1024)

    manifest = {
        "tracks": {
            "sad_piano.wav": {
                "title": "Sad Piano",
                "creator": "Composer A",
                "source_url": "https://example.org/sad",
                "license_name": "CC0 1.0",
                "attribution_instructions": "Sad Piano by Composer A (CC0)",
                "commercial_use_allowed": True,
                "mood_tags": ["sad", "melancholic", "piano"],
            },
            "hopeful_sunrise.wav": {
                "title": "Hopeful Sunrise",
                "creator": "Composer B",
                "source_url": "https://example.org/hope",
                "license_name": "CC BY 4.0",
                "attribution_instructions": "Hopeful Sunrise by Composer B (CC BY 4.0)",
                "commercial_use_allowed": True,
                "mood_tags": ["hopeful", "uplifting", "hope"],
            },
        }
    }
    (lic_dir / "music_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    cfg = AppConfig(music_directory=music_dir, license_directory=lic_dir)
    selector = MusicSelector(config=cfg)

    selected = selector.select_music("melancholic and sad")
    assert selected is not None
    assert selected.title == "Sad Piano"


def test_music_selector_rejects_unlicensed_audio_and_falls_back_to_narration_only(
    tmp_path: Path,
) -> None:
    music_dir = tmp_path / "music"
    lic_dir = tmp_path / "licenses"
    music_dir.mkdir(parents=True)
    lic_dir.mkdir(parents=True)

    # Track exists on disk but has NO entry in music_manifest.json
    unlicensed = music_dir / "mystery_song.mp3"
    unlicensed.write_bytes(b"ID3" + b"\x00" * 2048)

    cfg = AppConfig(
        music_directory=music_dir,
        license_directory=lic_dir,
        allow_narration_only_fallback=True,
    )
    selector = MusicSelector(config=cfg)
    assert selector.select_music("sad") is None

    # When fallback is disabled, raise MusicSelectionError
    cfg_strict = AppConfig(
        music_directory=music_dir,
        license_directory=lic_dir,
        allow_narration_only_fallback=False,
    )
    with pytest.raises(MusicSelectionError, match="No licensed background music"):
        MusicSelector(config=cfg_strict).select_music("sad")


def test_piper_missing_model_and_corrupt_wav_fail_clearly(tmp_path: Path) -> None:
    cfg = AppConfig(
        piper_model_path=tmp_path / "nonexistent_voice.onnx",
        piper_config_path=tmp_path / "nonexistent_voice.onnx.json",
        allow_Offline_tts_fallback=False,
    )
    vg = VoiceGenerator(config=cfg)
    with pytest.raises(VoiceGenerationError, match="Piper ONNX voice model not found"):
        vg.verify_model_and_license()

    bad_wav = tmp_path / "empty.wav"
    bad_wav.write_bytes(b"RIFF1234")
    with pytest.raises(VoiceGenerationError, match="empty or contains only a header"):
        validate_wav_file(bad_wav)


def test_piper_model_compatibility_check_and_offline_synthesis(tmp_path: Path) -> None:
    model_file = tmp_path / "en_US-ljspeech-medium.onnx"
    config_file = tmp_path / "en_US-ljspeech-medium.onnx.json"
    model_file.write_bytes(b"\x08\x01" * 1024)
    config_file.write_text(
        json.dumps(
            {
                "audio": {"sample_rate": 22050},
                "espeak": {"voice": "en-us"},
                "num_speakers": 1,
                "phoneme_type": "espeak",
            }
        ),
        encoding="utf-8",
    )

    cfg = AppConfig(
        piper_voice="en_US-ljspeech-medium",
        piper_model_path=model_file,
        piper_config_path=config_file,
        allow_Offline_tts_fallback=True,
    )
    vg = VoiceGenerator(config=cfg)
    info = vg.verify_model_and_license()
    assert info["sample_rate"] == 22050
    assert "Public Domain" in info["license_status"]

    scenes = [
        SceneSegment(
            narration="Every evening at six, Aris waited on Platform Four.",
            image_query="train platform evening",
            visual_mood="sadness",
        ),
        SceneSegment(
            narration="One storm never ruins the whole blueprint, he said softly.",
            image_query="sunrise train",
            visual_mood="hope",
        ),
    ]
    segments, master_wav, total_dur = vg.generate_scene_narrations(
        scenes=scenes, work_dir=tmp_path / "tts_out"
    )
    assert len(segments) == 2
    assert master_wav.exists()
    assert total_dur > 3.0
