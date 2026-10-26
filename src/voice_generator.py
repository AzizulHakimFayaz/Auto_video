"""Local Text-to-Speech module powered by Piper TTS.

Converts story scene narrations into individual WAV audio segments for
frame-accurate subtitle synchronization, validates model compatibility and
licensing metadata, applies configurable speech rate and volume scaling,
and verifies output WAV duration and integrity.
"""

from __future__ import annotations

import json
import logging
import math
import re
import shutil
import struct
import subprocess
import wave
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.config import AppConfig
from src.story_generator import SceneSegment

logger = logging.getLogger(__name__)

# Verified licenses for official rhasspy/piper-voices models on Hugging Face
VERIFIED_PIPER_VOICE_LICENSES: Dict[str, Dict[str, Any]] = {
    "en_US-ljspeech-medium": {
        "dataset": "LJ Speech Dataset",
        "license": "Public Domain (CC0 / Unencumbered)",
        "commercial_use_permitted": True,
        "language": "English",
        "locale": "en_US",
        "official_repo": "https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ljspeech/medium",
    },
    "en_US-ljspeech-high": {
        "dataset": "LJ Speech Dataset",
        "license": "Public Domain (CC0 / Unencumbered)",
        "commercial_use_permitted": True,
        "language": "English",
        "locale": "en_US",
        "official_repo": "https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/ljspeech/high",
    },
    "en_US-libritts_r-medium": {
        "dataset": "LibriTTS-R",
        "license": "CC BY 4.0",
        "commercial_use_permitted": True,
        "language": "English",
        "locale": "en_US",
        "official_repo": "https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/libritts_r/medium",
    },
    "en_US-lessac-medium": {
        "dataset": "Blizzard Challenge 2013 (Lessac)",
        "license": "Blizzard 2013 Research / Non-Commercial License",
        "commercial_use_permitted": False,
        "language": "English",
        "locale": "en_US",
        "official_repo": "https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_US/lessac/medium",
    },
}


class VoiceGenerationError(RuntimeError):
    """Raised when Piper TTS model validation or audio synthesis fails."""


@dataclass
class SpokenChunk:
    """A single continuous utterance within a scene, followed by an optional dramatic pause."""

    text: str
    duration_seconds: float
    pause_after_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


PAUSE_REGEX = re.compile(r"\[pause(?::\s*([0-9]+(?:\.[0-9]+)?)s?)?\]", re.IGNORECASE)


def split_narration_pauses(text: str) -> List[Tuple[str, float]]:
    """Split narration text on [pause: Xs] or [pause] tags into (speech_text, pause_after_seconds)."""
    matches = list(PAUSE_REGEX.finditer(text))
    if not matches:
        return [(text.strip(), 0.0)]

    chunks: List[Tuple[str, float]] = []
    last_end = 0
    for m in matches:
        speech = text[last_end : m.start()].strip()
        pause_sec = float(m.group(1)) if m.group(1) else 1.0
        if speech:
            chunks.append((speech, pause_sec))
        last_end = m.end()

    trailing = text[last_end:].strip()
    if trailing:
        chunks.append((trailing, 0.0))
    return chunks


def create_silence_wav(
    output_path: Path,
    duration_seconds: float,
    sample_rate: int = 22050,
    n_channels: int = 1,
    sampwidth: int = 2,
) -> None:
    """Create a PCM 16-bit silence WAV file of specified duration."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_frames = max(1, int(round(duration_seconds * sample_rate)))
    silence_bytes = b"\x00" * (num_frames * n_channels * sampwidth)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(sample_rate)
        wf.writeframes(silence_bytes)


@dataclass
class AudioSegmentResult:
    """Metadata and timing for a synthesized scene narration WAV file."""

    scene_index: int
    narration_text: str
    audio_path: str
    duration_seconds: float
    sample_rate: int
    voice_used: str
    license_status: str
    spoken_chunks: List[SpokenChunk] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["spoken_chunks"] = [c.to_dict() for c in self.spoken_chunks]
        return data


def validate_wav_file(wav_path: Path, min_duration_seconds: float = 0.2) -> Tuple[float, int]:
    """Validate that a WAV file exists, is well-formed PCM audio, and meets minimum duration.

    Returns (duration_seconds, sample_rate).
    """
    if not wav_path.exists() or not wav_path.is_file():
        raise VoiceGenerationError(f"Generated WAV file does not exist: {wav_path}")

    try:
        file_size = wav_path.stat().st_size
    except OSError as exc:
        raise VoiceGenerationError(f"Cannot stat WAV file {wav_path}: {exc}") from exc

    if file_size <= 44:
        raise VoiceGenerationError(
            f"Generated WAV file is empty or contains only a header ({file_size} bytes): {wav_path}"
        )

    try:
        with wave.open(str(wav_path), "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            n_frames = wf.getnframes()
    except (wave.Error, EOFError, OSError) as exc:
        raise VoiceGenerationError(f"Corrupt or invalid WAV file {wav_path}: {exc}") from exc

    if sample_rate <= 0 or n_channels <= 0 or sample_width <= 0:
        raise VoiceGenerationError(
            f"Invalid WAV format parameters in {wav_path} (rate={sample_rate}, ch={n_channels})"
        )

    duration = n_frames / float(sample_rate)
    if duration < min_duration_seconds:
        raise VoiceGenerationError(
            f"Generated WAV duration ({duration:.3f}s) is shorter than minimum {min_duration_seconds:.3f}s: {wav_path}"
        )

    return duration, sample_rate


def scale_wav_volume(wav_path: Path, volume_factor: float) -> None:
    """Apply lossless clamped PCM amplitude scaling when volume_factor != 1.0."""
    if abs(volume_factor - 1.0) < 1e-3:
        return
    if volume_factor <= 0.0:
        raise VoiceGenerationError("PIPER_VOLUME must be greater than 0.0.")

    with wave.open(str(wav_path), "rb") as rf:
        params = rf.getparams()
        if params.sampwidth != 2:
            return
        raw_frames = rf.readframes(params.nframes)

    sample_count = len(raw_frames) // 2
    samples = struct.unpack(f"<{sample_count}h", raw_frames)
    scaled = [
        max(-32768, min(32767, int(round(s * volume_factor)))) for s in samples
    ]
    packed = struct.pack(f"<{sample_count}h", *scaled)

    temp_out = wav_path.with_suffix(".vol.wav")
    with wave.open(str(temp_out), "wb") as wf:
        wf.setparams(params)
        wf.writeframes(packed)
    temp_out.replace(wav_path)


def concatenate_wav_files(
    segment_paths: List[Path],
    output_path: Path,
    inter_scene_pauses: Optional[List[float]] = None,
    inter_scene_pause_seconds: float = 0.55,
) -> float:
    """Combine multiple scene WAV files into a single master narration WAV file."""
    if not segment_paths:
        raise VoiceGenerationError("Cannot concatenate an empty list of WAV segments.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    first_duration, sample_rate = validate_wav_file(segment_paths[0])

    with wave.open(str(segment_paths[0]), "rb") as first_wf:
        n_channels = first_wf.getnchannels()
        sampwidth = first_wf.getsampwidth()

    with wave.open(str(output_path), "wb") as out_wf:
        out_wf.setnchannels(n_channels)
        out_wf.setsampwidth(sampwidth)
        out_wf.setframerate(sample_rate)

        for idx, seg_path in enumerate(segment_paths):
            with wave.open(str(seg_path), "rb") as in_wf:
                if (
                    in_wf.getframerate() != sample_rate
                    or in_wf.getnchannels() != n_channels
                    or in_wf.getsampwidth() != sampwidth
                ):
                    raise VoiceGenerationError(
                        f"WAV format mismatch across narration segments at {seg_path}."
                    )
                out_wf.writeframes(in_wf.readframes(in_wf.getnframes()))

            if idx < len(segment_paths) - 1:
                pause_s = (
                    inter_scene_pauses[idx]
                    if inter_scene_pauses and idx < len(inter_scene_pauses)
                    else inter_scene_pause_seconds
                )
                if pause_s > 0.0:
                    pause_frames = int(round(pause_s * sample_rate))
                    silence_bytes = b"\x00" * (pause_frames * n_channels * sampwidth)
                    out_wf.writeframes(silence_bytes)

    total_duration, _ = validate_wav_file(output_path)
    return total_duration


class VoiceGenerator:
    """Synthesizes natural local speech per scene using Piper TTS."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _resolve_model_paths(self) -> Tuple[Path, Path]:
        """Locate the configured .onnx model and .onnx.json config file."""
        if self.config.piper_model_path:
            model_path = self.config.piper_model_path
        else:
            voices_dir = self.config.project_root / "models" / "piper"
            model_path = voices_dir / f"{self.config.piper_voice}.onnx"

        if self.config.piper_config_path:
            config_path = self.config.piper_config_path
        else:
            config_path = Path(f"{model_path}.json")

        return model_path, config_path

    def verify_model_and_license(self) -> Dict[str, Any]:
        """Verify that the Piper voice model and JSON config exist, are compatible, and check license."""
        model_path, config_path = self._resolve_model_paths()

        if not model_path.exists() or not model_path.is_file():
            raise VoiceGenerationError(
                f"Piper ONNX voice model not found at {model_path}. "
                "Please download a verified voice model (e.g., en_US-ljspeech-medium) "
                "from https://huggingface.co/rhasspy/piper-voices and set PIPER_MODEL_PATH."
            )
        if model_path.stat().st_size < 1024:
            raise VoiceGenerationError(
                f"Piper ONNX model file at {model_path} appears corrupt or truncated."
            )

        if not config_path.exists() or not config_path.is_file():
            raise VoiceGenerationError(
                f"Piper model configuration JSON not found at {config_path}. "
                "Both the .onnx and .onnx.json files are required."
            )

        try:
            model_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise VoiceGenerationError(
                f"Failed to parse Piper model config {config_path}: {exc}"
            ) from exc

        if not isinstance(model_cfg, dict) or "audio" not in model_cfg:
            raise VoiceGenerationError(
                f"Incompatible Piper model config {config_path}: missing required 'audio' section."
            )

        sample_rate = model_cfg.get("audio", {}).get("sample_rate")
        if not isinstance(sample_rate, int) or sample_rate < 8000:
            raise VoiceGenerationError(
                f"Incompatible Piper sample_rate ({sample_rate!r}) in {config_path}."
            )

        phoneme_type = model_cfg.get("phoneme_type", "espeak")
        if phoneme_type not in {"espeak", "text"}:
            raise VoiceGenerationError(
                f"Unsupported Piper phoneme_type {phoneme_type!r} in {config_path}."
            )

        voice_key = self.config.piper_voice or model_path.stem
        license_info = VERIFIED_PIPER_VOICE_LICENSES.get(voice_key)
        if license_info is None:
            logger.warning(
                "Piper voice '%s' does not have a pre-verified license entry. "
                "Treating license as UNVERIFIED — verify the voice's MODEL_CARD before redistribution.",
                voice_key,
            )
            license_status = "UNVERIFIED (Check MODEL_CARD)"
        else:
            license_status = f"{license_info['license']} ({license_info['dataset']})"
            if not license_info.get("commercial_use_permitted", False):
                logger.warning(
                    "Notice: Piper voice '%s' uses a non-commercial license (%s). "
                    "For monetized channels, use 'en_US-ljspeech-medium' (Public Domain) "
                    "or 'en_US-libritts_r-medium' (CC BY 4.0).",
                    voice_key,
                    license_info["license"],
                )

        return {
            "model_path": str(model_path),
            "config_path": str(config_path),
            "sample_rate": sample_rate,
            "num_speakers": int(model_cfg.get("num_speakers", 1)),
            "voice_key": voice_key,
            "license_status": license_status,
        }

    def _locate_piper_binary(self) -> Optional[str]:
        exe = self.config.piper_executable
        resolved = shutil.which(exe)
        if resolved:
            return resolved
        candidate_path = Path(exe)
        if candidate_path.exists() and candidate_path.is_file():
            return str(candidate_path.resolve())
        return None

    def _synthesize_with_piper_cli(
        self,
        text: str,
        output_wav: Path,
        model_path: Path,
        config_path: Path,
        piper_bin: str,
    ) -> None:
        cmd: List[str] = [
            piper_bin,
            "--model",
            str(model_path),
            "--config",
            str(config_path),
            "--output_file",
            str(output_wav),
            "--length_scale",
            f"{self.config.piper_length_scale:.2f}",
            "--sentence_silence",
            f"{self.config.piper_sentence_silence:.2f}",
        ]
        if self.config.piper_speaker_id is not None:
            cmd.extend(["--speaker", str(self.config.piper_speaker_id)])

        try:
            proc = subprocess.run(
                cmd,
                input=text.encode("utf-8"),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VoiceGenerationError(f"Failed to execute Piper binary: {exc}") from exc

        if proc.returncode != 0:
            stderr_msg = proc.stderr.decode("utf-8", errors="replace").strip()
            raise VoiceGenerationError(
                f"Piper TTS exited with code {proc.returncode}: {stderr_msg}"
            )

    def _synthesize_offline_fallback_wav(self, text: str, output_wav: Path) -> None:
        """Deterministic local acoustic speech-cadence synthesizer for offline testing when explicitly enabled."""
        words = [w for w in text.split() if w.strip()]
        word_count = max(1, len(words))
        # Natural speaking rate ~2.4 words/sec plus punctuation pauses
        sentence_breaks = sum(text.count(p) for p in (".", "!", "?", ";"))
        duration = max(
            1.5,
            (word_count / 2.4) * self.config.piper_length_scale
            + sentence_breaks * self.config.piper_sentence_silence,
        )
        sample_rate = 22050
        total_samples = int(round(duration * sample_rate))

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        samples = bytearray(total_samples * 2)
        base_freq = 165.0

        for i in range(total_samples):
            t = i / float(sample_rate)
            # Modulate envelope by syllable cadence (~4 Hz) with sentence pauses
            syllable_env = 0.55 + 0.45 * math.sin(2.0 * math.pi * 4.2 * t)
            pitch_mod = base_freq + 18.0 * math.sin(2.0 * math.pi * 0.8 * t)
            fundamental = math.sin(2.0 * math.pi * pitch_mod * t)
            harmonic2 = 0.35 * math.sin(2.0 * math.pi * (pitch_mod * 2.0) * t)
            harmonic3 = 0.15 * math.sin(2.0 * math.pi * (pitch_mod * 3.0) * t)
            fade = min(1.0, t / 0.08, (duration - t) / 0.12)
            amp = int(
                max(
                    -32767,
                    min(
                        32767,
                        5500.0
                        * self.config.piper_volume
                        * fade
                        * syllable_env
                        * (fundamental + harmonic2 + harmonic3),
                    ),
                )
            )
            struct.pack_into("<h", samples, i * 2, amp)

        with wave.open(str(output_wav), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(bytes(samples))

    def _synthesize_speech_audio(self, text: str, output_wav: Path) -> Tuple[str, str]:
        """Synthesize a single continuous speech utterance using Piper or offline fallback."""
        piper_bin = self._locate_piper_binary()
        if piper_bin is not None:
            model_meta = self.verify_model_and_license()
            self._synthesize_with_piper_cli(
                text=text,
                output_wav=output_wav,
                model_path=Path(model_meta["model_path"]),
                config_path=Path(model_meta["config_path"]),
                piper_bin=piper_bin,
            )
            return model_meta["voice_key"], model_meta["license_status"]

        if self.config.allow_Offline_tts_fallback:
            self._synthesize_offline_fallback_wav(text, output_wav)
            return "offline-local-synth-fallback", "Public Domain (Procedural Local Audio)"

        self.verify_model_and_license()
        raise VoiceGenerationError(
            f"Piper executable '{self.config.piper_executable}' was not found on PATH. "
            "Install piper-tts (`pip install piper-tts` or download the official Piper binary release) "
            "and configure PIPER_EXECUTABLE and PIPER_MODEL_PATH."
        )

    def synthesize_segment(
        self,
        text: str,
        output_wav: Path,
        scene_index: int = 0,
    ) -> AudioSegmentResult:
        """Synthesize a single scene narration into a validated WAV file, respecting pause markers."""
        chunks = split_narration_pauses(text)
        if not chunks or (len(chunks) == 1 and not chunks[0][0]):
            raise VoiceGenerationError(
                f"Scene {scene_index + 1} narration text is empty; cannot synthesize audio."
            )

        output_wav.parent.mkdir(parents=True, exist_ok=True)
        has_explicit_pauses = len(chunks) > 1 or (len(chunks) == 1 and chunks[0][1] > 0.0)

        if has_explicit_pauses:
            temp_dir = output_wav.parent / f"_pause_parts_{output_wav.stem}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            try:
                sub_paths: List[Path] = []
                spoken_chunks: List[SpokenChunk] = []
                voice_used = ""
                license_status = ""
                sample_rate = 22050

                for c_idx, (speech_text, pause_sec) in enumerate(chunks):
                    if speech_text:
                        c_wav = temp_dir / f"chunk_{c_idx}.wav"
                        voice_used, license_status = self._synthesize_speech_audio(speech_text, c_wav)
                        dur, sample_rate = validate_wav_file(c_wav)
                        sub_paths.append(c_wav)
                        spoken_chunks.append(
                            SpokenChunk(
                                text=speech_text,
                                duration_seconds=round(dur, 3),
                                pause_after_seconds=round(pause_sec, 3),
                            )
                        )
                    if pause_sec > 0.0 and (c_idx < len(chunks) - 1 or speech_text):
                        p_wav = temp_dir / f"pause_{c_idx}.wav"
                        create_silence_wav(p_wav, pause_sec, sample_rate=sample_rate)
                        sub_paths.append(p_wav)

                if not sub_paths:
                    raise VoiceGenerationError(f"No audio produced for scene {scene_index + 1}.")

                concatenate_wav_files(
                    segment_paths=sub_paths,
                    output_path=output_wav,
                    inter_scene_pause_seconds=0.0,
                )
                scale_wav_volume(output_wav, self.config.piper_volume)
                total_dur, sr = validate_wav_file(output_wav)
                clean_text = " ".join(c[0] for c in chunks if c[0])
                return AudioSegmentResult(
                    scene_index=scene_index,
                    narration_text=clean_text,
                    audio_path=str(output_wav.resolve()),
                    duration_seconds=round(total_dur, 3),
                    sample_rate=sr,
                    voice_used=voice_used,
                    license_status=license_status,
                    spoken_chunks=spoken_chunks,
                )
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

        # Single continuous speech without pause tags:
        clean_text = " ".join(chunks[0][0].split())
        voice_used, license_status = self._synthesize_speech_audio(clean_text, output_wav)
        scale_wav_volume(output_wav, self.config.piper_volume)
        duration, sample_rate = validate_wav_file(output_wav)
        return AudioSegmentResult(
            scene_index=scene_index,
            narration_text=clean_text,
            audio_path=str(output_wav.resolve()),
            duration_seconds=round(duration, 3),
            sample_rate=sample_rate,
            voice_used=voice_used,
            license_status=license_status,
            spoken_chunks=[
                SpokenChunk(
                    text=clean_text,
                    duration_seconds=round(duration, 3),
                    pause_after_seconds=0.0,
                )
            ],
        )

    def generate_scene_narrations(
        self,
        scenes: List[SceneSegment],
        work_dir: Path,
        inter_scene_pause_seconds: float = 0.55,
    ) -> Tuple[List[AudioSegmentResult], Path, float]:
        """Generate individual WAV files for every scene plus a combined narration WAV.

        Returns (segment_results, combined_wav_path, total_duration_seconds).
        """
        if not scenes:
            raise VoiceGenerationError("Cannot generate narration for an empty scene list.")

        work_dir.mkdir(parents=True, exist_ok=True)
        segment_results: List[AudioSegmentResult] = []
        segment_paths: List[Path] = []

        for idx, scene in enumerate(scenes):
            seg_wav = work_dir / f"scene_{idx + 1:02d}.wav"
            result = self.synthesize_segment(
                text=scene.narration, output_wav=seg_wav, scene_index=idx
            )
            segment_results.append(result)
            segment_paths.append(seg_wav)
            logger.info(
                "Scene %d narration generated: %.2fs (%s)",
                idx + 1,
                result.duration_seconds,
                seg_wav.name,
            )

        inter_pauses = [getattr(s, "pause_after", inter_scene_pause_seconds) for s in scenes[:-1]]
        combined_wav = work_dir / "narration_master.wav"
        total_duration = concatenate_wav_files(
            segment_paths=segment_paths,
            output_path=combined_wav,
            inter_scene_pauses=inter_pauses,
            inter_scene_pause_seconds=inter_scene_pause_seconds,
        )
        logger.info(
            "Master narration audio assembled: %.2fs total duration (%s)",
            total_duration,
            combined_wav.name,
        )
        return segment_results, combined_wav, round(total_duration, 3)
