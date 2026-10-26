"""Subtitle synchronization and SRT/ASS file generator.

Calculates subtitle cue timings directly from the measured durations of each
synthesized scene WAV segment (accounting for inter-scene pauses and offsets),
splits sentences into mobile-friendly short phrases while preserving punctuation,
and guarantees monotonically increasing, non-overlapping timestamps bounded by
the total video duration.
"""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.voice_generator import AudioSegmentResult

logger = logging.getLogger(__name__)


class SubtitleGenerationError(RuntimeError):
    """Raised when subtitle cue calculation or file generation fails."""


@dataclass
class SubtitleCue:
    """Single timed subtitle entry."""

    index: int
    scene_index: int
    start_seconds: float
    end_seconds: float
    text: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def format_srt_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp `HH:MM:SS,mmm`."""
    clamped = max(0.0, seconds)
    total_ms = int(round(clamped * 1000.0))
    hours = total_ms // 3_600_000
    rem = total_ms % 3_600_000
    minutes = rem // 60_000
    rem %= 60_000
    secs = rem // 1000
    millis = rem % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_ass_timestamp(seconds: float) -> str:
    """Format seconds as ASS timestamp `H:MM:SS.cc`."""
    clamped = max(0.0, seconds)
    total_cs = int(round(clamped * 100.0))
    hours = total_cs // 360_000
    rem = total_cs % 360_000
    minutes = rem // 6_000
    rem %= 6_000
    secs = rem // 100
    centis = rem % 100
    return f"{hours:d}:{minutes:02d}:{secs:02d}.{centis:02d}"


def split_text_into_mobile_phrases(
    text: str,
    max_chars_per_cue: int = 56,
    max_words_per_cue: int = 10,
) -> List[str]:
    """Split a scene narration into readable mobile phrases while preserving punctuation."""
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        return []

    # First split on major sentence/clause punctuation while keeping the delimiter attached
    raw_clauses = re.split(r"(?<=[.!?;:—])\s+|(?<=,)\s+", cleaned)
    phrases: List[str] = []

    current_words: List[str] = []
    current_len = 0

    for clause in raw_clauses:
        clause = clause.strip()
        if not clause:
            continue
        words = clause.split()
        for word in words:
            projected_len = current_len + (1 if current_words else 0) + len(word)
            if current_words and (
                projected_len > max_chars_per_cue or len(current_words) >= max_words_per_cue
            ):
                phrases.append(" ".join(current_words))
                current_words = [word]
                current_len = len(word)
            else:
                current_words.append(word)
                current_len = projected_len

            # Natural break if word ends with strong sentence punctuation and phrase is substantial
            if word.endswith((".", "!", "?", ";")) and len(current_words) >= 4:
                phrases.append(" ".join(current_words))
                current_words = []
                current_len = 0

    if current_words:
        phrases.append(" ".join(current_words))

    return phrases


def wrap_cue_lines(text: str, max_line_chars: int = 32) -> str:
    """Wrap a phrase into at most two balanced lines for vertical mobile screens."""
    words = text.strip().split()
    if not words:
        return ""
    if len(text) <= max_line_chars or len(words) <= 3:
        return text

    # Split near the middle of character length
    mid = len(text) / 2.0
    best_idx = 1
    best_diff = float("inf")
    running = 0
    for i in range(1, len(words)):
        running += len(words[i - 1]) + (1 if i > 1 else 0)
        diff = abs(running - mid)
        if diff < best_diff:
            best_diff = diff
            best_idx = i

    line1 = " ".join(words[:best_idx])
    line2 = " ".join(words[best_idx:])
    return f"{line1}\n{line2}"


class SubtitleGenerator:
    """Generates synchronized SRT and ASS subtitles from measured audio segments."""

    def __init__(
        self,
        max_chars_per_cue: int = 56,
        max_words_per_cue: int = 9,
        max_line_chars: int = 30,
    ) -> None:
        self.max_chars_per_cue = max_chars_per_cue
        self.max_words_per_cue = max_words_per_cue
        self.max_line_chars = max_line_chars

    def build_cues(
        self,
        audio_segments: List[AudioSegmentResult],
        inter_scene_pause_seconds: float = 0.25,
        max_video_duration: Optional[float] = None,
        initial_offset_seconds: float = 0.0,
    ) -> List[SubtitleCue]:
        """Compute subtitle cues strictly anchored to each audio segment's real duration."""
        if not audio_segments:
            raise SubtitleGenerationError("Cannot build subtitles from empty audio segments.")

        cues: List[SubtitleCue] = []
        cursor = max(0.0, initial_offset_seconds)
        cue_index = 1

        for seg_idx, segment in enumerate(audio_segments):
            seg_duration = max(0.2, float(segment.duration_seconds))
            scene_start = cursor
            scene_end = scene_start + seg_duration

            if max_video_duration is not None:
                if scene_start >= max_video_duration:
                    break
                scene_end = min(scene_end, max_video_duration)

            effective_seg_duration = max(0.1, scene_end - scene_start)
            phrases = split_text_into_mobile_phrases(
                segment.narration_text,
                max_chars_per_cue=self.max_chars_per_cue,
                max_words_per_cue=self.max_words_per_cue,
            )
            if not phrases:
                phrases = [segment.narration_text.strip()]

            # Weight each phrase within this measured audio segment by spoken character count + punctuation pause weight
            weights: List[float] = []
            for p in phrases:
                spoken_chars = len(re.findall(r"[A-Za-z0-9]", p))
                punct_bonus = 2.5 if p.endswith((".", "!", "?")) else (1.0 if p.endswith(",") else 0.0)
                weights.append(max(1.0, float(spoken_chars) + punct_bonus))

            total_weight = sum(weights)
            phrase_cursor = scene_start

            for p_idx, (phrase, weight) in enumerate(zip(phrases, weights)):
                if p_idx == len(phrases) - 1:
                    phrase_end = scene_end
                else:
                    share = (weight / total_weight) * effective_seg_duration
                    phrase_end = min(scene_end, phrase_cursor + max(0.25, share))

                # Ensure non-overlapping monotonically increasing timestamps
                start_t = round(max(0.0, phrase_cursor), 3)
                end_t = round(max(start_t + 0.08, phrase_end), 3)
                if max_video_duration is not None:
                    end_t = round(min(end_t, max_video_duration), 3)
                    if start_t >= end_t:
                        break

                wrapped = wrap_cue_lines(phrase, max_line_chars=self.max_line_chars)
                cues.append(
                    SubtitleCue(
                        index=cue_index,
                        scene_index=segment.scene_index,
                        start_seconds=start_t,
                        end_seconds=end_t,
                        text=wrapped,
                    )
                )
                cue_index += 1
                phrase_cursor = end_t

            cursor = scene_end + (
                inter_scene_pause_seconds if seg_idx < len(audio_segments) - 1 else 0.0
            )

        if not cues:
            raise SubtitleGenerationError("No valid subtitle cues were generated.")

        return cues

    def write_srt(self, cues: List[SubtitleCue], output_path: Path) -> Path:
        """Write standard SubRip (.srt) subtitle file."""
        if not cues:
            raise SubtitleGenerationError("Cannot write an empty SRT subtitle file.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        blocks: List[str] = []
        for idx, cue in enumerate(cues, start=1):
            start_str = format_srt_timestamp(cue.start_seconds)
            end_str = format_srt_timestamp(cue.end_seconds)
            blocks.append(f"{idx}\n{start_str} --> {end_str}\n{cue.text}\n")

        output_path.write_text("\n".join(blocks), encoding="utf-8")
        logger.info("Generated SRT subtitle file with %d cues: %s", len(cues), output_path)
        return output_path

    def write_ass(
        self,
        cues: List[SubtitleCue],
        output_path: Path,
        video_width: int = 1080,
        video_height: int = 1920,
        font_name: str = "Arial",
        font_size: int = 52,
        margin_v: int = 240,
    ) -> Path:
        """Write Advanced SubStation Alpha (.ass) file with high-contrast mobile styling."""
        if not cues:
            raise SubtitleGenerationError("Cannot write an empty ASS subtitle file.")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        header = f"""[Script Info]
Title: Auto Story Video Subtitles
ScriptType: v4.00+
PlayResX: {video_width}
PlayResY: {video_height}
WrapStyle: 1
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font_name},{font_size},&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,4,2,2,80,80,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
        events: List[str] = []
        for cue in cues:
            start_str = format_ass_timestamp(cue.start_seconds)
            end_str = format_ass_timestamp(cue.end_seconds)
            ass_text = cue.text.replace("\n", r"\N")
            events.append(
                f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{ass_text}"
            )

        output_path.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
        logger.info("Generated ASS subtitle file with %d cues: %s", len(cues), output_path)
        return output_path
