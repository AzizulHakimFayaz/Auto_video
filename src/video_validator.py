"""FFprobe video validation module.

Inspects the rendered MP4 container and streams to verify video codec (H.264),
audio codec (AAC), exact 1080x1920 (9:16) resolution, target frame rate (30 FPS),
file size, and duration alignment against the narration audio.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from src.config import AppConfig

logger = logging.getLogger(__name__)


class VideoValidationError(RuntimeError):
    """Raised when the rendered MP4 fails FFprobe structural or stream validation."""


@dataclass
class VideoValidationReport:
    """Verified stream and container attributes of a rendered MP4."""

    file_path: str
    file_size_bytes: int
    video_codec: str
    audio_codec: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    pix_fmt: str
    is_valid: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _parse_frame_rate(rate_str: str) -> float:
    if not rate_str or rate_str == "0/0":
        return 0.0
    if "/" in rate_str:
        parts = rate_str.split("/", 1)
        try:
            num = float(parts[0])
            den = float(parts[1])
            return num / den if den != 0 else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate_str)
    except ValueError:
        return 0.0


class VideoValidator:
    """Validates rendered MP4 files using FFprobe."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def validate(
        self,
        video_path: Path,
        expected_duration_seconds: Optional[float] = None,
        duration_tolerance_seconds: float = 2.0,
    ) -> VideoValidationReport:
        """Inspect `video_path` with FFprobe and verify all vertical short requirements."""
        if not video_path.exists() or not video_path.is_file():
            raise VideoValidationError(f"Rendered video file does not exist: {video_path}")

        file_size = video_path.stat().st_size
        if file_size < 4096:
            raise VideoValidationError(
                f"Rendered video file is suspiciously small or empty ({file_size} bytes): {video_path}"
            )

        cmd = [
            self.config.ffprobe_binary,
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(video_path.resolve()),
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise VideoValidationError(f"Failed to execute ffprobe: {exc}") from exc

        if proc.returncode != 0:
            err_msg = proc.stderr.decode("utf-8", errors="replace").strip()
            raise VideoValidationError(
                f"FFprobe exited with code {proc.returncode} on {video_path}: {err_msg}"
            )

        try:
            probe_data = json.loads(proc.stdout.decode("utf-8", errors="replace"))
        except json.JSONDecodeError as exc:
            raise VideoValidationError(f"FFprobe returned invalid JSON: {exc}") from exc

        streams = probe_data.get("streams", [])
        fmt = probe_data.get("format", {})
        if not isinstance(streams, list) or not streams:
            raise VideoValidationError("FFprobe found no media streams in MP4 file.")

        video_stream = next(
            (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "video"),
            None,
        )
        audio_stream = next(
            (s for s in streams if isinstance(s, dict) and s.get("codec_type") == "audio"),
            None,
        )

        if video_stream is None:
            raise VideoValidationError("MP4 file is missing a video stream.")
        if audio_stream is None:
            raise VideoValidationError("MP4 file is missing an audio stream.")

        v_codec = str(video_stream.get("codec_name", "")).lower()
        a_codec = str(audio_stream.get("codec_name", "")).lower()
        width = int(video_stream.get("width", 0))
        height = int(video_stream.get("height", 0))
        pix_fmt = str(video_stream.get("pix_fmt", "")).lower()

        fps = _parse_frame_rate(
            str(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1")
        )
        duration = float(
            fmt.get("duration") or video_stream.get("duration") or 0.0
        )

        if v_codec != "h264":
            raise VideoValidationError(
                f"Expected H.264 ('h264') video codec, got {v_codec!r}."
            )
        if a_codec != "aac":
            raise VideoValidationError(
                f"Expected AAC ('aac') audio codec, got {a_codec!r}."
            )
        if width != self.config.video_width or height != self.config.video_height:
            raise VideoValidationError(
                f"Expected resolution {self.config.video_width}x{self.config.video_height}, "
                f"got {width}x{height}."
            )
        if abs(fps - float(self.config.video_fps)) > 1.5:
            raise VideoValidationError(
                f"Expected frame rate ~{self.config.video_fps} FPS, got {fps:.2f} FPS."
            )
        if duration <= 0.5:
            raise VideoValidationError(
                f"Invalid video duration reported by FFprobe: {duration:.2f}s."
            )

        if expected_duration_seconds is not None:
            drift = abs(duration - expected_duration_seconds)
            if drift > duration_tolerance_seconds:
                raise VideoValidationError(
                    f"Video duration ({duration:.2f}s) deviates from expected narration duration "
                    f"({expected_duration_seconds:.2f}s) by {drift:.2f}s (tolerance={duration_tolerance_seconds:.2f}s)."
                )

        report = VideoValidationReport(
            file_path=str(video_path.resolve()),
            file_size_bytes=file_size,
            video_codec=v_codec,
            audio_codec=a_codec,
            width=width,
            height=height,
            fps=round(fps, 2),
            duration_seconds=round(duration, 3),
            pix_fmt=pix_fmt,
            is_valid=True,
        )
        logger.info(
            "Video validation passed: %dx%d %s/%s @ %.1f FPS, %.2fs (%d KB)",
            width,
            height,
            v_codec,
            a_codec,
            fps,
            duration,
            file_size // 1024,
        )
        return report
