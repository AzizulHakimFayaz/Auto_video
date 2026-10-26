"""FFmpeg vertical video renderer (1080x1920, 9:16, 30 FPS, H.264/AAC).

Combines scene images with aspect-preserving scale/crop, Ken Burns zoom/pan,
scene-to-scene crossfades, a subtle dark readability overlay, opening story
title card, burned-in synchronized subtitles, and ducked/faded background music.
Manages temporary workspace files and preserves diagnostics on failure.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.config import AppConfig
from src.image_selector import ImageMetadata
from src.music_selector import MusicTrackMetadata
from src.voice_generator import AudioSegmentResult

logger = logging.getLogger(__name__)


class VideoRenderError(RuntimeError):
    """Raised when FFmpeg command execution or video rendering fails."""


@dataclass
class RenderResult:
    """Details of a completed MP4 render."""

    output_video_path: str
    subtitle_path: str
    target_duration_seconds: float
    scene_count: int
    music_used: bool
    ffmpeg_command: List[str]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def escape_ffmpeg_filter_path(path: Path) -> str:
    """Escape a filesystem path for safe embedding inside an FFmpeg filtergraph string."""
    posix_str = path.resolve().as_posix()
    escaped = posix_str.replace("\\", "/")
    escaped = escaped.replace(":", r"\:")
    escaped = escaped.replace("'", r"\'")
    escaped = escaped.replace("[", r"\[").replace("]", r"\]")
    return escaped


def escape_ffmpeg_drawtext(text: str) -> str:
    """Escape a title string for FFmpeg's `drawtext` filter."""
    cleaned = " ".join(text.strip().split())
    cleaned = cleaned.replace("\\", "\\\\")
    cleaned = cleaned.replace(":", r"\:")
    cleaned = cleaned.replace("'", "")
    cleaned = cleaned.replace("%", r"\%")
    cleaned = cleaned.replace(",", r"\,")
    cleaned = cleaned.replace(";", r"\;")
    return cleaned[:70]


class VideoRenderer:
    """Renders the daily 9:16 vertical short story MP4 using FFmpeg."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def build_ffmpeg_command(
        self,
        images: List[ImageMetadata],
        audio_segments: List[AudioSegmentResult],
        master_narration_wav: Path,
        subtitle_file: Path,
        output_mp4: Path,
        story_title: str,
        total_duration_seconds: float,
        music_track: Optional[MusicTrackMetadata] = None,
        inter_scene_pauses: Optional[List[float]] = None,
        inter_scene_pause_seconds: float = 0.55,
    ) -> List[str]:
        """Construct the complete FFmpeg CLI command for vertical video rendering."""
        if not images or not audio_segments:
            raise VideoRenderError("Cannot build FFmpeg command with empty images or audio segments.")
        if len(images) != len(audio_segments):
            raise VideoRenderError(
                f"Scene count mismatch: {len(images)} images vs {len(audio_segments)} audio segments."
            )

        w = self.config.video_width
        h = self.config.video_height
        fps = self.config.video_fps
        xfade_dur = max(0.1, min(1.0, self.config.crossfade_duration_seconds))
        num_scenes = len(images)

        cmd: List[str] = [self.config.ffmpeg_binary, "-y"]

        # Calculate exact per-scene slot durations so post-xfade length equals total_duration_seconds
        slot_durations: List[float] = []
        for idx, seg in enumerate(audio_segments):
            pause = (
                inter_scene_pauses[idx]
                if inter_scene_pauses and idx < len(inter_scene_pauses)
                else (inter_scene_pause_seconds if idx < num_scenes - 1 else 0.0)
            )
            slot_durations.append(max(1.0, float(seg.duration_seconds) + pause))

        # Adjust final slot if needed to match total_duration_seconds
        slot_sum = sum(slot_durations)
        if abs(slot_sum - total_duration_seconds) > 0.02:
            diff = total_duration_seconds - slot_sum
            slot_durations[-1] = max(1.0, slot_durations[-1] + diff)

        # Add still image inputs (zoompan synthesizes the exact frame count per scene)
        for img in images:
            cmd.extend(["-i", img.local_path])

        # Add master narration audio input (index = num_scenes)
        narration_input_idx = num_scenes
        cmd.extend(["-i", str(master_narration_wav.resolve())])

        # Add optional background music input (index = num_scenes + 1)
        music_input_idx: Optional[int] = None
        if music_track is not None:
            music_input_idx = num_scenes + 1
            cmd.extend(
                [
                    "-stream_loop",
                    "-1",
                    "-t",
                    f"{total_duration_seconds:.3f}",
                    "-i",
                    music_track.file_path,
                ]
            )

        filter_parts: List[str] = []

        # 1. Per-scene scale/crop (no stretching) and Ken Burns zoom/pan
        for idx in range(num_scenes):
            clip_dur = slot_durations[idx] + (xfade_dur if idx < num_scenes - 1 else 0.0)
            frames = max(1, int(round(clip_dur * fps)))

            if idx % 2 == 0:
                z_expr = "min(zoom+0.0007,1.12)"
                x_expr = "iw/2-(iw/zoom/2)"
                y_expr = "ih/2-(ih/zoom/2)"
            else:
                z_expr = "if(eq(on,1),1.12,max(1.0,zoom-0.0007))"
                x_expr = "iw/2-(iw/zoom/2)"
                y_expr = "ih/2-(ih/zoom/2)"

            scene_filter = (
                f"[{idx}:v]scale={w}:{h}:force_original_aspect_ratio=increase,"
                f"crop={w}:{h},setsar=1,"
                f"zoompan=z='{z_expr}':x='{x_expr}':y='{y_expr}':d={frames}:s={w}x{h}:fps={fps},"
                f"trim=duration={clip_dur:.3f},setpts=PTS-STARTPTS,"
                f"format=yuv420p[v{idx}]"
            )
            filter_parts.append(scene_filter)

        # 2. Chain scenes with crossfade transitions
        if num_scenes == 1:
            current_v_label = "[v0]"
        else:
            current_v_label = "[v0]"
            cumulative_offset = 0.0
            for idx in range(1, num_scenes):
                cumulative_offset += slot_durations[idx - 1]
                out_label = f"[vx{idx}]"
                filter_parts.append(
                    f"{current_v_label}[v{idx}]xfade=transition=fade:"
                    f"duration={xfade_dur:.3f}:offset={cumulative_offset:.3f}{out_label}"
                )
                current_v_label = out_label

        # 3. Global visual polish: subtle dark overlay, fade-in/out, opening title, and burned subtitles
        fade_out_start = max(0.0, total_duration_seconds - 1.0)
        escaped_sub_path = escape_ffmpeg_filter_path(subtitle_file)
        escaped_fonts_dir = escape_ffmpeg_filter_path(self.config.font_directory)
        safe_title = escape_ffmpeg_drawtext(story_title)
        title_end = min(self.config.title_overlay_seconds, max(1.5, total_duration_seconds * 0.25))
        overlay_alpha = max(0.05, min(0.65, self.config.dark_overlay_opacity))

        font_file_clause = ""
        bundled_font = self.config.font_directory / "Arial.ttf"
        if bundled_font.exists():
            font_file_clause = f"fontfile='{escape_ffmpeg_filter_path(bundled_font)}':"

        if subtitle_file.suffix.lower() == ".ass":
            sub_filter = f"ass='{escaped_sub_path}':fontsdir='{escaped_fonts_dir}'"
        else:
            style_str = (
                f"FontName={self.config.subtitle_font_name},"
                f"FontSize={self.config.subtitle_font_size},"
                "PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BackColour=&H80000000,"
                "Bold=1,BorderStyle=1,Outline=2,Shadow=1,Alignment=2,"
                f"MarginV={self.config.subtitle_margin_v},MarginL=35,MarginR=35"
            )
            sub_filter = (
                f"subtitles='{escaped_sub_path}':fontsdir='{escaped_fonts_dir}':"
                f"force_style='{style_str}'"
            )

        post_v_chain = (
            f"{current_v_label}"
            f"drawbox=x=0:y=0:w={w}:h={h}:color=black@{overlay_alpha:.2f}:t=fill,"
            f"drawtext={font_file_clause}text='{safe_title}':fontcolor=white:fontsize=44:"
            f"box=1:boxcolor=black@0.55:boxborderw=18:x=(w-text_w)/2:y=180:"
            f"enable='between(t,0.2,{title_end:.2f})',"
            f"{sub_filter},"
            f"fade=t=in:st=0:d=0.6,"
            f"fade=t=out:st={fade_out_start:.3f}:d=1.0,"
            f"format=yuv420p[vout]"
        )
        filter_parts.append(post_v_chain)

        # 4. Audio processing (narration + optional ducked & faded background music + anti-clipping limiter)
        if music_input_idx is not None:
            m_vol = max(0.02, min(0.45, self.config.music_volume))
            m_fade_in = min(self.config.music_fade_in_seconds, total_duration_seconds * 0.25)
            m_fade_out = min(self.config.music_fade_out_seconds, total_duration_seconds * 0.3)
            m_fade_out_start = max(0.0, total_duration_seconds - m_fade_out)

            if self.config.enable_audio_ducking:
                audio_chain = (
                    f"[{narration_input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                    f"asplit=2[narr_main][narr_sc];"
                    f"[{music_input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                    f"atrim=0:{total_duration_seconds:.3f},asetpts=PTS-STARTPTS,"
                    f"volume={m_vol:.3f},"
                    f"afade=t=in:st=0:d={m_fade_in:.2f},"
                    f"afade=t=out:st={m_fade_out_start:.3f}:d={m_fade_out:.2f}[music_pre];"
                    f"[music_pre][narr_sc]sidechaincompress=threshold=0.08:ratio=3:attack=50:release=700[music_ducked];"
                    f"[narr_main][music_ducked]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
                    f"alimiter=limit=0.95[aout]"
                )
            else:
                audio_chain = (
                    f"[{narration_input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo[narr_main];"
                    f"[{music_input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                    f"atrim=0:{total_duration_seconds:.3f},asetpts=PTS-STARTPTS,"
                    f"volume={m_vol:.3f},"
                    f"afade=t=in:st=0:d={m_fade_in:.2f},"
                    f"afade=t=out:st={m_fade_out_start:.3f}:d={m_fade_out:.2f}[music_faded];"
                    f"[narr_main][music_faded]amix=inputs=2:duration=first:dropout_transition=2:normalize=0,"
                    f"alimiter=limit=0.95[aout]"
                )
        else:
            audio_chain = (
                f"[{narration_input_idx}:a]aformat=sample_fmts=fltp:sample_rates=44100:channel_layouts=stereo,"
                f"alimiter=limit=0.95[aout]"
            )

        filter_parts.append(audio_chain)
        filter_complex = ";".join(filter_parts)

        cmd.extend(
            [
                "-filter_complex",
                filter_complex,
                "-map",
                "[vout]",
                "-map",
                "[aout]",
                "-c:v",
                "libx264",
                "-preset",
                self.config.ffmpeg_preset,
                "-crf",
                str(self.config.ffmpeg_crf),
                "-profile:v",
                "high",
                "-pix_fmt",
                "yuv420p",
                "-r",
                str(fps),
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-ar",
                "44100",
                "-t",
                f"{total_duration_seconds:.3f}",
                "-movflags",
                "+faststart",
                str(output_mp4.resolve()),
            ]
        )
        return cmd

    def _prepare_fontconfig_env(self, temp_dir: Path) -> Dict[str, str]:
        """Create a self-contained fontconfig XML in temp_dir so libass/drawtext work on any OS."""
        fc_cache = temp_dir / "fccache"
        fc_cache.mkdir(parents=True, exist_ok=True)
        font_dir_posix = self.config.font_directory.resolve().as_posix()
        cache_dir_posix = fc_cache.resolve().as_posix()

        fonts_conf = temp_dir / "fonts.conf"
        fonts_conf.write_text(
            f'<?xml version="1.0"?>\n'
            f'<!DOCTYPE fontconfig SYSTEM "fonts.dtd">\n'
            f"<fontconfig>\n"
            f"  <dir>{font_dir_posix}</dir>\n"
            f"  <dir>/usr/share/fonts</dir>\n"
            f"  <cachedir>{cache_dir_posix}</cachedir>\n"
            f"</fontconfig>\n",
            encoding="utf-8",
        )

        env = os.environ.copy()
        env["FONTCONFIG_FILE"] = str(fonts_conf.resolve())
        env["FONTCONFIG_PATH"] = str(temp_dir.resolve())
        return env

    def render(
        self,
        images: List[ImageMetadata],
        audio_segments: List[AudioSegmentResult],
        master_narration_wav: Path,
        subtitle_file: Path,
        output_mp4: Path,
        story_title: str,
        total_duration_seconds: float,
        music_track: Optional[MusicTrackMetadata] = None,
        inter_scene_pauses: Optional[List[float]] = None,
        inter_scene_pause_seconds: float = 0.55,
        diagnostics_dir: Optional[Path] = None,
    ) -> RenderResult:
        """Execute the FFmpeg render pipeline using a temporary workspace and clean up on success."""
        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        temp_dir = Path(tempfile.mkdtemp(prefix="auto_video_render_"))

        try:
            local_sub = temp_dir / f"subs{subtitle_file.suffix.lower() or '.srt'}"
            shutil.copy2(subtitle_file, local_sub)
            env = self._prepare_fontconfig_env(temp_dir)

            cmd = self.build_ffmpeg_command(
                images=images,
                audio_segments=audio_segments,
                master_narration_wav=master_narration_wav,
                subtitle_file=local_sub,
                output_mp4=output_mp4,
                story_title=story_title,
                total_duration_seconds=total_duration_seconds,
                music_track=music_track,
                inter_scene_pauses=inter_scene_pauses,
                inter_scene_pause_seconds=inter_scene_pause_seconds,
            )

            logger.info(
                "Starting FFmpeg render (%dx%d @ %dfps, %.2fs, scenes=%d, music=%s)...",
                self.config.video_width,
                self.config.video_height,
                self.config.video_fps,
                total_duration_seconds,
                len(images),
                bool(music_track),
            )

            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                timeout=600,
                check=False,
            )

            if proc.returncode != 0 or not output_mp4.exists() or output_mp4.stat().st_size < 1024:
                stderr_text = proc.stderr.decode("utf-8", errors="replace")
                diag_folder = diagnostics_dir or (output_mp4.parent / "diagnostics")
                diag_folder.mkdir(parents=True, exist_ok=True)
                diag_log = diag_folder / f"{output_mp4.stem}_ffmpeg_error.log"
                diag_log.write_text(
                    f"Command:\n{' '.join(cmd)}\n\nReturn Code: {proc.returncode}\n\nSTDERR:\n{stderr_text}",
                    encoding="utf-8",
                )
                logger.error(
                    "FFmpeg render failed (exit code %d). Diagnostics saved to %s",
                    proc.returncode,
                    diag_log,
                )
                raise VideoRenderError(
                    f"FFmpeg video rendering failed with exit code {proc.returncode}. "
                    f"See diagnostic log at {diag_log}"
                )

            logger.info("Successfully rendered MP4 video: %s", output_mp4)
            return RenderResult(
                output_video_path=str(output_mp4.resolve()),
                subtitle_path=str(subtitle_file.resolve()),
                target_duration_seconds=total_duration_seconds,
                scene_count=len(images),
                music_used=music_track is not None,
                ffmpeg_command=cmd,
            )

        finally:
            if not self.config.keep_temp_files and temp_dir.exists():
                shutil.rmtree(temp_dir, ignore_errors=True)
