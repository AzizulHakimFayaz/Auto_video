"""Bootstrap utility to initialize the local asset library with CC0 Public Domain visuals and ambient tracks.

Generates high-resolution (1080x1920) procedural atmospheric illustrations across all
eight mood folders (`sadness`, `love`, `family`, `hope`, `tragedy`, `peace`, `nature`, `city`)
and synthesizes five gentle ambient CC0 instrumental tracks (`sad_piano_ambient.wav`,
`hopeful_dawn_chords.wav`, `romantic_warm_pad.wav`, `suspense_atmosphere.wav`,
`peaceful_breeze_ambient.wav`) along with complete `images_manifest.json` and
`music_manifest.json` license records in `assets/licenses/`.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path
import shutil
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw, ImageFilter

from src.config import PROJECT_ROOT

THEME_SPECS: Dict[str, List[Tuple[str, Tuple[int, int, int], Tuple[int, int, int], List[str]]]] = {
    "sadness": [
        (
            "rainy_platform_evening_umbrella.jpg",
            (18, 28, 48),
            (52, 72, 98),
            ["sadness", "rainy", "train", "station", "platform", "evening", "lonely", "umbrella", "window"],
        ),
        (
            "melancholic_window_rain_dusk.jpg",
            (22, 30, 44),
            (68, 82, 105),
            ["sadness", "melancholic", "rain", "window", "reflection", "dusk", "tears", "conversation"],
        ),
    ],
    "love": [
        (
            "warm_lantern_kindness_light.jpg",
            (64, 28, 38),
            (185, 110, 75),
            ["love", "sharing", "yellow", "umbrella", "kindness", "warm", "light", "generosity"],
        ),
        (
            "golden_embrace_sunset_glow.jpg",
            (75, 35, 42),
            (210, 135, 88),
            ["love", "romantic", "warmth", "compassion", "sunset", "heartfelt"],
        ),
    ],
    "family": [
        (
            "nostalgic_home_hearth_memory.jpg",
            (48, 34, 26),
            (145, 105, 72),
            ["family", "winter", "train", "tracks", "nostalgic", "memory", "warmth", "home"],
        ),
        (
            "generations_walk_autumn_path.jpg",
            (55, 38, 24),
            (168, 118, 68),
            ["family", "reunion", "parents", "children", "autumn", "together"],
        ),
    ],
    "hope": [
        (
            "sunrise_horizon_golden_rays.jpg",
            (28, 45, 72),
            (235, 175, 85),
            ["hope", "train", "headlights", "glowing", "rain", "hopeful", "sunrise", "dawn"],
        ),
        (
            "morning_light_breaking_clouds.jpg",
            (34, 58, 88),
            (242, 195, 110),
            ["hope", "uplifting", "light", "clouds", "future", "inspiring"],
        ),
    ],
    "tragedy": [
        (
            "stormy_night_bench_downpour.jpg",
            (12, 16, 26),
            (42, 48, 64),
            ["tragedy", "heavy", "rain", "bench", "city", "night", "sad", "wet", "storm"],
        ),
        (
            "shadowed_corridor_silent_loss.jpg",
            (16, 18, 24),
            (52, 54, 66),
            ["tragedy", "loss", "grief", "empty", "dark", "stormy"],
        ),
    ],
    "peace": [
        (
            "still_lake_mist_morning.jpg",
            (24, 52, 62),
            (110, 162, 168),
            ["peace", "calm", "serene", "lake", "mist", "quiet", "gentle"],
        ),
        (
            "starlit_meadow_quiet_sky.jpg",
            (18, 36, 54),
            (85, 128, 152),
            ["peace", "night", "stars", "meadow", "harmony", "rest"],
        ),
    ],
    "nature": [
        (
            "misty_pine_forest_sunbeams.jpg",
            (18, 44, 34),
            (78, 132, 96),
            ["nature", "forest", "trees", "rain", "woods", "green", "leaves"],
        ),
        (
            "ocean_waves_twilight_shore.jpg",
            (16, 38, 58),
            (72, 124, 156),
            ["nature", "ocean", "sea", "shore", "horizon", "breeze"],
        ),
    ],
    "city": [
        (
            "evening_streetlights_bokeh_rain.jpg",
            (20, 24, 38),
            (88, 96, 128),
            ["city", "street", "urban", "lights", "evening", "station", "commuter"],
        ),
        (
            "quiet_cafe_corner_twilight.jpg",
            (28, 26, 36),
            (112, 92, 82),
            ["city", "window", "bench", "architecture", "buildings", "road"],
        ),
    ],
}

MUSIC_SPECS: List[Tuple[str, str, List[float], List[str]]] = [
    (
        "sad_piano_ambient.wav",
        "Quiet Rain Reflections",
        [220.0, 261.63, 329.63, 392.0],  # A minor7 gentle chord
        ["sad", "sadness", "melancholic", "melancholy", "lonely", "piano"],
    ),
    (
        "hopeful_dawn_chords.wav",
        "First Light of Tomorrow",
        [261.63, 329.63, 392.0, 493.88],  # C major7 uplifting chord
        ["hopeful", "hope", "uplifting", "inspiring", "sunrise"],
    ),
    (
        "romantic_warm_pad.wav",
        "Warmth Under the Shelter",
        [174.61, 220.0, 261.63, 349.23],  # F major warm voicing
        ["romantic", "love", "warm", "kindness", "family", "heartfelt"],
    ),
    (
        "suspense_atmosphere.wav",
        "Distant Thunder Horizon",
        [146.83, 174.61, 220.0, 277.18],  # D minor atmospheric tension
        ["suspenseful", "suspense", "tense", "tragedy", "atmospheric"],
    ),
    (
        "peaceful_breeze_ambient.wav",
        "Still Water Sanctuary",
        [196.0, 246.94, 293.66, 392.0],  # G major open ambient voicing
        ["peaceful", "peace", "calm", "ambient", "gentle", "nature"],
    ),
]


def _render_atmospheric_image(
    dest_path: Path,
    top_rgb: Tuple[int, int, int],
    bottom_rgb: Tuple[int, int, int],
    seed_offset: int = 0,
    width: int = 1080,
    height: int = 1920,
) -> None:
    """Create a rich 1080x1920 atmospheric vertical illustration with soft bokeh and horizon glow."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (width, height))
    draw = ImageDraw.Draw(img)

    for y in range(height):
        t = y / float(height - 1)
        # Smooth S-curve interpolation
        s = t * t * (3.0 - 2.0 * t)
        r = int(top_rgb[0] + (bottom_rgb[0] - top_rgb[0]) * s)
        g = int(top_rgb[1] + (bottom_rgb[1] - top_rgb[1]) * s)
        b = int(top_rgb[2] + (bottom_rgb[2] - top_rgb[2]) * s)
        draw.line([(0, y), (width, y)], fill=(r, g, b))

    # Add soft glowing atmospheric light orbs / bokeh and silhouette horizon
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    ov_draw = ImageDraw.Draw(overlay)

    for i in range(14):
        cx = ((i * 197 + seed_offset * 83) % (width - 160)) + 80
        cy = ((i * 263 + seed_offset * 131) % (height - 300)) + 150
        radius = 65 + ((i * 37) % 130)
        glow_r = min(255, bottom_rgb[0] + 45)
        glow_g = min(255, bottom_rgb[1] + 45)
        glow_b = min(255, bottom_rgb[2] + 45)
        alpha = 26 + (i % 4) * 10
        ov_draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            fill=(glow_r, glow_g, glow_b, alpha),
        )

    # Lower horizon silhouette for cinematic depth
    horizon_y = int(height * 0.78)
    ov_draw.polygon(
        [
            (0, height),
            (0, horizon_y + 40),
            (int(width * 0.35), horizon_y - 25),
            (int(width * 0.7), horizon_y + 15),
            (width, horizon_y - 35),
            (width, height),
        ],
        fill=(max(0, top_rgb[0] - 12), max(0, top_rgb[1] - 12), max(0, top_rgb[2] - 12), 185),
    )

    blurred_overlay = overlay.filter(ImageFilter.GaussianBlur(radius=18))
    combined = Image.alpha_composite(img.convert("RGBA"), blurred_overlay).convert("RGB")
    combined.save(dest_path, format="JPEG", quality=92)


def _synthesize_ambient_track(
    dest_path: Path,
    frequencies: List[float],
    duration_seconds: float = 16.0,
    sample_rate: int = 22050,
) -> None:
    """Synthesize a gentle, loopable harmonic ambient pad WAV file."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = int(round(duration_seconds * sample_rate))
    buf = bytearray(total_samples * 2)

    for i in range(total_samples):
        t = i / float(sample_rate)
        env = min(1.0, t / 1.5, (duration_seconds - t) / 1.5)
        lfo = 0.82 + 0.18 * math.sin(2.0 * math.pi * 0.25 * t)
        val = 0.0
        for idx, freq in enumerate(frequencies):
            detune = 1.0 + (idx - 1.5) * 0.0008
            val += math.sin(2.0 * math.pi * freq * detune * t) / len(frequencies)
        sample_int = int(max(-32767, min(32767, val * env * lfo * 6500.0)))
        struct.pack_into("<h", buf, i * 2, sample_int)

    with wave.open(str(dest_path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(bytes(buf))


def bootstrap_local_assets(root_dir: Path = PROJECT_ROOT) -> Dict[str, Any]:
    """Ensure all required asset directories, CC0 fallback images, music tracks, and license manifests exist."""
    bg_dir = root_dir / "assets" / "backgrounds"
    music_dir = root_dir / "assets" / "music"
    font_dir = root_dir / "assets" / "fonts"
    license_dir = root_dir / "assets" / "licenses"

    bg_dir.mkdir(parents=True, exist_ok=True)
    music_dir.mkdir(parents=True, exist_ok=True)
    font_dir.mkdir(parents=True, exist_ok=True)
    license_dir.mkdir(parents=True, exist_ok=True)

    images_manifest: Dict[str, Dict[str, Any]] = {}
    created_images = 0

    for theme_idx, (theme, specs) in enumerate(THEME_SPECS.items()):
        theme_folder = bg_dir / theme
        theme_folder.mkdir(parents=True, exist_ok=True)
        for item_idx, (filename, top_c, bot_c, tags) in enumerate(specs):
            img_path = theme_folder / filename
            if not img_path.exists():
                _render_atmospheric_image(
                    img_path, top_c, bot_c, seed_offset=theme_idx * 10 + item_idx
                )
                created_images += 1
            rel_key = f"{theme}/{filename}"
            images_manifest[rel_key] = {
                "creator": "Auto Story Video Procedural Studio",
                "source_url": f"local://assets/backgrounds/{rel_key}",
                "license": "CC0 1.0 Universal (Public Domain Dedication)",
                "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                "attribution": f"Procedural artwork '{filename}' dedicated to the Public Domain (CC0 1.0).",
                "tags": tags,
            }

    (license_dir / "images_manifest.json").write_text(
        json.dumps(images_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    music_manifest_tracks: Dict[str, Dict[str, Any]] = {}
    created_tracks = 0

    for filename, title, freqs, mood_tags in MUSIC_SPECS:
        track_path = music_dir / filename
        if not track_path.exists():
            _synthesize_ambient_track(track_path, freqs)
            created_tracks += 1
        music_manifest_tracks[filename] = {
            "title": title,
            "creator": "Auto Story Video Acoustic Synth",
            "source_url": f"local://assets/music/{filename}",
            "license_name": "CC0 1.0 Universal (Public Domain Dedication)",
            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
            "attribution_instructions": f"'{title}' synthesized locally under CC0 1.0 Public Domain.",
            "commercial_use_allowed": True,
            "mood_tags": mood_tags,
            "restrictions": "None (CC0 Public Domain)",
        }

    (license_dir / "music_manifest.json").write_text(
        json.dumps({"version": 1, "tracks": music_manifest_tracks}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # Ensure a fallback TrueType font is available in assets/fonts for subtitle/title rendering
    target_font = font_dir / "Arial.ttf"
    if not target_font.exists():
        system_font_candidates = [
            Path("C:/Windows/Fonts/arialbd.ttf"),
            Path("C:/Windows/Fonts/arial.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
            Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
            Path("/usr/share/fonts/TTF/DejaVuSans.ttf"),
        ]
        for candidate in system_font_candidates:
            if candidate.exists() and candidate.is_file():
                try:
                    shutil.copy2(candidate, target_font)
                    break
                except OSError:
                    pass

    # Create a README inside assets/fonts and assets/licenses explaining usage
    (font_dir / "README.md").write_text(
        "# Custom Fonts\n\nPlace `.ttf` or `.otf` font files here (e.g., `DejaVuSans-Bold.ttf` or `Montserrat-Bold.ttf`) "
        "and set `SUBTITLE_FONT_NAME` in `.env`.\n",
        encoding="utf-8",
    )

    return {
        "images_total": len(images_manifest),
        "images_created": created_images,
        "tracks_total": len(music_manifest_tracks),
        "tracks_created": created_tracks,
    }


if __name__ == "__main__":
    summary = bootstrap_local_assets()
    print(f"Bootstrapped local assets: {summary}")
