# Free Daily AI Story Video Automation System

A production-oriented, zero-budget Python automation system that generates one original emotional short story video (`1080×1920`, `9:16`, `30 FPS`, `H.264/AAC` MP4) every day and publishes it to **YouTube Shorts**, **Instagram Reels**, **Facebook Reels**, and **TikTok** via official platform APIs.

---

## 1. Key Capabilities & Zero-Budget Safeguards

- **AI Story Generation (`src/story_generator.py`)**: Uses OpenRouter's free model endpoint (`nvidia/nemotron-3-ultra-550b-a55b:free` by default) via an OpenAI-compatible client. Enforces a hard zero-budget policy: any model identifier not ending in `:free` is rejected at startup.
- **5-Stage Scene Image Selection (`src/image_selector.py`)**:
  1. Local library tagged match (`assets/backgrounds/<theme>/`).
  2. Official **Pixabay API** search (when `PIXABAY_API_KEY` is configured).
  3. Official **Openverse API** search with Creative Commons license verification (`CC0`, `PDM`, `CC BY`, `CC BY-SA`).
  4. Local mood-based fallback image library.
  5. Clean pipeline stop if no valid image exists.
- **Local Neural Narration (`src/voice_generator.py`)**: Synthesizes per-scene WAV audio locally using **Piper TTS** (`en_US-ljspeech-medium` Public Domain voice by default), verifying `.onnx` and `.onnx.json` model compatibility and license terms.
- **Licensed Emotional Music (`src/music_selector.py`)**: Selects mood-matched soundtracks (`sad`, `hopeful`, `romantic`, `suspenseful`, `peaceful`) from `assets/music/` only when verified in `assets/licenses/music_manifest.json`. Never assumes an unlisted audio file is copyright-free. Automatically falls back to narration-only audio if no licensed track is present.
- **Frame-Accurate Subtitles (`src/subtitle_generator.py`)**: Computes mobile-friendly phrase timings from actual measured WAV durations per scene (never guessing solely from word counts), generating both `.srt` and `.ass` files and burning subtitles into the video.
- **Cinematic Vertical Rendering (`src/video_renderer.py`)**: Uses **FFmpeg** to render `1080×1920` (`9:16`) at `30 FPS` (`H.264` + `AAC`) with aspect-ratio-preserving scale/crop, Ken Burns slow zoom/pan (`zoompan`), crossfades (`xfade`), a dark readability overlay, opening story title card, burned-in subtitles, and sidechain audio ducking (`sidechaincompress` + `alimiter`).
- **FFprobe Verification (`src/video_validator.py`)**: Validates container, codecs (`h264`/`aac`), resolution (`1080×1920`), frame rate (`30 FPS`), and duration alignment before any upload attempt.
- **Official Platform Publishing (`src/publishers/`)**: Independent adapters for YouTube Data API v3, Instagram Graph API, Facebook Page Reels API, and TikTok Content Posting API v2. Defaults to `DRY_RUN=true`.
- **Daily GitHub Actions Automation (`.github/workflows/daily.yml`)**: Runs daily at **09:00 AM Bangladesh Time (`03:00 UTC`)** on `ubuntu-latest` and preserves the generated MP4, subtitles, metadata, and logs as workflow artifacts.

---

## 2. Project Structure

```text
.
├── .github/
│   └── workflows/
│       └── daily.yml                  # Daily 03:00 UTC (09:00 AM BST) GitHub Actions workflow
├── assets/
│   ├── backgrounds/                   # Mood-organized local image library (CC0 bootstrapped)
│   │   ├── sadness/
│   │   ├── love/
│   │   ├── family/
│   │   ├── hope/
│   │   ├── tragedy/
│   │   ├── peace/
│   │   ├── nature/
│   │   └── city/
│   ├── music/                         # Local licensed background music tracks
│   ├── fonts/                         # TrueType fonts and fontconfig settings
│   └── licenses/                      # images_manifest.json & music_manifest.json
├── config/
│   ├── __init__.py
│   └── settings.py                    # Re-exports canonical configuration from src/config.py
├── data/
│   ├── history.json                   # Persistent run history & duplicate prevention state
│   └── sample_story.json              # Example validated story JSON payload
├── src/
│   ├── __init__.py
│   ├── main.py                        # 16-stage daily pipeline orchestrator & CLI
│   ├── config.py                      # Single configuration module & secret-redacting log filter
│   ├── asset_bootstrap.py             # Generates CC0 1080x1920 fallback visuals & ambient tracks
│   ├── story_generator.py             # OpenRouter free-tier LLM client & schema validator
│   ├── image_selector.py              # Local, Pixabay, and Openverse image selection
│   ├── voice_generator.py             # Local Piper TTS wrapper & WAV validator
│   ├── music_selector.py              # Mood-based music selector & license verifier
│   ├── subtitle_generator.py          # Audio-synced SRT and ASS subtitle generator
│   ├── video_renderer.py              # FFmpeg Ken Burns + xfade + ducking renderer
│   ├── video_validator.py             # FFprobe stream & duration verifier
│   ├── metadata_generator.py          # Platform-specific titles, captions, tags & credits
│   ├── history_manager.py             # Story similarity & per-platform upload state manager
│   └── publishers/
│       ├── __init__.py
│       ├── base.py                    # Dry-run, credential, and retry safeguards
│       ├── youtube.py                 # Official YouTube Data API v3 resumable uploader
│       ├── instagram.py               # Official Meta Instagram Graph API Reels publisher
│       ├── facebook.py                # Official Meta Facebook Page Reels 3-step publisher
│       └── tiktok.py                  # Official TikTok Content Posting API v2 publisher
├── tests/
│   ├── test_story_generator.py
│   ├── test_image_selector.py
│   ├── test_music_selector.py
│   ├── test_subtitles.py
│   ├── test_video_renderer.py
│   ├── test_history.py
│   └── test_publishers.py
├── output/                            # Dated production outputs (<YYYY-MM-DD>/*.mp4, *.srt, *.json)
├── .env.example                       # Documented configuration template
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 3. Prerequisites & Local Installation

### System Requirements
1. **Python 3.11+** (tested on Python 3.12 and 3.13).
2. **FFmpeg & FFprobe** installed and available on `PATH` (must include `libx264`, `libass`, `libfreetype`).
   - **Ubuntu/Debian**: `sudo apt-get update && sudo apt-get install -y ffmpeg espeak-ng libsndfile1`
   - **Windows (winget/choco)**: `winget install Gyan.FFmpeg`

### Python Setup
```bash
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Bootstrap Local Fallback Assets
Generate the initial 16 high-resolution (`1080×1920`) CC0 atmospheric background images, 5 CC0 ambient music tracks, and license manifests:
```bash
python -m src.asset_bootstrap
```

### Configure Environment Variables
```bash
cp .env.example .env
```
Edit `.env` to add your `OPENROUTER_API_KEY` and any optional provider keys.

---

## 4. Configuring Free Services & Licensing Rules

### 4.1 OpenRouter Free Story Generation
1. Create a free account at [https://openrouter.ai](https://openrouter.ai) and generate an API key at [https://openrouter.ai/keys](https://openrouter.ai/keys).
2. Set in `.env`:
   ```ini
   OPENROUTER_API_KEY=sk-or-v1-...
   OPENROUTER_MODEL=nvidia/nemotron-3-ultra-550b-a55b:free
   ```
3. **Free-Tier Quotas & Safeguards**:
   - OpenRouter `:free` models cost `$0.00` per prompt/completion token, but are subject to dynamic rate limits (typically ~20 requests/minute and 50–200 requests/day depending on account tier) and upstream provider capacity.
   - If the primary free model is temporarily unavailable, you can set a secondary free model in `OPENROUTER_FALLBACK_MODEL` (e.g., `meta-llama/llama-3.3-70b-instruct:free`).
   - The application strictly validates that both `OPENROUTER_MODEL` and `OPENROUTER_FALLBACK_MODEL` end with `:free` and **never** falls back to paid endpoints.

### 4.2 Piper Local Text-to-Speech Voice Models & Licensing
Download a verified voice model (`.onnx` and `.onnx.json`) from the official [`rhasspy/piper-voices`](https://huggingface.co/rhasspy/piper-voices) repository on Hugging Face:

```bash
mkdir -p models/piper
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx" -o models/piper/en_US-ljspeech-medium.onnx
curl -L "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ljspeech/medium/en_US-ljspeech-medium.onnx.json" -o models/piper/en_US-ljspeech-medium.onnx.json
```

**Important Piper Voice Licensing Comparison**:
| Voice Model ID | Training Dataset | License | Commercial / Monetized Use Permitted? |
| :--- | :--- | :--- | :--- |
| **`en_US-ljspeech-medium`** *(Default)* | LJ Speech Dataset | **Public Domain** | **Yes** (Unencumbered) |
| **`en_US-libritts_r-medium`** | LibriTTS-R | **CC BY 4.0** | **Yes** (With attribution in description) |
| `en_US-lessac-medium` | Blizzard Challenge 2013 | Non-Commercial Research | **No** (Research/non-commercial only) |
| `en_US-hfc_female-medium` | Hi-Fi CAPTAIN | CC BY-NC-SA 4.0 | **No** (Non-commercial only) |

To narrate in another language, set `STORY_LANGUAGE` in `.env` and configure `PIPER_VOICE`, `PIPER_MODEL_PATH`, and `PIPER_CONFIG_PATH` to point to a compatible `.onnx` voice model for that language after checking its `MODEL_CARD` license.

### 4.3 Image Providers & Attribution
- **Local Library (`assets/backgrounds/<theme>/`)**: Used first when keywords match, and as Stage 4 fallback when offline.
- **Pixabay API**: Create a free account at [https://pixabay.com/api/docs/](https://pixabay.com/api/docs/) and set `PIXABAY_API_KEY`. Rate limit is 100 requests/60s. Per Pixabay API terms, downloaded images are cached locally rather than hotlinked.
- **Openverse API**: Set `OPENVERSE_ENABLED=true`. Only queries permissive licenses (`cc0`, `pdm`, `by`, `by-sa`) and automatically records creator and license URLs into `assets/licenses/selected_images_attribution.json` and the video description credits.

### 4.4 Adding Licensed Background Music
Never download random copyrighted tracks from YouTube or streaming sites.
To add your own legally licensed tracks (e.g., CC0 or CC BY 4.0 tracks):
1. Copy the audio file (`.wav`, `.mp3`, `.ogg`) into `assets/music/`.
2. Add an explicit license record in `assets/licenses/music_manifest.json` under `"tracks"`:
   ```json
   "my_track.mp3": {
     "title": "Gentle Piano Dawn",
     "creator": "Artist Name",
     "source_url": "https://example.org/track",
     "license_name": "CC BY 4.0",
     "license_url": "https://creativecommons.org/licenses/by/4.0/",
     "attribution_instructions": "\"Gentle Piano Dawn\" by Artist Name (CC BY 4.0)",
     "commercial_use_allowed": true,
     "mood_tags": ["sad", "piano", "melancholic"],
     "restrictions": "Requires attribution in video description"
   }
   ```
Any audio file in `assets/music/` without a complete manifest entry or `.json` sidecar is automatically skipped.

---

## 5. Setting Up Official Platform Publishing APIs

By default, `DRY_RUN=true` and all `ENABLE_*` flags are `false`. Each platform uses its official API only and has specific eligibility and review requirements:

### 5.1 YouTube Shorts (YouTube Data API v3)
1. In the [Google Cloud Console](https://console.cloud.google.com/), create a project and enable **YouTube Data API v3**.
2. Configure the OAuth Consent Screen (add your YouTube account as a Test User if the app is in Testing mode) and create **OAuth 2.0 Client ID** credentials (`Desktop app` or `Web application`).
3. Exchange an authorization code with scope `https://www.googleapis.com/auth/youtube.upload` for a `refresh_token`.
4. Set `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, and `YOUTUBE_REFRESH_TOKEN`.
5. **Important Restriction**: Unaudited Google Cloud projects created after July 2020 will have uploaded videos locked to `private` (`private` viewing restriction) until the project passes Google's API compliance audit. Default `YOUTUBE_PRIVACY_STATUS=private` works immediately without audit.

### 5.2 Instagram Reels (Meta Instagram Graph API)
1. Requires an **Instagram Professional Account** (Business or Creator) linked to a Facebook Page. Personal Instagram accounts are not supported by the official Content Publishing API.
2. Create a Meta App in the [Meta App Dashboard](https://developers.facebook.com/) and grant `instagram_business_basic` and `instagram_business_content_publish` permissions.
3. Because Meta's Reels container endpoint (`POST /{ig_user_id}/media`) fetches the video from a public HTTPS URL (`video_url`), set `INSTAGRAM_VIDEO_PUBLIC_URL_BASE` to the public HTTPS base URL where your video is hosted, alongside `INSTAGRAM_ACCOUNT_ID` and `INSTAGRAM_ACCESS_TOKEN`.

### 5.3 Facebook Reels (Meta Page Reels Graph API)
1. Requires a **Facebook Page** and a Page Access Token with `pages_manage_posts`, `pages_read_engagement`, and `pages_show_list` permissions.
2. Uses the 3-step binary upload protocol (`upload_phase=start` -> binary POST to `rupload.facebook.com` -> `upload_phase=finish`), so no external hosting URL is required.
3. Set `FACEBOOK_PAGE_ID` and `FACEBOOK_PAGE_ACCESS_TOKEN`.

### 5.4 TikTok (Official Content Posting API v2)
1. Register an app at the [TikTok Developer Portal](https://developers.tiktok.com/), enable the **Content Posting API** product, and request the `video.publish` scope.
2. Obtain a user OAuth2 `access_token` and set `TIKTOK_ACCESS_TOKEN`.
3. **Important Restriction**: Until your TikTok developer application passes TikTok's official **Content Posting API Audit**, TikTok restricts all direct posts to private viewing mode (`TIKTOK_PRIVACY_LEVEL=SELF_ONLY`). Our adapter queries `/v2/post/publish/creator_info/query/` first and automatically enforces `SELF_ONLY` when public posting is not yet permitted for the account.

---

## 6. Local Dry-Run Demonstration & Verification Commands

### 6.1 Run Automated Unit & Integration Tests
```bash
python -m pytest -v
```

### 6.2 Run a Local End-to-End Dry-Run Video Generation
Using the included sample story (`data/sample_story.json`) without consuming any API quota:
```bash
# PowerShell:
$env:ALLOW_OFFLINE_TTS_FALLBACK="true"
$env:DRY_RUN="true"
python -m src.main --date 2026-10-26 --story-file data/sample_story.json --force-regenerate
```
Or with live OpenRouter story generation and local Piper TTS:
```bash
python -m src.main --date 2026-10-26
```

### 6.3 Inspect the Rendered MP4 with FFprobe
```bash
ffprobe -v error -show_format -show_streams -of json output/2026-10-26/*.mp4
```

### 6.4 Retry Failed Platform Publishing Only (Without Regenerating Story/Video)
If video generation succeeded on `2026-10-26` but one platform failed due to a temporary token or network issue:
```bash
python -m src.main --date 2026-10-26 --retry-publish
```
Platforms already recorded as `"published"` in `data/history.json` for `2026-10-26` are automatically skipped to prevent duplicate uploads.

---

## 7. Enabling Daily GitHub Actions Automation

1. Push this repository to GitHub.
2. Go to **Settings → Secrets and variables → Actions**.
3. Under the **Secrets** tab, add:
   - `OPENROUTER_API_KEY` (Required)
   - `PIXABAY_API_KEY` (Optional)
   - `YOUTUBE_CLIENT_ID`, `YOUTUBE_CLIENT_SECRET`, `YOUTUBE_REFRESH_TOKEN` (Optional)
   - `INSTAGRAM_ACCOUNT_ID`, `INSTAGRAM_ACCESS_TOKEN` (Optional)
   - `FACEBOOK_PAGE_ID`, `FACEBOOK_PAGE_ACCESS_TOKEN` (Optional)
   - `TIKTOK_ACCESS_TOKEN` (Optional)
4. Under the **Variables** tab, configure:
   - `DRY_RUN` (`true` for artifact-only generation, `false` to enable live publishing)
   - `ENABLE_YOUTUBE`, `ENABLE_INSTAGRAM`, `ENABLE_FACEBOOK`, `ENABLE_TIKTOK` (`true` or `false`)
5. Open the **Actions** tab, select **Daily Emotional Story Video Pipeline**, and click **Run workflow** to trigger a manual test run (`workflow_dispatch`), or let the daily cron schedule (`0 3 * * *` = **09:00 AM Bangladesh Time**) run automatically.
6. **Public vs. Private Repository Limits**:
   - **Public Repositories**: GitHub Actions is completely free with unlimited standard runner minutes. Never commit secrets or `.env` files; always use GitHub Actions Secrets.
   - **Private Repositories**: GitHub Free accounts include **2,000 free Linux runner minutes/month** and 500 MB artifact storage (a daily ~3–5 minute run uses ~100–150 minutes/month, well within the free quota).

---

## 8. Troubleshooting Guide

- **`StoryAPIError: HTTP 429 Too Many Requests` / `404 Not Found`**:
  Free models on OpenRouter share community capacity. Configure `OPENROUTER_FALLBACK_MODEL` with a second `:free` model (e.g. `meta-llama/llama-3.3-70b-instruct:free`) or re-run later.
- **`VoiceGenerationError: Piper ONNX voice model not found`**:
  Ensure both `.onnx` and `.onnx.json` files are downloaded to `models/piper/` and match `PIPER_MODEL_PATH` and `PIPER_CONFIG_PATH`. For quick local dry-run testing without downloading the 63 MB ONNX model, set `ALLOW_OFFLINE_TTS_FALLBACK=true`.
- **`Fontconfig error` on Windows FFmpeg**:
  Handled automatically by `VideoRenderer._prepare_fontconfig_env()`, which creates an isolated `fonts.conf` and font cache in the temporary render directory.
- **YouTube Video Locked to Private (`uploadStatus=uploaded`)**:
  Google Cloud projects in "Testing" / unverified status restrict API uploads to `private`. Set `YOUTUBE_PRIVACY_STATUS=private` or submit your OAuth app for Google verification.
- **TikTok API `unaudited_client_can_only_post_to_private_accounts`**:
  Keep `TIKTOK_PRIVACY_LEVEL=SELF_ONLY` until your TikTok app completes the Content Posting API audit.
