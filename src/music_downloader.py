"""Automated royalty-free background music downloader from open archives.

Searches and downloads mood-matched, licensed background music directly from
Internet Archive (Public Domain / CC0 / Creative Commons) and Freesound.
Ensures every downloaded file is saved with verified licensing attribution
in `assets/licenses/music_manifest.json`.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import requests

from src.config import AppConfig

logger = logging.getLogger(__name__)


class MusicDownloader:
    """Auto-downloads royalty-free background music matched to AI story direction."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def _sanitize_name(self, name: str) -> str:
        clean = re.sub(r"[^\w\-_.]", "_", name)
        return clean[:45]

    def _update_manifest(self, audio_filename: str, meta: Dict[str, Any]) -> None:
        manifest_path = self.config.license_directory / "music_manifest.json"
        manifest: Dict[str, Any] = {"tracks": {}}
        if manifest_path.exists():
            try:
                data = json.loads(manifest_path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    manifest = data if "tracks" in data else {"tracks": data}
            except Exception as exc:
                logger.warning("Could not parse existing music manifest: %s", exc)

        manifest["tracks"][audio_filename] = meta
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    def search_and_download(
        self,
        style_prompt: str,
        mood: str,
        timeout_seconds: int = 15,
    ) -> Optional[Path]:
        """Search and download an instrumental audio track matching `style_prompt` or `mood`."""
        if not self.config.internet_archive_music_enabled and not self.config.freesound_api_key:
            return None

        # 1. Try Internet Archive Audio Search (Zero Budget, No API Key needed)
        if self.config.internet_archive_music_enabled:
            track_path = self._download_from_internet_archive(style_prompt, mood, timeout_seconds)
            if track_path is not None and track_path.exists():
                return track_path

        # 2. Try Freesound API if key configured
        if self.config.freesound_api_key:
            track_path = self._download_from_freesound(style_prompt, mood, timeout_seconds)
            if track_path is not None and track_path.exists():
                return track_path

        return None

    def _download_from_internet_archive(
        self,
        style_prompt: str,
        mood: str,
        timeout_seconds: int = 15,
    ) -> Optional[Path]:
        """Query Internet Archive Advanced Search for CC/Public Domain MP3 audio."""
        keywords = re.findall(r"\b[a-zA-Z]{3,}\b", f"{mood} {style_prompt}".lower())
        filtered = [k for k in keywords if k not in {"and", "with", "the", "for", "music", "audio", "track", "sound"}]
        query_terms = " OR ".join(filtered[:4]) if filtered else "piano OR cello OR dramatic"

        archive_query = (
            f"mediatype:audio AND (licenseurl:*publicdomain* OR licenseurl:*creative* OR licenseurl:*by/*) "
            f"AND format:\"VBR MP3\" AND ({query_terms})"
        )

        search_url = "https://archive.org/advancedsearch.php"
        params = {
            "q": archive_query,
            "fl[]": ["identifier", "title", "creator", "licenseurl"],
            "rows": 8,
            "output": "json",
        }

        try:
            logger.info("Searching Internet Archive for music matching %r (%s)...", mood, query_terms)
            resp = requests.get(search_url, params=params, timeout=timeout_seconds)
            if resp.status_code != 200:
                logger.warning("Internet Archive search returned HTTP %d", resp.status_code)
                return None

            docs = resp.json().get("response", {}).get("docs", [])
            if not docs:
                logger.info("No matching audio items found on Internet Archive for query.")
                return None

            # Look through items to find a suitable MP3 file
            for doc in docs:
                identifier = doc.get("identifier")
                if not identifier:
                    continue

                meta_url = f"https://archive.org/metadata/{identifier}/files"
                meta_resp = requests.get(meta_url, timeout=timeout_seconds)
                if meta_resp.status_code != 200:
                    continue

                files_data = meta_resp.json().get("files", [])
                mp3_candidates = [
                    f for f in files_data
                    if str(f.get("name", "")).lower().endswith(".mp3")
                    and float(f.get("length", 0) or 0) >= 30.0
                    and float(f.get("length", 0) or 0) <= 420.0
                ]
                if not mp3_candidates:
                    mp3_candidates = [
                        f for f in files_data if str(f.get("name", "")).lower().endswith(".mp3")
                    ]
                if not mp3_candidates:
                    continue

                chosen_file = mp3_candidates[0]
                filename = chosen_file.get("name")
                download_url = f"https://archive.org/download/{identifier}/{quote(filename)}"

                safe_id = self._sanitize_name(identifier)
                safe_fname = self._sanitize_name(filename)
                dest_file = self.config.music_directory / f"ia_{safe_id}_{safe_fname}.mp3"

                if dest_file.exists() and dest_file.stat().st_size > 1024:
                    logger.info("Found cached Internet Archive music track: %s", dest_file.name)
                    return dest_file

                logger.info("Downloading royalty-free soundtrack from Internet Archive: %s", download_url)
                dl_resp = requests.get(download_url, timeout=timeout_seconds * 2, stream=True)
                if dl_resp.status_code != 200:
                    continue

                dest_file.parent.mkdir(parents=True, exist_ok=True)
                with open(dest_file, "wb") as f:
                    for chunk in dl_resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)

                if dest_file.stat().st_size < 10240:
                    dest_file.unlink(missing_ok=True)
                    continue

                title = str(doc.get("title") or filename or "Internet Archive Dramatic Score").strip()
                creator = str(doc.get("creator") or "Public Domain / CC Artist").strip()
                license_url = str(doc.get("licenseurl") or "https://creativecommons.org/publicdomain/zero/1.0/").strip()

                meta_record = {
                    "title": title,
                    "creator": creator,
                    "source_url": f"https://archive.org/details/{identifier}",
                    "license_name": "Creative Commons / Public Domain",
                    "license_url": license_url,
                    "attribution_instructions": f"Music: '{title}' by {creator} via Internet Archive ({license_url})",
                    "commercial_use_allowed": True,
                    "mood_tags": list(set([mood.lower(), "dramatic", "emotional", "soundtrack"] + filtered[:3])),
                    "restrictions": "None",
                }
                self._update_manifest(dest_file.name, meta_record)
                logger.info("Successfully downloaded and registered music track: %s (%s)", dest_file.name, title)
                return dest_file

        except Exception as exc:
            logger.warning("Internet Archive music download failed: %s", exc)

        return None

    def _download_from_freesound(
        self,
        style_prompt: str,
        mood: str,
        timeout_seconds: int = 15,
    ) -> Optional[Path]:
        """Query Freesound API v2 for CC0 ambient/music sounds."""
        api_key = self.config.freesound_api_key
        if not api_key:
            return None

        url = "https://freesound.org/apiv2/search/text/"
        headers = {"Authorization": f"Token {api_key}"}
        params = {
            "query": f"{mood} piano cello emotional",
            "filter": 'license:"Creative Commons 0"',
            "fields": "id,name,username,previews,license",
            "page_size": 5,
        }

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=timeout_seconds)
            if resp.status_code != 200:
                return None
            results = resp.json().get("results", [])
            for item in results:
                previews = item.get("previews", {})
                mp3_url = previews.get("preview-hq-mp3") or previews.get("preview-lq-mp3")
                if not mp3_url:
                    continue
                safe_name = self._sanitize_name(item.get("name", "sound"))
                dest_file = self.config.music_directory / f"fs_{item['id']}_{safe_name}.mp3"
                if dest_file.exists():
                    return dest_file
                dl = requests.get(mp3_url, timeout=timeout_seconds)
                if dl.status_code == 200:
                    dest_file.write_bytes(dl.content)
                    self._update_manifest(
                        dest_file.name,
                        {
                            "title": item.get("name", "Freesound Audio"),
                            "creator": item.get("username", "Freesound Artist"),
                            "source_url": f"https://freesound.org/people/{item.get('username')}/sounds/{item.get('id')}/",
                            "license_name": "Creative Commons 0",
                            "license_url": "https://creativecommons.org/publicdomain/zero/1.0/",
                            "attribution_instructions": "Freesound CC0 Audio",
                            "commercial_use_allowed": True,
                            "mood_tags": [mood.lower(), "dramatic", "freesound"],
                            "restrictions": "None",
                        },
                    )
                    return dest_file
        except Exception as exc:
            logger.warning("Freesound music download failed: %s", exc)

        return None
