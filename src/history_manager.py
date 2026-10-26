"""Production history, deduplication, and recovery state manager.

Tracks generated stories, used image assets, music licenses, and per-platform
publication outcomes so interrupted runs can resume or retry publishing without
generating duplicate stories or re-uploading already published videos.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


def _normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def compute_story_similarity(text_a: str, text_b: str) -> float:
    """Compute a combined token-Jaccard and sequence similarity score in [0.0, 1.0]."""
    norm_a = _normalize_text(text_a)
    norm_b = _normalize_text(text_b)
    if not norm_a or not norm_b:
        return 0.0
    if norm_a == norm_b:
        return 1.0

    tokens_a = set(norm_a.split())
    tokens_b = set(norm_b.split())
    union = tokens_a | tokens_b
    jaccard = len(tokens_a & tokens_b) / len(union) if union else 0.0
    seq_ratio = SequenceMatcher(None, norm_a, norm_b).ratio()
    return max(jaccard, seq_ratio)


@dataclass
class PlatformUploadRecord:
    """Status of a single platform publishing attempt."""

    platform: str
    status: str  # "published", "dry_run", "skipped", "failed"
    platform_post_id: Optional[str] = None
    url: Optional[str] = None
    error_message: Optional[str] = None
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ProductionRunRecord:
    """Complete record of a daily production run."""

    run_id: str
    production_date: str
    title: str
    story_hash: str
    story_text: str
    mood: str
    language: str
    video_path: Optional[str] = None
    metadata_path: Optional[str] = None
    selected_images: List[Dict[str, Any]] = field(default_factory=list)
    music_metadata: Optional[Dict[str, Any]] = None
    platform_uploads: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    stage_status: Dict[str, str] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    updated_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class HistoryManager:
    """Manages persistent JSON history for stories, assets, and publishing state."""

    def __init__(self, history_file: Path, similarity_threshold: float = 0.65) -> None:
        self.history_file = history_file
        self.similarity_threshold = similarity_threshold
        self._ensure_file()

    def _ensure_file(self) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        if not self.history_file.exists():
            self._write_data({"version": 1, "runs": []})

    def _read_data(self) -> Dict[str, Any]:
        self._ensure_file()
        try:
            raw = self.history_file.read_text(encoding="utf-8").strip()
            if not raw:
                return {"version": 1, "runs": []}
            data = json.loads(raw)
            if not isinstance(data, dict) or "runs" not in data:
                return {"version": 1, "runs": []}
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("History file unreadable (%s); initializing clean state.", exc)
            return {"version": 1, "runs": []}

    def _write_data(self, data: Dict[str, Any]) -> None:
        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        temp_file = self.history_file.with_suffix(".tmp")
        temp_file.write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        temp_file.replace(self.history_file)

    @staticmethod
    def compute_run_id(production_date: str, title: str, story_text: str) -> str:
        """Create a deterministic unique ID for a production date and story."""
        digest = hashlib.sha256(
            f"{production_date}:{_normalize_text(title)}:{_normalize_text(story_text)}".encode(
                "utf-8"
            )
        ).hexdigest()[:12]
        return f"run_{production_date}_{digest}"

    @staticmethod
    def compute_story_hash(story_text: str) -> str:
        return hashlib.sha256(_normalize_text(story_text).encode("utf-8")).hexdigest()

    def get_all_runs(self) -> List[Dict[str, Any]]:
        return list(self._read_data().get("runs", []))

    def get_run_by_date(self, production_date: str) -> Optional[Dict[str, Any]]:
        """Return the most recent run record for the given YYYY-MM-DD date."""
        runs = self.get_all_runs()
        for run in reversed(runs):
            if run.get("production_date") == production_date:
                return run
        return None

    def get_run_by_id(self, run_id: str) -> Optional[Dict[str, Any]]:
        for run in reversed(self.get_all_runs()):
            if run.get("run_id") == run_id:
                return run
        return None

    def is_platform_already_published(self, production_date: str, platform: str) -> bool:
        """Return True if the specified platform has already confirmed publication for this date."""
        run = self.get_run_by_date(production_date)
        if not run:
            return False
        uploads = run.get("platform_uploads", {})
        platform_record = uploads.get(platform, {})
        return platform_record.get("status") == "published"

    def get_recent_titles(self, limit: int = 30) -> List[str]:
        runs = self.get_all_runs()[-limit:]
        return [r.get("title", "") for r in runs if r.get("title")]

    def get_recently_used_images(self, limit_runs: int = 10) -> Set[str]:
        """Return image identifiers or paths used in recent runs to avoid repetition."""
        used: Set[str] = set()
        for run in self.get_all_runs()[-limit_runs:]:
            for img in run.get("selected_images", []):
                for key in ("id", "local_path", "source_url"):
                    val = img.get(key)
                    if val:
                        used.add(str(val))
        return used

    def check_story_uniqueness(
        self, title: str, story_text: str, recent_limit: int = 50
    ) -> Optional[str]:
        """Check whether a candidate story duplicates or closely resembles a previous run.

        Returns None if unique, or a descriptive rejection reason if too similar.
        """
        norm_title = _normalize_text(title)
        story_hash = self.compute_story_hash(story_text)

        for past in self.get_all_runs()[-recent_limit:]:
            past_title = _normalize_text(past.get("title", ""))
            if norm_title and norm_title == past_title:
                return f"Duplicate story title detected: {title!r}"
            if past.get("story_hash") == story_hash:
                return "Exact duplicate story text detected in history."
            past_story = past.get("story_text", "")
            similarity = compute_story_similarity(story_text, past_story)
            if similarity >= self.similarity_threshold:
                return (
                    f"Story is too similar (score={similarity:.2f} >= {self.similarity_threshold:.2f}) "
                    f"to past story {past.get('title')!r} from {past.get('production_date')}."
                )
        return None

    def save_or_update_run(self, record: ProductionRunRecord) -> None:
        """Insert or update a production run record atomically."""
        data = self._read_data()
        runs: List[Dict[str, Any]] = data.get("runs", [])
        record.updated_at = datetime.now(timezone.utc).isoformat()
        payload = asdict(record)

        for idx, existing in enumerate(runs):
            if existing.get("run_id") == record.run_id:
                runs[idx] = payload
                data["runs"] = runs
                self._write_data(data)
                return

        runs.append(payload)
        data["runs"] = runs
        self._write_data(data)

    def record_platform_outcome(
        self,
        run_id: str,
        platform: str,
        status: str,
        platform_post_id: Optional[str] = None,
        url: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> None:
        """Record the outcome of a single platform upload independently."""
        data = self._read_data()
        runs: List[Dict[str, Any]] = data.get("runs", [])
        now_iso = datetime.now(timezone.utc).isoformat()

        for run in runs:
            if run.get("run_id") == run_id:
                uploads = run.setdefault("platform_uploads", {})
                uploads[platform] = asdict(
                    PlatformUploadRecord(
                        platform=platform,
                        status=status,
                        platform_post_id=platform_post_id,
                        url=url,
                        error_message=error_message,
                        updated_at=now_iso,
                    )
                )
                run["updated_at"] = now_iso
                self._write_data(data)
                return

        raise KeyError(f"Run ID {run_id!r} not found in history.")
