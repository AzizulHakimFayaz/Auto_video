"""Unit tests for production history tracking and duplicate prevention."""

from __future__ import annotations

from pathlib import Path

from src.history_manager import HistoryManager, ProductionRunRecord


def test_history_manager_detects_duplicate_titles_and_similar_stories(
    tmp_path: Path,
) -> None:
    hm = HistoryManager(history_file=tmp_path / "history.json", similarity_threshold=0.60)
    story_text = (
        "Every evening at six, retired stationmaster Aris stood beneath the leaking roof "
        "of Platform Four, holding a bright yellow umbrella he never opened for himself."
    )
    run_id = hm.compute_run_id("2026-10-26", "The Last Umbrella", story_text)
    record = ProductionRunRecord(
        run_id=run_id,
        production_date="2026-10-26",
        title="The Last Umbrella",
        story_hash=hm.compute_story_hash(story_text),
        story_text=story_text,
        mood="sad then hopeful",
        language="English",
    )
    hm.save_or_update_run(record)

    # Duplicate title rejected
    dup_title_reason = hm.check_story_uniqueness(
        "the last umbrella!", "A completely different story text about mountains and rivers."
    )
    assert dup_title_reason is not None
    assert "Duplicate story title" in dup_title_reason

    # Nearly identical story text with different title rejected
    similar_story = (
        "Every evening at six, retired stationmaster Aris stood beneath the leaking roof "
        "of Platform Four, holding a bright red umbrella he never opened for himself."
    )
    sim_reason = hm.check_story_uniqueness("Platform Four Rain", similar_story)
    assert sim_reason is not None
    assert "too similar" in sim_reason

    # Distinct story accepted
    unique_reason = hm.check_story_uniqueness(
        "The Baker's Midnight Loaf",
        "In a quiet mountain village, an elderly baker left warm bread on the windowsill for travelers.",
    )
    assert unique_reason is None


def test_history_tracks_per_platform_upload_status_and_prevents_duplicates(
    tmp_path: Path,
) -> None:
    hm = HistoryManager(history_file=tmp_path / "history.json")
    record = ProductionRunRecord(
        run_id="run_2026-10-26_abc123",
        production_date="2026-10-26",
        title="Test Story",
        story_hash="hash123",
        story_text="Sample story text.",
        mood="hopeful",
        language="English",
    )
    hm.save_or_update_run(record)

    assert not hm.is_platform_already_published("2026-10-26", "youtube")

    hm.record_platform_outcome(
        run_id="run_2026-10-26_abc123",
        platform="youtube",
        status="published",
        platform_post_id="yt_vid_999",
        url="https://www.youtube.com/shorts/yt_vid_999",
    )
    hm.record_platform_outcome(
        run_id="run_2026-10-26_abc123",
        platform="tiktok",
        status="failed",
        error_message="HTTP 403 unaudited client",
    )

    assert hm.is_platform_already_published("2026-10-26", "youtube")
    assert not hm.is_platform_already_published("2026-10-26", "tiktok")
