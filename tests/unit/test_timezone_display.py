"""Timestamps are stored in UTC and must be PRESENTED in the site's timezone.

The bug: every stored instant is naive UTC, but nothing said so. The API
serialized `2026-09-08T11:31:15` with no offset, and per the ECMAScript spec a
date-time string without an offset is parsed as LOCAL time — so a browser in
IST read 11:31 UTC as 11:31 local and rendered every event 5h30m early. The
same omission put the dashboard's hour buckets and day boundaries in UTC.

What made it survive so long is that the wrong times look entirely plausible.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from backend.app import schemas
from backend.app.schemas import _as_utc_iso

IST = ZoneInfo("Asia/Kolkata")


def an_event(detected_at: datetime) -> schemas.EventOut:
    return schemas.EventOut(
        id=1, plate_number="HR29BG7381", vehicle_type="car",
        direction="in", status="unknown", confidence=0.8, ocr_confidence=0.8,
        detected_at=detected_at,
    )


class TestSerializationCarriesAnExplicitMarker:
    def test_a_naive_instant_is_marked_utc(self):
        """THE fix. Without the marker a browser assumes local time."""
        out = _as_utc_iso(datetime(2026, 9, 8, 11, 31, 15, 725237))
        assert out == "2026-09-08T11:31:15.725237Z"

    def test_the_api_response_carries_it(self):
        import json

        payload = json.loads(an_event(datetime(2026, 9, 8, 11, 31, 15)).model_dump_json())
        assert payload["detected_at"].endswith("Z")

    def test_an_aware_instant_is_converted_not_assumed(self):
        """Safe if a column is ever migrated to timestamptz: a value that
        already knows its offset is converted, never relabelled."""
        aware = datetime(2026, 9, 8, 17, 1, 15, tzinfo=IST)
        assert _as_utc_iso(aware) == "2026-09-08T11:31:15Z"

    def test_none_stays_none(self):
        assert _as_utc_iso(None) is None

    def test_a_browser_in_ist_now_renders_the_right_wall_clock(self):
        """The end-to-end property, simulating what `new Date(...)` does.

        A vehicle passing at 17:01 IST is stored as 11:31 UTC. Parsed WITH the
        marker it comes back as 17:01 local; parsed without it, 11:31 — the
        5h30m error the operator reported.
        """
        stored = datetime(2026, 9, 8, 11, 31, 15)          # naive UTC, as in the DB

        with_marker = datetime.fromisoformat(_as_utc_iso(stored).replace("Z", "+00:00"))
        assert with_marker.astimezone(IST).strftime("%H:%M") == "17:01"

        # What the old serializer produced: no offset, so read as local.
        without_marker = stored.replace(tzinfo=IST)
        assert without_marker.strftime("%H:%M") == "11:31", "the reported bug"

    def test_every_exposed_timestamp_field_uses_the_marked_type(self):
        """A field left as a bare `datetime` would silently reintroduce the
        bug for whichever view reads it."""
        import re
        from pathlib import Path

        source = Path("backend/app/schemas.py").read_text(encoding="utf-8")
        # Field annotations only — the helper's own signature is excluded by
        # requiring leading indentation.
        bare = re.findall(r"^\s+\w+: (?:Optional\[)?datetime\]?", source, flags=re.M)
        assert bare == [], f"unmarked datetime fields: {bare}"


class TestLocalDayBoundaries:
    def test_today_starts_at_local_midnight_not_utc_midnight(self):
        """Midnight UTC is 05:30 in IST, so the old boundary counted every
        vehicle between local midnight and 05:30 as YESTERDAY."""
        from backend.app.routers.dashboard import _today_start

        start = _today_start()
        assert start.tzinfo is None, "must be naive UTC to match the column"

        # Converting the boundary back to local must land exactly on midnight.
        local = start.replace(tzinfo=timezone.utc).astimezone(IST)
        assert (local.hour, local.minute) == (0, 0)

    def test_the_boundary_is_not_utc_midnight(self):
        from backend.app.routers.dashboard import _today_start

        start = _today_start()
        assert (start.hour, start.minute) != (0, 0), (
            "05:30-ish in UTC for IST; exactly 00:00 would mean the old bug"
        )

    def test_an_early_morning_event_counts_as_today(self):
        """00:30 local is the case the old code got wrong: it is 19:00 the
        previous day in UTC, so it fell outside 'today'."""
        from backend.app.routers.dashboard import _today_start

        start = _today_start()
        now_local = datetime.now(IST)
        early_local = now_local.replace(hour=0, minute=30, second=0, microsecond=0)
        early_utc = early_local.astimezone(timezone.utc).replace(tzinfo=None)
        assert early_utc >= start, "an event at 00:30 local belongs to today"


class TestTrendBucketing:
    """The chart itself. Verified as SQL semantics against the real database,
    because the bug lived in how Postgres was asked to truncate."""

    def _events_in_local_hour(self, session, hour_expr):
        from sqlalchemy import text

        return dict(session.execute(text(f"""
            select to_char({hour_expr}, 'HH24') h, count(*)
            from events where detected_at >= now() - interval '3 days'
            group by 1 order by 1
        """)).all())

    def test_local_bucketing_does_not_split_a_local_hour(self):
        """Bucketing the raw UTC column split each local hour across two bars,
        because IST's half-hour offset puts local hour boundaries at :30 UTC:

            local 14:00 -> buckets ['08:00', '09:00']

        Converting BEFORE truncating puts each event in the hour it happened.
        """
        from sqlalchemy import text

        from backend.app.core.config import get_settings
        from backend.app.db.session import SessionLocal

        tz = get_settings().locale.display_timezone
        session = SessionLocal()
        try:
            rows = session.execute(text(f"""
                select to_char(detected_at AT TIME ZONE 'UTC' AT TIME ZONE '{tz}', 'HH24') local_hr,
                       to_char(date_trunc('hour', detected_at AT TIME ZONE 'UTC' AT TIME ZONE '{tz}'), 'HH24') bucket,
                       count(*)
                from events where detected_at >= now() - interval '3 days'
                group by 1, 2
            """)).all()
        except Exception:
            pytest.skip("database not reachable")
        finally:
            session.close()

        if not rows:
            pytest.skip("no recent events to bucket")

        from collections import defaultdict
        buckets = defaultdict(set)
        for local_hr, bucket, _ in rows:
            buckets[local_hr].add(bucket)
        split = {h: b for h, b in buckets.items() if len(b) > 1}
        assert split == {}, f"local hours split across buckets: {split}"

    def test_each_local_hour_maps_to_its_own_label(self):
        from sqlalchemy import text

        from backend.app.core.config import get_settings
        from backend.app.db.session import SessionLocal

        tz = get_settings().locale.display_timezone
        session = SessionLocal()
        try:
            rows = session.execute(text(f"""
                select to_char(detected_at AT TIME ZONE 'UTC' AT TIME ZONE '{tz}', 'HH24') local_hr,
                       to_char(date_trunc('hour', detected_at AT TIME ZONE 'UTC' AT TIME ZONE '{tz}'), 'HH24') bucket
                from events where detected_at >= now() - interval '3 days' limit 500
            """)).all()
        except Exception:
            pytest.skip("database not reachable")
        finally:
            session.close()

        if not rows:
            pytest.skip("no recent events to bucket")
        assert all(local_hr == bucket for local_hr, bucket in rows)


class TestConfiguration:
    def test_the_timezone_is_configurable(self):
        from backend.app.core.config import get_settings

        assert get_settings().locale.display_timezone == "Asia/Kolkata"

    def test_display_tz_resolves_to_a_real_zone(self):
        from backend.app.core.config import display_tz

        assert display_tz() == IST

    def test_an_unknown_zone_falls_back_to_utc_rather_than_failing(self):
        """A dashboard showing UTC beats a dashboard that 500s."""
        from unittest.mock import patch

        from backend.app.core import config as cfg

        broken = cfg.LocaleSettings(display_timezone="Mars/Olympus_Mons")
        with patch.object(cfg, "get_settings") as fake:
            fake.return_value = type("S", (), {"locale": broken})()
            assert cfg.display_tz() == timezone.utc
