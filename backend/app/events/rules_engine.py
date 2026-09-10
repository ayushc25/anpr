"""Alert rules.

Rules evaluate a PERSISTED event, deliberately separate from event creation.
That separation buys two things: a rule change can be backfilled over history,
and a bug in a rule can never prevent a gate event from being recorded.

Adding a rule means writing a Rule subclass, registering its code, and
inserting an ``alert_rules`` row — no migration, no pipeline change.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import Callable, Optional

from ..db.base import AlertSeverity, VehicleStatus

logger = logging.getLogger("anpr.events.rules")


@dataclass
class RuleContext:
    """Everything a rule may need that is not on the event itself."""

    now: datetime
    #: (plate, since) -> count, so a rule can ask "how often lately?"
    recent_event_count: Callable[[str, datetime], int] = lambda plate, since: 0
    camera_name: str = ""
    registered: bool = False


@dataclass
class AlertDraft:
    code: str
    title: str
    message: str
    severity: AlertSeverity = AlertSeverity.warning
    context: dict = field(default_factory=dict)


class Rule(ABC):
    code: str = ""
    default_severity: AlertSeverity = AlertSeverity.warning

    def __init__(self, params: dict | None = None, severity: AlertSeverity | None = None):
        self.params = params or {}
        self.severity = severity or self.default_severity

    @abstractmethod
    def check(self, event, ctx: RuleContext) -> Optional[AlertDraft]: ...


class BlacklistedVehicleRule(Rule):
    code = "blacklisted_vehicle"
    default_severity = AlertSeverity.critical

    def check(self, event, ctx):
        if event.status != VehicleStatus.blacklist:
            return None
        owner = event.vehicle.display_owner if event.vehicle else ""
        return AlertDraft(
            code=self.code,
            title=f"Blacklisted vehicle {event.plate_number}",
            message=(
                f"{event.plate_number} was detected at {ctx.camera_name or 'a gate camera'} "
                f"({event.direction.value if event.direction else 'unknown'})."
                + (f" Registered to {owner}." if owner else "")
            ),
            severity=self.severity,
            context={"plate": event.plate_number, "camera_id": event.camera_id},
        )


class UnknownVehicleRule(Rule):
    code = "unknown_vehicle"
    default_severity = AlertSeverity.info

    def check(self, event, ctx):
        if event.status != VehicleStatus.unknown:
            return None
        # Only alert on entries by default: an unregistered vehicle leaving is
        # the normal end of a visit and alerting on it doubles the noise.
        directions = self.params.get("directions", ["in"])
        if event.direction and event.direction.value not in directions:
            return None
        min_conf = float(self.params.get("min_confidence", 0.0))
        if event.plate_confidence < min_conf:
            return None
        return AlertDraft(
            code=self.code,
            title=f"Unregistered vehicle {event.plate_number}",
            message=f"{event.plate_number} entered at {ctx.camera_name or 'a gate camera'} and is not in the registry.",
            severity=self.severity,
            context={"plate": event.plate_number, "camera_id": event.camera_id},
        )


class ExpiredPermitRule(Rule):
    code = "expired_permit"
    default_severity = AlertSeverity.warning

    def check(self, event, ctx):
        vehicle = event.vehicle
        if vehicle is None or not vehicle.valid_until:
            return None
        if vehicle.valid_until >= ctx.now:
            return None
        return AlertDraft(
            code=self.code,
            title=f"Expired permit for {event.plate_number}",
            message=(
                f"{event.plate_number} ({vehicle.display_owner or 'unknown owner'}, "
                f"flat {vehicle.display_flat or '-'}) has a permit that expired on "
                f"{vehicle.valid_until:%d %b %Y}."
            ),
            severity=self.severity,
            context={"plate": event.plate_number, "valid_until": vehicle.valid_until.isoformat()},
        )


class AfterHoursEntryRule(Rule):
    code = "after_hours_entry"
    default_severity = AlertSeverity.warning

    def check(self, event, ctx):
        if event.direction and event.direction.value != "in":
            return None
        start = _parse_time(self.params.get("from", "23:00"))
        end = _parse_time(self.params.get("to", "05:00"))
        moment = (event.detected_at or ctx.now).time()
        # The window normally wraps midnight, which is the whole point of it.
        inside = (start <= moment or moment < end) if start > end else (start <= moment < end)
        if not inside:
            return None
        if self.params.get("registered_exempt", True) and ctx.registered:
            return None
        return AlertDraft(
            code=self.code,
            title=f"After-hours entry: {event.plate_number}",
            message=f"{event.plate_number} entered at {moment:%H:%M} via {ctx.camera_name or 'a gate camera'}.",
            severity=self.severity,
            context={"plate": event.plate_number, "time": moment.isoformat()},
        )


class RepeatedUnknownRule(Rule):
    code = "repeated_unknown"
    default_severity = AlertSeverity.warning

    def check(self, event, ctx):
        if event.status != VehicleStatus.unknown:
            return None
        window_minutes = int(self.params.get("window_minutes", 60))
        threshold = int(self.params.get("threshold", 3))
        since = ctx.now - timedelta(minutes=window_minutes)
        seen = ctx.recent_event_count(event.plate_number, since)
        if seen < threshold:
            return None
        return AlertDraft(
            code=self.code,
            title=f"Repeated unregistered vehicle {event.plate_number}",
            message=f"{event.plate_number} has been seen {seen} times in the last {window_minutes} minutes.",
            severity=self.severity,
            context={"plate": event.plate_number, "count": seen},
        )


class LowConfidenceRule(Rule):
    code = "low_confidence"
    default_severity = AlertSeverity.info

    def check(self, event, ctx):
        threshold = float(self.params.get("threshold", 0.65))
        if event.plate_confidence >= threshold and not event.is_disputed:
            return None
        return AlertDraft(
            code=self.code,
            title=f"Low-confidence read: {event.plate_number}",
            message=(
                f"{event.plate_number} was recorded at {event.plate_confidence:.0%} confidence "
                f"from {event.read_count} reads. Review the frame evidence and correct it if needed."
            ),
            severity=self.severity,
            context={"plate": event.plate_number, "confidence": event.plate_confidence},
        )


RULES: dict[str, type[Rule]] = {
    cls.code: cls
    for cls in (
        BlacklistedVehicleRule,
        UnknownVehicleRule,
        ExpiredPermitRule,
        AfterHoursEntryRule,
        RepeatedUnknownRule,
        LowConfidenceRule,
    )
}


def _parse_time(value: str) -> time:
    try:
        hour, minute = str(value).split(":")[:2]
        return time(int(hour), int(minute))
    except (ValueError, TypeError):
        return time(0, 0)


class RulesEngine:
    """Evaluates the enabled ``alert_rules`` rows against one event."""

    def __init__(self, rule_rows: list, cooldown_check: Callable[[str, str, int], bool] | None = None):
        self.rule_rows = rule_rows
        #: (code, plate, cooldown_seconds) -> True when an alert was already
        #: raised recently and this one should be suppressed.
        self.cooldown_check = cooldown_check or (lambda code, plate, seconds: False)

    def evaluate(self, event, ctx: RuleContext) -> list[tuple[object, AlertDraft]]:
        results: list[tuple[object, AlertDraft]] = []
        for row in self.rule_rows:
            if not row.is_enabled:
                continue
            if row.camera_ids and event.camera_id not in row.camera_ids:
                continue
            rule_cls = RULES.get(row.code)
            if rule_cls is None:
                logger.warning("alert_rules row %s names unknown code %r", row.id, row.code)
                continue
            try:
                draft = rule_cls(row.params or {}, row.severity).check(event, ctx)
            except Exception:
                # One broken rule must not stop the others, and must never
                # affect the event that has already been written.
                logger.exception("rule %s failed on event %s", row.code, event.id)
                continue
            if draft is None:
                continue
            if self.cooldown_check(draft.code, event.plate_number, row.cooldown_seconds or 0):
                logger.debug("alert %s for %s suppressed by cooldown", draft.code, event.plate_number)
                continue
            results.append((row, draft))
        return results


DEFAULT_RULES = [
    {
        "code": "blacklisted_vehicle",
        "name": "Blacklisted vehicle detected",
        "description": "Raised whenever a vehicle marked blacklisted is seen at any gate.",
        "severity": AlertSeverity.critical,
        "channels": ["smtp"],
        "cooldown_seconds": 60,
        "params": {},
    },
    {
        "code": "unknown_vehicle",
        "name": "Unregistered vehicle entered",
        "description": "Raised when a vehicle that is not in the registry enters.",
        "severity": AlertSeverity.info,
        "channels": [],
        "cooldown_seconds": 600,
        "params": {"directions": ["in"], "min_confidence": 0.6},
    },
    {
        "code": "low_confidence",
        "name": "Low-confidence recognition",
        "description": "Flags reads that need an operator to confirm the plate.",
        "severity": AlertSeverity.info,
        "channels": [],
        "cooldown_seconds": 0,
        "params": {"threshold": 0.65},
    },
]
