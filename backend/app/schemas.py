from datetime import datetime, timezone
from typing import Annotated, Optional, List
from pydantic import BaseModel, BeforeValidator, ConfigDict, PlainSerializer

from .models import VehicleStatus, CameraDirection, EventDirection


def _as_utc_iso(value: Optional[datetime]) -> Optional[str]:
    """Serialize a stored instant with an explicit UTC marker.

    Every timestamp in this system is stored as naive UTC — the columns are
    ``timestamp without time zone`` and the writers convert to UTC first. That
    is fine as storage, but Pydantic then serialized it as
    ``2026-09-08T11:31:15`` with no offset, and per the ECMAScript spec a
    date-time string WITHOUT an offset is parsed as LOCAL time. So the browser
    read 11:31 UTC as 11:31 local and rendered every event 5h30m early in
    IST — the times looked plausible, which is why it went unnoticed.

    Appending the marker is the whole fix for every table view: with an offset
    present, ``new Date(...).toLocaleString()`` already converts correctly to
    whatever timezone the viewer is in. No frontend change is needed, and the
    stored values are correct and untouched.

    A value that already carries a timezone is converted rather than assumed,
    so this is safe if a column is ever migrated to ``timestamptz``.
    """
    if value is None:
        return None
    aware = value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    return aware.isoformat().replace("+00:00", "Z")


#: A stored instant, serialized with an explicit UTC marker. Use this instead
#: of a bare ``datetime`` for anything the API hands to a client.
UtcDatetime = Annotated[datetime, PlainSerializer(_as_utc_iso, return_type=str, when_used="json")]


# ---------- Auth ----------
class LoginRequest(BaseModel):
    username: str
    password: str


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


# ---------- User ----------
class UserBase(BaseModel):
    username: str
    full_name: str = ""
    role_name: str = ""
    permissions: List[str] = []
    is_active: bool = True


class UserCreate(UserBase):
    password: str


class UserUpdate(BaseModel):
    full_name: Optional[str] = None
    role_name: Optional[str] = None
    permissions: Optional[List[str]] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


class UserOut(UserBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: UtcDatetime


class PermissionOut(BaseModel):
    key: str
    label: str


Token.model_rebuild()


# ---------- Location ----------
class LocationBase(BaseModel):
    name: str
    description: str = ""


class LocationCreate(LocationBase):
    pass


class LocationOut(LocationBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: UtcDatetime


# ---------- Camera ----------
class CameraBase(BaseModel):
    name: str
    rtsp_url: str
    location_id: Optional[int] = None
    direction: CameraDirection = CameraDirection.both
    is_active: bool = True


class CameraCreate(CameraBase):
    pass


class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    location_id: Optional[int] = None
    direction: Optional[CameraDirection] = None
    is_active: Optional[bool] = None


class CameraOut(CameraBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    is_online: bool
    last_seen_at: Optional[UtcDatetime] = None
    created_at: UtcDatetime


# ---------- Vehicle ----------
def _blank_if_none(value: Optional[str]) -> str:
    """These columns are nullable in the database, and legacy rows predating
    their defaults hold NULL. A single such row must not fail validation for
    the whole list response, so coerce NULL to an empty string."""
    return value or ""


NullableStr = Annotated[str, BeforeValidator(_blank_if_none)]


class VehicleBase(BaseModel):
    plate_number: str
    owner_name: NullableStr = ""
    flat_number: NullableStr = ""
    vehicle_type: NullableStr = "car"
    status: VehicleStatus = VehicleStatus.registered
    valid_until: Optional[UtcDatetime] = None
    notes: NullableStr = ""


class VehicleCreate(VehicleBase):
    pass


class VehicleUpdate(BaseModel):
    owner_name: Optional[str] = None
    flat_number: Optional[str] = None
    vehicle_type: Optional[str] = None
    status: Optional[VehicleStatus] = None
    valid_until: Optional[UtcDatetime] = None
    notes: Optional[str] = None


class VehicleOut(VehicleBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: UtcDatetime
    updated_at: UtcDatetime


# ---------- Event ----------
class EventOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    plate_number: str
    vehicle_id: Optional[int] = None
    camera_id: Optional[int] = None
    vehicle_type: str
    vehicle_color: Optional[str] = None
    plate_color: Optional[str] = None
    direction: EventDirection
    status: VehicleStatus
    # An event records what actually happened, so it uses EventDirection
    # (in / out / unknown), not the camera's configured role which may be
    # "both". Serving "unknown" through a CameraDirection field raised a
    # validation error the moment a camera had no virtual line.
    confidence: float
    ocr_confidence: float
    image_path: Optional[str] = None
    detected_at: UtcDatetime
    camera_name: Optional[str] = None
    owner_name: Optional[str] = None


class EventCreate(BaseModel):
    plate_number: str
    camera_id: Optional[int] = None
    vehicle_type: str = "car"
    direction: EventDirection = EventDirection.unknown
    confidence: float = 0.0
    ocr_confidence: float = 0.0
    image_path: Optional[str] = None


# ---------- Dashboard ----------
class DashboardStats(BaseModel):
    total_vehicles_today: int
    vehicles_inside: int
    blacklisted_count: int
    unknown_today: int
    active_cameras: int
    total_cameras: int
    entries_today: int
    exits_today: int


class TrendPoint(BaseModel):
    hour: str
    count: int


# ---------- Reports ----------
class DailySummaryRow(BaseModel):
    date: str
    total: int
    entries: int
    exits: int
    registered: int
    whitelist: int
    blacklist: int
    unknown: int


class DailySummaryPage(BaseModel):
    items: List[DailySummaryRow]
    total: int
    page: int
    page_size: int


# ---------- Logs ----------
class ActivityLogOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    username: str
    action: str
    details: str
    created_at: UtcDatetime


class ActivityLogPage(BaseModel):
    items: List[ActivityLogOut]
    total: int
    page: int
    page_size: int
