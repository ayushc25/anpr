from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, ConfigDict

from .models import VehicleStatus, CameraDirection


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
    created_at: datetime


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
    created_at: datetime


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
    last_seen_at: Optional[datetime] = None
    created_at: datetime


# ---------- Vehicle ----------
class VehicleBase(BaseModel):
    plate_number: str
    owner_name: str = ""
    flat_number: str = ""
    vehicle_type: str = "car"
    status: VehicleStatus = VehicleStatus.registered
    valid_until: Optional[datetime] = None
    notes: str = ""


class VehicleCreate(VehicleBase):
    pass


class VehicleUpdate(BaseModel):
    owner_name: Optional[str] = None
    flat_number: Optional[str] = None
    vehicle_type: Optional[str] = None
    status: Optional[VehicleStatus] = None
    valid_until: Optional[datetime] = None
    notes: Optional[str] = None


class VehicleOut(VehicleBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    updated_at: datetime


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
    direction: CameraDirection
    status: VehicleStatus
    confidence: float
    ocr_confidence: float
    image_path: Optional[str] = None
    detected_at: datetime
    camera_name: Optional[str] = None
    owner_name: Optional[str] = None


class EventCreate(BaseModel):
    plate_number: str
    camera_id: Optional[int] = None
    vehicle_type: str = "car"
    direction: CameraDirection = CameraDirection.both
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
    created_at: datetime
