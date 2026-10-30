from datetime import datetime
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app import models, schemas
from backend.app.routers.events import _serialize

client = TestClient(app)


def test_setup_status_endpoint():
    res = client.get("/setup/status")
    assert res.status_code == 200
    data = res.json()
    assert "setup_required" in data
    assert "database_configured" in data
    assert "database_connected" in data
    assert "has_users" in data


def test_setup_status_via_api_prefix():
    res = client.get("/api/setup/status")
    assert res.status_code == 200
    data = res.json()
    assert "setup_required" in data


def test_test_db_invalid_url():
    res = client.post("/setup/test-db", json={"database_url": "invalid://user:pass@host/db"})
    assert res.status_code == 400
    assert "must start with 'postgresql://'" in res.json()["detail"]


def test_test_db_unreachable_host():
    res = client.post("/setup/test-db", json={"database_url": "postgresql://invalid:invalid@127.0.0.1:54399/nonexistent"})
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is False
    assert "error" in data


def test_spa_root_and_routes_serving():
    res_root = client.get("/")
    assert res_root.status_code == 200
    assert "<!doctype html>" in res_root.text.lower() or "<html" in res_root.text.lower()

    res_setup = client.get("/setup")
    assert res_setup.status_code == 200
    assert "<!doctype html>" in res_setup.text.lower() or "<html" in res_setup.text.lower()


def test_event_out_serialization_both_images():
    ev = models.Event(
        id=99999,
        plate_number="MH12AB1234",
        vehicle_type="car",
        direction="in",
        status="registered",
        detect_confidence=0.95,
        plate_confidence=0.98,
        vehicle_image_path="2026/10/30/cam1_MH12AB1234.jpg",
        plate_image_path="2026/10/30/cam1_MH12AB1234_plate.jpg",
        detected_at=datetime.utcnow(),
    )
    serialized = _serialize(ev)
    assert serialized.vehicle_image_path == "2026/10/30/cam1_MH12AB1234.jpg"
    assert serialized.plate_image_path == "2026/10/30/cam1_MH12AB1234_plate.jpg"
    assert serialized.image_path == "2026/10/30/cam1_MH12AB1234.jpg"
