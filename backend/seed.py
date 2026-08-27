"""Seed initial admin/manager/guard-equivalent users and sample data."""
from app.database import Base, engine, SessionLocal
from app import models
from app.security import hash_password
from app.permissions import PERMISSION_KEYS

Base.metadata.create_all(bind=engine)

db = SessionLocal()

MANAGER_PERMISSIONS = [k for k in PERMISSION_KEYS if k != "users"]
GUARD_PERMISSIONS = ["dashboard", "live", "events", "vehicles"]

DEFAULT_USERS = [
    ("admin", "Admin User", "admin123", "Administrator", PERMISSION_KEYS),
    ("manager", "Manager User", "manager123", "Manager", MANAGER_PERMISSIONS),
    ("guard", "Guard User", "guard123", "Guard", GUARD_PERMISSIONS),
]

for username, full_name, password, role_name, permissions in DEFAULT_USERS:
    existing = db.query(models.User).filter(models.User.username == username).first()
    if existing:
        continue
    user = models.User(
        username=username,
        full_name=full_name,
        password_hash=hash_password(password),
        role_name=role_name,
        permissions=permissions,
        is_active=True,
    )
    db.add(user)

if db.query(models.Location).count() == 0:
    db.add(models.Location(name="Main Gate", description="Primary entrance/exit"))
    db.add(models.Location(name="North Gate", description="Secondary entrance"))

db.commit()
db.close()

print("Seed complete. Default logins:")
for username, full_name, password, role_name, permissions in DEFAULT_USERS:
    print(f"  {username} / {password}  ({role_name})")
