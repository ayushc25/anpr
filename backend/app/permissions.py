"""Catalog of menu/page permission keys that can be granted to a user.

Each key gates both a sidebar menu item on the frontend and the matching
write-endpoints on the backend (see deps.require_permission usage in the
routers). This is the single source of truth for what "rights" means -
there are no fixed roles, an admin creating a user just picks a role name
and a subset of these keys.
"""

PERMISSIONS = [
    {"key": "dashboard", "label": "Dashboard"},
    {"key": "live", "label": "Live View"},
    {"key": "events", "label": "ANPR Events"},
    {"key": "vehicles", "label": "Vehicle Search"},
    {"key": "lists", "label": "Blacklist / Whitelist"},
    {"key": "cameras", "label": "Cameras"},
    {"key": "locations", "label": "Locations"},
    {"key": "reports", "label": "Reports"},
    {"key": "users", "label": "Users"},
    {"key": "logs", "label": "Logs"},
]

PERMISSION_KEYS = [p["key"] for p in PERMISSIONS]
