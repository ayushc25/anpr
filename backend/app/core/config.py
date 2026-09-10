"""Configuration: environment for secrets and connections, YAML for tuning.

The split is deliberate. Anything an operator might tune while watching a gate
(thresholds, intervals, retention) belongs in configs/*.yaml where it can be
reviewed and versioned; anything that is a secret or differs per machine
belongs in the environment.

The legacy backend/app/config.py still exists and is still imported by the
prototype routers. It re-exports from here, so there is exactly one source of
truth during the migration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[3]
CONFIG_DIR = ROOT_DIR / "configs"
load_dotenv(ROOT_DIR / ".env")


def _load_yaml(path: Path) -> dict:
    if not path.exists():
        return {}
    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _subset(cls, data: Mapping[str, Any]) -> dict:
    """Keep only the keys a dataclass actually declares.

    A YAML file that has drifted ahead of the code should log a warning, not
    crash a camera worker at 2am.
    """
    known = {f.name for f in fields(cls)}
    return {k: v for k, v in (data or {}).items() if k in known}


@dataclass(frozen=True)
class LocaleSettings:
    """How stored UTC instants are presented.

    Everything in this system is stored in UTC and that does not change: UTC
    is the only sane storage format for instants, and the columns already
    hold it. What was missing is a statement of what to CONVERT it to for a
    human, and that gap is what made every displayed time wrong.

    A single site-level timezone rather than per-user, because a gate serves
    one physical place; the vehicles pass in that place's local time and an
    operator standing at the gate reads the clock on the wall.
    """

    #: IANA name. Resolved with zoneinfo, so anything in the tz database
    #: works and DST is handled where it applies.
    display_timezone: str = "Asia/Kolkata"


@dataclass(frozen=True)
class RuntimeSettings:
    reserve_cores: int = 1
    model_cache_dir: str = ".cache/openvino"
    preview_fps: float = 5.0
    preview_jpeg_quality: int = 70


@dataclass(frozen=True)
class ReaderSettings:
    reconnect_min: float = 2.0
    reconnect_max: float = 60.0
    read_timeout: float = 15.0
    open_timeout_ms: int = 8000


@dataclass(frozen=True)
class StorageSettings:
    media_root: str = "backend/storage/events"
    keep_plate_crops: bool = True
    retention_days: int = 90
    event_reads_retention_days: int = 30


@dataclass(frozen=True)
class Settings:
    """Everything the process needs, resolved once."""

    database_url: str
    secret_key: str
    access_token_expire_minutes: int
    algorithm: str

    root_dir: Path
    models_dir: Path

    pipeline: dict = field(default_factory=dict)
    validation: dict = field(default_factory=dict)
    dedupe: dict = field(default_factory=dict)
    debug: dict = field(default_factory=dict)
    locale: LocaleSettings = field(default_factory=LocaleSettings)
    reader: ReaderSettings = field(default_factory=ReaderSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    storage: StorageSettings = field(default_factory=StorageSettings)
    models: dict = field(default_factory=dict)

    #: Shared secret the camera workers present on /internal/events.
    worker_token: str = ""
    api_base_url: str = "http://127.0.0.1:8002"
    redis_url: str = ""

    def resolve(self, relative: str | Path) -> Path:
        """Config paths are written relative to the repository root so the
        same file works in Docker and on a bare-metal edge box."""
        path = Path(relative)
        return path if path.is_absolute() else (self.root_dir / path)

    @property
    def media_root(self) -> Path:
        path = self.resolve(self.storage.media_root)
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_cache_dir(self) -> Path:
        return self.resolve(self.runtime.model_cache_dir)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    defaults = _load_yaml(CONFIG_DIR / "default.yaml")
    models = _load_yaml(CONFIG_DIR / "models.yaml")

    database_url = os.getenv("database_url") or os.getenv("DATABASE_URL")
    if not database_url:
        raise RuntimeError(
            "database_url is not set. Add it to .env, e.g.\n"
            "  database_url=postgresql://user:pass@localhost:5432/anpr"
        )

    return Settings(
        database_url=database_url,
        secret_key=os.getenv("SECRET_KEY", "anpr-dev-secret-change-me"),
        access_token_expire_minutes=int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "480")),
        algorithm="HS256",
        root_dir=ROOT_DIR,
        models_dir=ROOT_DIR / "models",
        pipeline=defaults.get("pipeline", {}),
        validation=defaults.get("validation", {}),
        dedupe=defaults.get("dedupe", {}),
        debug=defaults.get("debug", {}),
        locale=LocaleSettings(**_subset(LocaleSettings, defaults.get("locale", {}))),
        reader=ReaderSettings(**_subset(ReaderSettings, defaults.get("reader", {}))),
        runtime=RuntimeSettings(**_subset(RuntimeSettings, defaults.get("runtime", {}))),
        storage=StorageSettings(**_subset(StorageSettings, defaults.get("storage", {}))),
        models=models,
        worker_token=os.getenv("WORKER_TOKEN", ""),
        api_base_url=os.getenv("API_BASE_URL", "http://127.0.0.1:8002"),
        redis_url=os.getenv("REDIS_URL", ""),
    )


def build_debug_config():
    """DebugConfig from YAML. Off unless configs/default.yaml says otherwise."""
    from ..debug.recorder import DebugConfig

    return DebugConfig(**_subset(DebugConfig, get_settings().debug))


def build_pipeline_config(overrides: Mapping[str, Any] | None = None):
    """PipelineConfig from YAML, with per-camera overrides applied on top.

    ``pipeline.ocr`` is a nested block rather than a flat prefix because the
    OCR scheduler is its own decision with its own dataclass; a camera may
    override the whole block or any single key inside it.
    """
    from ..video.frame_processor import PipelineConfig
    from ..ai.plate_recognizer.grammar_decode import GrammarDecodeConfig
    from ..video.plate_association import AssociationConfig
    from ..video.ocr_scheduler import OcrPolicy

    merged = dict(get_settings().pipeline)
    # The ocr block merges KEY BY KEY. A camera that overrides one OCR knob
    # must keep the site's tuning for the rest; a whole-block replace would
    # silently drop it back to the dataclass defaults, which is invisible
    # until the day the YAML and the defaults differ.
    ocr = dict(merged.get("ocr") or {})
    grammar = dict(merged.get("grammar_decode") or {})
    association = dict(merged.get("association") or {})
    nested = {"ocr": ocr, "grammar_decode": grammar, "association": association}
    for key, value in (overrides or {}).items():
        if value is None:
            continue
        if key in nested:
            nested[key].update({k: v for k, v in (value or {}).items() if v is not None})
        else:
            merged[key] = value

    data = _subset(PipelineConfig, merged)
    if isinstance(data.get("skip_classes"), list):
        data["skip_classes"] = tuple(data["skip_classes"])
    # OcrPolicy validates its own invariants in __post_init__, so a bad YAML
    # value fails loudly at worker startup rather than quietly at 2am.
    data["ocr"] = OcrPolicy(**_subset(OcrPolicy, ocr))
    data["grammar_decode"] = GrammarDecodeConfig(**_subset(GrammarDecodeConfig, grammar))
    data["association"] = AssociationConfig(**_subset(AssociationConfig, association))
    return PipelineConfig(**data)


def build_validation_config(overrides: Mapping[str, Any] | None = None):
    from ..events.multi_frame_validator import ValidationConfig

    merged = dict(get_settings().validation)
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return ValidationConfig(**_subset(ValidationConfig, merged))


def build_dedupe_config(overrides: Mapping[str, Any] | None = None):
    from ..events.dedupe import DedupeConfig

    merged = dict(get_settings().dedupe)
    merged.update({k: v for k, v in (overrides or {}).items() if v is not None})
    return DedupeConfig(**_subset(DedupeConfig, merged))


def model_config(stage: str) -> dict:
    """One stage's model config, with its artifact path made absolute."""
    settings = get_settings()
    cfg = dict(settings.models.get(stage, {}))
    for key in ("artifact", "charset"):
        if cfg.get(key):
            cfg[key] = str(settings.resolve(cfg[key]))
    return cfg


def display_tz():
    """The site's timezone, as a tzinfo. Falls back to UTC if the configured
    name is not in the tz database, because a dashboard showing UTC is a lot
    better than a dashboard that 500s."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

    name = get_settings().locale.display_timezone
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, KeyError):
        import logging
        logging.getLogger("anpr.config").warning(
            "unknown display_timezone %r; falling back to UTC", name
        )
        from datetime import timezone
        return timezone.utc
