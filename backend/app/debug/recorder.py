"""Debug artifact recorder — the evaluation/error dataset builder.

Writes, for every OCR call, everything needed to work out afterwards WHY a
plate was misread:

    <dir>/<camera>/<track>/<frame>/vehicle.jpg         the vehicle crop
                                  /plate_raw.jpg       unenhanced, un-deskewed
                                  /plate_warped.jpg    what the crop became
                                  /plate_enhanced.jpg  what OCR actually saw
                                  /ocr.json            text, per-char probs,
                                                       alternatives, quality,
                                                       scheduler verdict
    <dir>/<camera>/<track>/resolved.json               the final verdict

The four images exist to separate causes that look identical in the logs. A
misread can come from the detector framing the wrong box, from the deskew, or
from the CLAHE enhancement crushing a low-contrast plate — and with only the
final string there is no way to tell which. With all four, one glance does it.

``ocr.json`` carries the per-character probabilities and the runner-up
characters. That is the raw material for a real confusion matrix: rather than
guessing which glyph pairs this model confuses, the pairs can be counted from
recorded evidence. Until that data exists the confusion maps in
``postprocess`` stay exactly as they are.

OFF BY DEFAULT, AND BOUNDED
---------------------------
Recording every OCR call is a temporary measure for dataset building, not a
production mode. It writes four JPEGs and a JSON per recognizer call, which at
a busy gate is tens of megabytes an hour. So:

  * ``enabled`` defaults to False and nothing in this module is touched when
    it is — the hot path checks one boolean;
  * ``max_tracks`` caps how many distinct tracks are ever recorded;
  * ``max_bytes`` caps total bytes written, checked before each write;
  * every failure is swallowed and logged once. A debug facility that can take
    down a camera worker is worse than no debug facility.
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

logger = logging.getLogger("anpr.debug")


@dataclass(frozen=True)
class DebugConfig:
    enabled: bool = False
    dir: str = "backend/storage/debug"
    #: Distinct tracks recorded before the recorder stops accepting new ones.
    #: Existing tracks keep recording, so a partially captured vehicle is
    #: never left with half its frames.
    max_tracks: int = 50
    #: Total bytes this recorder may write. 500 MB is a few thousand OCR calls
    #: — enough for an initial error dataset, small enough not to fill a disk.
    max_bytes: int = 500_000_000
    jpeg_quality: int = 92


class DebugRecorder:
    """Per-worker. Thread-safe because the preview thread may also touch it."""

    def __init__(self, cfg: DebugConfig, camera_id: int, root: Optional[Path] = None):
        self.cfg = cfg
        self.camera_id = camera_id
        self._lock = threading.Lock()
        self._tracks: set[int] = set()
        self._bytes = 0
        self._exhausted_logged = False
        self._base: Optional[Path] = None
        if cfg.enabled:
            base = Path(root) if root else Path(cfg.dir)
            self._base = base / f"camera_{camera_id}"
            try:
                self._base.mkdir(parents=True, exist_ok=True)
                logger.warning(
                    "DEBUG RECORDING ENABLED for camera %s -> %s (max %d tracks, %d MB). "
                    "This writes artifacts for EVERY OCR call; disable in production.",
                    camera_id, self._base, cfg.max_tracks, cfg.max_bytes // 1_000_000,
                )
            except Exception:
                logger.exception("debug recorder could not create %s; disabling", self._base)
                self._base = None

    @property
    def enabled(self) -> bool:
        return self._base is not None

    # -- budget ------------------------------------------------------------
    def _accepting(self, track_id: int) -> bool:
        if self._base is None:
            return False
        if self._bytes >= self.cfg.max_bytes:
            if not self._exhausted_logged:
                logger.warning(
                    "debug recorder for camera %s hit its %d MB budget; recording stops",
                    self.camera_id, self.cfg.max_bytes // 1_000_000,
                )
                self._exhausted_logged = True
            return False
        # A track already being recorded keeps recording, so no vehicle is
        # left with half its frames captured.
        if track_id in self._tracks:
            return True
        return len(self._tracks) < self.cfg.max_tracks

    # -- writes ------------------------------------------------------------
    def record_ocr(
        self,
        track_id: int,
        frame_idx: int,
        *,
        vehicle_crop: Optional[np.ndarray] = None,
        plate_raw: Optional[np.ndarray] = None,
        plate_warped: Optional[np.ndarray] = None,
        plate_enhanced: Optional[np.ndarray] = None,
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """One OCR call's artifacts. Never raises."""
        if self._base is None:
            return
        try:
            with self._lock:
                if not self._accepting(track_id):
                    return
                self._tracks.add(track_id)
                target = self._base / f"track_{track_id:06d}" / f"frame_{frame_idx:06d}"
                target.mkdir(parents=True, exist_ok=True)
                for name, image in (
                    ("vehicle", vehicle_crop),
                    ("plate_raw", plate_raw),
                    ("plate_warped", plate_warped),
                    ("plate_enhanced", plate_enhanced),
                ):
                    self._write_image(target / f"{name}.jpg", image)
                if payload is not None:
                    self._write_json(target / "ocr.json", payload)
        except Exception:
            logger.exception("debug recorder failed on track %s frame %s", track_id, frame_idx)

    def record_resolution(self, track_id: int, payload: dict[str, Any]) -> None:
        """The track's final verdict, alongside its frames."""
        if self._base is None:
            return
        try:
            with self._lock:
                if track_id not in self._tracks:
                    # Never recorded any of this track's frames, so a lone
                    # verdict would have nothing to explain.
                    return
                target = self._base / f"track_{track_id:06d}"
                target.mkdir(parents=True, exist_ok=True)
                self._write_json(target / "resolved.json", payload)
        except Exception:
            logger.exception("debug recorder failed to record resolution for %s", track_id)

    def _write_image(self, path: Path, image: Optional[np.ndarray]) -> None:
        if image is None or getattr(image, "size", 0) == 0:
            return
        ok, buffer = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality])
        if not ok:
            return
        data = buffer.tobytes()
        path.write_bytes(data)
        self._bytes += len(data)

    def _write_json(self, path: Path, payload: dict[str, Any]) -> None:
        text = json.dumps(payload, indent=2, default=_encode)
        path.write_text(text, encoding="utf-8")
        self._bytes += len(text)

    @property
    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "tracks_recorded": len(self._tracks),
            "bytes_written": self._bytes,
            "budget_bytes": self.cfg.max_bytes,
        }


def _encode(value: Any) -> Any:
    """JSON fallback for numpy scalars, dataclasses and enums."""
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    return str(value)


class NullRecorder:
    """Stand-in when debugging is off, so the pipeline needs no None checks."""

    enabled = False

    def record_ocr(self, *_args, **_kwargs) -> None:
        return None

    def record_resolution(self, *_args, **_kwargs) -> None:
        return None

    @property
    def stats(self) -> dict:
        return {"enabled": False}
