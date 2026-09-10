"""RTSP capture with a latest-frame policy.

The important property: a reader thread pulls frames as fast as the stream
delivers them into a slot of size one, and the pipeline takes whatever is in
that slot. When the pipeline is slower than the camera — always, on a CPU box —
old frames are DISCARDED rather than queued.

This is what keeps event timestamps honest. The prototype reads and processes
on the same thread, so any inference stall is absorbed by the decoder's
internal buffer and the system silently starts recognising plates from
several seconds ago.
"""
from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

logger = logging.getLogger("anpr.video.rtsp")

# TCP transport for RTSP: UDP loses packets on a congested society LAN and
# produces smeared frames that look exactly like motion blur to the recognizer.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"
)


def _looks_like_file(url: str) -> bool:
    """True for a local video file rather than a network stream."""
    if not url:
        return False
    lowered = url.lower()
    if lowered.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://")):
        return False
    try:
        return Path(url).is_file()
    except OSError:
        return False


@dataclass
class ReaderStats:
    frames_read: int = 0
    frames_dropped: int = 0
    reconnects: int = 0
    last_frame_at: float = 0.0
    fps: float = 0.0
    connected: bool = False
    last_error: str = ""


class RtspReader:
    """One stream, one thread, one frame slot."""

    def __init__(
        self,
        url: str,
        name: str = "camera",
        reconnect_min: float = 2.0,
        reconnect_max: float = 60.0,
        read_timeout: float = 15.0,
        open_timeout_ms: int = 8000,
        loop_file: bool = True,
    ):
        self.url = url
        self.name = name
        self.reconnect_min = reconnect_min
        self.reconnect_max = reconnect_max
        self.read_timeout = read_timeout
        self.open_timeout_ms = open_timeout_ms

        # A local video file is a legitimate source: it is how clips are
        # replayed for tuning and evaluation. It needs different handling from
        # a live camera, because cv2 decodes a file as fast as the CPU allows
        # (~300 fps for 1080p) — the latest-frame drop policy would then throw
        # away ~98% of the clip before the pipeline ever saw it.
        self.is_file = _looks_like_file(url)
        self.loop_file = loop_file
        self.source_fps = 0.0

        self.stats = ReaderStats()
        self._frame: Optional[np.ndarray] = None
        self._frame_at = 0.0
        self._frame_seq = 0
        self._last_taken_seq = -1
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fps_window: list[float] = []

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "RtspReader":
        if self._thread and self._thread.is_alive():
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"rtsp-{self.name}", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def __enter__(self) -> "RtspReader":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    # -- consumer API ------------------------------------------------------
    def latest(self) -> Optional[np.ndarray]:
        """Newest frame, or None if none has arrived since the last call.

        Returning None for an already-consumed frame (rather than the same
        frame again) keeps the pipeline from re-processing a still image when
        a stream stalls, which would otherwise inflate the read count of a
        parked vehicle's track.
        """
        with self._lock:
            if self._frame is None or self._frame_seq == self._last_taken_seq:
                return None
            self._last_taken_seq = self._frame_seq
            return self._frame

    def latest_with_timestamp(self) -> tuple[Optional[np.ndarray], float]:
        """``latest()`` plus the wall-clock time the frame was published.

        The consumer needs the capture time, not its own clock, to know how
        far behind it is running. Measuring that inside the pipeline is
        impossible — by then the only timestamp available is "now" — so the
        reader hands it out with the frame.
        """
        with self._lock:
            if self._frame is None or self._frame_seq == self._last_taken_seq:
                return None, 0.0
            self._last_taken_seq = self._frame_seq
            return self._frame, self._frame_at

    def peek(self) -> Optional[np.ndarray]:
        """Newest frame regardless of whether it was already consumed — for
        the preview overlay and for /cameras/{id}/snapshot."""
        with self._lock:
            return self._frame

    @property
    def connected(self) -> bool:
        return self.stats.connected

    # -- thread ------------------------------------------------------------
    def _open(self) -> Optional[cv2.VideoCapture]:
        capture = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        try:
            # A small decoder buffer reinforces the latest-frame policy at the
            # driver level, where the backend honours it.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, self.open_timeout_ms)
            capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(self.read_timeout * 1000))
        except Exception:
            pass  # not all builds expose every property
        if not capture.isOpened():
            capture.release()
            return None
        if self.is_file:
            self.source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        return capture

    def _publish(self, frame: np.ndarray) -> None:
        now = time.time()
        with self._lock:
            if self._frame is not None and self._frame_seq != self._last_taken_seq:
                self.stats.frames_dropped += 1
            self._frame = frame
            self._frame_at = now
            self._frame_seq += 1
        self.stats.frames_read += 1
        self.stats.last_frame_at = now

        self._fps_window.append(now)
        if len(self._fps_window) > 30:
            self._fps_window.pop(0)
        span = self._fps_window[-1] - self._fps_window[0]
        self.stats.fps = round((len(self._fps_window) - 1) / span, 2) if span > 0 else 0.0

    def _run(self) -> None:
        backoff = self.reconnect_min
        while not self._stop.is_set():
            capture = self._open()
            if capture is None:
                self.stats.connected = False
                self.stats.last_error = "could not open stream"
                logger.warning("%s: cannot open stream, retrying in %.0fs", self.name, backoff)
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, self.reconnect_max)
                self.stats.reconnects += 1
                continue

            logger.info("%s: connected", self.name)
            self.stats.connected = True
            self.stats.last_error = ""
            backoff = self.reconnect_min
            last_ok = time.time()

            # Play a file at its native rate so the pipeline sees the clip the
            # way it would see a live camera. Without this the reader finishes
            # a two-minute clip in ten seconds and the pacer discards almost
            # all of it.
            frame_period = (1.0 / self.source_fps) if (self.is_file and self.source_fps > 0) else 0.0
            next_frame_at = time.monotonic()

            while not self._stop.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    if self.is_file:
                        # End of clip, not a network fault.
                        self.stats.last_error = "end of file"
                        logger.info("%s: end of clip", self.name)
                        break
                    # A few dropped reads are normal on a busy LAN; only give
                    # up on the connection once nothing has arrived for a while.
                    if time.time() - last_ok > self.read_timeout:
                        self.stats.last_error = "read timeout"
                        break
                    time.sleep(0.02)
                    continue
                last_ok = time.time()
                self._publish(frame)

                if frame_period:
                    next_frame_at += frame_period
                    delay = next_frame_at - time.monotonic()
                    if delay > 0:
                        if self._stop.wait(delay):
                            break
                    else:
                        next_frame_at = time.monotonic()

            capture.release()
            self.stats.connected = False
            if self.is_file and not self.loop_file:
                logger.info("%s: clip finished, reader stopping", self.name)
                break
            if not self._stop.is_set():
                self.stats.reconnects += 1
                logger.warning("%s: stream lost (%s), reconnecting", self.name, self.stats.last_error or "eof")
                if self._stop.wait(self.reconnect_min):
                    break

        self.stats.connected = False
        logger.info("%s: reader stopped", self.name)


def grab_single_frame(url: str, timeout: float = 10.0) -> Optional[np.ndarray]:
    """One frame, for /cameras/{id}/test and the ROI editor's snapshot.

    Deliberately not a reader: the API must never hold a long-lived capture.
    """
    capture = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    try:
        capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout * 1000))
    except Exception:
        pass
    try:
        if not capture.isOpened():
            return None
        deadline = time.time() + timeout
        while time.time() < deadline:
            ok, frame = capture.read()
            if ok and frame is not None:
                return frame
            time.sleep(0.05)
        return None
    finally:
        capture.release()
