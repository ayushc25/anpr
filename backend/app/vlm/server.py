"""The one shared VLM service.

One process, one model, any number of camera workers connected to it. Memory is
the reason: Qwen3-VL-2B INT4 is ~2 GB resident, and a model per camera would
turn a four-camera box into an eight-gigabyte one.

Three kinds of thread, deliberately kept apart:

  accept     takes new connections
  handler    one per connection; reads a job, answers "overflow" immediately
             if the bounded queue is full, otherwise enqueues. It never runs
             inference, so a full queue is reported in microseconds and the
             client is never left waiting on a refusal.
  inference  exactly one. A single worker thread is the point: the CPU is the
             bottleneck, a second concurrent generation would just halve the
             speed of both and double the memory.

A job that has waited longer than ``max_job_age_s`` by the time it reaches the
front is discarded UNRUN and answered "expired". Its camera has already given
up on it (``DeferredFinalizer`` has its own deadline), and running it would
spend seconds of CPU producing an answer nobody is listening for — the way a
backlog turns into a permanent lag.
"""
from __future__ import annotations

import logging
import queue
import signal
import threading
import time
from multiprocessing.connection import Listener
from typing import Optional

from ..ai.plate_recognizer.base import PlateRecognizer
from .executor import execute
from .protocol import (
    STATUS_ERROR,
    STATUS_EXPIRED,
    STATUS_OVERFLOW,
    VlmResult,
    decode_job,
    encode_result,
    failure,
)

logger = logging.getLogger("anpr.vlm.server")


class _Reply:
    """A connection plus the lock that serializes writes to it (the handler
    thread answers overflows while the inference thread answers results)."""

    def __init__(self, conn):
        self.conn = conn
        self.lock = threading.Lock()
        self.alive = True

    def send(self, result: VlmResult) -> None:
        if not self.alive:
            return
        try:
            with self.lock:
                self.conn.send_bytes(encode_result(result))
        except (OSError, EOFError, ConnectionError):
            self.alive = False


class VlmServer:
    def __init__(
        self,
        recognizer: PlateRecognizer,
        host: str = "127.0.0.1",
        port: int = 8765,
        authkey: bytes = b"anpr-vlm",
        queue_size: int = 8,
        max_job_age_s: float = 90.0,
    ):
        self.recognizer = recognizer
        self.address = (host, int(port))
        self.authkey = authkey
        self.max_job_age_s = max_job_age_s
        self._queue: "queue.Queue[tuple[_Reply, object, float]]" = queue.Queue(maxsize=max(1, queue_size))
        self._stop = threading.Event()
        self._listener: Optional[Listener] = None
        self._threads: list[threading.Thread] = []
        self.stats = {"jobs_run": 0, "jobs_overflow": 0, "jobs_expired": 0, "jobs_failed": 0}

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        self._listener = Listener(self.address, authkey=self.authkey)
        # The port may have been 0 (tests): report the one actually bound.
        self.address = self._listener.address
        for target, name in ((self._accept_loop, "vlm-accept"), (self._inference_loop, "vlm-infer")):
            thread = threading.Thread(target=target, name=name, daemon=True)
            thread.start()
            self._threads.append(thread)
        logger.info("VLM service listening on %s:%d (queue=%d)", *self.address, self._queue.maxsize)

    def stop(self) -> None:
        self._stop.set()
        if self._listener is not None:
            try:
                self._listener.close()
            except Exception:
                pass

    def serve_forever(self) -> None:
        self.start()
        signal.signal(signal.SIGINT, lambda *_: self.stop())
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        last_report = time.monotonic()
        while not self._stop.is_set():
            time.sleep(0.5)
            if time.monotonic() - last_report > 60:
                last_report = time.monotonic()
                logger.info("vlm service: depth=%d %s", self._queue.qsize(), self.stats)
        logger.info("VLM service stopped: %s", self.stats)

    # -- threads -----------------------------------------------------------
    def _accept_loop(self) -> None:
        while not self._stop.is_set():
            try:
                conn = self._listener.accept()
            except Exception:
                if self._stop.is_set():
                    return
                # A failed handshake (wrong authkey) raises here; keep serving.
                logger.warning("rejected a connection to the VLM service", exc_info=True)
                continue
            thread = threading.Thread(target=self._handle, args=(conn,), name="vlm-conn", daemon=True)
            thread.start()

    def _handle(self, conn) -> None:
        reply = _Reply(conn)
        while not self._stop.is_set() and reply.alive:
            try:
                payload = conn.recv_bytes()
            except (EOFError, OSError, ConnectionError):
                break
            try:
                job = decode_job(payload)
            except Exception as exc:
                logger.warning("undecodable job: %s", exc)
                self.stats["jobs_failed"] += 1
                continue
            try:
                self._queue.put_nowait((reply, job, time.monotonic()))
            except queue.Full:
                self.stats["jobs_overflow"] += 1
                reply.send(failure(job.job_id, job.kind, STATUS_OVERFLOW, "vlm service queue full"))
        reply.alive = False
        try:
            conn.close()
        except Exception:
            pass

    def _inference_loop(self) -> None:
        while not self._stop.is_set():
            try:
                reply, job, enqueued = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue
            waited = time.monotonic() - enqueued
            if self.max_job_age_s > 0 and waited > self.max_job_age_s:
                self.stats["jobs_expired"] += 1
                reply.send(failure(job.job_id, job.kind, STATUS_EXPIRED, f"waited {waited:.0f}s"))
                continue
            try:
                result = execute(self.recognizer, job)
            except Exception as exc:
                logger.exception("job %s failed", job.job_id)
                self.stats["jobs_failed"] += 1
                result = failure(job.job_id, job.kind, STATUS_ERROR, str(exc))
            result.queue_ms = waited * 1000.0
            self.stats["jobs_run"] += 1
            reply.send(result)


def run_service(recognizer: PlateRecognizer, host: str, port: int, authkey: bytes,
                queue_size: int, max_job_age_s: float) -> None:
    VlmServer(recognizer, host, port, authkey, queue_size, max_job_age_s).serve_forever()
