"""How a camera worker talks to the VLM.

The contract, and the only thing the pipeline relies on:

  * ``submit`` NEVER blocks. It returns False when there is no room, and the
    caller turns that into a class B / "queue overflow" result. A full queue is
    reported, not waited on and not silently dropped.
  * ``poll`` NEVER blocks. It returns whatever has finished since the last call.
  * Every submitted job produces exactly one result eventually — a real answer,
    or a failure status (unavailable, overflow, expired, error). The client's
    own deadline is the caller's job (``DeferredFinalizer``), because only the
    caller knows what the job was for.

Two implementations:

``SocketVlmClient``  production. Talks to the one shared service process.
``LocalVlmClient``   runs a recognizer on a thread in THIS process. For tests,
                     and for a single-camera dev box where a second process is
                     more ceremony than it is worth. It loads its own model, so
                     it is NOT what you run with several cameras.
"""
from __future__ import annotations

import collections
import logging
import queue
import threading
import time
from abc import ABC, abstractmethod
from multiprocessing.connection import Client
from typing import Optional

from ..ai.plate_recognizer.base import PlateRecognizer
from .executor import execute
from .protocol import (
    STATUS_OVERFLOW,
    STATUS_UNAVAILABLE,
    VlmJob,
    VlmResult,
    decode_result,
    encode_job,
    failure,
)

logger = logging.getLogger("anpr.vlm.client")


class VlmClient(ABC):
    @abstractmethod
    def submit(self, job: VlmJob) -> bool:
        """Queue a job. False means refused (no room); nothing was queued."""

    @abstractmethod
    def poll(self) -> list[VlmResult]:
        """Results finished since the last call. Never blocks."""

    @property
    @abstractmethod
    def depth(self) -> int:
        """Jobs submitted and not yet answered."""

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------
class LocalVlmClient(VlmClient):
    """Bounded queue plus one worker thread around an in-process recognizer."""

    def __init__(self, recognizer: PlateRecognizer, max_pending: int = 8):
        self._recognizer = recognizer
        self._jobs: "queue.Queue[tuple[VlmJob, float]]" = queue.Queue(maxsize=max(1, max_pending))
        self._results: collections.deque[VlmResult] = collections.deque()
        self._lock = threading.Lock()
        self._outstanding = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vlm-local", daemon=True)
        self._thread.start()

    def submit(self, job: VlmJob) -> bool:
        try:
            self._jobs.put_nowait((job, time.perf_counter()))
        except queue.Full:
            return False
        with self._lock:
            self._outstanding += 1
        return True

    def poll(self) -> list[VlmResult]:
        with self._lock:
            out = list(self._results)
            self._results.clear()
        return out

    @property
    def depth(self) -> int:
        with self._lock:
            return self._outstanding

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                job, enqueued = self._jobs.get(timeout=0.1)
            except queue.Empty:
                continue
            result = execute(self._recognizer, job)
            result.queue_ms = (time.perf_counter() - enqueued) * 1000.0 - result.run_ms
            with self._lock:
                self._results.append(result)
                self._outstanding -= 1


# --------------------------------------------------------------------------
class SocketVlmClient(VlmClient):
    """One persistent connection to the VLM service, managed on its own thread.

    The pipeline thread only touches two in-memory collections under a lock;
    every socket operation, including connect, happens on the I/O thread, so a
    dead or slow service cannot stall frame processing.
    """

    def __init__(
        self,
        host: str,
        port: int,
        authkey: bytes,
        max_pending: int = 8,
        reconnect_interval: float = 2.0,
    ):
        self._address = (host, int(port))
        self._authkey = authkey
        self._max_pending = max(1, max_pending)
        self._reconnect_interval = reconnect_interval

        self._lock = threading.Lock()
        self._outbox: collections.deque[VlmJob] = collections.deque()
        self._inflight: dict[str, VlmJob] = {}
        self._results: collections.deque[VlmResult] = collections.deque()

        self._conn = None
        self._next_connect_at = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="vlm-client", daemon=True)
        self._thread.start()

    # -- pipeline-facing ---------------------------------------------------
    def submit(self, job: VlmJob) -> bool:
        with self._lock:
            if len(self._outbox) + len(self._inflight) >= self._max_pending:
                return False
            self._outbox.append(job)
        return True

    def poll(self) -> list[VlmResult]:
        with self._lock:
            out = list(self._results)
            self._results.clear()
        return out

    @property
    def depth(self) -> int:
        with self._lock:
            return len(self._outbox) + len(self._inflight)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3.0)
        self._drop_connection()

    # -- I/O thread --------------------------------------------------------
    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("vlm client loop error")
                self._drop_connection()
                time.sleep(0.2)

    def _tick(self) -> None:
        if self._conn is None:
            if not self._connect():
                # Nobody to send to. Fail what is queued NOW so the pipeline
                # sees "unavailable" in a poll or two instead of waiting out
                # its deadline for an answer that cannot come.
                self._fail_all(STATUS_UNAVAILABLE, "vlm service not reachable")
                self._stop.wait(0.25)
                return

        self._send_outbox()
        if self._conn is not None and self._conn.poll(0.05):
            payload = self._conn.recv_bytes()
            result = decode_result(payload)
            with self._lock:
                self._inflight.pop(result.job_id, None)
                self._results.append(result)
        elif not self._outbox:
            self._stop.wait(0.02)

    def _connect(self) -> bool:
        now = time.monotonic()
        if now < self._next_connect_at:
            return False
        try:
            self._conn = Client(self._address, authkey=self._authkey)
            logger.info("connected to the VLM service at %s:%d", *self._address)
            return True
        except Exception:
            self._conn = None
            self._next_connect_at = now + self._reconnect_interval
            return False

    def _send_outbox(self) -> None:
        while True:
            with self._lock:
                if not self._outbox:
                    return
                job = self._outbox.popleft()
                self._inflight[job.job_id] = job
            try:
                self._conn.send_bytes(encode_job(job))
            except (OSError, EOFError, ConnectionError):
                self._drop_connection()
                return

    def _drop_connection(self) -> None:
        conn, self._conn = self._conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
        self._fail_inflight(STATUS_UNAVAILABLE, "connection to the vlm service was lost")

    def _fail_inflight(self, status: str, error: str) -> None:
        with self._lock:
            for job in self._inflight.values():
                self._results.append(failure(job.job_id, job.kind, status, error))
            self._inflight.clear()

    def _fail_all(self, status: str, error: str) -> None:
        self._fail_inflight(status, error)
        with self._lock:
            while self._outbox:
                job = self._outbox.popleft()
                self._results.append(failure(job.job_id, job.kind, status, error))


def overflow_result(job: VlmJob) -> VlmResult:
    return failure(job.job_id, job.kind, STATUS_OVERFLOW, "client queue full")


def build_client(
    mode: str,
    host: str,
    port: int,
    authkey: bytes,
    max_pending: int,
    recognizer: Optional[PlateRecognizer] = None,
) -> VlmClient:
    if mode == "local":
        if recognizer is None:
            raise ValueError("vlm client mode 'local' needs a recognizer")
        return LocalVlmClient(recognizer, max_pending=max_pending)
    return SocketVlmClient(host, port, authkey, max_pending=max_pending)
