"""Spawns and supervises the camera worker processes.

Runs as its own process (``anpr supervisor``), separate from FastAPI. Its
whole job is reconciliation: the set of running workers should equal the set
of enabled cameras, and a worker that dies should come back with backoff.

Workers are spawned as subprocesses rather than multiprocessing children so
that a crash in a video decoder cannot corrupt the parent, and so systemd or
Docker can see and signal them normally.
"""
from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Optional

from ..ai.inference.threading import compute_budget, physical_cores

logger = logging.getLogger("anpr.supervisor")

MIN_BACKOFF = 2.0
MAX_BACKOFF = 60.0


@dataclass
class ManagedWorker:
    camera_id: int
    process: subprocess.Popen
    started_at: float
    restarts: int = 0
    backoff: float = MIN_BACKOFF
    restart_at: float = 0.0


@dataclass
class SupervisorStatus:
    cores: int
    intra_op: int
    workers: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"cores": self.cores, "intra_op_per_worker": self.intra_op, "workers": self.workers}


class Supervisor:
    def __init__(self, reconcile_interval: float = 5.0, reserve_cores: int = 1, dry_run: bool = False):
        self.reconcile_interval = reconcile_interval
        self.reserve_cores = reserve_cores
        self.dry_run = dry_run
        self._workers: dict[int, ManagedWorker] = {}
        self._pending: dict[int, float] = {}   # camera_id -> restart_at
        self._backoff: dict[int, float] = {}
        self._stop = False

    # -- desired state -----------------------------------------------------
    def desired_camera_ids(self) -> set[int]:
        """Enabled cameras, read fresh each cycle.

        The supervisor is the only process that queries cameras for this
        purpose; workers receive their config on the command line and never
        poll the database.
        """
        from ..db.session import SessionLocal
        from ..models.camera import Camera

        session = SessionLocal()
        try:
            rows = session.query(Camera.id).filter(Camera.is_active.is_(True)).all()
            return {row[0] for row in rows}
        except Exception:
            logger.exception("could not read the camera list; keeping current workers")
            return set(self._workers) | set(self._pending)
        finally:
            session.close()

    # -- process control ---------------------------------------------------
    def _worker_env(self, n_workers: int) -> dict:
        """Thread budget goes in the environment because OpenMP reads it at
        import time — setting it inside the worker would be too late."""
        budget = compute_budget(n_workers, reserve=self.reserve_cores)
        env = dict(os.environ)
        env.update(budget.as_env)
        env["ANPR_INTRA_OP"] = str(budget.intra_op)
        return env

    def _spawn(self, camera_id: int, n_workers: int) -> None:
        command = [sys.executable, "-m", "backend.app.cli", "worker", "--camera-id", str(camera_id)]
        if self.dry_run:
            command.append("--dry-run")
        logger.info("starting worker for camera %s", camera_id)
        process = subprocess.Popen(command, env=self._worker_env(n_workers))
        self._workers[camera_id] = ManagedWorker(
            camera_id=camera_id, process=process, started_at=time.monotonic()
        )

    def _stop_worker(self, camera_id: int, timeout: float = 12.0) -> None:
        managed = self._workers.pop(camera_id, None)
        if managed is None:
            return
        logger.info("stopping worker for camera %s", camera_id)
        try:
            # SIGTERM so the worker flushes in-flight tracks into events
            # rather than dropping a vehicle that is mid-gate.
            managed.process.terminate()
            managed.process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("worker %s did not exit, killing", camera_id)
            managed.process.kill()
        except Exception:
            logger.debug("error stopping worker %s", camera_id, exc_info=True)

    # -- reconciliation ----------------------------------------------------
    def reconcile(self) -> None:
        desired = self.desired_camera_ids()
        running = set(self._workers)
        now = time.monotonic()

        for camera_id in running - desired:
            self._stop_worker(camera_id)
            self._pending.pop(camera_id, None)

        for camera_id in list(self._pending):
            if camera_id not in desired:
                self._pending.pop(camera_id, None)

        # Restart anything that died, with per-camera exponential backoff so a
        # permanently broken camera does not spin.
        for camera_id, managed in list(self._workers.items()):
            if managed.process.poll() is None:
                continue
            code = managed.process.returncode
            uptime = now - managed.started_at
            logger.warning("worker %s exited (code %s) after %.0fs", camera_id, code, uptime)
            self._workers.pop(camera_id)
            if uptime > 120:
                # It ran fine for a while; treat this as a fresh failure
                # rather than compounding an old backoff.
                self._backoff[camera_id] = MIN_BACKOFF
            else:
                self._backoff[camera_id] = min(self._backoff.get(camera_id, MIN_BACKOFF) * 2, MAX_BACKOFF)
            self._pending[camera_id] = now + self._backoff[camera_id]

        due = [cid for cid, at in self._pending.items() if at <= now]
        to_start = (desired - set(self._workers) - set(self._pending)) | set(due)
        if not to_start:
            return

        total = len(desired) or 1
        for camera_id in sorted(to_start):
            self._pending.pop(camera_id, None)
            self._spawn(camera_id, total)

    def status(self) -> SupervisorStatus:
        cores = physical_cores()
        budget = compute_budget(max(1, len(self._workers)), reserve=self.reserve_cores, cores=cores)
        workers = {
            camera_id: {
                "pid": managed.process.pid,
                "alive": managed.process.poll() is None,
                "uptime_s": round(time.monotonic() - managed.started_at, 1),
            }
            for camera_id, managed in self._workers.items()
        }
        for camera_id, at in self._pending.items():
            workers[camera_id] = {"pid": None, "alive": False, "restart_in_s": round(at - time.monotonic(), 1)}
        return SupervisorStatus(cores=cores, intra_op=budget.intra_op, workers=workers)

    # -- lifecycle ---------------------------------------------------------
    def stop(self, *_args) -> None:
        self._stop = True

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        cores = physical_cores()
        logger.info("supervisor starting on %d physical cores (reserving %d)", cores, self.reserve_cores)
        try:
            while not self._stop:
                try:
                    self.reconcile()
                except Exception:
                    logger.exception("reconcile failed; retrying next cycle")
                time.sleep(self.reconcile_interval)
        finally:
            logger.info("supervisor stopping %d workers", len(self._workers))
            for camera_id in list(self._workers):
                self._stop_worker(camera_id)
