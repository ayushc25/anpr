"""Job and result records, and their wire format.

Pure data plus (de)serialization — no sockets, no model — so the same records
flow unchanged through the in-process client used in tests and the socket
client used in production.

WIRE FORMAT
-----------
One ``send_bytes`` message per job:  ``[u32 header length][header JSON][PNG][PNG]...``
One JSON ``send_bytes`` message per result.

``multiprocessing.connection`` offers ``send()``/``recv()`` too, but those
pickle, and unpickling bytes from a socket is code execution for whoever can
reach it. The authkey handshake already restricts who that is; there is still
no reason to leave the door open when JSON and PNG carry everything needed.
Crops travel as PNG, not JPEG: a lossy re-encode would change the very pixels
the model is being asked to read.
"""
from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

KIND_RECOGNIZE = "recognize"
KIND_VERIFY = "verify"

STATUS_OK = "ok"
#: The service's queue was full: refused at the door, nothing was run.
STATUS_OVERFLOW = "overflow"
#: Nobody to talk to: service down, connection lost.
STATUS_UNAVAILABLE = "unavailable"
#: The job sat in the queue past its useful life and was discarded unrun.
STATUS_EXPIRED = "expired"
#: Raised by the CLIENT when its own deadline passes.
STATUS_TIMEOUT = "timeout"
STATUS_ERROR = "error"


@dataclass
class VlmJob:
    job_id: str
    kind: str
    camera_id: int
    track_id: int
    crops: list[np.ndarray]
    #: One tag per crop, echoed back so a result can be matched to the
    #: observation it came from without relying on order.
    crop_ids: list[int] = field(default_factory=list)
    #: Verification only: the plate string being checked.
    candidate: str = ""
    submitted_at: float = field(default_factory=time.time)
    #: Recognize only: stop reading this vehicle's remaining crops once this
    #: many reads agree exactly. 0 reads every crop. See executor.execute.
    stop_after_agree: int = 0


@dataclass
class VlmRead:
    crop_id: int
    #: Empty when the model said UNREADABLE (or produced nothing usable).
    text: str = ""
    raw_text: str = ""
    confidence: float = 0.0
    per_char_confidence: list[float] = field(default_factory=list)
    per_char_alternatives: list[list[list]] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def readable(self) -> bool:
        return bool(self.text)


@dataclass
class VlmResult:
    job_id: str
    kind: str
    status: str = STATUS_OK
    reads: list[VlmRead] = field(default_factory=list)
    #: Verification only: P(yes) per crop, None where the model gave no answer.
    answers: list[Optional[float]] = field(default_factory=list)
    error: str = ""
    queue_ms: float = 0.0
    run_ms: float = 0.0

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK


# -- encoding --------------------------------------------------------------
def encode_job(job: VlmJob) -> bytes:
    blobs: list[bytes] = []
    for crop in job.crops:
        ok, buffer = cv2.imencode(".png", crop)
        if not ok:
            raise ValueError("could not encode a crop for the VLM service")
        blobs.append(buffer.tobytes())
    header = json.dumps(
        {
            "job_id": job.job_id,
            "kind": job.kind,
            "camera_id": job.camera_id,
            "track_id": job.track_id,
            "crop_ids": job.crop_ids,
            "candidate": job.candidate,
            "submitted_at": job.submitted_at,
            "stop_after_agree": job.stop_after_agree,
            "sizes": [len(b) for b in blobs],
        }
    ).encode("utf-8")
    return struct.pack(">I", len(header)) + header + b"".join(blobs)


def decode_job(payload: bytes) -> VlmJob:
    (header_len,) = struct.unpack(">I", payload[:4])
    header = json.loads(payload[4 : 4 + header_len].decode("utf-8"))
    offset = 4 + header_len
    crops: list[np.ndarray] = []
    for size in header["sizes"]:
        raw = np.frombuffer(payload[offset : offset + size], dtype=np.uint8)
        offset += size
        image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("undecodable crop in a VLM job")
        crops.append(image)
    return VlmJob(
        job_id=header["job_id"],
        kind=header["kind"],
        camera_id=int(header["camera_id"]),
        track_id=int(header["track_id"]),
        crops=crops,
        crop_ids=list(header.get("crop_ids", [])),
        candidate=header.get("candidate", ""),
        submitted_at=float(header.get("submitted_at", time.time())),
        stop_after_agree=int(header.get("stop_after_agree", 0)),
    )


def encode_result(result: VlmResult) -> bytes:
    return json.dumps(
        {
            "job_id": result.job_id,
            "kind": result.kind,
            "status": result.status,
            "reads": [vars(r) for r in result.reads],
            "answers": result.answers,
            "error": result.error,
            "queue_ms": result.queue_ms,
            "run_ms": result.run_ms,
        }
    ).encode("utf-8")


def decode_result(payload: bytes) -> VlmResult:
    data = json.loads(payload.decode("utf-8"))
    return VlmResult(
        job_id=data["job_id"],
        kind=data["kind"],
        status=data.get("status", STATUS_OK),
        reads=[VlmRead(**r) for r in data.get("reads", [])],
        answers=list(data.get("answers", [])),
        error=data.get("error", ""),
        queue_ms=float(data.get("queue_ms", 0.0)),
        run_ms=float(data.get("run_ms", 0.0)),
    )


def failure(job_id: str, kind: str, status: str, error: str = "") -> VlmResult:
    return VlmResult(job_id=job_id, kind=kind, status=status, error=error)
