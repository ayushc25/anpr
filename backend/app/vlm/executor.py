"""Run one job against one recognizer.

The only place a ``VlmJob`` meets a ``PlateRecognizer``, shared by the socket
service and the in-process client so both behave identically.

One crop failing does not fail the job: a crop the model cannot read is a normal
outcome (``readable`` False), and one that raises is logged and reported as
unreadable rather than discarding the other crops' answers.
"""
from __future__ import annotations

import logging
import time

from ..ai.plate_recognizer.base import PlateRecognizer
from .protocol import (
    KIND_RECOGNIZE,
    KIND_VERIFY,
    STATUS_ERROR,
    VlmJob,
    VlmRead,
    VlmResult,
)

logger = logging.getLogger("anpr.vlm.executor")


def execute(recognizer: PlateRecognizer, job: VlmJob) -> VlmResult:
    started = time.perf_counter()
    result = VlmResult(job_id=job.job_id, kind=job.kind)

    if job.kind == KIND_RECOGNIZE:
        agreeing: dict[str, int] = {}
        for index, crop in enumerate(job.crops):
            crop_id = job.crop_ids[index] if index < len(job.crop_ids) else index
            read = _read_one(recognizer, crop, crop_id)
            result.reads.append(read)
            # Crops arrive best-first. Once enough of them say the SAME thing,
            # the rest cannot change the outcome much and each costs ~2 s of
            # the one resource this whole architecture is short of. Stopping
            # here is what lets a clear plate cost 3 reads instead of 6.
            if job.stop_after_agree and read.readable:
                agreeing[read.text] = agreeing.get(read.text, 0) + 1
                if agreeing[read.text] >= job.stop_after_agree:
                    break
    elif job.kind == KIND_VERIFY:
        for crop in job.crops:
            try:
                result.answers.append(recognizer.verify(crop, job.candidate))
            except Exception:
                logger.exception("verification failed for job %s", job.job_id)
                result.answers.append(None)
    else:
        result.status = STATUS_ERROR
        result.error = f"unknown job kind {job.kind!r}"

    result.run_ms = (time.perf_counter() - started) * 1000.0
    return result


def _read_one(recognizer: PlateRecognizer, crop, crop_id: int) -> VlmRead:
    started = time.perf_counter()
    try:
        read = recognizer.recognize(crop)
    except Exception:
        logger.exception("recognizer failed on crop %s", crop_id)
        read = None
    latency = (time.perf_counter() - started) * 1000.0
    if read is None:
        return VlmRead(crop_id=crop_id, latency_ms=latency)
    return VlmRead(
        crop_id=crop_id,
        text=read.text,
        raw_text=read.raw_text or read.text,
        confidence=float(read.confidence),
        per_char_confidence=[float(p) for p in read.per_char_confidence],
        per_char_alternatives=[[[c, float(p)] for c, p in position] for position in read.per_char_alternatives],
        latency_ms=latency,
    )
