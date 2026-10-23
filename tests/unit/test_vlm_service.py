"""The VLM job protocol, the clients and the shared service — with a fake model.

Real sockets on loopback, no model: the thing under test is the plumbing
(bounded queues, overflow, expiry, never blocking), not recognition.
"""
from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from backend.app.ai.plate_recognizer.base import PlateRecognizer
from backend.app.ai.types import PlateRead
from backend.app.vlm.client import LocalVlmClient, SocketVlmClient
from backend.app.vlm.executor import execute
from backend.app.vlm.protocol import (
    KIND_RECOGNIZE,
    KIND_VERIFY,
    STATUS_EXPIRED,
    STATUS_OK,
    STATUS_OVERFLOW,
    STATUS_UNAVAILABLE,
    VlmJob,
    decode_job,
    decode_result,
    encode_job,
    encode_result,
)
from backend.app.vlm.server import VlmServer

KEY = b"test-key"


class FakeRecognizer(PlateRecognizer):
    """Reads the grey level of the crop as a 'plate', optionally slowly."""

    name = "fake"

    def __init__(self, delay=0.0):
        self.delay = delay
        self.gate = threading.Event()
        self.gate.set()
        self.calls = 0

    def recognize(self, plate_image):
        self.gate.wait(5)
        time.sleep(self.delay)
        self.calls += 1
        level = int(plate_image.mean())
        if level == 0:
            return None                       # "UNREADABLE"
        return PlateRead(
            text=f"UP32AB{level:04d}", confidence=0.9, per_char_confidence=[0.9] * 10,
            raw_text=f"UP32AB{level:04d}",
            per_char_alternatives=[(), (), (("0", 0.05),), (), (), (), (), (), (), ()],
        )

    @property
    def supports_verification(self):
        return True

    def verify(self, plate_image, candidate):
        self.gate.wait(5)
        return 0.9 if candidate.endswith(f"{int(plate_image.mean()):04d}") else 0.1


def crop(level):
    return np.full((40, 120, 3), level, dtype=np.uint8)


def job(job_id, levels=(50,), kind=KIND_RECOGNIZE, candidate=""):
    return VlmJob(
        job_id=job_id, kind=kind, camera_id=1, track_id=9,
        crops=[crop(v) for v in levels], crop_ids=list(range(len(levels))), candidate=candidate,
    )


def wait_for(client, count, timeout=10.0):
    got = []
    deadline = time.time() + timeout
    while len(got) < count and time.time() < deadline:
        got.extend(client.poll())
        time.sleep(0.02)
    return got


# ---- protocol ---------------------------------------------------------------
def test_job_round_trips_pixel_exact():
    original = job("a", levels=(10, 200), kind=KIND_VERIFY, candidate="UP32AB0010")
    clone = decode_job(encode_job(original))
    assert (clone.job_id, clone.kind, clone.candidate, clone.crop_ids) == ("a", KIND_VERIFY, "UP32AB0010", [0, 1])
    # PNG, not JPEG: the model must be shown the pixels the camera produced.
    assert all(np.array_equal(a, b) for a, b in zip(original.crops, clone.crops))


def test_result_round_trips():
    result = execute(FakeRecognizer(), job("r", levels=(50, 0)))
    clone = decode_result(encode_result(result))
    assert clone.reads[0].text == "UP32AB0050"
    assert clone.reads[0].per_char_alternatives[2] == [["0", 0.05]]
    assert clone.reads[1].readable is False


def test_one_unreadable_crop_does_not_fail_the_job():
    result = execute(FakeRecognizer(), job("r", levels=(0, 70)))
    assert result.status == STATUS_OK
    assert [r.readable for r in result.reads] == [False, True]


def test_a_crashing_crop_is_reported_unreadable_not_fatal():
    class Boom(FakeRecognizer):
        def recognize(self, plate_image):
            if int(plate_image.mean()) == 99:
                raise RuntimeError("model fell over")
            return super().recognize(plate_image)

    result = execute(Boom(), job("r", levels=(99, 70)))
    assert result.status == STATUS_OK
    assert [r.readable for r in result.reads] == [False, True]


# ---- local client -----------------------------------------------------------
def test_local_client_answers_and_bounds_its_queue():
    recognizer = FakeRecognizer()
    recognizer.gate.clear()                         # hold the worker thread
    client = LocalVlmClient(recognizer, max_pending=2)
    try:
        assert client.submit(job("1"))              # picked up by the thread, blocked
        time.sleep(0.2)
        assert client.submit(job("2"))
        assert client.submit(job("3"))
        assert client.submit(job("4")) is False     # full: refused, not queued, not blocking
        recognizer.gate.set()
        results = wait_for(client, 3)
        assert sorted(r.job_id for r in results) == ["1", "2", "3"]
        assert all(r.ok for r in results)
    finally:
        client.close()


def test_submit_and_poll_never_block():
    recognizer = FakeRecognizer(delay=0.5)
    client = LocalVlmClient(recognizer, max_pending=4)
    try:
        started = time.perf_counter()
        client.submit(job("1"))
        client.poll()
        assert time.perf_counter() - started < 0.1
    finally:
        client.close()


# ---- socket service ---------------------------------------------------------
@pytest.fixture
def service():
    recognizer = FakeRecognizer()
    server = VlmServer(recognizer, host="127.0.0.1", port=0, authkey=KEY, queue_size=2, max_job_age_s=60)
    server.start()
    try:
        yield server, recognizer
    finally:
        server.stop()


def connect(server, **kw):
    return SocketVlmClient(*server.address, authkey=KEY, **kw)


def test_socket_round_trip(service):
    server, _ = service
    client = connect(server)
    try:
        assert client.submit(job("s1", levels=(40, 80)))
        (result,) = wait_for(client, 1)
        assert result.ok and [r.text for r in result.reads] == ["UP32AB0040", "UP32AB0080"]
        assert client.submit(job("s2", kind=KIND_VERIFY, candidate="UP32AB0050"))
        (verdict,) = wait_for(client, 1)
        assert verdict.answers == [0.9]
    finally:
        client.close()


def test_two_cameras_share_one_service(service):
    server, recognizer = service
    first, second = connect(server), connect(server)
    try:
        first.submit(job("c1", levels=(30,)))
        second.submit(job("c2", levels=(60,)))
        a, b = wait_for(first, 1), wait_for(second, 1)
        assert a[0].job_id == "c1" and b[0].job_id == "c2"
        assert recognizer.calls == 2                # ONE model served both
    finally:
        first.close()
        second.close()


def test_service_refuses_when_its_queue_is_full(service):
    server, recognizer = service
    recognizer.gate.clear()                         # inference thread stalls on the first job
    client = connect(server, max_pending=20)
    try:
        for i in range(6):                          # 1 running + 2 queued + 3 refused
            assert client.submit(job(f"q{i}"))
        refused = []
        deadline = time.time() + 5
        while len(refused) < 3 and time.time() < deadline:
            refused += [r for r in client.poll() if r.status == STATUS_OVERFLOW]
            time.sleep(0.02)
        assert len(refused) == 3                    # reported, not silently dropped
        recognizer.gate.set()
        done = wait_for(client, 3)
        assert all(r.ok for r in done)
    finally:
        client.close()


def test_client_refuses_past_its_own_bound(service):
    server, recognizer = service
    recognizer.gate.clear()
    client = connect(server, max_pending=2)
    try:
        assert client.submit(job("a")) and client.submit(job("b"))
        assert client.submit(job("c")) is False
        recognizer.gate.set()
    finally:
        client.close()


def test_stale_job_is_discarded_unrun():
    recognizer = FakeRecognizer()
    server = VlmServer(recognizer, host="127.0.0.1", port=0, authkey=KEY, queue_size=4, max_job_age_s=0.2)
    server.start()
    client = connect(server)
    try:
        recognizer.gate.clear()
        client.submit(job("slow"))
        time.sleep(0.15)
        client.submit(job("stale"))
        time.sleep(0.5)                              # 'stale' outlives max_job_age_s in the queue
        recognizer.gate.set()
        results = {r.job_id: r for r in wait_for(client, 2)}
        assert results["slow"].ok
        assert results["stale"].status == STATUS_EXPIRED
        assert recognizer.calls == 1                 # never spent CPU on a dead job
    finally:
        client.close()
        server.stop()


def test_unreachable_service_fails_jobs_fast_not_slowly():
    client = SocketVlmClient("127.0.0.1", 1, authkey=KEY, max_pending=4, reconnect_interval=0.1)
    try:
        started = time.time()
        assert client.submit(job("x"))
        (result,) = wait_for(client, 1, timeout=5)
        assert result.status == STATUS_UNAVAILABLE
        assert time.time() - started < 3
    finally:
        client.close()


def test_wrong_authkey_is_refused(service):
    server, _ = service
    client = SocketVlmClient(*server.address, authkey=b"wrong", max_pending=2, reconnect_interval=0.1)
    try:
        client.submit(job("x"))
        (result,) = wait_for(client, 1, timeout=5)
        assert result.status == STATUS_UNAVAILABLE
    finally:
        client.close()


# ---- early stop -------------------------------------------------------------
def test_early_stop_skips_the_remaining_crops_once_enough_agree():
    """The service is the bottleneck, so a clear plate must not cost six reads."""
    recognizer = FakeRecognizer()
    request = job("e", levels=(50, 50, 50, 50, 50, 50))
    request.stop_after_agree = 3
    result = execute(recognizer, request)
    assert recognizer.calls == 3
    assert len(result.reads) == 3 and all(r.text == "UP32AB0050" for r in result.reads)


def test_early_stop_does_not_trigger_on_disagreement():
    recognizer = FakeRecognizer()
    request = job("e", levels=(10, 20, 30, 10, 20, 30))      # never three alike until the end
    request.stop_after_agree = 3
    result = execute(recognizer, request)
    assert recognizer.calls == 6 and len(result.reads) == 6


def test_unreadable_crops_do_not_count_towards_agreement():
    recognizer = FakeRecognizer()
    request = job("e", levels=(0, 0, 0, 50, 50, 50))
    request.stop_after_agree = 3
    assert len(execute(recognizer, request).reads) == 6


def test_early_stop_disabled_reads_everything():
    recognizer = FakeRecognizer()
    request = job("e", levels=(50, 50, 50, 50))
    assert request.stop_after_agree == 0
    assert len(execute(recognizer, request).reads) == 4 and recognizer.calls == 4


def test_the_setting_survives_the_wire():
    request = job("e", levels=(50,))
    request.stop_after_agree = 3
    assert decode_job(encode_job(request)).stop_after_agree == 3
