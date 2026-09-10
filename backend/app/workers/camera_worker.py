"""One camera, one OS process.

Structure of the loop, and why each part is where it is:

  * The reader runs on its own thread with a latest-frame policy, so decode
    never blocks the pipeline and the pipeline never processes stale frames.
  * The pacer takes frames at ``processing_fps`` rather than as fast as
    possible: running flat out would consume the whole box for no accuracy
    gain, since consecutive frames of a slow-moving vehicle are nearly
    identical.
  * Submission goes through the ingest client, which spools to disk on
    failure. Nothing in this loop may block on the network.
"""
from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2

from ..ai.inference.threading import ThreadBudget, configure_opencv
from ..ai.registry import build_plate_detector, build_plate_recognizer, build_tracker, build_vehicle_detector
from ..debug.recorder import DebugConfig, DebugRecorder, NullRecorder
from ..events.dedupe import DedupeConfig, EventDeduplicator
from ..events.event_builder import EventBuilder
from ..events.multi_frame_validator import MultiFrameValidator, SnapshotRegistry, ValidationConfig
from ..video.frame_processor import FrameProcessor, PipelineConfig
from ..video.line_crossing import CrossDirection, VirtualLine
from ..video.roi import RoiPolygon
from ..video.rtsp_reader import RtspReader

logger = logging.getLogger("anpr.worker")


@dataclass
class CameraConfig:
    """Everything a worker needs, resolved from the database by the supervisor
    so the worker itself never opens a session."""

    id: int
    code: str
    name: str
    rtsp_url: str
    rtsp_url_sub: str = ""
    direction: str = "both"
    processing_fps: float = 6.0
    roi: list = field(default_factory=list)
    #: Optional near-field band inside the ROI where a plate on this camera is
    #: actually large enough to read. Empty means "the whole ROI", which is
    #: how every camera behaves until an operator calibrates one.
    read_zone: list = field(default_factory=list)
    line: list = field(default_factory=list)
    line_forward: str = "in"
    pipeline_overrides: dict = field(default_factory=dict)

    @property
    def stream_url(self) -> str:
        """The sub-stream is the AI path when the camera exposes one.

        Decoding 720p instead of 1080p is the cheapest large saving available,
        and at gate distance it costs no measurable accuracy.
        """
        return self.rtsp_url_sub or self.rtsp_url


@dataclass
class WorkerHealth:
    camera_id: int
    connected: bool = False
    stream_fps: float = 0.0
    processed_fps: float = 0.0
    last_frame_at: float = 0.0
    frames_processed: int = 0
    events_emitted: int = 0
    events_suppressed: int = 0
    reconnects: int = 0
    spool_depth: int = 0
    avg_stage_ms: dict = field(default_factory=dict)
    last_error: str = ""
    #: Plates localized vs. plates actually recognized. The ratio is the OCR
    #: scheduler's saving, and the first number to look at when tuning a read
    #: zone: all observation and no OCR means the zone or the readable-width
    #: floor is too strict for this mounting.
    plates_observed: int = 0
    ocr_calls: int = 0
    ocr_skipped: int = 0
    #: Debug recorder state, so an operator can see it is on and how much it
    #: has written. Empty when disabled.
    debug: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "connected": self.connected,
            "stream_fps": round(self.stream_fps, 2),
            "processed_fps": round(self.processed_fps, 2),
            "last_frame_at": self.last_frame_at,
            "frames_processed": self.frames_processed,
            "events_emitted": self.events_emitted,
            "events_suppressed": self.events_suppressed,
            "plates_observed": self.plates_observed,
            "ocr_calls": self.ocr_calls,
            "ocr_skipped": self.ocr_skipped,
            "reconnects": self.reconnects,
            "spool_depth": self.spool_depth,
            "avg_stage_ms": {k: round(v, 2) for k, v in self.avg_stage_ms.items()},
            "last_error": self.last_error,
            "debug": self.debug,
        }


class CameraWorker:
    def __init__(
        self,
        camera: CameraConfig,
        pipeline_cfg: PipelineConfig,
        validation_cfg: ValidationConfig,
        dedupe_cfg: DedupeConfig,
        model_cfgs: dict,
        threads: ThreadBudget,
        ingest,
        registry: SnapshotRegistry | None = None,
        cache_dir=None,
        reader_kwargs: dict | None = None,
        debug_cfg: DebugConfig | None = None,
        debug_root=None,
    ):
        self.camera = camera
        self.ingest = ingest
        self.health = WorkerHealth(camera_id=camera.id)
        self._stop = False
        self._preview_jpeg: Optional[bytes] = None
        self._stage_totals: dict[str, float] = {}
        self._processed_window: list[float] = []

        debug_cfg = debug_cfg or DebugConfig()
        self.recorder = (
            DebugRecorder(debug_cfg, camera.id, debug_root)
            if debug_cfg.enabled
            else NullRecorder()
        )

        configure_opencv(threads)

        logger.info("camera %s: loading models (intra_op=%d)", camera.code, threads.intra_op)
        self.detector = build_vehicle_detector(model_cfgs["vehicle_detector"], threads, cache_dir)
        self.tracker = build_tracker(model_cfgs["tracker"])
        self.plate_detector = build_plate_detector(model_cfgs["plate_detector"], threads, cache_dir)
        self.recognizer = build_plate_recognizer(model_cfgs["plate_recognizer"], threads, cache_dir)

        # Warm up before the first vehicle arrives: on the OpenVINO EP the
        # first inference pays for graph compilation, which would otherwise
        # land on a real plate.
        for component in (self.detector, self.plate_detector, self.recognizer):
            latency = component.warmup(2)
            logger.info("camera %s: %s warm-up %.1f ms", camera.code, component.name, latency)

        self.processor = FrameProcessor(
            detector=self.detector,
            tracker=self.tracker,
            plate_detector=self.plate_detector,
            recognizer=self.recognizer,
            roi=RoiPolygon.from_config(camera.roi),
            line=VirtualLine.from_config(camera.line, camera.line_forward),
            cfg=pipeline_cfg,
            camera_id=camera.id,
            read_zone=RoiPolygon.from_config(camera.read_zone, name="read_zone"),
            recorder=self.recorder,
        )
        self.validator = MultiFrameValidator(validation_cfg, registry)
        self.dedupe = EventDeduplicator(dedupe_cfg)
        self.builder = EventBuilder(camera_id=camera.id)

        self.reader = RtspReader(camera.stream_url, name=camera.code, **(reader_kwargs or {}))
        self._frame_interval = 1.0 / max(0.5, camera.processing_fps)
        self._preview_wanted = False

    # -- lifecycle ---------------------------------------------------------
    def stop(self, *_args) -> None:
        self._stop = True

    def run(self) -> None:
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)

        self.reader.start()
        logger.info("camera %s: worker started (%.1f fps target)", self.camera.code, self.camera.processing_fps)
        next_frame_at = time.monotonic()

        try:
            while not self._stop:
                now = time.monotonic()
                if now < next_frame_at:
                    time.sleep(min(0.02, next_frame_at - now))
                    continue
                next_frame_at = now + self._frame_interval

                self._sync_health()
                frame, captured_at = self.reader.latest_with_timestamp()
                if frame is None:
                    # No new frame: the stream is stalled or slower than our
                    # target rate. Either way there is nothing to process.
                    self.ingest.drain_spool()
                    continue

                self._process(frame, captured_at)
        finally:
            self._shutdown()

    def _process(self, frame, captured_at: float = 0.0) -> None:
        wall = time.time()
        # Frame time is the CAPTURE time where we have one. It is what the
        # event timestamp should say, and the gap between it and `wall` is how
        # far behind the pipeline is running — which is what tells the OCR
        # scheduler whether this frame is still worth its most expensive call.
        frame_ts = captured_at or wall
        frame_age_ms = (wall - captured_at) * 1000.0 if captured_at else None
        try:
            result = self.processor.process(frame, ts=frame_ts, frame_age_ms=frame_age_ms)
        except Exception:
            logger.exception("camera %s: pipeline failed on a frame", self.camera.code)
            self.health.last_error = "pipeline error"
            return

        self.health.frames_processed += 1
        self._record_timing(result.stats)

        for state in result.finalized:
            self._finalize(state, wall)

        if self._preview_wanted:
            self._render_preview(frame, result.tracks)

    def _finalize(self, state, wall: float) -> None:
        # One last, bounded attempt on crops the track banked but never read —
        # only when the plate is a character or two from settled. Runs before
        # validation so the extra reads join the same ballot.
        self.processor.resolve_pending(state, wall)

        final = self.validator.validate(state)
        if final is None:
            return

        direction = state.direction
        if direction == CrossDirection.NONE:
            # No line configured, or the track never crossed it: fall back to
            # the camera's declared role so the event is still usable.
            declared = self.camera.direction
            direction = CrossDirection(declared) if declared in ("in", "out") else CrossDirection.NONE

        if not self.dedupe.check_and_record(final.text, direction, wall):
            self.health.events_suppressed += 1
            logger.debug("camera %s: suppressed duplicate %s", self.camera.code, final.text)
            return

        draft = self.builder.build(state, final, direction)
        if self.recorder.enabled:
            # The final verdict, filed next to the frames that produced it —
            # so an error can be traced from the resolved plate back to the
            # exact crop and per-character probabilities behind it.
            self.recorder.record_resolution(state.track_id, {
                "camera_id": self.camera.id,
                "track_id": state.track_id,
                "plate_number": final.text,
                "plate_raw": draft.plate_raw,
                "recognition_state": final.state.value,
                "anpr_confidence": final.confidence,
                "ocr_confidence": final.ocr_confidence,
                "char_support": final.char_support,
                "weakest_char_posterior": final.weakest_char_posterior,
                "support": final.support,
                "unstable_positions": final.unstable_positions,
                "cap_reason": final.cap_reason,
                "grammar_valid": final.grammar_valid,
                "corrections": final.corrections,
                "read_count": final.read_count,
                "distinct_variants": final.distinct_variants,
                "ocr_calls": state.ocr_calls,
                "observations": state.observation_count,
                "retained_observations": len(state.plate_observations),
                "retry_positions": state.retry_positions,
                "finalize_reason": state.finalize_reason,
                "direction": draft.direction,
                "reads": [
                    {
                        "raw_text": r.raw_text,
                        "normalized_text": r.text,
                        "rec_confidence": round(r.rec_confidence, 4),
                        "quality": round(r.quality, 4),
                        "weight": round(r.weight, 5),
                        "frame_idx": r.frame_idx,
                        "corrections": r.corrections,
                    }
                    for r in state.reads
                ],
            })
        self.ingest.submit(draft)
        self.health.events_emitted += 1
        logger.info(
            "camera %s: %s %s conf=%.2f support=%.2f reads=%d (%s) "
            "media=%s vehicle@f%d plate@f%d win=[%.1f..%.1f] track=%s",
            self.camera.code, final.text, draft.direction, final.confidence,
            final.support, final.read_count, state.finalize_reason,
            draft.media_source, draft.vehicle_image_frame, draft.plate_image_frame,
            draft.winning_window[0], draft.winning_window[1], state.track_id,
        )

    # -- preview -----------------------------------------------------------
    def set_preview_enabled(self, enabled: bool) -> None:
        """Preview JPEGs are only encoded while someone is watching. The
        prototype encoded every decoded frame regardless, which on a four
        camera box is a whole core spent on nobody."""
        self._preview_wanted = enabled

    def _render_preview(self, frame, tracks) -> None:
        try:
            canvas = self.processor.draw_overlay(frame.copy(), tracks)
            ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
            if ok:
                self._preview_jpeg = buffer.tobytes()
        except Exception:
            logger.debug("preview render failed", exc_info=True)

    @property
    def preview_jpeg(self) -> Optional[bytes]:
        return self._preview_jpeg

    # -- health ------------------------------------------------------------
    def _record_timing(self, stats) -> None:
        self.health.plates_observed += stats.plates_observed
        self.health.ocr_calls += stats.ocr_run
        self.health.debug = self.recorder.stats
        self.health.ocr_skipped += (
            stats.ocr_waited + stats.ocr_skipped_small
            + stats.ocr_skipped_budget + stats.ocr_skipped_stale
        )

        timing = stats.timing
        for key in ("vehicle_detect", "track", "plate_detect", "recognize", "total"):
            self._stage_totals[key] = self._stage_totals.get(key, 0.0) + getattr(timing, key)
        count = max(1, self.health.frames_processed)
        self.health.avg_stage_ms = {k: v / count for k, v in self._stage_totals.items()}

        now = time.monotonic()
        self._processed_window.append(now)
        if len(self._processed_window) > 30:
            self._processed_window.pop(0)
        span = self._processed_window[-1] - self._processed_window[0]
        self.health.processed_fps = (len(self._processed_window) - 1) / span if span > 0 else 0.0

    def _sync_health(self) -> None:
        stats = self.reader.stats
        self.health.connected = stats.connected
        self.health.stream_fps = stats.fps
        self.health.last_frame_at = stats.last_frame_at
        self.health.reconnects = stats.reconnects
        self.health.spool_depth = self.ingest.spool_depth
        if stats.last_error:
            self.health.last_error = stats.last_error

    # -- shutdown ----------------------------------------------------------
    def _shutdown(self) -> None:
        logger.info("camera %s: stopping", self.camera.code)
        self.reader.stop()

        # A vehicle mid-gate at shutdown should still produce its event.
        wall = time.time()
        for state in self.processor.flush():
            try:
                self._finalize(state, wall)
            except Exception:
                logger.exception("camera %s: flush failed", self.camera.code)

        self.ingest.close()
        self.processor.close()
        logger.info(
            "camera %s: stopped after %d frames, %d events",
            self.camera.code, self.health.frames_processed, self.health.events_emitted,
        )
