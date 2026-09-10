"""The cascade.

The only class that knows the order of the pipeline stages, and the only place
the gating rules live. It owns no camera state beyond the TrackStates it is
holding, does no I/O, and never blocks — everything that touches the network or
the disk happens in the CameraWorker around it.

Gating, in order of how much CPU each one saves:
  * ROI crop before the vehicle detector
  * vehicle detector on 1-in-N processed frames
  * plate stages only for tracks inside the ROI and the read zone, above a
    size floor, with OCR budget left, and not already locked
  * recognizer only for plate crops the OCR scheduler picks — above a
    READABLE size (which is not the same as a detectable one), past a quality
    floor, off the approach and onto the frames worth paying for

The last of those is the expensive one and gets its own module: the decision
of whether a localized plate is worth recognizing lives in ``ocr_scheduler``,
which is pure and independently tested. This class owns the ORDER of the
stages and nothing else.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from ..ai.color import plate_color, vehicle_color
from ..debug.recorder import NullRecorder
from ..ai.inference.ops import crop as crop_box
from ..ai.plate_detector.base import PlateDetector
from ..ai.plate_recognizer.base import PlateRecognizer
from ..ai.plate_recognizer.grammar_decode import GrammarDecodeConfig
from ..ai.quality import crop_hash
from ..ai.quality.plate_quality import QualityBreakdown, enhance_for_ocr, score_plate
from ..ai.types import FrameStats, PlateObservation, StageTiming, Track
from ..ai.vehicle_detector.base import VehicleDetector
from ..ai.vehicle_tracker.base import VehicleTracker
from ..events.track_state import TrackState
from .line_crossing import CrossDirection, VirtualLine
from .ocr_scheduler import BACKOFF_REASONS, OcrContext, OcrPolicy, OcrVerdict, decide
from .plate_association import AssociationConfig, validate as validate_association, vehicle_laterally_visible
from .roi import RoiPolygon

logger = logging.getLogger("anpr.video.pipeline")

#: Scheduler reasons that get their own counter. Anything else lands in the
#: generic ``ocr_waited`` bucket; these three are broken out because they are
#: the ones an operator tunes against — too small means the read zone or the
#: mounting is wrong, budget means the vehicle dwelled, stale means the box is
#: undersized for the camera count.
_DECLINE_COUNTERS = {
    "plate_too_small": "ocr_skipped_small",
    "budget_exhausted": "ocr_skipped_budget",
    "stale_frame": "ocr_skipped_stale",
}


def _first_associated(candidates, vehicle_bbox, frame_shape, cfg: AssociationConfig):
    """Highest-confidence candidate that is geometrically plausible.

    Falls through the list rather than only testing the best one: when the
    detector's top pick is a bumper sticker and its second is the real plate,
    taking the second is free and taking neither loses the vehicle.
    """
    for candidate in candidates:
        if validate_association(candidate.bbox, vehicle_bbox, frame_shape, cfg):
            return candidate
    return None


def _quality_of(observation: PlateObservation) -> QualityBreakdown:
    """Rebuild a QualityBreakdown from a stored observation.

    Only the composite score is used downstream for weighting, and it was
    already computed when the observation was taken. Re-scoring the crop would
    give the same answer for a few hundred more microseconds.
    """
    return QualityBreakdown(
        score=observation.quality,
        width_px=observation.width,
        sharpness=0.0,
        resolution_score=0.0,
        sharpness_score=0.0,
        aspect_score=0.0,
        exposure_score=0.0,
    )


def _count_decline(stats: FrameStats, reason: str) -> None:
    counter = _DECLINE_COUNTERS.get(reason)
    if counter is None:
        stats.ocr_waited += 1
    else:
        setattr(stats, counter, getattr(stats, counter) + 1)


@dataclass
class PipelineConfig:
    #: Run the vehicle detector on 1 in N processed frames. The tracker still
    #: runs on every frame and coasts in between.
    detect_interval: int = 2
    #: Frames between plate attempts for one track. Too low wastes CPU on
    #: near-identical crops; too high starves the validator of reads.
    plate_interval: int = 2
    max_plate_attempts: int = 40
    #: A vehicle smaller than this fraction of the frame is too far away for
    #: its plate to be legible.
    min_vehicle_area_ratio: float = 0.010
    #: Fraction of the vehicle box that must lie inside the ROI.
    min_roi_overlap: float = 0.30
    #: How a vehicle is tested against the READ ZONE (see FrameProcessor):
    #: "bottom_center" asks whether the plate-bearing end of the vehicle has
    #: reached the zone, which is the right question at a low gate mount and
    #: costs a point-in-polygon test; "overlap" uses the same area fraction
    #: the ROI uses and costs a rasterization.
    read_zone_anchor: str = "bottom_center"
    #: Only consulted when read_zone_anchor == "overlap".
    min_read_zone_overlap: float = 0.20
    #: Skip recognition below this crop quality.
    min_plate_quality: float = 0.35
    #: Crop expansion around the detected plate before recognition; a tight
    #: box often clips the first or last character.
    plate_crop_pad: float = 0.06
    #: Classes that never carry a readable plate at gate distance.
    skip_classes: tuple[str, ...] = ("bicycle",)
    #: Finalize a track that has been alive this long regardless of geometry —
    #: vehicles do park inside a gate ROI.
    max_track_seconds: float = 45.0
    #: Whether to deskew using the detector's quad, when it provides one.
    use_quad_warp: bool = True
    enhance_before_ocr: bool = True
    roi_crop_pad: float = 0.05
    lock_min_reads: int = 4
    lock_min_support: float = 0.90
    #: When to spend the pipeline's most expensive call. Its own dataclass
    #: because it is its own decision with its own tests; see ocr_scheduler.
    ocr: OcrPolicy = field(default_factory=OcrPolicy)
    #: Guards on evidence-driven character substitution. See grammar_decode;
    #: the map-based `postprocess.resolve` path is unaffected by these.
    grammar_decode: GrammarDecodeConfig = field(default_factory=GrammarDecodeConfig)
    #: Geometric plausibility of a plate against the vehicle it is claimed to
    #: belong to, plus the partially-visible-vehicle guard. See
    #: video/plate_association.
    association: AssociationConfig = field(default_factory=AssociationConfig)
    #: Processed frames a track must have survived before any plate work.
    #:
    #: A brand-new track is the least reliable thing in the pipeline: its box
    #: has had one detection, its class may still flip, and ByteTrack has not
    #: yet confirmed it is a vehicle rather than a shadow. Spending the most
    #: expensive stage in the cascade on it is the worst possible bet, and a
    #: track that vanishes after two frames was never a vehicle to read.
    min_track_age: int = 2


@dataclass
class ProcessResult:
    tracks: list[Track] = field(default_factory=list)
    finalized: list[TrackState] = field(default_factory=list)
    stats: FrameStats = field(default_factory=FrameStats)


@dataclass
class _PlateWork:
    """A localized, scored plate awaiting the per-frame OCR decision.

    Lives for the duration of one ``process`` call and never longer. It is the
    closest thing in the cascade to a work item, and it is bounded by the
    number of tracks in the frame precisely so it cannot become a backlog.
    """

    track: Track
    state: TrackState
    observation: PlateObservation
    verdict: OcrVerdict
    plate_crop: np.ndarray
    vehicle_crop: np.ndarray
    quality: QualityBreakdown
    det_confidence: float
    raw_crop: Optional[np.ndarray] = None


class FrameProcessor:
    def __init__(
        self,
        detector: VehicleDetector,
        tracker: VehicleTracker,
        plate_detector: PlateDetector,
        recognizer: PlateRecognizer,
        roi: RoiPolygon,
        line: Optional[VirtualLine] = None,
        cfg: PipelineConfig | None = None,
        camera_id: int = 0,
        read_zone: Optional[RoiPolygon] = None,
        recorder=None,
    ):
        self.detector = detector
        self.tracker = tracker
        self.plate_detector = plate_detector
        self.recognizer = recognizer
        self.roi = roi
        self.line = line
        self.cfg = cfg or PipelineConfig()
        self.camera_id = camera_id

        # The READ ZONE: the part of the ROI where a plate on THIS camera is
        # actually big enough to read. The ROI says where to look for
        # vehicles; the read zone says where it is worth trying to read one.
        #
        # On a fixed mount that region is a fixed, measurable band — plate
        # pixel width is a function of position in the frame and nothing else
        # — so gating on geometry BEFORE inference is free, where gating on
        # measured quality afterwards costs a plate-detector call first.
        #
        # An empty read zone means "the whole ROI", so a camera nobody has
        # calibrated behaves exactly as it did before.
        self.read_zone = read_zone if read_zone is not None else RoiPolygon.from_config([], name="read_zone")

        # Off by default; NullRecorder makes every call site a no-op without
        # a None check in the hot path.
        self.recorder = recorder or NullRecorder()

        self.states: dict[int, TrackState] = {}
        self._frame_idx = 0

    # -- main entry point --------------------------------------------------
    def process(
        self,
        frame: np.ndarray,
        frame_idx: int | None = None,
        ts: float | None = None,
        frame_age_ms: float | None = None,
    ) -> ProcessResult:
        """Run one frame through the cascade.

        ``frame_age_ms`` is how long ago the frame was captured, measured by
        the caller — the worker knows both the capture time and the clock, and
        this class should not have to guess at either. When it exceeds
        ``cfg.ocr.max_frame_age_ms`` the frame still gets detected and tracked
        (dropping those would break tracks) but no OCR call is spent on it: a
        stale frame no longer describes where the vehicle is, and recognition
        is the one stage expensive enough that skipping it is how the pipeline
        catches back up. Pass None (offline replay, tests) to disable the check.
        """
        cfg = self.cfg
        now = ts if ts is not None else time.time()
        self._frame_idx = frame_idx if frame_idx is not None else self._frame_idx + 1
        timing = StageTiming()
        stats = FrameStats(frame_idx=self._frame_idx, timing=timing)
        stats.frame_age_ms = float(frame_age_ms or 0.0)
        stale = bool(
            frame_age_ms is not None
            and cfg.ocr.max_frame_age_ms > 0
            and frame_age_ms > cfg.ocr.max_frame_age_ms
        )
        started = time.perf_counter()

        frame_shape = frame.shape[:2]
        roi_frame, roi_offset = self.roi.crop(frame, cfg.roi_crop_pad)

        # --- vehicle detection, 1-in-N -------------------------------------
        detections = []
        run_detector = (self._frame_idx % max(1, cfg.detect_interval)) == 0
        if run_detector:
            stage_start = time.perf_counter()
            detections = self.detector.detect(roi_frame, roi_offset)
            timing.vehicle_detect = (time.perf_counter() - stage_start) * 1000
            stats.detector_ran = True
        stats.vehicles = len(detections)

        # --- tracking, every frame -----------------------------------------
        stage_start = time.perf_counter()
        tracks = self.tracker.update(detections, frame_shape, detector_ran=run_detector)
        timing.track = (time.perf_counter() - stage_start) * 1000

        for state in self.states.values():
            state.frames_since_plate_attempt += 1
            state.frames_since_ocr += 1

        # --- plate localization, then one budgeted OCR pass -----------------
        #
        # Two passes over the frame's tracks rather than one. Localizing a
        # plate tells us how good the crop is; only then can the frame's OCR
        # calls go to the best crops rather than to whichever track the
        # tracker happened to list first. With a queue at the gate that
        # ordering is the difference between reading the vehicle at the boom
        # and reading the one behind it.
        live_ids = set()
        pending: list[_PlateWork] = []
        for track in tracks:
            live_ids.add(track.track_id)
            state = self.states.get(track.track_id)
            if state is None:
                state = TrackState(track_id=track.track_id, camera_id=self.camera_id, first_seen=now)
                self.states[track.track_id] = state
            state.touch(track, now)

            if self._should_detect_plate(track, state, frame_shape):
                work = self._observe_plate(frame, track, state, now, stats, timing, stale)
                if work is not None:
                    pending.append(work)

        self._spend_ocr_budget(pending, now, stats, timing)

        # --- finalize -------------------------------------------------------
        finalized = self._collect_finalized(live_ids, frame_shape, now)

        timing.total = (time.perf_counter() - started) * 1000
        return ProcessResult(tracks=tracks, finalized=finalized, stats=stats)

    # -- gating ------------------------------------------------------------
    def _should_detect_plate(self, track: Track, state: TrackState, frame_shape: tuple[int, int]) -> bool:
        """Whether to spend a plate-DETECTOR call on this track this frame.

        Ordered cheapest test first. Every one of these runs per track per
        frame, so anything expensive belongs after everything that might make
        it unnecessary.
        """
        cfg = self.cfg
        if track.class_name in cfg.skip_classes:
            return False
        if state.emitted:
            # Already through the gate and reported; further reads cannot
            # change the event and only cost CPU.
            return False
        # A track we are deliberately waiting on gets its localization cadence
        # stretched too. Declining to read a crop but still paying to produce
        # it three times a second is the same waste one stage earlier.
        interval = cfg.plate_interval
        if state.last_ocr_reason in BACKOFF_REASONS:
            interval *= max(1, cfg.ocr.wait_backoff)
        if not state.should_attempt_plate(interval, cfg.max_plate_attempts):
            return False
        if state.ocr_remaining(cfg.ocr.budget_per_track) <= 0 and not self._grant_retry(state):
            # Plate localization exists only to feed recognition. With the OCR
            # budget spent there is nothing left for a candidate to become, so
            # stopping here saves the detector call as well as the OCR one.
            return False
        if track.age < cfg.min_track_age:
            return False
        if cfg.association.require_lateral_visibility:
            visible = vehicle_laterally_visible(track.bbox, frame_shape, cfg.association)
            if not visible:
                # A partially visible vehicle must not trigger a plate read:
                # its box is a measurement of a fragment, so every size and
                # position gate downstream is measuring the wrong thing.
                state.note_ocr_wait(visible.reason)
                return False
        frame_area = frame_shape[0] * frame_shape[1]
        if frame_area and (track.area / frame_area) < cfg.min_vehicle_area_ratio:
            return False
        if not self.roi.is_empty:
            if self.roi.overlap_ratio(track.bbox, frame_shape) < cfg.min_roi_overlap:
                return False
        if not self._in_read_zone(track, frame_shape):
            return False
        return True

    def _grant_retry(self, state: TrackState) -> bool:
        """Extend a track's budget when a little more evidence would settle it.

        Targeted at the PLATES worth more calls, not at individual characters:
        the extra reads go through the same recognizer on the same crops, so
        this cannot aim at position 4 specifically. What it can do — and what
        the fused posteriors make possible — is tell a plate that is one
        ambiguous character from certain apart from one the model simply
        cannot read, and spend the extra budget only on the first.

        Character-targeted retry (re-recognizing a sub-crop around an
        ambiguous position, or a second pass with different preprocessing to
        decorrelate the error) needs a change to how crops reach the
        recognizer and is deliberately left for later.
        """
        policy = self.cfg.ocr
        if policy.retry_bonus <= 0 or state.retry_granted:
            return False
        positions = state.nearly_settled(
            max_unstable=policy.retry_max_unstable,
            min_reads=policy.retry_min_reads,
        )
        if positions is None:
            return False
        state.grant_retry(policy.retry_bonus, positions)
        logger.info(
            "camera %s: track %s granted %d retry reads for unstable positions %s",
            self.camera_id, state.track_id, policy.retry_bonus, positions,
        )
        return True

    def _in_read_zone(self, track: Track, frame_shape: tuple[int, int]) -> bool:
        """Has the plate-bearing end of this vehicle reached the read zone?

        The default anchor is the bottom-centre of the vehicle box. At a 3-4 ft
        gate mount that is where the plate sits on both an approaching vehicle
        (front plate) and a departing one (rear plate), and it answers the
        question a read zone is actually asking — "is the plate close enough
        yet" — better than any fraction of the whole vehicle box does. A large
        vehicle overlaps a near-field band with its roof long before its plate
        arrives; the bottom edge does not.
        """
        zone = self.read_zone
        if zone is None or zone.is_empty:
            return True
        x1, _y1, x2, y2 = track.bbox
        if self.cfg.read_zone_anchor == "overlap":
            return zone.overlap_ratio(track.bbox, frame_shape) >= self.cfg.min_read_zone_overlap
        return zone.contains(((x1 + x2) / 2.0, float(y2)), frame_shape)

    # -- plate stages ------------------------------------------------------
    def _observe_plate(
        self,
        frame: np.ndarray,
        track: Track,
        state: TrackState,
        now: float,
        stats: FrameStats,
        timing: StageTiming,
        stale: bool,
    ) -> Optional[_PlateWork]:
        """Localize, crop and score one plate — everything short of reading it.

        Always records the observation, even for a crop the scheduler will
        refuse: a too-small plate is exactly the evidence that tells the
        scheduler the vehicle is still approaching, and throwing it away is
        what forces a size gate to be a hard cut-off instead of a "wait".
        """
        cfg = self.cfg
        state.note_plate_attempt()

        vehicle_crop = crop_box(frame, track.bbox)
        if vehicle_crop is None:
            return None
        state.note_vehicle_crop(vehicle_crop)

        stage_start = time.perf_counter()
        candidates = self.plate_detector.detect(vehicle_crop, offset=(track.bbox[0], track.bbox[1]))
        timing.plate_detect += (time.perf_counter() - stage_start) * 1000
        if not candidates:
            return None
        stats.plates_detected += len(candidates)

        candidate = _first_associated(candidates, track.bbox, frame.shape[:2], cfg.association)
        if candidate is None:
            stats.plates_rejected_geometry += 1
            state.note_ocr_wait("plate_not_on_vehicle")
            return None
        # Raw first, then the (possibly deskewed) crop the recognizer sees.
        # Keeping both is what lets a debug dump answer "was the crop wrong,
        # or did the warp/enhancement break it?" — a question the pipeline
        # could not previously answer at all.
        raw_crop = crop_box(frame, candidate.bbox, pad=self.cfg.plate_crop_pad)
        plate_crop = self._extract_plate(frame, candidate)
        if plate_crop is None:
            return None

        quality = score_plate(plate_crop)
        height, width = plate_crop.shape[:2]
        observation = PlateObservation(
            frame_idx=self._frame_idx,
            ts=now,
            width=int(width),
            height=int(height),
            quality=quality.score,
            det_confidence=candidate.confidence,
            crop=plate_crop,
            raw_crop=raw_crop,
            appearance=crop_hash.crop_hash(plate_crop),
        )
        state.note_observation(
            observation,
            keep=cfg.ocr.keep_observations,
            growth_window=cfg.ocr.growth_window,
            min_distance=cfg.ocr.diversity_hamming,
        )
        # Now that the plate has a score, use it to rank the vehicle image as
        # well: a sharp, large plate means a good view of the vehicle too.
        state.note_vehicle_crop(vehicle_crop, quality.score)
        stats.plates_observed += 1

        verdict = decide(
            OcrContext(
                quality=quality.score,
                width=observation.width,
                quality_floor=cfg.min_plate_quality,
                frames_since_ocr=state.frames_since_ocr,
                ocr_spent=state.ocr_calls,
                consecutive_waits=state.consecutive_ocr_waits,
                peak_width=state.peak_plate_width,
                approach_growth=state.approach_growth(cfg.ocr.growth_window),
                stale=stale,
            ),
            cfg.ocr,
        )
        if not verdict.run:
            state.note_ocr_wait(verdict.reason)
            _count_decline(stats, verdict.reason)
            return None

        return _PlateWork(
            track=track,
            state=state,
            observation=observation,
            verdict=verdict,
            plate_crop=plate_crop,
            vehicle_crop=vehicle_crop,
            quality=quality,
            det_confidence=candidate.confidence,
            raw_crop=raw_crop,
        )

    def _spend_ocr_budget(
        self, pending: list[_PlateWork], now: float, stats: FrameStats, timing: StageTiming
    ) -> None:
        """Run at most ``ocr.max_per_frame`` recognizer calls, best crops first.

        The cap is the hard guarantee against a backlog: however many vehicles
        are queued at the gate, one frame can only ever cost this many
        recognizer calls, so per-frame cost has a ceiling that does not depend
        on how busy the gate is. Tracks that lose the contest are not queued —
        they are told to wait and will compete again next frame with a fresh,
        current crop.
        """
        if not pending:
            return
        # Quality first, then plate size as the tie-break. An excellent frame
        # therefore pre-empts a merely good one on another track, which is the
        # "prioritize the excellent frame" rule doing its work across tracks
        # rather than just across time.
        pending.sort(key=lambda w: (w.verdict.priority, w.observation.width), reverse=True)

        allowed = max(1, self.cfg.ocr.max_per_frame)
        for work in pending[allowed:]:
            work.state.note_ocr_wait("frame_ocr_cap")
            stats.ocr_waited += 1

        for work in pending[:allowed]:
            self._run_ocr(work, now, stats, timing)

    def _run_ocr(self, work: _PlateWork, now: float, stats: FrameStats, timing: StageTiming) -> None:
        state = work.state
        # Counted before the call, not after: a recognizer that returns
        # nothing has still consumed the budget it was given, and a track
        # whose crops the model cannot read must not retry forever.
        state.note_ocr_call()
        stats.ocr_run += 1

        ocr_input = enhance_for_ocr(work.plate_crop) if self.cfg.enhance_before_ocr else work.plate_crop
        stage_start = time.perf_counter()
        try:
            read = self.recognizer.recognize(ocr_input)
        except Exception:
            logger.exception("camera %s: recognizer failed", self.camera_id)
            return
        finally:
            timing.recognize += (time.perf_counter() - stage_start) * 1000
        if self.recorder.enabled:
            self._record_ocr(work, read, ocr_input)
        if read is None:
            return
        stats.plates_recognized += 1
        work.observation.ocr_ran = True

        # Sampled on the same frames the plate is read on, so colour costs
        # nothing extra in gating and is voted over the same evidence.
        state.consider_colors(vehicle_color(work.vehicle_crop), plate_color(work.plate_crop))

        state.add_read(
            read,
            plate_det_confidence=work.det_confidence,
            quality=work.quality,
            frame_ts=now,
            plate_crop=work.plate_crop,
            vehicle_crop=work.vehicle_crop,
            grammar_cfg=self.cfg.grammar_decode,
            frame_idx=work.observation.frame_idx,
            appearance=work.observation.appearance,
        )
        state.consider_lock(self.cfg.lock_min_reads, self.cfg.lock_min_support)

    def _record_ocr(self, work: _PlateWork, read, ocr_input) -> None:
        """Dump one OCR call's four crops and its full evidence.

        The four images separate causes that are indistinguishable in a log:
        a badly framed detector box, a bad deskew, and an enhancement that
        crushed a low-contrast plate all produce the same wrong string.
        """
        observation = work.observation
        self.recorder.record_ocr(
            work.state.track_id,
            observation.frame_idx,
            vehicle_crop=work.vehicle_crop,
            plate_raw=work.raw_crop,
            plate_warped=work.plate_crop,
            plate_enhanced=ocr_input,
            payload={
                "camera_id": self.camera_id,
                "track_id": work.state.track_id,
                "frame_idx": observation.frame_idx,
                "ts": observation.ts,
                "vehicle": {
                    "bbox": list(work.track.bbox),
                    "class_name": work.track.class_name,
                    "confidence": round(work.track.confidence, 4),
                    "age": work.track.age,
                },
                "plate": {
                    "width": observation.width,
                    "height": observation.height,
                    "det_confidence": round(observation.det_confidence, 4),
                    "appearance_hash": observation.appearance,
                },
                "quality": work.quality.as_dict(),
                "scheduler": {
                    "action": work.verdict.action.value,
                    "reason": work.verdict.reason,
                    "priority": round(work.verdict.priority, 4),
                    "ocr_calls_before": work.state.ocr_calls - 1,
                },
                # The raw material for a real confusion matrix. Until this
                # data exists, the confusion maps in postprocess stay as they
                # are rather than being edited from single examples.
                "ocr": None if read is None else {
                    "raw_text": read.raw_text,
                    "normalized_text": read.text,
                    "confidence": round(read.confidence, 4),
                    "per_char_confidence": [round(p, 4) for p in read.per_char_confidence],
                    "per_char_alternatives": [
                        [[c, round(p, 4)] for c, p in position]
                        for position in read.per_char_alternatives
                    ],
                    "has_alternatives": read.has_alternatives,
                },
            },
        )

    def _extract_plate(self, frame: np.ndarray, candidate) -> Optional[np.ndarray]:
        if self.cfg.use_quad_warp and candidate.quad is not None:
            from ..ai.inference.ops import warp_quad

            x1, y1, x2, y2 = candidate.bbox
            width, height = max(32, x2 - x1), max(16, y2 - y1)
            try:
                return warp_quad(frame, candidate.quad, (width, height))
            except Exception:
                logger.debug("quad warp failed, falling back to axis-aligned crop", exc_info=True)
        return crop_box(frame, candidate.bbox, pad=self.cfg.plate_crop_pad)

    # -- finalization ------------------------------------------------------
    def _collect_finalized(
        self, live_ids: set[int], frame_shape: tuple[int, int], now: float
    ) -> list[TrackState]:
        finalized: list[TrackState] = []

        # A line crossing finalizes immediately: waiting for the track to be
        # retired would delay the event by seconds and misreport its timestamp.
        #
        # The state is NOT removed here. The track is still live, so removing
        # it would let the next frame build a fresh state for the same
        # track_id, whose history still shows the crossing — emitting a new
        # event every frame until the vehicle left. It stays until the tracker
        # retires it, marked `emitted` so nothing fires twice.
        if self.line is not None:
            for track_id in live_ids:
                state = self.states.get(track_id)
                if state is None or state.emitted or state.direction != CrossDirection.NONE:
                    continue
                direction = self.line.check(state.centroid_history, frame_shape)
                if direction != CrossDirection.NONE:
                    state.direction = direction
                    state.crossed_at = now
                    state.finalize_reason = "line_crossing"
                    state.emitted = True
                    finalized.append(state)

        for track_id in self.tracker.removed_track_ids():
            state = self.states.pop(track_id, None)
            if state is None or state.emitted:
                continue
            if state.direction == CrossDirection.NONE and self.line is not None:
                state.direction = self.line.check(state.centroid_history, frame_shape)
            state.finalize_reason = "track_retired"
            state.emitted = True
            finalized.append(state)

        # A track that has been alive too long is finalized where it stands.
        #
        # The state is NOT removed, for exactly the reason the line-crossing
        # branch above documents: the track is still live, so removing its
        # state lets the next frame build a fresh one for the same track_id,
        # which then accumulates new reads and emits AGAIN when the timer
        # expires. That produced repeated events from one live track — track
        # 56 emitting `DL5CU1624` (car) and then `UP16BA8695` (motorcycle) 42
        # seconds apart, track 19 emitting twice 101 seconds apart.
        #
        # Marked `emitted` and kept as a tombstone, the same track can never
        # emit twice; `_should_detect_plate` already refuses further plate
        # work for an emitted state, so the tombstone costs nothing per frame
        # and stops accumulating. Cleanup is the tracker's job: the
        # `removed_track_ids` pass above pops the state once the track is
        # genuinely retired, which is the only moment the track_id can no
        # longer come back.
        for state in self.states.values():
            if state.emitted:
                continue
            if now - state.first_seen <= self.cfg.max_track_seconds:
                continue
            if state.direction == CrossDirection.NONE and self.line is not None:
                state.direction = self.line.check(state.centroid_history, frame_shape)
            state.finalize_reason = "max_duration"
            state.emitted = True
            finalized.append(state)

        return finalized

    # -- targeted retry ----------------------------------------------------
    def resolve_pending(self, state: TrackState, now: float | None = None) -> int:
        """Spend retry calls on RETAINED crops, resolving a nearly-settled plate.

        Called at finalize, once, per track. The difference from the
        budget-extension mechanism is what the extra calls are spent ON: not
        the next frame that happens along, but the best crops the track
        already banked and the recognizer has not yet seen.

        That matters for three reasons:

          * a finalized track has no next frame — the vehicle has crossed the
            line or the tracker has retired it, so future-frame budget is
            worthless exactly when the plate is still in doubt;
          * the retained set is diversity-filtered, so an unread crop is a
            genuinely DIFFERENT view rather than another sample of the one the
            recognizer already got wrong;
          * a deterministic model re-run on a crop it has already seen returns
            the same characters, so retrying anything else is a wasted call.

        Returns the number of calls spent. Bounded by ``ocr.retry_bonus`` and
        by how many unread crops the track actually kept, so the worst case is
        known in advance and cannot depend on how long the vehicle dwelled.
        """
        policy = self.cfg.ocr
        if policy.retry_bonus <= 0 or state.retry_resolved:
            return 0
        state.retry_resolved = True

        positions = state.nearly_settled(
            max_unstable=policy.retry_max_unstable, min_reads=policy.retry_min_reads
        )
        if positions is None:
            return 0
        pool = [o for o in state.unread_observations() if o.quality >= self.cfg.min_plate_quality]
        if not pool:
            return 0

        state.retry_positions = list(positions)
        spent = 0
        now = now if now is not None else time.time()
        for observation in pool[: policy.retry_bonus]:
            read = self._recognize(observation.crop)
            observation.ocr_ran = True
            state.ocr_calls += 1
            spent += 1
            if read is None:
                continue
            state.add_read(
                read,
                plate_det_confidence=observation.det_confidence,
                quality=_quality_of(observation),
                frame_ts=observation.ts,
                plate_crop=observation.crop,
                grammar_cfg=self.cfg.grammar_decode,
                frame_idx=observation.frame_idx,
                appearance=observation.appearance,
            )
        if spent:
            logger.info(
                "camera %s: track %s spent %d retry reads on banked crops for positions %s",
                self.camera_id, state.track_id, spent, positions,
            )
        return spent

    def _recognize(self, plate_crop: Optional[np.ndarray]):
        if plate_crop is None or plate_crop.size == 0:
            return None
        ocr_input = enhance_for_ocr(plate_crop) if self.cfg.enhance_before_ocr else plate_crop
        try:
            return self.recognizer.recognize(ocr_input)
        except Exception:
            logger.exception("camera %s: recognizer failed on a retry crop", self.camera_id)
            return None

    def flush(self) -> list[TrackState]:
        """Finalize everything still in flight — called on shutdown so a
        vehicle mid-gate at restart still produces its event."""
        pending = [s for s in self.states.values() if not s.emitted]
        for state in pending:
            state.finalize_reason = state.finalize_reason or "shutdown"
            state.emitted = True
        self.states.clear()
        return pending

    def draw_overlay(self, frame: np.ndarray, tracks: list[Track], labels: dict[int, str] | None = None) -> np.ndarray:
        """Annotate a copy for the live preview. Only ever called when a
        viewer is actually subscribed."""
        canvas = frame
        self.roi.draw(canvas)
        # A dimmer inner polygon: the operator needs to see the read zone to
        # calibrate it, and needs to see that it is not the ROI.
        self.read_zone.draw(canvas, color=(90, 220, 120), thickness=1)
        if self.line is not None:
            self.line.draw(canvas)
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (64, 190, 200), 2)
            state = self.states.get(track.track_id)
            label = (labels or {}).get(track.track_id) or ""
            if not label and state and state.reads:
                label = max(state.reads, key=lambda r: r.weight).text
            label = label or f"{track.class_name} #{track.track_id}"
            cv2.putText(canvas, label, (x1, max(y1 - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (64, 190, 200), 2)
        return canvas

    def close(self) -> None:
        for component in (self.detector, self.plate_detector, self.recognizer):
            try:
                component.close()
            except Exception:
                logger.debug("close failed for %s", component, exc_info=True)
