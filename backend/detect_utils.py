import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os
import time
from collections import defaultdict, deque

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yolov8m.pt")
STALE_TIMEOUT = 30  # seconds before a track is forgotten
YOLO_IOU = 0.45
LIVE_IMGSZ = 960
IMAGE_IMGSZ = 1280
LIVE_BASE_CONF = 0.25
IMAGE_BASE_CONF = 0.25
MAX_RENDER_DETECTIONS = int(os.getenv("YOLO_MAX_RENDER_DETECTIONS", "20"))

_model = None
_device = "cuda" if torch.cuda.is_available() else "cpu"

# Persistent human labels: track_id → "Human N"
_human_id_map: dict[int, str] = {}
_human_counter: int = 0
_track_last_seen: dict[int, float] = {}   # track_id → epoch seconds

# Temporal tracking for confidence boosting
_detection_history: dict[int, deque] = defaultdict(lambda: deque(maxlen=10))
_inference_times: list = []

FALLBACK_COLORS = [
    (0, 200, 100), (255, 115, 60), (60, 150, 255), (255, 210, 0),
    (180, 100, 255), (0, 210, 255), (255, 140, 0), (100, 255, 180),
]

CLASS_COLORS = {
    "person": (0, 200, 100),       # green
    "bicycle": (255, 120, 40),     # blue in BGR
    "motorcycle": (255, 210, 0),   # cyan in BGR
    "car": (0, 140, 255),          # orange in BGR
    "bus": (0, 120, 255),
    "truck": (0, 110, 230),
}

CLASS_THRESHOLDS = {
    # COCO class ids. Keep these balanced so small/side-facing objects survive.
    0: 0.40,    # person
    1: 0.35,    # bicycle
    2: 0.38,    # car
    3: 0.35,    # motorcycle
    5: 0.38,    # bus
    7: 0.38,    # truck
    
    # Animals are often smaller/side-facing: do not over-filter.
    14: 0.35,   # bird
    15: 0.35,   # cat
    16: 0.35,   # dog
    17: 0.35,   # horse
    18: 0.35,   # sheep
    19: 0.35,   # cow
    20: 0.35,   # elephant
    21: 0.35,   # bear
    22: 0.35,   # zebra
    23: 0.35,   # giraffe
    56: 0.35,   # chair
    57: 0.35,   # couch
    60: 0.35,   # dining table
    
    # Lower confidence: Small objects and accessories
    24: 0.35,   # backpack
    25: 0.35,   # umbrella
    26: 0.35,   # handbag
    27: 0.35,   # tie
    28: 0.35,   # suitcase
    31: 0.35,   # skis
    32: 0.35,   # snowboard
    33: 0.35,   # sports ball
    
    # Electronics and kitchen items
    63: 0.35,   # laptop
    64: 0.35,   # mouse
    65: 0.35,   # remote
    66: 0.35,   # keyboard
    67: 0.35,   # cell phone
    
    # Default for any other class
    "default": 0.35
}

ANIMAL_CLASSES = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
SMALL_OBJECT_CLASSES = {24, 25, 26, 27, 28, 31, 32, 33, 64, 65, 67}
HUMAN_CLASSES = {"person", "human"}
VEHICLE_CLASSES = {"car", "bus", "truck", "motorcycle", "bicycle", "train"}


def get_color_for_class(label_raw: str, cls_id: int) -> tuple:
    return CLASS_COLORS.get(label_raw, FALLBACK_COLORS[cls_id % len(FALLBACK_COLORS)])


def _normalize_class_name(value: str) -> str:
    return str(value or "").strip().lower().replace("_", " ")


def get_category(label_raw: str, display_label: str = "") -> str:
    raw = _normalize_class_name(label_raw)
    display = _normalize_class_name(display_label).split()[0]
    if raw in HUMAN_CLASSES or display in HUMAN_CLASSES:
        return "humans"
    if raw in VEHICLE_CLASSES or display in VEHICLE_CLASSES:
        return "vehicles"
    return "other"


def _bbox_iou(a: list, b: list) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-6))


def suppress_duplicate_detections(detections: list, iou_threshold: float = YOLO_IOU) -> tuple:
    kept = []
    removed = 0
    for det in sorted(detections, key=lambda d: d["confidence_raw"], reverse=True):
        duplicate = any(
            det["class"] == prev["class"] and _bbox_iou(det["bbox"], prev["bbox"]) >= iou_threshold
            for prev in kept
        )
        if duplicate:
            removed += 1
            continue
        kept.append(det)
    return kept, removed


def build_counts(detections: list) -> tuple:
    class_counts = defaultdict(int)
    category_counts = {"humans": 0, "vehicles": 0, "other": 0}
    for det in detections:
        class_counts[det["class"]] += 1
        category_counts[det["category"]] += 1
    return dict(class_counts), category_counts

def get_threshold_for_class(cls_id: int) -> float:
    """Return optimal confidence threshold for a specific class."""
    return CLASS_THRESHOLDS.get(cls_id, CLASS_THRESHOLDS["default"])

def is_valid_detection(bbox: list, frame_shape: tuple, cls_id: int, conf: float) -> tuple:
    """
    Apply multiple filters to remove false positives.
    Returns: (is_valid: bool, reason: str)
    """
    x1, y1, x2, y2 = bbox
    frame_h, frame_w = frame_shape[:2]
    
    # 1. Size filter - Remove tiny detections
    width = x2 - x1
    height = y2 - y1
    area = width * height
    
    if cls_id in ANIMAL_CLASSES:
        min_area = 100
    elif cls_id in SMALL_OBJECT_CLASSES:
        min_area = 80
    elif cls_id == 0:
        min_area = 180
    else:
        min_area = 120
    if area < min_area:
        return False, f"too_small (area={area})"
    
    aspect_ratio = width / (height + 1e-6)
    
    if aspect_ratio < 0.1 or aspect_ratio > 10.0:
        return False, f"weird_aspect ({aspect_ratio:.2f})"
    
    # 3. Edge filter - Ignore detections at frame edges (often partial/cut-off)
    edge_margin = 10  # pixels from edge
    at_edge = (x1 < edge_margin or y1 < edge_margin or 
               x2 > frame_w - edge_margin or y2 > frame_h - edge_margin)
    
    # For high-confidence detections, allow edge cases
    edge_conf = 0.28 if cls_id in ANIMAL_CLASSES or cls_id in SMALL_OBJECT_CLASSES else 0.38
    if at_edge and conf < edge_conf:
        return False, "at_edge"
    
    # 4. Maximum size filter - Remove detections that are entire frame (false positives)
    frame_coverage = area / (frame_w * frame_h)
    if frame_coverage > 0.95:  # Covers 95%+ of frame
        return False, f"too_large (coverage={frame_coverage:.2%})"
    
    return True, "valid"


def apply_temporal_boost(track_id: int, confidence: float) -> float:
    """
    Boost confidence for consistently detected objects.
    Objects detected across multiple frames get higher confidence.
    """
    if track_id is None:
        return confidence
    
    # Add to history
    _detection_history[track_id].append(confidence)
    history = list(_detection_history[track_id])
    
    # Need at least 5 frames for boosting
    if len(history) < 5:
        return confidence
    
    # Calculate stability metrics
    avg_conf = np.mean(history)
    std_conf = np.std(history)
    stability = 1.0 - min(std_conf, 1.0)  # High stability = low variance
    
    # Boost formula: reward consistent detections
    temporal_boost = 0.0
    if len(history) >= 5 and stability > 0.7:
        # Stable detection for 5+ frames
        temporal_boost = min(0.15, stability * 0.20)
    
    boosted_conf = min(1.0, confidence + temporal_boost)
    
    # Log significant boosts
    if temporal_boost > 0.05:
        pass  # Can enable logging if needed
        # print(f"[Temporal] Track {track_id}: {confidence:.2f} → {boosted_conf:.2f} (boost: +{temporal_boost:.2f})")
    
    return boosted_conf


def get_model() -> YOLO:
    """Load and optimize YOLOv8m model."""
    global _model
    if _model is None:
        print(f"[YOLO] Loading YOLOv8m on device: {_device}")
        _model = YOLO(MODEL_PATH)
        _model.to(_device)
        
        # GPU optimization: FP16 (half-precision) for 2x speedup
        if _device == "cuda":
            try:
                _model.model.half()
                print("[YOLO] FP16 (half-precision) enabled on GPU")
            except Exception as e:
                print(f"[YOLO] FP16 failed: {e}")
        
        print("[YOLO] Model ready.")
    return _model


def _prune_stale_tracks():
    """Remove tracks not seen in STALE_TIMEOUT seconds."""
    now = time.time()
    stale = [tid for tid, t in _track_last_seen.items() if now - t > STALE_TIMEOUT]
    for tid in stale:
        _track_last_seen.pop(tid, None)
        _human_id_map.pop(tid, None)
        _detection_history.pop(tid, None)
    
    if stale:
        print(f"[Track] Pruned {len(stale)} stale tracks")


def _get_human_label(track_id: int) -> str:
    """Return consistent 'Human N' label for a given ByteTrack ID."""
    global _human_counter
    now = time.time()
    _track_last_seen[track_id] = now
    if track_id not in _human_id_map:
        _human_counter += 1
        _human_id_map[track_id] = f"Human {_human_counter}"
        print(f"[Track] New human detected: {_human_id_map[track_id]} (ID: {track_id})")
    return _human_id_map[track_id]


def _rects_overlap(a, b) -> bool:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def _draw_label_pill(img, rect, color, text):
    x1, y1, x2, y2 = rect
    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.68, img, 0.32, 0, img)
    cv2.putText(img, text, (x1 + 5, y2 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)


def _draw_fancy_box(img, x1, y1, x2, y2, color, label, conf, occupied_labels=None):
    """Clean thin bounding box with a compact, collision-aware label."""
    if occupied_labels is None:
        occupied_labels = []

    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    clen = min(16, max(8, int(min(x2 - x1, y2 - y1) * 0.20)))
    for (sx, sy), (ex, ey) in [
        ((x1, y1), (x1 + clen, y1)), ((x1, y1), (x1, y1 + clen)),
        ((x2 - clen, y1), (x2, y1)), ((x2, y1), (x2, y1 + clen)),
        ((x1, y2 - clen), (x1, y2)), ((x1, y2), (x1 + clen, y2)),
        ((x2 - clen, y2), (x2, y2)), ((x2, y2 - clen), (x2, y2)),
    ]:
        cv2.line(img, (sx, sy), (ex, ey), color, 2, cv2.LINE_AA)

    text = f"{label} | {conf:.0%}"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    pad = 5
    label_w = tw + pad * 2
    label_h = th + pad * 2
    max_x = max(0, img.shape[1] - label_w - 2)
    lx = int(max(2, min(x1, max_x)))
    candidates = [
        (lx, y1 - label_h - 4, lx + label_w, y1 - 4),
        (lx, y1 + 4, lx + label_w, y1 + label_h + 4),
        (lx, y2 + 4, lx + label_w, y2 + label_h + 4),
    ]
    chosen = None
    for rect in candidates:
        if rect[1] < 2 or rect[3] > img.shape[0] - 2:
            continue
        if not any(_rects_overlap(rect, prev) for prev in occupied_labels):
            chosen = rect
            break
    if chosen is None:
        chosen = candidates[1]
        chosen = (
            chosen[0],
            max(2, min(chosen[1], img.shape[0] - label_h - 2)),
            chosen[2],
            max(label_h + 2, min(chosen[3], img.shape[0] - 2)),
        )
    occupied_labels.append(chosen)
    _draw_label_pill(img, chosen, color, text)


# ═══════════════════════════════════════════════════════════════════════════
# STREAM METADATA API (WebSocket)
# ═══════════════════════════════════════════════════════════════════════════

def detect_for_stream(frame_bgr: np.ndarray, conf_threshold: float = LIVE_BASE_CONF,
                      imgsz: int = LIVE_IMGSZ, enable_filtering: bool = True) -> dict:
    """
    Run YOLOv8m + ByteTrack and return pure JSON metadata.
    Does NOT draw on the frame — overlays are rendered client-side on canvas.
    
    Args:
        frame_bgr: Input frame in BGR format
        conf_threshold: Base confidence threshold (will be overridden per-class)
        imgsz: Detection resolution (640, 1280, 1920)
        enable_filtering: Apply size/shape/edge filters
    
    Returns:
        {
            detections: [...],
            humans: int,
            total: int,
            inference_ms: int,
            filtered_count: int  # How many were filtered out
        }
    """
    model = get_model()
    _prune_stale_tracks()
    frame_h, frame_w = frame_bgr.shape[:2]
    
    t0 = time.time()

    try:
        # Use LOW base threshold - we'll filter per-class afterward
        results = model.track(
            frame_bgr, 
            conf=conf_threshold,
            iou=YOLO_IOU,
            imgsz=imgsz,
            tracker="bytetrack.yaml", 
            persist=True,
            device=_device, 
            verbose=False,
        )[0]
    except Exception as e:
        print(f"[YOLO stream] track error: {e}")
        return {
            "detections": [], 
            "humans": 0, 
            "total": 0, 
            "inference_ms": 0,
            "filtered_count": 0,
            "frame_width": int(frame_w),
            "frame_height": int(frame_h),
        }
    
    inference_ms = int((time.time() - t0) * 1000)
    _inference_times.append(inference_ms)
    if len(_inference_times) > 100:
        _inference_times.pop(0)

    candidates = []
    human_count = 0
    boxes = results.boxes
    filtered_count = 0
    frame_shape = frame_bgr.shape

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label_raw = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = [x1, y1, x2, y2]
        print(
            f"[YOLO candidate stream] class_id={cls_id} label={label_raw} "
            f"confidence={conf:.2f} bbox={bbox} image={frame_w}x{frame_h}"
        )
        
        # Apply per-class threshold
        min_conf = get_threshold_for_class(cls_id)
        if conf < min_conf:
            filtered_count += 1
            print(
                f"[YOLO filtered stream] label={label_raw} confidence={conf:.2f} "
                f"min={min_conf:.2f} reason=class_threshold bbox={bbox}"
            )
            continue
        
        # Apply filtering
        if enable_filtering:
            is_valid, reason = is_valid_detection(bbox, frame_shape, cls_id, conf)
            if not is_valid:
                filtered_count += 1
                print(
                    f"[YOLO filtered stream] label={label_raw} confidence={conf:.2f} "
                    f"reason={reason} bbox={bbox}"
                )
                continue
        
        # Apply temporal confidence boosting (for tracked objects)
        track_id = int(box.id[0]) if box.id is not None else None
        if track_id is not None:
            conf = apply_temporal_boost(track_id, conf)
        
        color = get_color_for_class(label_raw, cls_id)

        # Human labeling with persistent IDs
        if label_raw == "person":
            if track_id is not None:
                display_label = _get_human_label(track_id)
            else:
                human_count += 1
                display_label = f"Human {human_count}"
        else:
            display_label = label_raw.replace("_", " ").title()

        candidates.append({
            "label": display_label,
            "class": label_raw,
            "category": get_category(label_raw, display_label),
            "confidence": round(conf * 100, 1),
            "confidence_raw": conf,
            "bbox": bbox,
            "frame_width": int(frame_w),
            "frame_height": int(frame_h),
            "color": list(color),          # [B, G, R] for canvas rendering
            "is_human": label_raw == "person",
            "track_id": track_id,
        })
        print(
            f"[YOLO detected stream] Detected: {label_raw} | confidence: {conf:.2f} "
            f"| class_id={cls_id} | bbox={bbox} | image={frame_w}x{frame_h}"
        )

    detections, duplicate_count = suppress_duplicate_detections(candidates)
    filtered_count += duplicate_count
    detections = detections[:MAX_RENDER_DETECTIONS]
    for det in detections:
        det.pop("confidence_raw", None)

    humans = sum(1 for d in detections if d["is_human"])
    class_counts, category_counts = build_counts(detections)
    
    # Performance logging (can be disabled in production)
    if len(detections) > 0:
        avg_conf = np.mean([d["confidence"] for d in detections])
        print(f"[Stream] {imgsz}p: {inference_ms}ms | {len(detections)} objects (filtered: {filtered_count}) | avg_conf: {avg_conf:.1f}%")
    
    return {
        "detections": detections, 
        "humans": humans, 
        "total": len(detections),
        "class_counts": class_counts,
        "category_counts": category_counts,
        "inference_ms": inference_ms,
        "filtered_count": filtered_count,
        "frame_width": int(frame_w),
        "frame_height": int(frame_h),
    }


# ═══════════════════════════════════════════════════════════════════════════
# ANNOTATED IMAGE API (Manual Detection)
# ═══════════════════════════════════════════════════════════════════════════

def detect_objects(frame_bgr: np.ndarray, conf_threshold: float = IMAGE_BASE_CONF,
                   use_tracking: bool = False, imgsz: int = IMAGE_IMGSZ,
                   enable_filtering: bool = True) -> dict:
    """
    Run YOLOv8m detection (+ optional ByteTrack) on a BGR frame.
    Returns annotated frame + JSON detections.
    
    Args:
        frame_bgr: Input frame in BGR format
        conf_threshold: Base confidence (overridden by per-class thresholds)
        use_tracking: Enable ByteTrack for persistent Human IDs
        imgsz: Detection resolution (640, 1280, 1920)
        enable_filtering: Apply size/shape/edge filters
    
    Returns:
        {
            frame: Annotated BGR frame,
            detections: [...],
            humans: int,
            total: int,
            inference_ms: int,
            filtered_count: int
        }
    """
    model = get_model()
    t0 = time.time()
    frame_h, frame_w = frame_bgr.shape[:2]

    if use_tracking:
        results = model.track(
            frame_bgr, 
            conf=conf_threshold,
            iou=YOLO_IOU,
            imgsz=imgsz,
            tracker="bytetrack.yaml", 
            persist=True,
            device=_device, 
            verbose=False
        )[0]
    else:
        results = model(
            frame_bgr, 
            conf=conf_threshold,
            iou=YOLO_IOU,
            imgsz=imgsz,
            device=_device, 
            verbose=False
        )[0]
    
    inference_ms = int((time.time() - t0) * 1000)

    annotated = frame_bgr.copy()
    candidates = []
    human_count = 0
    boxes = results.boxes
    filtered_count = 0
    frame_shape = frame_bgr.shape

    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label_raw = model.names[cls_id]
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        bbox = [x1, y1, x2, y2]
        print(
            f"[YOLO candidate image] class_id={cls_id} label={label_raw} "
            f"confidence={conf:.2f} bbox={bbox} image={frame_w}x{frame_h}"
        )
        
        # Apply per-class threshold
        min_conf = get_threshold_for_class(cls_id)
        if conf < min_conf:
            filtered_count += 1
            print(
                f"[YOLO filtered image] label={label_raw} confidence={conf:.2f} "
                f"min={min_conf:.2f} reason=class_threshold bbox={bbox}"
            )
            continue
        
        # Apply filtering
        if enable_filtering:
            is_valid, reason = is_valid_detection(bbox, frame_shape, cls_id, conf)
            if not is_valid:
                filtered_count += 1
                print(
                    f"[YOLO filtered image] label={label_raw} confidence={conf:.2f} "
                    f"reason={reason} bbox={bbox}"
                )
                continue
        
        color = get_color_for_class(label_raw, cls_id)

        # Human labeling
        if label_raw == "person":
            if use_tracking and box.id is not None:
                display_label = _get_human_label(int(box.id[0]))
            else:
                human_count += 1
                display_label = f"Human {human_count}"
        else:
            display_label = label_raw.replace("_", " ").title()

        candidates.append({
            "label": display_label,
            "class": label_raw,
            "category": get_category(label_raw, display_label),
            "confidence": round(conf * 100, 1),
            "confidence_raw": conf,
            "bbox": bbox,
            "frame_width": int(frame_w),
            "frame_height": int(frame_h),
            "color": list(color),
            "is_human": label_raw == "person",
        })
        print(
            f"[YOLO detected image] Detected: {label_raw} | confidence: {conf:.2f} "
            f"| class_id={cls_id} | bbox={bbox} | image={frame_w}x{frame_h}"
        )

    detections, duplicate_count = suppress_duplicate_detections(candidates)
    filtered_count += duplicate_count
    detections = detections[:MAX_RENDER_DETECTIONS]
    occupied_labels = []
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        color = tuple(det["color"])
        _draw_fancy_box(annotated, x1, y1, x2, y2, color, det["label"], det["confidence_raw"], occupied_labels)
        det.pop("confidence_raw", None)

    total = len(detections)
    humans = sum(1 for d in detections if d["is_human"])
    class_counts, category_counts = build_counts(detections)

    # HUD overlay with enhanced info
    overlay = annotated.copy()
    cv2.rectangle(overlay, (8, 8), (420, 95), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.60, annotated, 0.40, 0, annotated)
    
    gpu_str = "GPU" if _device == "cuda" else "CPU"
    mode_str = "ByteTrack" if use_tracking else "Detection"
    
    # Line 1: Detection counts
    cv2.putText(annotated, f"Objects: {total}  |  Humans: {humans}  |  Filtered: {filtered_count}",
                (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 136), 1, cv2.LINE_AA)
    
    # Line 2: Model info
    cv2.putText(annotated, f"YOLOv8m @ {imgsz}p | {mode_str} | {gpu_str}",
                (16, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
    
    # Line 3: Performance
    avg_inf = int(np.mean(_inference_times)) if _inference_times else inference_ms
    fps = int(1000 / inference_ms) if inference_ms > 0 else 0
    cv2.putText(annotated, f"Inference: {inference_ms}ms | Avg: {avg_inf}ms | FPS: {fps}",
                (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return {
        "frame": annotated, 
        "detections": detections, 
        "humans": humans, 
        "total": total,
        "class_counts": class_counts,
        "category_counts": category_counts,
        "inference_ms": inference_ms,
        "filtered_count": filtered_count,
        "frame_width": int(frame_w),
        "frame_height": int(frame_h),
    }


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_performance_stats() -> dict:
    """Get performance statistics."""
    if not _inference_times:
        return {
            "avg_ms": 0,
            "min_ms": 0,
            "max_ms": 0,
            "fps": 0
        }
    
    return {
        "device": _device,
        "active_tracks": len(_human_id_map),
        "avg_inference_ms": int(np.mean(_inference_times)),
        "min_inference_ms": int(np.min(_inference_times)),
        "max_inference_ms": int(np.max(_inference_times)),
        "avg_fps": int(1000 / np.mean(_inference_times)) if _inference_times else 0,
    }


def reset_tracking():
    """Reset all tracking state (useful for new sessions)."""
    global _human_id_map, _human_counter, _track_last_seen, _detection_history
    _human_id_map.clear()
    _track_last_seen.clear()
    _detection_history.clear()
    _human_counter = 0
    print("[Track] Reset all tracking state")
