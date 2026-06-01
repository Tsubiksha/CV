import os
import pickle
import numpy as np
import cv2
import threading
import time
from collections import Counter, deque
from datetime import datetime

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False
    print("[WARNING] DeepFace not installed.")

FACES_DIR       = os.path.join(os.path.dirname(__file__), "faces")
ENCODINGS_FILE  = os.path.join(FACES_DIR, "encodings.pkl")
os.makedirs(FACES_DIR, exist_ok=True)

MODEL      = "ArcFace"    # 512-D, 99.4% LFW accuracy
DETECTOR   = "opencv"     # fast, no dlib/CMake required
THRESHOLD  = float(os.getenv("FACE_DISTANCE_THRESHOLD", "0.55"))
MIN_FACE_SIZE = int(os.getenv("FACE_MIN_SIZE", "64"))
BLUR_THRESHOLD = float(os.getenv("FACE_BLUR_THRESHOLD", "35"))
FACE_MARGIN = float(os.getenv("FACE_CROP_MARGIN", "0.18"))

RECOGNITION_INTERVAL = 5
VOTE_WINDOW = 5
MIN_STABLE_VOTES = 3
UNKNOWN_STABLE_FRAMES = 4
TRACK_TTL_SECONDS = 2.0

_stream_lock = threading.RLock()
_stream_frame_no = 0
_stream_tracks = {}
_next_track_id = 1
_cached_stream_faces = []

def _normalize_embedding(emb):
    arr = np.asarray(emb, dtype=np.float32).reshape(-1)
    norm = np.linalg.norm(arr)
    if norm <= 1e-8:
        return None
    return arr / norm


def load_encodings():
    migrated = False
    if os.path.exists(ENCODINGS_FILE):
        with open(ENCODINGS_FILE, "rb") as f:
            data = pickle.load(f)

        if "people" in data:
            people = {}
            timestamps = data.get("timestamps", {})
            if isinstance(timestamps, list):
                timestamps = {}
                migrated = True
            for raw_name, value in data.get("people", {}).items():
                name = raw_name.strip().lower()
                if isinstance(value, dict):
                    embeddings = value.get("embeddings", [])
                    ts = value.get("timestamps", timestamps.get(name, []))
                else:
                    embeddings = value
                    ts = timestamps.get(name, [])
                if isinstance(embeddings, np.ndarray):
                    embeddings = [embeddings]
                people[name] = [e for e in (_normalize_embedding(e) for e in embeddings) if e is not None]
                timestamps[name] = list(ts) if isinstance(ts, (list, tuple)) else []
            normalized = {"people": people, "timestamps": timestamps}
        elif "names" in data and ("embeddings" in data or "encodings" in data):
            embeddings = data.get("embeddings", data.get("encodings", []))
            timestamps_old = data.get("timestamps", [])
            people = {}
            timestamps = {}
            for idx, (raw_name, emb) in enumerate(zip(data.get("names", []), embeddings)):
                name = raw_name.strip().lower()
                normalized_emb = _normalize_embedding(emb)
                if normalized_emb is not None:
                    people.setdefault(name, []).append(normalized_emb)
                ts = timestamps_old[idx] if idx < len(timestamps_old) else ""
                timestamps.setdefault(name, []).append(ts)
            normalized = {"people": people, "timestamps": timestamps}
            migrated = True
        else:
            # Very old format: {"subiksha": embedding} or {"subiksha": [emb1, emb2]}.
            people = {}
            timestamps = {}
            for raw_name, value in data.items():
                if raw_name in {"timestamps", "names", "embeddings", "encodings"}:
                    continue
                name = raw_name.strip().lower()
                if isinstance(value, np.ndarray):
                    embeddings = [value]
                elif isinstance(value, list) and value and not np.isscalar(value[0]):
                    embeddings = value
                else:
                    embeddings = [value]
                people[name] = [e for e in (_normalize_embedding(e) for e in embeddings) if e is not None]
                timestamps[name] = [""] * len(people[name])
            normalized = {"people": people, "timestamps": timestamps}
            migrated = True

        # Compatibility aliases for older call sites while new code uses people.
        normalized["names"] = [
            name for name, embeddings in normalized["people"].items() for _ in embeddings
        ]
        normalized["embeddings"] = [
            emb for embeddings in normalized["people"].values() for emb in embeddings
        ]
        if migrated:
            save_encodings(normalized)
            total = sum(len(v) for v in normalized["people"].values())
            print(f"[FaceDB] Migrated encodings.pkl to multi-sample format ({total} samples)")
        return normalized
    return {"people": {}, "timestamps": {}, "names": [], "embeddings": []}


def save_encodings(data):
    if "people" in data:
        data = {
            "people": data["people"],
            "timestamps": data.get("timestamps", {}),
        }
    with open(ENCODINGS_FILE, "wb") as f:
        pickle.dump(data, f)


def cosine_dist(a, b):
    a_norm = _normalize_embedding(a)
    b_norm = _normalize_embedding(b)
    if a_norm is None or b_norm is None:
        return 1.0
    return float(1 - np.dot(a_norm, b_norm))


def _frame_size(frame_bgr):
    h, w = frame_bgr.shape[:2]
    return {"frame_width": int(w), "frame_height": int(h)}


def _clip_bbox(x1, y1, x2, y2, frame_bgr):
    h, w = frame_bgr.shape[:2]
    return [
        int(max(0, min(w - 1, x1))),
        int(max(0, min(h - 1, y1))),
        int(max(0, min(w - 1, x2))),
        int(max(0, min(h - 1, y2))),
    ]


def _crop_with_margin(frame_bgr, bbox, margin=FACE_MARGIN):
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    pad_x = int(w * margin)
    pad_y = int(h * margin)
    cx1, cy1, cx2, cy2 = _clip_bbox(x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y, frame_bgr)
    crop = frame_bgr[cy1:cy2, cx1:cx2]
    return crop, [cx1, cy1, cx2, cy2]


def _is_blurry(crop):
    if crop is None or crop.size == 0:
        return True, 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return score < BLUR_THRESHOLD, score


def _extract_face_regions(frame_bgr):
    try:
        face_objs = DeepFace.extract_faces(
            img_path=frame_bgr,
            detector_backend=DETECTOR,
            enforce_detection=False,
            align=False,
        )
    except Exception as e:
        print(f"[Face detect] error={e}")
        return []

    regions = []
    for face_obj in face_objs:
        if _face_detection_confidence(face_obj) < 0.7:
            continue
        area = face_obj.get("facial_area", {})
        x = int(area.get("x", 0))
        y = int(area.get("y", 0))
        w = int(area.get("w", 0))
        h = int(area.get("h", 0))
        bbox = _clip_bbox(x, y, x + w, y + h, frame_bgr)
        if bbox[2] - bbox[0] < MIN_FACE_SIZE or bbox[3] - bbox[1] < MIN_FACE_SIZE:
            print(f"[Face quality] rejected small face bbox={bbox}")
            continue
        crop, crop_bbox = _crop_with_margin(frame_bgr, bbox)
        blurry, blur_score = _is_blurry(crop)
        if blurry:
            print(f"[Face quality] rejected blurry face blur={blur_score:.1f} bbox={bbox}")
            continue
        regions.append({
            "bbox": bbox,
            "crop_bbox": crop_bbox,
            "crop": crop,
            "blur": blur_score,
            "confidence": _face_detection_confidence(face_obj),
        })
    return regions


def _bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _center_distance(a, b):
    ax, ay = _bbox_center(a)
    bx, by = _bbox_center(b)
    return float(((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5)


def _confidence_from_distance(distance):
    if distance is None:
        return 0.0
    similarity = 1.0 - float(distance)
    confidence = similarity * 100.0
    return round(max(0.0, min(100.0, confidence)), 1)


def _face_detection_confidence(face_obj):
    confidence = face_obj.get("confidence")
    if confidence is None:
        return 1.0
    try:
        return float(confidence)
    except (TypeError, ValueError):
        return 1.0


def _predict_identity(emb, data):
    emb = _normalize_embedding(emb)
    if emb is None:
        return "Unknown Face", 0.0, 1.0, None, None, False

    samples = [
        (name, sample_idx, sample)
        for name, embeddings in data.get("people", {}).items()
        for sample_idx, sample in enumerate(embeddings)
    ]
    if not samples:
        print("[Face match] no registered faces available")
        return "Unknown Face", 0.0, 1.0, None, None, False

    dists = [cosine_dist(emb, sample) for _, _, sample in samples]
    best_idx = int(np.argmin(dists))
    best_dist = float(dists[best_idx])
    best_name, best_sample_idx, _sample = samples[best_idx]
    confidence = _confidence_from_distance(best_dist)
    verified = best_dist <= THRESHOLD
    if not verified:
        confidence = 0.0
    predicted = best_name.title() if verified else "Unknown Face"

    per_person = {}
    for (candidate_name, _sample_idx, _sample), dist in zip(samples, dists):
        per_person[candidate_name] = min(per_person.get(candidate_name, 1.0), float(dist))
    for candidate_name, dist in sorted(per_person.items(), key=lambda item: item[1]):
        print(
            f"[Face match] Candidate: {candidate_name.title()} "
            f"distance={dist:.4f} confidence={_confidence_from_distance(dist):.1f}%"
        )
    print(
        f"[Face match] Best distance: {best_dist:.4f} Threshold: {THRESHOLD:.2f} "
        f"Final: {predicted}"
    )
    return predicted, confidence, best_dist, best_name, best_sample_idx, verified


def _prune_stream_tracks(now):
    stale = [tid for tid, tr in _stream_tracks.items() if now - tr["last_seen"] > TRACK_TTL_SECONDS]
    for tid in stale:
        _stream_tracks.pop(tid, None)


def _match_track(bbox, now):
    global _next_track_id
    best_id = None
    best_dist = None
    for track_id, track in _stream_tracks.items():
        dist = _center_distance(bbox, track["bbox"])
        max_jump = max(80, 0.75 * max(bbox[2] - bbox[0], bbox[3] - bbox[1]))
        if dist <= max_jump and (best_dist is None or dist < best_dist):
            best_id = track_id
            best_dist = dist
    if best_id is None:
        best_id = _next_track_id
        _next_track_id += 1
        _stream_tracks[best_id] = {
            "bbox": bbox,
            "votes": deque(maxlen=VOTE_WINDOW),
            "stable_name": "Unknown Face",
            "stable_confidence": 0.0,
            "unknown_count": 0,
            "last_seen": now,
        }
    return best_id, _stream_tracks[best_id]


def _stable_identity(track, predicted_name, confidence):
    track["votes"].append(predicted_name)
    counts = Counter(track["votes"])
    majority_name, majority_count = counts.most_common(1)[0]

    if predicted_name in {"Unknown", "Unknown Face"}:
        track["unknown_count"] += 1
    else:
        track["unknown_count"] = 0

    previous = track.get("stable_name", "Unknown")
    if majority_name not in {"Unknown", "Unknown Face"} and majority_count >= MIN_STABLE_VOTES:
        track["stable_name"] = majority_name
        track["stable_confidence"] = confidence
    elif previous not in {"Unknown", "Unknown Face"} and track["unknown_count"] < UNKNOWN_STABLE_FRAMES:
        # Hold the last known identity through short weak/unknown bursts.
        track["stable_name"] = previous
        track["stable_confidence"] = max(track.get("stable_confidence", 0.0), confidence)
    elif track["unknown_count"] >= UNKNOWN_STABLE_FRAMES:
        track["stable_name"] = "Unknown Face"
        track["stable_confidence"] = 0.0

    return track["stable_name"], round(track.get("stable_confidence", confidence), 1), majority_name, majority_count


def _embed_face_crop(face_bgr):
    """Get ArcFace embedding from an already-cropped face image."""
    try:
        res = DeepFace.represent(
            img_path=face_bgr,
            model_name=MODEL,
            detector_backend="skip",
            enforce_detection=False,
            align=False,
        )
        if res:
            return _normalize_embedding(res[0]["embedding"])
    except Exception as e:
        print(f"[Embed] {e}")
    return None


def _embed(img_bgr):
    """Get ArcFace embedding for the clearest detected face in a frame."""
    regions = _extract_face_regions(img_bgr)
    if not regions:
        return None
    return _embed_face_crop(regions[0]["crop"])


def register_face(name, image_bytes):
    if not DEEPFACE_OK:
        return {"success": False, "error": "DeepFace not installed"}
    display_name = name.strip()
    key_name = display_name.lower()
    if not key_name:
        return {"success": False, "error": "Name cannot be empty"}

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return {"success": False, "error": "Could not decode image"}

    regions = _extract_face_regions(img_bgr)
    print(f"[Face register] Registered name: {display_name}")
    print(f"[Face register] Face detected: {'yes' if regions else 'no'}")
    if not regions:
        return {"success": False, "error": "Face not clear. Please capture again."}
    if len(regions) > 1:
        print(f"[Face register] Multiple faces detected: {len(regions)}")
        return {"success": False, "error": "Multiple faces detected. Please capture only one face."}

    emb = _embed_face_crop(regions[0]["crop"])
    if emb is None:
        print("[Face register] Embedding generated: no")
        return {"success": False, "error": "Face not clear. Please capture again."}
    print(f"[Face register] Embedding generated: yes")
    print(f"[Face register] Embedding shape: {emb.shape}")

    data = load_encodings()
    data.setdefault("people", {})
    data.setdefault("timestamps", {})
    data["people"].setdefault(key_name, []).append(emb)
    data["timestamps"].setdefault(key_name, []).append(datetime.now().isoformat())
    save_encodings(data)

    count = len(data["people"][key_name])
    print(f"[Face register] Total samples for {display_name}: {count}")
    cv2.imwrite(os.path.join(FACES_DIR, f"{key_name}_{count-1}.jpg"), regions[0]["crop"])
    return {
        "success": True,
        "name": display_name.title(),
        "message": f"Added new sample for {display_name.title()}. Total samples: {count}",
        "sample_count": count
    }


def get_registered_faces():
    data = load_encodings()
    result = []
    for name in sorted(data.get("people", {})):
        count = len(data["people"][name])
        ts = data.get("timestamps", {}).get(name, [])
        latest = max([t for t in ts if t], default="")
        result.append({
            "name": name,
            "samples": count,
            "sample_count": count,
            "registered_at": latest,
        })
    return result


def get_face_debug_info():
    data = load_encodings()
    people = data.get("people", {})
    return {
        "encoding_file_exists": os.path.exists(ENCODINGS_FILE),
        "encoding_file": ENCODINGS_FILE,
        "threshold": THRESHOLD,
        "detector": DETECTOR,
        "model": MODEL,
        "registered_names": [name.title() for name in sorted(people)],
        "sample_counts": {name.title(): len(samples) for name, samples in sorted(people.items())},
        "total_samples": sum(len(samples) for samples in people.values()),
        "deepface_available": DEEPFACE_OK,
    }


def delete_face(name):
    name = name.strip().lower()
    data = load_encodings()
    if name not in data.get("people", {}):
        return {"success": False, "error": f"No face found for '{name}'"}
    removed = len(data["people"].get(name, []))
    data["people"].pop(name, None)
    data.get("timestamps", {}).pop(name, None)
    save_encodings(data)
    for f in os.listdir(FACES_DIR):
        if f.startswith(f"{name}_") and f.endswith(".jpg"):
            try:
                os.remove(os.path.join(FACES_DIR, f))
            except Exception:
                pass
    return {"success": True, "message": f"Removed '{name}' and {removed} sample(s)"}

# ── Two-step recognition (annotated — legacy) ─────────────────────────────────
def recognize_faces_in_frame(frame_bgr):
    """
    Detect faces → extract ArcFace embeddings → match vs DB.
    Returns list of face dicts with bounding box coords.
    Used by the legacy /api/process annotated-image endpoint.
    """
    if not DEEPFACE_OK:
        return []
    data = load_encodings()

    regions = _extract_face_regions(frame_bgr)
    people = data.get("people", {})
    print(f"[Face legacy] registered_persons={len(people)} sample_counts={ {name.title(): len(samples) for name, samples in people.items()} }")
    print(f"[Face legacy] live_face_detected={'yes' if regions else 'no'} count={len(regions)}")

    results = []
    for region in regions:
        bbox = region["bbox"]
        emb = _embed_face_crop(region["crop"])
        print(f"[Face legacy] embedding_generated={'yes' if emb is not None else 'no'} bbox={bbox}")
        if emb is None:
            continue

        name, confidence_pct, best_dist, best_name, sample_idx, verified = _predict_identity(emb, data)
        print(
            f"[Face legacy] detected_faces={len(regions)} predicted={name} "
            f"distance={best_dist:.3f} confidence={confidence_pct:.1f}% "
            f"sample={best_name}:{sample_idx} bbox={bbox}"
        )

        x1, y1, x2, y2 = bbox
        results.append({
            "name": name,
            "confidence": confidence_pct,
            "verified": verified,
            "known": verified,
            "top": y1, "right": x2, "bottom": y2, "left": x1,
            "bbox": bbox,
            "distance": round(best_dist, 4),
            **_frame_size(frame_bgr),
        })

    return results


# ── Stream metadata (WebSocket) ───────────────────────────────────────────────

def recognize_for_stream(frame_bgr):
    """
    Detect faces → extract embeddings → match vs DB.
    Returns pure JSON metadata for WebSocket emission (no frame drawing).
    """
    global _stream_frame_no, _cached_stream_faces

    if not DEEPFACE_OK:
        return []
    data = load_encodings()

    with _stream_lock:
        _stream_frame_no += 1
        frame_no = _stream_frame_no
        size = _frame_size(frame_bgr)

        if frame_no % RECOGNITION_INTERVAL != 1 and _cached_stream_faces:
            cached = [{**face, **size, "cached": True, "frame_no": frame_no} for face in _cached_stream_faces]
            print(
                f"[Face stream] frame={frame_no} reused cached recognition "
                f"faces={len(cached)} frame={size['frame_width']}x{size['frame_height']}"
            )
            return cached

        now = time.time()
        _prune_stream_tracks(now)

        people = data.get("people", {})
        print(
            f"[Face stream] frame={frame_no} registered_persons={len(people)} "
            f"sample_counts={ {name.title(): len(samples) for name, samples in people.items()} }"
        )
        regions = _extract_face_regions(frame_bgr)
        print(f"[Face stream] frame={frame_no} live_face_detected={'yes' if regions else 'no'} detected_faces={len(regions)}")

        results = []
        for region in regions:
            bbox = region["bbox"]
            emb = _embed_face_crop(region["crop"])
            print(f"[Face stream] frame={frame_no} embedding_generated={'yes' if emb is not None else 'no'} bbox={bbox}")
            if emb is None:
                continue

            predicted, confidence_pct, best_dist, best_name, sample_idx, verified = _predict_identity(emb, data)
            track_id, track = _match_track(bbox, now)
            stable_name, stable_conf, majority_name, majority_count = _stable_identity(
                track, predicted, confidence_pct
            )
            track["bbox"] = bbox
            track["last_seen"] = now
            stable_verified = stable_name not in {"Unknown", "Unknown Face"}

            print(
                f"[Face stream] frame={frame_no} track={track_id} predicted={predicted} "
                f"distance={best_dist:.3f} confidence={confidence_pct:.1f}% "
                f"sample={best_name}:{sample_idx} "
                f"majority={majority_name}/{majority_count} stable={stable_name} "
                f"threshold={THRESHOLD:.2f} final={stable_name} cached=False bbox={bbox}"
            )

            results.append({
                "name": stable_name,
                "confidence": stable_conf,
                "bbox": bbox,
                "known": stable_verified,
                "verified": stable_verified,
                "distance": round(best_dist, 4),
                "predicted_name": predicted,
                "predicted_verified": verified,
                "track_id": track_id,
                "cached": False,
                "frame_no": frame_no,
                **size,
            })

        _cached_stream_faces = results
        return results
