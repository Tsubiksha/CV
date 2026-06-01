import os
import io
import base64
import json
import threading
import time
import socket

import cv2
import numpy as np
from flask import Flask, Response, request, jsonify
from flask_cors import CORS
from flask_socketio import SocketIO, emit

from camera_utils import get_camera_buffer, open_camera, find_external_camera
from face_utils    import (register_face, get_registered_faces, delete_face,
                           recognize_faces_in_frame, recognize_for_stream,
                           get_face_debug_info)
from detect_utils  import get_model, detect_objects, detect_for_stream
from ocr_utils     import run_easyocr, run_tesseract

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app, origins=["*"])

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
    logger=False,
    engineio_logger=False,
)

# ── Legacy camera singleton (for /api/process & /api/snapshot) ───────────────
_cam_lock  = threading.RLock()
_cam       = None
_cam_index = None

def get_camera():
    global _cam, _cam_index
    with _cam_lock:
        if _cam is None or not _cam.isOpened():
            _cam, _cam_index = open_camera()
    return _cam

def read_frame(resize_to=None):
    """Read one frame using the legacy singleton — used by /api/process."""
    # Try the shared buffer first (faster, no extra cap open)
    buf = get_camera_buffer()
    frame = buf.get_latest_frame()
    if frame is not None:
        if resize_to:
            frame = cv2.resize(frame, resize_to)
        return frame
    return None

def encode_jpeg(frame, quality=85):
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return buf.tobytes()

# ── Face annotation helper (legacy annotated endpoint) ───────────────────────
def draw_face_annotations(frame, faces):
    annotated = frame.copy()
    for face in faces:
        top, right, bottom, left = face["top"], face["right"], face["bottom"], face["left"]
        name, conf = face["name"], face["confidence"]
        color = (0, 200, 100) if name not in {"Unknown", "Unknown Face"} else (60, 60, 255)
        clen = 22
        for (sx, sy), (ex, ey) in [
            ((left, top), (left + clen, top)), ((left, top), (left, top + clen)),
            ((right - clen, top), (right, top)), ((right, top), (right, top + clen)),
            ((left, bottom - clen), (left, bottom)), ((left, bottom), (left + clen, bottom)),
            ((right - clen, bottom), (right, bottom)), ((right, bottom - clen), (right, bottom)),
        ]:
            cv2.line(annotated, (sx, sy), (ex, ey), color, 3)
        label = "Unknown Face" if name in {"Unknown", "Unknown Face"} else f"{name} | {conf:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 1)
        pad = 5
        cv2.rectangle(annotated, (left, top - th - 2 * pad - 2), (left + tw + 2 * pad, top - 2), color, -1)
        cv2.putText(annotated, label, (left + pad, top - pad - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 1, cv2.LINE_AA)
    return annotated


# ── AI Processing thread state ────────────────────────────────────────────────
_ai_state = {
    "active":  False,
    "mode":    "detect",
    "thread":  None,
    "stop_ev": threading.Event(),
}
_ai_lock = threading.Lock()


def _ai_processing_loop(mode: str, stop_event: threading.Event):
    """Runs in Thread 3. Pulls frames from buffer, runs AI, emits via WS."""
    target_fps = 10
    interval   = 1.0 / target_fps
    cam_buf    = get_camera_buffer()

    print(f"[AI] Processing thread started — mode={mode}")
    if mode == "recognize":
        print("[Face stream] face recognition mode started")
    socketio.emit("processing_status", {"active": True, "mode": mode})

    while not stop_event.is_set():
        t0 = time.time()
        frame = cam_buf.get_latest_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        try:
            if mode == "detect":
                result = detect_for_stream(frame)
                socketio.emit("detection_update", {
                    **result,
                    "timestamp":     time.time(),
                    "processing_ms": int((time.time() - t0) * 1000),
                })
            elif mode == "recognize":
                print("[Face stream] frame received")
                faces = recognize_for_stream(frame)
                known = sum(1 for f in faces if f["known"])
                frame_h, frame_w = frame.shape[:2]
                socketio.emit("recognition_update", {
                    "faces":         faces,
                    "message":       (
                        "No registered faces available"
                        if get_face_debug_info()["total_samples"] == 0
                        else ("No face detected" if not faces else "")
                    ),
                    "total_faces":   len(faces),
                    "known":         known,
                    "frame_width":    int(frame_w),
                    "frame_height":   int(frame_h),
                    "timestamp":     time.time(),
                    "processing_ms": int((time.time() - t0) * 1000),
                })
        except Exception as e:
            print(f"[AI] Error in processing loop: {e}")

        elapsed = time.time() - t0
        sleep   = max(0, interval - elapsed)
        time.sleep(sleep)

    socketio.emit("processing_status", {"active": False, "mode": mode})
    print(f"[AI] Processing thread stopped — mode={mode}")


# ── WebSocket events ──────────────────────────────────────────────────────────
@socketio.on("connect")
def on_connect():
    print(f"[WS] Client connected: {request.sid}")
    emit("processing_status", {
        "active": _ai_state["active"],
        "mode":   _ai_state["mode"],
    })

@socketio.on("disconnect")
def on_disconnect():
    print(f"[WS] Client disconnected: {request.sid}")

@socketio.on("start_processing")
def on_start_processing(data):
    mode = (data or {}).get("mode", "detect")
    if mode not in {"detect", "recognize"}:
        mode = "detect"
    with _ai_lock:
        # Stop existing thread if running
        if _ai_state["active"] and _ai_state["thread"]:
            _ai_state["stop_ev"].set()
            _ai_state["thread"].join(timeout=3)

        stop_ev = threading.Event()
        _ai_state.update({
            "active":  True,
            "mode":    mode,
            "stop_ev": stop_ev,
        })
        t = threading.Thread(
            target=_ai_processing_loop,
            args=(mode, stop_ev),
            daemon=True,
            name="AIProcessor",
        )
        _ai_state["thread"] = t
        t.start()

@socketio.on("stop_processing")
def on_stop_processing(_data=None):
    with _ai_lock:
        if _ai_state["active"]:
            _ai_state["stop_ev"].set()
            _ai_state["active"] = False
    emit("processing_status", {"active": False, "mode": _ai_state["mode"]})


# ── MJPEG Stream (Thread 2) ───────────────────────────────────────────────────
def _generate_mjpeg():
    """Generator: yield MJPEG frames from the camera buffer at ~30fps."""
    cam_buf = get_camera_buffer()
    while True:
        frame = cam_buf.get_latest_frame()
        if frame is None:
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera not available", (100, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
            frame = placeholder

        jpeg = encode_jpeg(frame, quality=75)

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
        )
        time.sleep(1 / 30)

@app.route("/api/stream")
def api_stream():
    resp = Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp


# ── Health ───────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    buf = get_camera_buffer()
    return jsonify({
        "status":       "ok",
        "camera_index": buf.index,
        "camera_ready": buf.is_ready(),
        "ai_active":    _ai_state["active"],
        "ai_mode":      _ai_state["mode"],
    })

@app.route("/api/camera-index")
def camera_index_route():
    return jsonify({"camera_index": find_external_camera()})


# ── Snapshot (legacy polling) ────────────────────────────────────────────────
@app.route("/api/snapshot")
def api_snapshot():
    frame = read_frame(resize_to=(640, 480))
    if frame is None:
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(placeholder, "Camera not available", (120, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (80, 80, 80), 2)
        jpeg = encode_jpeg(placeholder, quality=70)
    else:
        jpeg = encode_jpeg(frame, quality=80)
    resp = Response(jpeg, mimetype="image/jpeg")
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"]        = "no-cache"
    return resp


# ── Process (legacy: capture + annotate + base64) ────────────────────────────
@app.route("/api/process", methods=["POST"])
def api_process():
    data = request.json or {}
    mode = data.get("mode", "detect")
    t0   = time.time()
    ms   = lambda: int((time.time() - t0) * 1000)

    frame = read_frame()
    if frame is None:
        return jsonify({"success": False, "error": "Camera not available"}), 503

    if mode == "detect":
        result = detect_objects(frame, use_tracking=False)
        _, buf = cv2.imencode(".jpg", result["frame"], [cv2.IMWRITE_JPEG_QUALITY, 92])
        return jsonify({
            "success": True, "mode": "detect",
            "annotated_image": f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}",
            "detections": result["detections"],
            "total":      result["total"],
            "humans":     result["humans"],
            "ms":         ms(),
        })

    elif mode == "recognize":
        faces    = recognize_faces_in_frame(frame)
        annotated = draw_face_annotations(frame, faces)
        frame_h, frame_w = frame.shape[:2]
        _, buf   = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return jsonify({
            "success":      True, "mode": "recognize",
            "annotated_image": f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}",
            "faces":        faces,
            "message":      (
                "No registered faces available"
                if get_face_debug_info()["total_samples"] == 0
                else ("No face detected" if not faces else "")
            ),
            "frame_width":   int(frame_w),
            "frame_height":  int(frame_h),
            "total_faces":  len(faces),
            "known":        sum(1 for f in faces if f["name"] not in {"Unknown", "Unknown Face"}),
            "ms":           ms(),
        })

    else:  # raw — for registration capture
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        return jsonify({
            "success": True, "mode": "raw",
            "image":   f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}",
            "ms":      ms(),
        })


# ── Face management ──────────────────────────────────────────────────────────
@app.route("/api/register-face", methods=["POST"])
def api_register_face():
    data      = request.json
    name      = data.get("name", "").strip()
    image_b64 = data.get("image_b64", "")
    if not name or not image_b64:
        return jsonify({"success": False, "error": "Missing name or image"}), 400
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    try:
        image_bytes = base64.b64decode(image_b64)
    except Exception:
        return jsonify({"success": False, "error": "Invalid base64"}), 400
    result = register_face(name, image_bytes)
    return jsonify(result), (200 if result["success"] else 422)

@app.route("/api/registered-faces")
def api_registered_faces():
    try:
        return jsonify(get_registered_faces())
    except Exception as e:
        print(f"[API] registered-faces error: {e}")
        return jsonify([])

@app.route("/api/face-debug")
def api_face_debug():
    return jsonify(get_face_debug_info())

@app.route("/api/delete-face/<name>", methods=["DELETE"])
def api_delete_face(name):
    result = delete_face(name)
    return jsonify(result), (200 if result["success"] else 404)


# ── Image detection upload ────────────────────────────────────────────────────
@app.route("/api/detect-image", methods=["POST"])
def api_detect_image():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file"}), 400
    img_bgr = cv2.imdecode(
        np.frombuffer(request.files["image"].read(), np.uint8), cv2.IMREAD_COLOR
    )
    if img_bgr is None:
        return jsonify({"success": False, "error": "Could not decode image"}), 422
    result = detect_objects(img_bgr, use_tracking=False)
    _, buf = cv2.imencode(".jpg", result["frame"], [cv2.IMWRITE_JPEG_QUALITY, 92])
    return jsonify({
        "success":        True,
        "annotated_image": f"data:image/jpeg;base64,{base64.b64encode(buf.tobytes()).decode()}",
        "detections":     result["detections"],
        "total":          result["total"],
        "humans":         result["humans"],
    })


# ── OCR ───────────────────────────────────────────────────────────────────────
@app.route("/api/ocr", methods=["POST"])
def api_ocr():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file"}), 400
    engine  = request.form.get("engine", "easyocr")
    img_bgr = cv2.imdecode(
        np.frombuffer(request.files["image"].read(), np.uint8), cv2.IMREAD_COLOR
    )
    if img_bgr is None:
        return jsonify({"success": False, "error": "Could not decode image"}), 422
    return jsonify(run_tesseract(img_bgr) if engine == "tesseract" else run_easyocr(img_bgr))

@app.route("/api/ocr-webcam", methods=["POST"])
def api_ocr_webcam():
    engine = (request.json or {}).get("engine", "easyocr")
    frame  = read_frame()
    if frame is None:
        return jsonify({"success": False, "error": "Camera not available"}), 503
    return jsonify(run_tesseract(frame) if engine == "tesseract" else run_easyocr(frame))


# ── Startup ───────────────────────────────────────────────────────────────────
def _prewarm_deepface():
    try:
        from deepface import DeepFace as DF
        dummy = np.zeros((112, 112, 3), dtype=np.uint8)
        DF.represent(img_path=dummy, model_name="ArcFace",
                     detector_backend="skip", enforce_detection=False)
        print("[Prewarm] ArcFace model loaded ✓")
    except Exception as e:
        print(f"[Prewarm] DeepFace warmup skipped: {e}")


if __name__ == "__main__":
    # Resolve network IPs for display
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except Exception:
        local_ip = "unknown"

    print("[Server] Starting camera buffer...")
    get_camera_buffer()          # warm up + start capture thread

    print("[Server] Loading YOLO model...")
    get_model()

    print("[Server] Pre-warming DeepFace ArcFace in background...")
    threading.Thread(target=_prewarm_deepface, daemon=True).start()

    print(f"\n[Server] Running on:")
    print(f"  Local:   http://localhost:5000")
    print(f"  Network: http://{local_ip}:5000")
    print(f"[Mobile] Open on phone: http://{local_ip}:5000\n")

    socketio.run(
        app,
        host="0.0.0.0",
        port=5000,
        debug=False,
        allow_unsafe_werkzeug=True,
    )
