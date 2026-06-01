import cv2
import numpy as np
import easyocr
import pytesseract
from PIL import Image
import io
import os

# EasyOCR reader (loads model once - supports handwriting & printed text)
_ocr_reader = None


def get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        print("[OCR] Loading EasyOCR model (CNN+LSTM)...")
        _ocr_reader = easyocr.Reader(['en'], gpu=True, model_storage_directory='./models')
        print("[OCR] EasyOCR ready.")
    return _ocr_reader


def preprocess_for_ocr(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]

    # Upscale small images (better for OCR)
    if max(h, w) < 800:
        scale = 800 / max(h, w)
        img_bgr = cv2.resize(img_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Denoising
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Adaptive threshold for uneven lighting / handwriting
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 11
    )

    # Sharpening kernel
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)

    return sharpened


def correct_skew(img_bgr: np.ndarray) -> np.ndarray:
    """Auto-correct skewed document/handwriting angle using Hough transform."""
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
    if lines is None:
        return img_bgr

    angles = []
    for line in lines[:20]:
        rho, theta = line[0]
        angle = np.degrees(theta) - 90
        if -45 < angle < 45:
            angles.append(angle)

    if not angles:
        return img_bgr

    median_angle = float(np.median(angles))
    if abs(median_angle) < 0.5:
        return img_bgr

    h, w = img_bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), median_angle, 1.0)
    rotated = cv2.warpAffine(img_bgr, M, (w, h), flags=cv2.INTER_CUBIC,
                              borderMode=cv2.BORDER_REPLICATE)
    return rotated


def draw_ocr_overlay(img_bgr: np.ndarray, results: list) -> np.ndarray:
    """Draw bounding boxes + text labels on the original image."""
    overlay = img_bgr.copy()
    for item in results:
        bbox = item.get("bbox")
        text = item.get("text", "")
        conf = item.get("confidence", 0)
        if not bbox:
            continue
        pts = np.array(bbox, dtype=np.int32)
        color = (0, 255, 136) if conf > 70 else (255, 200, 0)
        cv2.polylines(overlay, [pts], isClosed=True, color=color, thickness=2)
        x, y = pts[0]
        label = f"{text[:30]}  {conf:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x, y - th - 8), (x + tw + 6, y - 2), color, -1)
        cv2.putText(overlay, label, (x + 3, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return overlay


def run_easyocr(img_bgr: np.ndarray) -> dict:

    reader = get_ocr_reader()
    img_bgr = correct_skew(img_bgr)
    results_raw = reader.readtext(img_bgr, detail=1, paragraph=False)

    results = []
    full_text_parts = []
    for (bbox, text, prob) in results_raw:
        conf = round(prob * 100, 1)
        if conf < 15:
            continue
        bbox_int = [[int(p[0]), int(p[1])] for p in bbox]
        results.append({"text": text, "confidence": conf, "bbox": bbox_int})
        full_text_parts.append(text)

    annotated = draw_ocr_overlay(img_bgr, results)
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    import base64
    annotated_b64 = base64.b64encode(buf.tobytes()).decode()

    return {
        "success": True,
        "engine": "EasyOCR (CNN+LSTM)",
        "mode": "handwriting + printed",
        "full_text": "\n".join(full_text_parts),
        "word_count": len(full_text_parts),
        "blocks": results,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
    }


def run_tesseract(img_bgr: np.ndarray) -> dict:
    """
    Secondary OCR engine: Tesseract LSTM for clean printed text.
    Applies aggressive preprocessing for maximum accuracy.
    """
    img_bgr = correct_skew(img_bgr)
    processed = preprocess_for_ocr(img_bgr)

    # Tesseract with OEM 3 (LSTM + legacy) and PSM 6 (assume block of text)
    config = r'--oem 3 --psm 6 -c preserve_interword_spaces=1'
    pil_img = Image.fromarray(processed)
    try:
        data = pytesseract.image_to_data(pil_img, config=config, output_type=pytesseract.Output.DICT)
    except Exception as e:
        return {"success": False, "error": f"Tesseract error: {str(e)}. Is Tesseract installed?"}

    results = []
    full_text_parts = []
    annotated = img_bgr.copy()
    n_boxes = len(data['text'])
    for i in range(n_boxes):
        word = data['text'][i].strip()
        conf = int(data['conf'][i])
        if not word or conf < 30:
            continue
        x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
        bbox = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
        color = (0, 255, 136) if conf > 70 else (255, 200, 0)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
        results.append({"text": word, "confidence": conf, "bbox": bbox})
        full_text_parts.append(word)

    import base64
    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 92])
    annotated_b64 = base64.b64encode(buf.tobytes()).decode()

    import re
    full_text = pytesseract.image_to_string(pil_img, config=config)
    full_text = re.sub(r'\n{3,}', '\n\n', full_text).strip()

    return {
        "success": True,
        "engine": "Tesseract LSTM (OEM3 + PSM6)",
        "mode": "printed text",
        "full_text": full_text,
        "word_count": len(full_text_parts),
        "blocks": results,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
    }