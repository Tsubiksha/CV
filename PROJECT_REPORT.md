# Vision AI Monitoring Platform - Project Report

## 1. Project Overview

The Vision AI Monitoring Platform is a full-stack computer vision system for live camera monitoring, object detection, face registration, face recognition, image-based object detection, and OCR. The backend is built with Flask and OpenCV, while the frontend is built with React.

The system is designed to use an external USB camera controlled by the backend. The backend streams live frames through MJPEG and sends AI metadata to the frontend through Flask-SocketIO. The frontend renders the stream on a canvas and draws clean, scaled overlays for detections and recognized faces.

## 2. Objectives

- Capture live video from an external USB camera.
- Display the camera stream in a React dashboard.
- Run real-time object detection using YOLOv8m.
- Detect objects in uploaded images.
- Register face samples using DeepFace ArcFace embeddings.
- Recognize registered faces from live camera frames.
- Support multiple face samples per person for improved recognition.
- Run OCR on uploaded images or webcam frames.
- Keep the UI readable and professional, especially in crowded detection scenes.

## 3. Technology Stack

### Backend

- Python
- Flask
- Flask-CORS
- Flask-SocketIO
- OpenCV
- NumPy
- Ultralytics YOLOv8m
- PyTorch
- DeepFace ArcFace
- EasyOCR
- Tesseract OCR
- Pillow

### Frontend

- React
- React Router
- Socket.IO Client
- GSAP
- Create React App
- Canvas API for live overlays

## 4. Project Structure

```text
DL/
  backend/
    app.py
    camera_utils.py
    detect_utils.py
    face_utils.py
    ocr_utils.py
    requirements.txt
    models/
      yolov8m.pt
    faces/
      encodings.pkl
      *.jpg

  frontend/
    frontend/
      package.json
      public/
      src/
        App.js
        App.css
        index.js
        index.css
        pages/
          LiveVision.js
          ImageDetection.js
          LiveDetection.js
          FaceRegistration.js
          FaceRecognition.js
        hooks/
          useWebSocket.js
          useCanvas.js
          useCamera.js
        components/
          ParticleBackground.js
```

## 5. Backend Architecture

The backend manages camera access, AI inference, face storage, OCR, API routes, and WebSocket communication.

### 5.1 Main Flask App

The main backend file is `backend/app.py`.

Important responsibilities:

- Creates the Flask application.
- Enables CORS.
- Configures Flask-SocketIO.
- Starts the camera buffer.
- Loads the YOLO model.
- Pre-warms DeepFace ArcFace.
- Defines REST API endpoints.
- Runs the live AI processing thread.

Important routes:

| Route | Method | Purpose |
|---|---:|---|
| `/api/health` | GET | Returns backend, camera, and AI status |
| `/api/camera-index` | GET | Returns selected camera index |
| `/api/stream` | GET | Streams MJPEG live camera frames |
| `/api/snapshot` | GET | Returns one camera frame |
| `/api/process` | POST | Processes one camera frame in detect, recognize, or raw mode |
| `/api/register-face` | POST | Registers a new face sample |
| `/api/registered-faces` | GET | Lists registered people and sample counts |
| `/api/face-debug` | GET | Returns face database/debug information |
| `/api/delete-face/<name>` | DELETE | Deletes a registered face identity |
| `/api/detect-image` | POST | Runs YOLO object detection on an uploaded image |
| `/api/ocr` | POST | Runs OCR on an uploaded image |
| `/api/ocr-webcam` | POST | Runs OCR on the current camera frame |

### 5.2 Camera System

Camera handling is implemented in `backend/camera_utils.py`.

The project uses a `CameraBuffer` class that continuously captures frames in a background thread and stores only the newest frame in a `deque(maxlen=1)`.

Features:

- Uses external USB camera index `1`.
- Captures at 1280x720 and 30 FPS.
- Uses OpenCV `CAP_DSHOW`, suitable for Windows.
- Provides thread-safe frame access.
- Detects frozen frames using MD5 frame hashes.
- Reopens the camera after repeated failures or frozen frames.
- Prevents multiple backend components from fighting over the same camera.

### 5.3 Object Detection

Object detection is implemented in `backend/detect_utils.py`.

The system uses YOLOv8m from Ultralytics. The model is loaded once and runs on CUDA when available, otherwise CPU.

Main functions:

- `detect_for_stream()` returns JSON metadata for live overlays.
- `detect_objects()` returns an annotated image and metadata for uploaded images.

Current detection features:

- YOLOv8m object detection.
- ByteTrack support for live tracking.
- Stable human labels such as `Human 1`, `Human 2`.
- Confidence filtering for crowded scenes.
- IoU threshold set around `0.45`.
- Extra same-class duplicate suppression.
- Optional maximum render limit using `YOLO_MAX_RENDER_DETECTIONS`, default `20`.
- Size, aspect ratio, edge, and frame-coverage filters.
- Class-based colors:
  - Human/person: green
  - Bicycle: blue
  - Motorcycle: cyan
  - Car, bus, truck: orange
- Clean annotation style with thinner boxes and compact labels.
- Label overlap reduction for uploaded image annotations.
- Category counts for humans, vehicles, and other objects.

Vehicle category includes:

```text
car, bus, truck, motorcycle, bicycle, train
```

Human category includes:

```text
person, human
```

### 5.4 Face Registration and Recognition

Face logic is implemented in `backend/face_utils.py`.

The project uses DeepFace with the ArcFace model. Each registered face sample is converted into a normalized embedding and stored in `backend/faces/encodings.pkl`.

Current face features:

- Multiple samples per person.
- New samples are appended, not replaced.
- Face crop is used for both registration and recognition.
- Embeddings are normalized before comparison.
- Cosine distance is used for matching.
- Confidence is calculated as:

```text
similarity = 1 - distance
confidence = similarity * 100
```

- Confidence is clamped between `0` and `100`.
- Default face distance threshold is configurable through `FACE_DISTANCE_THRESHOLD`, default `0.55`.
- Registration rejects poor samples when:
  - no face is detected
  - multiple faces are detected
  - face is too small
  - face is blurry
  - embedding generation fails
- Live recognition compares a face against every stored sample for every person.
- The best sample with the smallest distance is selected.
- Temporal smoothing and majority voting reduce flickering.
- Unknown results display as `Unknown Face`.
- Debug route `/api/face-debug` reports registered names, sample counts, encoding file status, model, detector, and threshold.

Recommended registration guidance:

```text
Register 5-8 clear samples from different angles for better accuracy.
```

### 5.5 OCR

OCR logic is implemented in `backend/ocr_utils.py`.

Supported OCR engines:

- EasyOCR
- Tesseract OCR

OCR preprocessing includes:

- image upscaling
- grayscale conversion
- denoising
- adaptive thresholding
- sharpening
- skew correction

OCR output includes text, confidence, word count, bounding boxes, and annotated images.

## 6. Real-Time Processing Flow

The live system uses three important loops:

1. Camera thread captures and stores the newest frame.
2. MJPEG endpoint streams frames to the frontend.
3. AI processing thread runs object detection or face recognition and emits metadata through WebSocket.

Socket.IO events:

| Event | Direction | Purpose |
|---|---|---|
| `start_processing` | Frontend to backend | Starts object detection or face recognition |
| `stop_processing` | Frontend to backend | Stops current AI processing |
| `processing_status` | Backend to frontend | Sends active mode and status |
| `detection_update` | Backend to frontend | Sends object detection metadata |
| `recognition_update` | Backend to frontend | Sends face recognition metadata |

## 7. Frontend Architecture

The frontend is a React application in `frontend/frontend`.

### 7.1 App Shell

`src/App.js` defines:

- main layout
- sidebar navigation
- topbar
- backend health polling
- routes
- backend URL selection

Current routes:

| Route | Page |
|---|---|
| `/` | Live Vision |
| `/image-detection` | Image Detection |

The sidebar keeps Image Detection as a separate page.

### 7.2 Live Vision Page

`src/pages/LiveVision.js` is the main live dashboard.

Current Live Vision tabs:

- Live Stream
- Face Registry

The Image Detect tab was removed from inside Live Vision because Image Detection already exists as a separate sidebar page.

Live Stream includes:

- live camera preview
- Start Object Detection button
- Start Face Recognition button
- Stop Detection button
- detection status badge
- detection results panel
- detection history panel

Face Registry includes:

- camera preview
- capture frame button
- name input
- add face sample button
- enrolled faces list
- sample counts
- delete button

### 7.3 WebSocket Hook

`src/hooks/useWebSocket.js` manages Socket.IO communication.

It provides:

- backend connection state
- detections
- recognized faces
- recognition messages
- AI status
- latency
- frame size
- start/stop processing functions

### 7.4 Canvas Overlay Hook

`src/hooks/useCanvas.js` draws live video and overlays.

It:

- draws the hidden MJPEG stream onto a visible canvas
- scales backend bounding boxes to the displayed canvas
- handles object-fit style containment offsets
- draws object boxes and labels
- draws face boxes and labels
- smooths object box movement
- uses class-based colors for objects
- uses confidence-based colors for faces
- reduces label overlap

Face labels display as:

```text
Subiksha | 87%
```

Unknown faces display as:

```text
Unknown Face
```

### 7.5 Image Detection Page

`src/pages/ImageDetection.js` handles uploaded image detection.

It shows:

- upload area
- image preview
- annotated result
- total object count
- human count
- vehicle count
- other object count
- detection lists

Category counting is normalized using lowercase class/label names.

Rules:

- `person` and `human` count as Humans.
- `car`, `bus`, `truck`, `motorcycle`, `bicycle`, and `train` count as Vehicles.
- all remaining classes count as Other Objects.

Example:

```text
Human, Human, Bicycle, Car

Total Objects: 4
Humans: 2
Vehicles: 2
Other Objects: 0
```

## 8. Data Flow

### 8.1 Live Object Detection

1. Backend captures frames from the USB camera.
2. Frontend displays `/api/stream`.
3. User clicks Start Object Detection.
4. Frontend emits `start_processing` with mode `detect`.
5. Backend runs `detect_for_stream()`.
6. Backend emits `detection_update`.
7. Frontend draws boxes and labels on the canvas.

### 8.2 Live Face Recognition

1. User captures a frame in Face Registry.
2. Backend detects and crops the face.
3. Backend generates an ArcFace embedding.
4. Embedding is appended to the person in `encodings.pkl`.
5. User starts Face Recognition.
6. Backend detects live faces and generates embeddings.
7. Live embedding is compared against all registered samples.
8. Best match is returned if it passes the threshold.
9. Frontend displays the recognized name and confidence.

### 8.3 Image Detection

1. User uploads an image.
2. Frontend sends it to `/api/detect-image`.
3. Backend runs YOLOv8m.
4. Backend filters low-confidence and duplicate detections.
5. Backend draws clean annotations.
6. Frontend displays the annotated image and counts.

### 8.4 OCR

1. User sends an image or webcam frame.
2. Backend preprocesses the image.
3. OCR engine extracts text.
4. Backend returns text and metadata.

## 9. Recent Improvements

Recent fixes added to the project:

- Removed Image Detect tab from Live Vision while keeping the Image Detection sidebar page.
- Fixed face recognition matching by using cropped face embeddings consistently.
- Added normalized multi-sample face matching.
- Added face registration quality checks.
- Added face confidence display as `Name | 87%`.
- Added `/api/face-debug`.
- Improved object detection visualization for crowded scenes.
- Added duplicate suppression for overlapping detections.
- Added cleaner image annotation labels.
- Added class-based object colors.
- Added category counts for humans, vehicles, and other objects.
- Fixed vehicle counting for bicycle and car detections.

## 10. Strengths

- Full-stack computer vision system with live and static image workflows.
- Backend owns camera access, preventing browser camera conflicts.
- MJPEG stream plus WebSocket metadata is efficient for live overlays.
- YOLOv8m provides strong general object detection.
- DeepFace ArcFace supports robust face embeddings.
- Multiple face samples improve recognition reliability.
- Detection filtering improves readability in crowded scenes.
- React canvas overlays allow precise frontend rendering.
- Modular backend files keep camera, detection, face, and OCR logic separated.

## 11. Limitations

- Camera index is hardcoded to `1`.
- Face data is stored in a pickle file, which is simple but not ideal for production.
- No authentication or role-based access control.
- CORS is open to all origins.
- Some legacy frontend pages are present but not routed in the current app.
- OCR backend exists, but OCR does not appear to have a main sidebar page in the current frontend.
- No automated test suite is currently included.
- The project folder is not currently a Git repository.
- Some older comments/text may contain encoding artifacts from emoji/line-drawing characters.

## 12. Future Enhancements

- Add camera selection in the UI.
- Add a dedicated OCR page.
- Replace pickle face storage with SQLite.
- Add authentication for non-local use.
- Add backend unit tests for detection filtering and face matching.
- Add frontend tests for major dashboard states.
- Add environment-based configuration for camera index, model path, CORS origins, and backend URL.
- Clean legacy pages or update them to use the current backend APIs.
- Add exportable detection reports.
- Add deployment documentation.

## 13. Running the Project

### Backend

```bash
cd backend
pip install -r requirements.txt
python app.py
```

Backend URL:

```text
http://localhost:5000
```

Useful optional environment variables:

```bash
FACE_DISTANCE_THRESHOLD=0.55
YOLO_MAX_RENDER_DETECTIONS=20
```

### Frontend

```bash
cd frontend/frontend
npm install
npm start
```

Frontend URL:

```text
http://localhost:3000
```

## 14. Conclusion

This project is a complete AI-based vision monitoring platform. It combines real-time video streaming, YOLO object detection, DeepFace ArcFace recognition, face registration, uploaded image detection, and OCR.

The strongest part of the project is its practical real-time architecture: the backend controls camera capture and AI inference, while the frontend renders a responsive dashboard with clean overlays and live status. With camera configuration, database-backed face storage, OCR UI integration, and automated testing, this can become a more polished and production-ready computer vision system.
