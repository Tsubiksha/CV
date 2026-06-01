# 👁️ Vision AI Monitoring Platform

A full-stack real-time Computer Vision platform built using **Flask, React, OpenCV, YOLOv8, and DeepFace ArcFace** for live monitoring, object detection, face recognition, and image-based AI analysis.

---

# 🚀 Features

## 🎥 Live Camera Monitoring

* Real-time MJPEG camera streaming
* USB camera support
* Backend-controlled camera access
* Stable threaded camera buffering

---

## 🧠 Real-Time Object Detection

* YOLOv8m-based object detection
* Live object tracking
* Human, vehicle, and object classification
* Accurate bounding box overlays
* Crowded-scene detection filtering
* Detection confidence scores

### Detects:

* Humans
* Cars
* Bicycles
* Motorcycles
* Bags
* Animals
* Common daily objects

---

## 😀 Face Registration & Recognition

* DeepFace ArcFace embeddings
* Multiple face samples per person
* Live face recognition
* Stable recognition logic
* Confidence-based matching
* Unknown face handling
* Face sample management

### Features:

* Add multiple samples for better accuracy
* Delete registered faces
* Real-time recognition from live camera

---

## 🖼️ Image Detection

* Upload image for AI detection
* YOLOv8 object analysis
* Detection summary cards
* Human/Vehicle/Object counts
* Annotated result images

---

## 📊 Professional Dashboard UI

* Modern React dashboard
* Responsive layout
* Light professional theme
* Detection history panel
* Live AI status indicators
* Real-time WebSocket updates

---

# 🛠️ Tech Stack

## Backend

* Python
* Flask
* Flask-SocketIO
* OpenCV
* NumPy
* YOLOv8m (Ultralytics)
* PyTorch
* DeepFace ArcFace

## Frontend

* React.js
* React Router
* Socket.IO Client
* Canvas API
* CSS

---

# 📂 Project Structure

```bash
DL/
│
├── backend/
│   ├── app.py
│   ├── camera_utils.py
│   ├── detect_utils.py
│   ├── face_utils.py
│   ├── requirements.txt
│   ├── models/
│   │   └── yolov8m.pt
│   └── faces/
│       ├── encodings.pkl
│       └── *.jpg
│
├── frontend/
│   └── frontend/
│       ├── package.json
│       ├── public/
│       └── src/
│           ├── App.js
│           ├── pages/
│           ├── hooks/
│           └── components/
│
└── PROJECT_REPORT.md
```

---

# ⚙️ Backend Setup

## 1️⃣ Navigate to backend

```bash
cd backend
```

## 2️⃣ Create virtual environment

```bash
python -m venv .venv
```

## 3️⃣ Activate virtual environment

### Windows PowerShell

```bash
.\.venv\Scripts\Activate.ps1
```

---

## 4️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

## 5️⃣ Run backend server

```bash
python app.py
```

Backend runs at:

```text
http://localhost:5000
```

---

# 💻 Frontend Setup

## 1️⃣ Navigate to frontend

```bash
cd frontend/frontend
```

## 2️⃣ Install frontend dependencies

```bash
npm install
```

## 3️⃣ Start React frontend

```bash
npm start
```

Frontend runs at:

```text
http://localhost:3000
```

---

# 📡 API Endpoints

| Route                     | Method | Purpose                 |
| ------------------------- | ------ | ----------------------- |
| `/api/health`             | GET    | Backend & camera status |
| `/api/stream`             | GET    | MJPEG live stream       |
| `/api/snapshot`           | GET    | Capture single frame    |
| `/api/process`            | POST   | Process frame           |
| `/api/register-face`      | POST   | Register face           |
| `/api/registered-faces`   | GET    | List registered faces   |
| `/api/delete-face/<name>` | DELETE | Delete face             |
| `/api/detect-image`       | POST   | Detect uploaded image   |
| `/api/face-debug`         | GET    | Face database debug     |

---

# 🔄 Real-Time Processing Flow

## Object Detection Flow

1. Camera captures live frames
2. Backend runs YOLOv8 detection
3. Detection metadata sent through WebSocket
4. Frontend draws overlays on canvas

---

## Face Recognition Flow

1. User registers face samples
2. ArcFace embeddings stored
3. Live face embeddings generated
4. Compared against stored samples
5. Best match displayed with confidence

---

# 🎯 Current Features

✅ Real-time object detection
✅ Live face recognition
✅ Multiple face samples
✅ Uploaded image detection
✅ Professional React dashboard
✅ Bounding box overlays
✅ Detection confidence scores
✅ Human/Vehicle/Object counts
✅ Detection history panel
✅ Stable recognition logic

---

# 🔥 Recent Improvements

* Improved crowded-scene detection filtering
* Fixed face recognition stability
* Added multiple face samples support
* Improved bounding box alignment
* Added confidence-based recognition
* Improved vehicle counting
* Cleaner professional UI
* Added detection history panel
* Removed cluttered OCR workflow

---

# ⚠️ Current Limitations

* Camera index currently configured manually
* Local deployment only
* No authentication system yet
* Face embeddings stored in pickle file
* GPU improves performance significantly

---

# 🚀 Future Enhancements

* Multi-camera support
* Database-backed face storage
* Authentication system
* Detection report export
* Cloud deployment
* AI analytics dashboard
* Advanced tracking system

---

# 📸 Screenshots

## Live Vision Dashboard

* Real-time camera monitoring
* Detection overlays
* Face recognition

## Image Detection

* Uploaded image analysis
* Object statistics
* Annotated results

## Face Registry

* Face enrollment
* Sample management
* Recognition improvements

---

# 👩‍💻 Author

## Subiksha T

AI & Data Science Student
Passionate about:

* Computer Vision
* AI Systems
* Backend Development
* Full-Stack AI Applications

---

# ⭐ Project Highlights

This project demonstrates practical implementation of:

* Real-time AI systems
* Computer Vision
* YOLOv8 object detection
* Face recognition pipelines
* Flask backend development
* React dashboard development
* WebSocket communication
* OpenCV integration


