# Sign Language Alphabet Recognition using Deep Learning

> **Final Year Major Project** — Production-Grade Computer Vision & Deep Learning Application for Real-Time Sign Language Alphabet Recognition, Sentence Buffering, and Text-to-Speech Synthesis.

---

## Architecture & Real-time Pipeline

```
Webcam Frame (RGB)
      │
      ▼
MediaPipe / cvzone HandDetector ──► Extracts 21 3D Joint Landmarks
      │
      ▼
OpenCV Skeleton Generator ──► Synthesizes Green Skeleton on White Canvas (400×400)
      │
      ▼
Preprocessing Service ──► Resizes to (128×128×3) & Normalizes Pixels to [0.0, 1.0]
      │
      ▼
Custom CNN Model (TensorFlow) ──► Predicts Categorical Probabilities (26 Softmax Classes)
      │
      ▼
FastAPI Server ──► Returns JSON: { "letter": "A", "confidence": 99.2 }
      │
      ▼
React Frontend UI (Vite + Tailwind) ──► Buffers Words & Triggers SpeechSynthesis Text-to-Speech
```

---

## Project Structure

```text
Sign-Language/
├── README.md
├── requirements.txt
├── .gitignore
├── .env
│
├── dataset/                      # A-Z subdirectories with skeleton dataset images
│   ├── A/ ... Z/
│
├── training/                     # Model Training & Evaluation Suite
│   ├── labels.py
│   ├── preprocess_dataset.py
│   ├── augmentation.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── models/                       # Exported model artifacts (best_model.keras)
│
├── backend/                      # Production FastAPI REST Backend
│   ├── app.py
│   ├── config.py
│   ├── api/
│   │   ├── health.py
│   │   └── predict.py
│   ├── services/
│   │   ├── hand_detector.py
│   │   ├── skeleton_generator.py
│   │   ├── preprocess.py
│   │   ├── predictor.py
│   │   └── sentence_builder.py
│   └── utils/
│       ├── image_utils.py
│       ├── labels.py
│       └── constants.py
│
└── frontend/                     # Modern React (Vite + Tailwind CSS) Web Interface
    ├── index.html
    ├── vite.config.js
    ├── tailwind.config.js
    ├── package.json
    └── src/
        ├── App.jsx
        ├── main.jsx
        ├── index.css
        └── components/
            ├── Navbar.jsx
            ├── WebcamFeed.jsx
            ├── PredictionBox.jsx
            └── SentenceBox.jsx
```

---

## Tech Stack

- **Frontend**: React 18, Vite, Tailwind CSS, Axios, `react-webcam`, `lucide-react`, Browser `SpeechSynthesis` API.
- **Backend**: FastAPI, Uvicorn, Pydantic, Python 3.10+.
- **Computer Vision & Deep Learning**: TensorFlow 2.x, OpenCV, MediaPipe, `cvzone`, NumPy, Scikit-Learn, Matplotlib.

---

## Getting Started

### 1. Environment Setup & Backend Installation
```bash
# Navigate to project root
cd Sign-Language

# Create and activate Python virtual environment
python -m venv venv
venv\Scripts\activate   # On Windows

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Model Training & Evaluation
```bash
# Run model training (30 epochs)
python training/train.py --epochs 30

# Evaluate trained model checkpoint on test set
python training/evaluate.py
```

### 3. Launching FastAPI Backend Server
```bash
# Start Uvicorn ASGI production server
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```
- API Documentation (Swagger UI): `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/api/v1/health`

### 4. Launching React Frontend
```bash
# Open new terminal and navigate to frontend folder
cd frontend

# Install Node dependencies
npm install

# Start Vite dev server
npm run dev
```
- Open Web Application: `http://localhost:5173`

---

## Features
- **Real-Time Landmark Synthesis**: Automatically extracts MediaPipe hand keypoints and draws green skeleton lines on white backgrounds to match training data modality.
- **Debounced Sentence Builder**: Buffers prediction stream, filters low-confidence noise, and appends characters dynamically.
- **Browser Text-to-Speech**: Speaks buffered sentences out loud via native `window.speechSynthesis` API.
- **Dark Mode UI**: Responsive dashboard with glassmorphism cards and glowing confidence metrics.
