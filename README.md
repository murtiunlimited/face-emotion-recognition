#  Facial Emotion Recognition (FER Project)

This project performs **real-time facial emotion detection** using a CNN model.  
It supports both:
-  OpenCV (desktop webcam)
-  Web browser (FastAPI + frontend)
-  Web Browser Version With Shell
-  Vercel and Render For Web Deployment
-  NEVER USE GIT INIT AFTER YOUR PROJECT IS COMPLETE LOL IT DELETED EVERYTHING

---

## 📁 Project Setup and Structure

```text
FER-PROJ-2/
├── .github/
│   ├── workflows/
│   ├── ci.yml
├── api/
│   ├── app.py
│   ├── __init__.py
├── data/
│   ├── processed/
│   │   ├── train/
│   │   └── validation/
│   └── raw/
│       ├── test/
│       └── train/
├── frontend/
│   └── index.html
├── tests/
│   ├── test_api.py
│   ├── test_data.py
│   ├── test_imports.py
│   ├── test_inference.py
│   ├── test_model.py
├── models/
│   ├── best_emotion_model.keras
│   └── final_emotion_model.keras
└── src/
    ├── data/
    │   ├── __pycache__/
    │   ├── __init__.py
    │   ├── preprocess.py
    ├── inference/
    │   ├── __init__.py
    │   ├── predict.py
    │   └── webcam.py # if user prefers openCV
    ├── models/
    │   ├── __init__.py
    │   ├── evaluate.py  # MLFLOW included
    │   ├── model.py
    │   └── train.py     # MLFLOW included
    ├── utils/
    │   ├── __init__.py
    │   └── config.py
    └── __init__.py
├── Dockerfile
├── README.md
├── requirements.txt
├── run_pipeline.py # if user prefers web version by python script
└── shellscript.sh  # if user prefers web version by shell script
```

---

### 1. Create Virtual Environment
```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows:**
```bash
venv\Scripts\activate
```

**Mac/Linux:**
```bash
source venv/bin/activate
```

---

### 3. Install Requirements
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

---

#  OpenCV Version (Desktop Webcam)

### 1. Preprocess Data
```bash
python -m src.data.preprocess
```

### 2. Train Model
```bash
python -m src.models.train
```

### 3. Run Webcam Detection
```bash
python -m src.inference.webcam
```

---

#  Web Browser Version (FastAPI + Frontend)

### Step 1: Preprocess Data
```bash
python -m src.data.preprocess
```

### Step 2: Train Model
```bash
python -m src.models.train
```

### Step 3: Start Backend Server
```bash
uvicorn api.app:app --reload
```

### Step 4: Launch Frontend
- Open `frontend/index.html` in your browser  
- Allow camera access  
- Start detecting emotions 🎉

---

#  Web Browser Version With Shell

### Enable Script Execution
```bash
chmod +x shellscript.sh   
```

### Run Full Pipeline (Preprocess + Train + Launch)
```bash
./shellscript.sh
```

---
##  Model Details
- Input: 48×48 grayscale face images  
- Architecture: Lightweight CNN  
- Classes:
  - Angry  
  - Disgust  
  - Fear  
  - Happy  
  - Sad  
  - Surprise  
  - Neutral  

---

##  Features
- Real-time emotion detection  
- Lightweight CNN (fast inference)  
- Works with webcam + browser  
- FastAPI backend for scalable deployment  

---

##  Notes
- Ensure your webcam is accessible  
- Backend must be running before opening the frontend  
- Model file (`final_emotion_model.keras`) must exist in model_artifacts
- Backend will be deployed using Render and Frontend with Vercel
****
