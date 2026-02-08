# SpeakEasy — AI Sign Language Translator

Sign-to-text with live webcam or video upload, optional translation to multiple Indian languages, and text-to-speech output.

## Features
- Live webcam and video upload inference.
- Keras LSTM model (TensorFlow backend) for sign classification.
- Translation via Google Translate (Indian languages prelisted).
- Text-to-speech via gTTS.
- Streamlit UI with WebRTC webcam preview.

## Requirements (macOS Sonoma 14.3, Apple Silicon)
- Python 3.10 (TensorFlow 2.14 wheels require it on ARM)
- `tensorflow-macos==2.14.0` + `tensorflow-metal==1.1.0`
- Other deps: `streamlit`, `streamlit-webrtc`, `av`, `opencv-python`, `mediapipe`, `gTTS`, `googletrans==4.0.0-rc1`, `numpy`
- Model file: place `action.h5` or `m1.h5` in the project root (`/Users/vedant/Downloads/SpeakEasy-main-2/`).
- Label file: `actions.json` must match the model’s class order.

## Setup
```bash
pyenv shell 3.10.13                # or use your Python 3.10 path
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install tensorflow-macos==2.14.0 tensorflow-metal==1.1.0
pip install streamlit streamlit-webrtc av opencv-python mediapipe gTTS googletrans==4.0.0-rc1 numpy
```

## Run
```bash
python inspect_model.py            # optional: verify model loads
streamlit run app.py               # opens the UI in your browser
```

## Usage
1) Ensure `action.h5` or `m1.h5` is in the project root and `actions.json` matches the label order.
2) In the UI:
   - Use **Live webcam** for real-time inference (allow camera access).
   - Or upload an MP4/AVI/MOV/MKV in **Upload a short sign video**.
3) Pick target language in the sidebar; enable TTS if desired.
4) View predicted sign, confidence, translated text, and (if enabled) hear audio output.

## Troubleshooting
- “Model or MediaPipe not available”: ensure TF 2.14.0 is installed, model file exists, and `mediapipe` is installed.
- “Live webcam requires streamlit-webrtc and av”: install those packages; restart the app.
- If the model fails to load with initializer errors, confirm you’re using Python 3.10 with `tensorflow-macos==2.14.0`.
- On Apple Silicon, prefer running inside the venv created above.


## Project Structure
- `app.py` — Streamlit UI (live webcam + upload, translation, TTS).
- `live_demo.py` — OpenCV webcam CLI demo.
- `inspect_model.py` — Loads and prints model summary.
- `actions.json` — Label list (must match model).
- `requirements.txt` — Dependency pins (TF 2.14.0).
