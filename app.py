import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import streamlit as st
import h5py

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import mediapipe as mp
    try:
        import mediapipe.solutions as mp_solutions  # newer/normal path
    except Exception:  # pragma: no cover
        mp_solutions = getattr(mp, "solutions", None)
except Exception:  # pragma: no cover
    mp = None
    mp_solutions = None

try:
    from streamlit_webrtc import webrtc_streamer
except Exception:  # pragma: no cover
    webrtc_streamer = None

try:
    import av
except Exception:  # pragma: no cover
    av = None

try:
    from tensorflow.keras.models import load_model
    from tensorflow import keras
    from tensorflow.keras import initializers as keras_initializers
except Exception:  # pragma: no cover
    load_model = None
    keras = None
    keras_initializers = None

try:
    from googletrans import Translator
except Exception:  # pragma: no cover
    Translator = None

try:
    from gtts import gTTS
except Exception:  # pragma: no cover
    gTTS = None

APP_ROOT = Path(__file__).resolve().parent

DEFAULT_ACTIONS = [
    "hello",
    "thanks",
    "iloveyou",
    "yes",
    "no",
    "please",
    "sorry",
    "help",
    "stop",
    "goodbye",
]

MODEL_CANDIDATES = [
    APP_ROOT / "action.h5",
    APP_ROOT / "m1.h5",
]

ACTIONS_PATH = APP_ROOT / "actions.json"

LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Punjabi": "pa",
    "Urdu": "ur",
    "Odia": "or",
    "Assamese": "as",
}


def load_actions() -> List[str]:
    if ACTIONS_PATH.exists():
        try:
            data = json.loads(ACTIONS_PATH.read_text())
            if isinstance(data, list) and all(isinstance(x, str) for x in data):
                return data
        except Exception:
            pass
    return DEFAULT_ACTIONS


def _clean_config(obj):
    if isinstance(obj, dict):
        return {k: _clean_config(v) for k, v in obj.items() if k not in {"module", "registered_name"}}
    if isinstance(obj, list):
        return [_clean_config(x) for x in obj]
    return obj


def _load_model_flexible(path: Path) -> Optional[object]:
    if load_model is None or keras is None:
        return None

    # Try normal load first.
    try:
        return load_model(path, compile=False)
    except Exception:
        pass

    # Fallback: clean config and load weights manually.
    try:
        with h5py.File(path, "r") as f:
            cfg = f.attrs.get("model_config")
            if cfg is None:
                return None
            if isinstance(cfg, bytes):
                cfg = cfg.decode("utf-8")
        cfg_dict = json.loads(cfg)
        cfg_clean = _clean_config(cfg_dict)
        model = keras.models.model_from_json(json.dumps(cfg_clean))
        model.load_weights(path)
        return model
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_sign_model() -> Optional[object]:
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            model = _load_model_flexible(candidate)
            if model is not None:
                return model
    return None


def extract_keypoints(results) -> np.ndarray:
    if results is None:
        return np.zeros(1662)

    pose = (
        np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten()
        if results.pose_landmarks
        else np.zeros(33 * 4)
    )
    face = (
        np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
        if results.face_landmarks
        else np.zeros(468 * 3)
    )
    left_hand = (
        np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
        if results.left_hand_landmarks
        else np.zeros(21 * 3)
    )
    right_hand = (
        np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
        if results.right_hand_landmarks
        else np.zeros(21 * 3)
    )
    return np.concatenate([pose, face, left_hand, right_hand])


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image, results


def draw_styled_landmarks(image, results) -> None:
    if mp_solutions is None:
        return
    mp_drawing = mp_solutions.drawing_utils
    mp_holistic = mp_solutions.holistic
    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1),
    )
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_holistic.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2),
    )
    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
    )


def sample_frames(video_path: Path, max_frames: int = 30) -> List[np.ndarray]:
    if cv2 is None:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    frames = []
    try:
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
    finally:
        cap.release()

    return frames


def predict_from_frames(frames: List[np.ndarray], actions: List[str]) -> Tuple[str, float]:
    model = load_sign_model()
    if model is None or mp_solutions is None:
        return "Model or MediaPipe not available", 0.0

    sequence = []
    with mp_solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        refine_face_landmarks=True,
    ) as holistic:
        for frame in frames:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(image)
            sequence.append(extract_keypoints(results))

    if not sequence:
        return "No frames processed", 0.0

    if len(sequence) < 30:
        pad = [np.zeros_like(sequence[0]) for _ in range(30 - len(sequence))]
        sequence.extend(pad)
    else:
        sequence = sequence[-30:]

    preds = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    label = actions[idx] if idx < len(actions) else "Unknown"
    return label, confidence


def translate_text(text: str, target_lang: str) -> str:
    if Translator is None or target_lang == "en":
        return text
    try:
        translator = Translator()
        return translator.translate(text, dest=target_lang).text
    except Exception:
        return text


def text_to_speech(text: str, lang: str) -> Optional[bytes]:
    if gTTS is None:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        audio_path = APP_ROOT / "_tts_output.mp3"
        tts.save(str(audio_path))
        return audio_path.read_bytes()
    except Exception:
        return None


def main() -> None:
    st.set_page_config(page_title="SpeakEasy", page_icon="🖐️", layout="centered")

    st.title("SpeakEasy")
    st.write("Sign language to text, translation, and voice output.")

    actions = load_actions()

    st.sidebar.header("Settings")
    target_language = st.sidebar.selectbox("Output language", list(LANGUAGES.keys()), index=0)
    confidence_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.6, 0.05)
    enable_tts = st.sidebar.checkbox("Enable text-to-speech", value=True)

    st.subheader("Live webcam")
    if webrtc_streamer is None or av is None:
        st.warning("Live webcam requires `streamlit-webrtc` and `av`.")
    elif load_sign_model() is None or mp_solutions is None or cv2 is None:
        st.info("Live webcam needs a model plus MediaPipe/OpenCV.")
    else:
        import threading

        st.caption("Allow camera access in the browser. Press Stop when finished.")

        if "live_state" not in st.session_state:
            st.session_state["live_state"] = {
                "sequence": [],
                "sentence": [],
                "label": "",
                "confidence": 0.0,
                "last_tts_label": "",
            }
        if "live_lock" not in st.session_state:
            st.session_state["live_lock"] = threading.Lock()

        live_state = st.session_state["live_state"]
        live_lock = st.session_state["live_lock"]

        model = load_sign_model()
        actions = load_actions()
        threshold = confidence_threshold

        holistic = mp_solutions.holistic.Holistic(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            image, results = mediapipe_detection(img, holistic)
            draw_styled_landmarks(image, results)

            keypoints = extract_keypoints(results)
            with live_lock:
                sequence = live_state["sequence"]
                sequence.append(keypoints)
                live_state["sequence"] = sequence[-30:]

                if len(live_state["sequence"]) == 30:
                    preds = model.predict(np.expand_dims(live_state["sequence"], axis=0), verbose=0)[0]
                    idx = int(np.argmax(preds))
                    confidence = float(preds[idx])
                    label = actions[idx] if idx < len(actions) else "Unknown"
                    live_state["label"] = label
                    live_state["confidence"] = confidence

                    if confidence >= threshold:
                        sentence = live_state["sentence"]
                        if not sentence or label != sentence[-1]:
                            sentence.append(label)
                            live_state["sentence"] = sentence[-5:]

                sentence_text = " ".join(live_state["sentence"][-3:])

            cv2.rectangle(image, (0, 0), (image.shape[1], 40), (245, 117, 16), -1)
            cv2.putText(
                image,
                sentence_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            return av.VideoFrame.from_ndarray(image, format="bgr24")

        webrtc_streamer(
            key="live",
            video_frame_callback=video_frame_callback,
            media_stream_constraints={"video": True, "audio": False},
        )

        with live_lock:
            label = live_state["label"]
            confidence = live_state["confidence"]

        if label:
            st.markdown(f"**Live prediction:** `{label}`  **Confidence:** `{confidence:.2f}`")
            if confidence >= confidence_threshold:
                target_code = LANGUAGES[target_language]
                translated = translate_text(label, target_code)
                st.markdown(f"**Translated ({target_language}):** `{translated}`")

                if enable_tts:
                    with live_lock:
                        last_tts_label = live_state["last_tts_label"]
                    if label != last_tts_label:
                        audio_bytes = text_to_speech(translated, target_code)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                            with live_lock:
                                live_state["last_tts_label"] = label
                        else:
                            st.warning("TTS unavailable. Install gTTS or check internet access.")

    st.subheader("Upload a short sign video")
    video_file = st.file_uploader("Upload MP4/AVI/MOV", type=["mp4", "avi", "mov", "mkv"])

    if video_file:
        temp_path = APP_ROOT / "_upload_video"
        temp_path.write_bytes(video_file.read())

        with st.spinner("Analyzing video..."):
            frames = sample_frames(temp_path)
            label, confidence = predict_from_frames(frames, actions)

        st.markdown(f"**Predicted sign:** `{label}`")
        st.markdown(f"**Confidence:** `{confidence:.2f}`")

        if confidence >= confidence_threshold and label not in {
            "Model or MediaPipe not available",
            "No frames processed",
        }:
            target_code = LANGUAGES[target_language]
            translated = translate_text(label, target_code)
            st.markdown(f"**Translated ({target_language}):** `{translated}`")

            if enable_tts:
                audio_bytes = text_to_speech(translated, target_code)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")
                else:
                    st.warning("TTS unavailable. Install gTTS or check internet access.")
        else:
            st.info("Low confidence or missing model. Try another video or adjust the threshold.")

    st.subheader("Quick checks")
    if load_sign_model() is None:
        st.warning("No model found. Place `action.h5` or `m1.h5` in the project root.")
    if cv2 is None:
        st.warning("OpenCV is missing. Install `opencv-python`.")
    if mp is None:
        st.warning("MediaPipe is missing. Install `mediapipe`.")
    if Translator is None:
        st.warning("Translation library missing. Install `googletrans`.")
    if gTTS is None:
        st.warning("TTS library missing. Install `gTTS`.")
    if webrtc_streamer is None or av is None:
        st.warning("Live webcam requires `streamlit-webrtc` and `av`.")


if __name__ == "__main__":
    main()
