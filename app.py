import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import tempfile

MODEL_PATH = "/media/oem/5662C8D262C8B7CF1/FYP ideas/ucf50ActionRecognitionProject/action_recognition_model_CNN_LSTM_with_Augmentation.h5"
model = tf.keras.models.load_model(MODEL_PATH)

selected_classes = [
    "Basketball", "Diving", "HorseRace", "JumpRope", "VolleyballSpiking",
    "WalkingWithDog", "BenchPress", "Biking", "GolfSwing", "HighJump",
    "Kayaking", "PullUps", "PushUps", "RopeClimbing", "SkateBoarding",
    "SoccerJuggling", "Swing", "TrampolineJumping"
]

NUM_FRAMES = 20
IMG_SIZE = 64



def extract_frames(video_path, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    interval = max(total // num_frames, 1)
    frames = []
    count = 0

    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if count % interval == 0:
            frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame / 255.0
            frames.append(frame)

        count += 1

    cap.release()

    while len(frames) < num_frames:
        frames.append(frames[-1])

    return np.array(frames)


def classify_video(video_path):
    frames = extract_frames(video_path, NUM_FRAMES)
    frames = np.expand_dims(frames, axis=0)  

    probabilities = model.predict(frames, verbose=0)[0]
    pred_idx = np.argmax(probabilities)
    pred_class = selected_classes[pred_idx]

    return pred_class, probabilities


st.title(" 18 selected Classes of  UCF50 Dataset  Action Recognition Model")
st.write("Upload a video and the model will classify the action.")

uploaded = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded:
    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded.read())
    video_path = temp.name

    st.video(video_path)
    st.write("⏳ Extracting Frames...")

    frames = extract_frames(video_path, NUM_FRAMES)

    st.subheader("📸 Extracted Frames (20 Frames Used by Model)")

    cols = st.columns(5)  # 5 columns -> grid of 5x4

    for i, frame in enumerate(frames):
        col = cols[i % 5]
        col.image(frame, caption=f"Frame {i+1}", use_container_width=True)  # <-- updated

    st.write("⏳ Predicting Action...")

    pred_class, probs = classify_video(video_path)

    st.success(f"🔥 **Predicted Action:** {pred_class}")

    st.subheader("📊 Class Probabilities:")
    for i, cls in enumerate(selected_classes):
        st.write(f"{cls}: {probs[i]*100:.2f}%")
