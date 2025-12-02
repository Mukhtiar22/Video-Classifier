# Video-Classifier
🎥 Human Action Recognition using CNN + LSTM

This project is an AI-based Human Action Recognition system built using a hybrid CNN + LSTM deep learning model.
It takes a video as input, extracts frames, processes them, and predicts the action being performed.
A fully interactive Streamlit web app is included for easy testing and visualization.

🚀 Features

1. Upload any video directly in the browser

2. Extract and preview 20 frames used for prediction

3. Classify actions using a CNN + LSTM model

4. Display prediction probabilities for all classes

5. Clean and interactive Streamlit UI

🧰 Tech Stack

TensorFlow / Keras

OpenCV

NumPy

Streamlit

Python

🎯 Dataset (UCF50)

This model is trained on a subset of the UCF50 dataset containing 18 action classes:

Basketball, Diving, HorseRace, JumpRope, VolleyballSpiking,
WalkingWithDog, BenchPress, Biking, GolfSwing, HighJump,
Kayaking, PullUps, PushUps, RopeClimbing, SkateBoarding,
SoccerJuggling, Swing, TrampolineJumping

🏗 How It Works

Video is uploaded through the Streamlit app

The system extracts 20 evenly spaced frames

Each frame is resized and preprocessed

CNN extracts spatial features

LSTM learns temporal patterns

The model outputs a prediction + probability distribution

▶️ Run the Project Locally
1. Clone the repository
git clone https://github.com/Mukhtiar22/Video-Classifier.git
cd Video-Classifier

2. Install dependencies
pip install -r requirements.txt

3. Run the app
streamlit run app.py

📂 Project Structure
│── app.py                     # Streamlit interface + inference
│── action_recognition_model.h5 # Trained CNN+LSTM model
│── README.md                  # Documentation

🧠 What I Learned

Video processing & frame sampling

CNN + LSTM sequence modeling

Real-time inference pipelines

Deploying ML models with Streamlit

Understanding action recognition challenges

📬 Contact

If you'd like to collaborate on AI or video-based projects:

Mukhtiar Ali 
GitHub: https://github.com/Mukhtiar22

LinkedIn: https://www.linkedin.com/in/mukhtiar-alie-