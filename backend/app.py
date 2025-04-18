import torch
import torch.nn as nn
import torchvision.transforms as transforms
import cv2
import os
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from backend.model import DeepfakeDetector
from backend.download_model import download_model

# Define constants
MODEL_PATH = "backend/models/deepfake_detector_v5.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_FOLDER = "backend/uploads"
ALLOWED_EXTENSIONS = {"mp4", "avi", "mov", "mkv"}

# Ensure model exists else download 
if not os.path.exists(MODEL_PATH):
    print("Model not found locally. Downloading...")
    download_model()

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define Flask app
app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the trained model
def load_model():
    model = DeepfakeDetector()
     # Load only the model's state_dict
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# Define transformations
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Function to extract frames from video
def extract_frames(video_path, frame_count=10):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None  # Error handling if video cannot be opened

    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames < frame_count:
        cap.release()
        return None

    step = max(total_frames // frame_count, 1)
    frame_indices = [i * step for i in range(frame_count)]

    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = transform(frame)
        frames.append(frame)

    cap.release()

    # Ensure exactly `frame_count` frames
    if len(frames) == frame_count:
        return torch.stack(frames).unsqueeze(0)
    return None

# File validation
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    print(file)
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        frames = extract_frames(file_path)
        if frames is None:
            os.remove(file_path)
            return jsonify({"error": "Could not process video frames"}), 400

        frames = frames.to(DEVICE)

        try:
            with torch.no_grad():
                output = model(frames)
                # probabilities = torch.nn.functional.softmax(output, dim=1)
                prediction = torch.argmax(output, dim=1).item()
                # confidence = probabilities[0][prediction].item()
                # print(probabilities)
                print(prediction)
            os.remove(file_path)  # Clean up uploaded file
            return jsonify({
                "prediction": "Fake" if prediction == 1 else "Real",
                # "confidence": round(confidence, 4)
            })

        except Exception as e:
            os.remove(file_path)
            return jsonify({"error": f"Model inference failed: {str(e)}"}), 500

    return jsonify({"error": "Invalid file format"}), 400

def run_app():
    app.run(host="0.0.0.0", port=5000, debug=True)
