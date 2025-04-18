import gdown
import os

def download_model():
    os.makedirs("backend/models", exist_ok=True)
    file_id = "1awRyKAvvmypeRQJKIzWA_C8B0xOHOrdB"
    url = f"https://drive.google.com/uc?id={file_id}"
    output = "backend/models/deepfake_detector_v5.pth"
    gdown.download(url, output, quiet=False)
