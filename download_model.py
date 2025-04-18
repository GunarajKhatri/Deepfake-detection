import gdown
import os

# Create the models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Google Drive file ID
file_id = "1awRyKAvvmypeRQJKIzWA_C8B0xOHOrdB"
url = f"https://drive.google.com/uc?id={file_id}"

# Download destination
output = "models/deepfake_detector_v5.pth"

# Download the file
gdown.download(url, output, quiet=False)
