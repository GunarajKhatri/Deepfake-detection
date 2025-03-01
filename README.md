# Deepfake Detection System

This is a Deepfake-Detection platform.

## Prerequisites
Ensure you have the following installed before proceeding:
- Python 3.8+
- pip (Python package manager)
- Node.js (for frontend setup)

## Backend Setup

### 1. Clone the repository
```sh
git clone https://github.com/GunarajKhatri/Deepfake-detection.git
cd project-directory
```

### 2. Install dependencies
```sh
pip install -r requirements.txt
```

### 3. Run the backend server
```sh
python3 app.py
```
The server will start at `http://127.0.0.1:5000/`.

## Frontend Setup

### 1. Navigate to the frontend folder
```sh
cd frontend
```

### 2. Install dependencies
```sh
npm install
```

### 3. Start the frontend
```sh
npm run dev
```

The frontend will start at `http://localhost:3000/`.

## Usage
1. Open the frontend in a browser (`http://localhost:3000/`).
2. Upload a video file.
3. The backend will process the video and return whether it's real or fake.
4. The result will be displayed in the frontend with a confidence score.

## API Endpoint
**POST `/predict`**
- Accepts a video file (`mp4`, `avi`, `mov`, `mkv`).
- Returns a JSON response with the prediction (`Fake` or `Real`) and confidence score.

Example Response:
```json
{
  "prediction": "Fake",
  "confidence": 0.97
}
```

## License
This project is open-source and available under the MIT License.

