# Face Recognition Attendance System

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/Maha200420/Face-Attendance-System)

A Python-based application for student attendance using face recognition (dlib, OpenCV, and Flask).

## Features
- Real-time face detection and alignment.
- Student signup and database registration.
- Daily attendance reporting (saved as CSV files in the `Attendance/` directory).
- Web-based local dashboard.

## Setup Instructions

1. **Install Dependencies**:
   Install the required Python packages:
   ```bash
   pip install opencv-python numpy scikit-learn Flask dlib mediapipe
   ```

2. **Download Landmarks Model**:
   Due to GitHub's file size limits, the `landmarks.dat` model file has been omitted from the repository. Follow these steps to set it up:
   - Download the compressed model file from [dlib.net](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
   - Extract the `.dat` file (using 7-Zip, WinRAR, or extraction commands).
   - Create a directory named `models` in the root of the project (if it doesn't exist).
   - Move the extracted file to `models/landmarks.dat` (rename the file to `landmarks.dat`).

3. **Run the Application**:
   Run the main application file:
   ```bash
   python app.py
   ```
   Open your browser and navigate to `http://127.0.0.1:5000` to access the interface.
