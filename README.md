# Real-Time Eyeglass Detection Using Classical Image Processing

This project detects whether a person is wearing eyeglasses in real time using classical image processing techniques. It does not rely on machine learning models, but rather uses edge detection and image filtering to determine the presence of eyeglass frames from webcam video.

## 🧠 Project Overview

The system uses a webcam feed, detects the user's face and facial landmarks using `dlib`, aligns the face based on the eyes, and then applies:
- Gaussian Blur to reduce noise
- Sobel Filter to detect vertical edges
- Otsu Thresholding to binarize the image

It then evaluates two specific regions (under the eyes and across the nose bridge) where eyeglasses typically appear. If edge density in these areas exceeds a defined threshold, the system classifies the input as "With Glasses", otherwise "No Glasses".

## 📁 Project Structure

- `eyeglass_detector.py` — Main script
- `data/shape_predictor_5_face_landmarks.dat` — Pretrained dlib model (must be downloaded manually)

## 🛠 Technologies Used

- Python 3.x
- OpenCV
- dlib
- NumPy

## 🚀 Installation & Running

### Environment Setup

You can create a conda environment with all required packages using the provided `environment.yml` file:

```bash
conda env create -f environment.yml
conda activate glasses-detection
```

Alternatively, manually install packages with the specified versions:

```bash
pip install opencv-python==3.4.0.12 dlib==19.7.0 numpy==1.14
```

> You may also need CMake and Boost installed for dlib.

### 1. Install dependencies

```bash
pip install opencv-python dlib numpy
```

> You may also need CMake and Boost installed for dlib.

### 2. Download the Dlib Landmark Model

Download from: http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2  
Extract it and place the `.dat` file in a folder named `data/`.

### 3. Run the Script

```bash
python eyeglass_detector.py
```

### 4. Controls

- The program starts webcam automatically.
- Detected face, aligned view, and filter steps are displayed.
- Press `ESC` to exit.

## ⚙️ How It Works

1. Convert webcam frame to grayscale.
2. Detect face and 5 landmark points using dlib.
3. Use eye positions to align the face.
4. Apply Gaussian blur to reduce noise.
5. Apply Sobel filter to detect vertical edges.
6. Use Otsu’s method to binarize the result.
7. Analyze two regions (under eyes and between eyes).
8. If the weighted edge density is above 0.15 → “With Glasses”.

## 📌 Notes

- Lightweight and real-time; no training required.
- May not be accurate in very low lighting or side angles.
- Can be enhanced using deep learning or MediaPipe in future.

## 📄 License

This project is under the MIT License.
