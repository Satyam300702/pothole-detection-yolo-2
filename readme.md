# 🚧 Pothole Detection & Tracking using YOLO

A Computer Vision project that detects potholes in road videos using a **custom-trained YOLO model** and tracks detected potholes across video frames using **ByteTrack**.

The project also includes a **Streamlit-based web interface** that allows users to upload road videos, adjust detection settings, process the video, visualize the results, and download the processed output.

---

## 📌 Project Overview

Potholes are a common road-safety problem that can damage vehicles, cause accidents, and make roads difficult to travel on.

The goal of this project was to build an end-to-end computer vision system capable of automatically detecting potholes from road images/videos and making the system easy for other people to use.

Instead of simply using a pre-trained pothole detection model, I worked on the complete pipeline myself:

```text
Dataset Collection
       ↓
Image Selection
       ↓
Manual Annotation
       ↓
Dataset Preparation
       ↓
YOLO Model Training
       ↓
Model Evaluation
       ↓
Multiple Training Experiments
       ↓
Video Detection & Tracking
       ↓
Streamlit UI
       ↓
Final Application

🎯 Project Objectives

The main objectives of this project were:

Detect potholes automatically using Computer Vision.
Train a custom YOLO model specifically for pothole detection.
Create and prepare my own annotated dataset.
Detect potholes in road videos.
Track potholes across multiple video frames.
Assign tracking IDs to detected potholes.
Build a simple and user-friendly interface using Streamlit.
Allow users to upload their own road videos.
Allow users to adjust confidence and image-size settings.
Display the processed video.
Allow users to download the final result.

📊 Dataset Preparation

Finding a suitable pothole dataset was one of the first challenges I faced.

Getting a dataset with:

Good-quality road images
Different pothole sizes
Different road conditions
Different lighting conditions
Different camera angles
Accurate annotations

was not easy.

I had to spend significant time searching for useful data and preparing it for training.

🔄 Model Training

After preparing the dataset, I trained a custom YOLO model for pothole detection.

The first training attempt was not the final model.

I had to train the model multiple times and experiment with different training configurations.

Some of the factors I worked with included:

Number of epochs
Image size
Batch size
Confidence threshold
Data augmentation
Learning-rate scheduling
Model configuration

The purpose of these experiments was to improve the model's ability to detect potholes while reducing incorrect detections.

🛠️ Technologies Used
Technology	Purpose
Python	Main programming language
YOLO	Pothole detection
Ultralytics	YOLO implementation
OpenCV	Video processing
ByteTrack	Object tracking
Streamlit	Web application
Git & GitHub	Version control and project hosting

📈 What I Learned From This Project

This project taught me much more than simply how to train a YOLO model.

I learned how to work through an end-to-end Computer Vision project.

Some of the major things I learned were:

Dataset collection and preparation
Image annotation
Object detection
YOLO model training
Model evaluation
Hyperparameter experimentation
Data augmentation
Video processing with OpenCV
Object tracking
ByteTrack
Streamlit application development
Handling uploaded files
Building a user-facing ML application
Debugging real-world ML problems
Iterating on models instead of expecting the first training run to work perfectly

💡 Challenges & Experience

This project involved several difficulties.

Finding appropriate data took considerable effort.

Creating accurate annotations manually was time-consuming.

The first trained models did not provide the performance I wanted, so I had to experiment with the dataset and training configuration and train the model multiple times.

I also faced challenges while integrating the trained model with video processing and object tracking.

Finally, converting the Python-based detection system into a simple Streamlit interface required additional work around file uploads, temporary files, video processing, progress tracking, and displaying downloadable results.

These challenges were an important part of the project because they helped me understand that real-world AI development involves continuous experimentation, debugging, and improvement.

🚀 Future Improvements

There are several improvements I would like to add in the future:

🖼️ Image pothole detection
📊 Detailed detection statistics
📈 Model performance dashboard
📏 Pothole size estimation
🗺️ GPS-based pothole mapping
📍 Pothole location tracking
📱 More responsive UI
☁️ Cloud deployment
⚡ Faster video processing
🧠 Further model optimization
📸 Automatic extraction of detected pothole images
📊 Generate a pothole detection report

👨‍💻 Author

Satyam Kumar

Computer Vision | Machine Learning | AI