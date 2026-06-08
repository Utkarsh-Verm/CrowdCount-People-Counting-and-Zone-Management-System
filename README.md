CrowdCount-AI: Real-Time Zone Management & Analytics

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)
![Flask](https://img.shields.io/badge/Flask-Backend-green.svg)

CrowdCount-AI is a high-performance computer vision application designed to automate crowd monitoring in public spaces. Developed as a capstone project, it utilizes deep learning to detect, track, and log human movement within user-defined security zones.

Key Features
 Real-Time Human Tracking: Implements YOLOv8 and centroid-based tracking to maintain unique IDs across video frames, ensuring high counting accuracy.
 Interactive Web Dashboard: A responsive Flask/TailwindCSS frontend that streams live video and displays asynchronous analytics (entries, exits, current occupancy).
 Automated Safety Alerts: Triggers dynamic UI warnings when the crowd density in a specific zone exceeds configured thresholds.
 Data Persistence: Uses JSON for spatial zone configurations and Pandas for logging historical crowd traffic into CSV files.

Technology Stack
 AI & Vision: OpenCV, Ultralytics YOLOv8
 Backend: Python, Flask Web Framework
 Frontend: HTML5, JavaScript, Tailwind CSS
 Data Analytics: Pandas, Matplotlib

How to Run
1. Clone the repository: `git clone https://github.com/yourusername/CrowdCount-AI.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Start the application: `python app.py`
4. Open your browser and navigate to `http://localhost:5000`

Architecture Overview
The system captures raw RTSP/Webcam feeds via OpenCV, processing them through a YOLOv8 neural network. The bounding boxes are analyzed using spatial algorithms to determine entry/exit events across a virtual tripwire. The processed frames and analytics are then streamed via HTTP to a JavaScript-powered web dashboard.