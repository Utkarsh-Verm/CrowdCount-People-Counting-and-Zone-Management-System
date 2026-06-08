import cv2
import json
import os
import time
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, Response, jsonify
from ultralytics import YOLO

app = Flask(__name__)

# --- Configuration ---
DATA_DIR = "data"
os.makedirs(DATA_DIR, exist_ok=True)
CONFIG_FILE = os.path.join(DATA_DIR, "zones_config.json")
LOG_FILE = os.path.join(DATA_DIR, "crowd_data.csv")
CROWD_LIMIT = 5

# --- Global State ---
model = YOLO('yolov8n.pt')
camera = cv2.VideoCapture(0)
system_stats = {"entries": 0, "exits": 0, "current_total": 0, "alert": False}
tracked_people = {}

def load_zones():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    return [{"name": "Main Zone", "coords": [100, 100, 500, 400]}] # Default zone

zones = load_zones()

def generate_frames():
    global system_stats, tracked_people
    
    while True:
        success, frame = camera.read()
        if not success:
            break
            
        h, w, _ = frame.shape
        line_y = h // 2
        
        # YOLOv8 Tracking
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        current_frame_total = 0
        system_stats["alert"] = False
        
        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
            ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            for box, pid in zip(boxes, ids):
                current_frame_total += 1
                x1, y1, x2, y2 = box
                cy = (y1 + y2) // 2
                
                # Entry/Exit Logic
                if pid in tracked_people:
                    if tracked_people[pid] < line_y and cy >= line_y:
                        system_stats["entries"] += 1
                    elif tracked_people[pid] > line_y and cy <= line_y:
                        system_stats["exits"] += 1
                tracked_people[pid] = cy
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Zone Monitoring
        for zone in zones:
            zx1, zy1, zx2, zy2 = zone["coords"]
            zone_count = 0
            if results[0].boxes.id is not None:
                for box in boxes:
                    bcx, bcy = (box[0] + box[2]) // 2, (box[1] + box[3]) // 2
                    if zx1 < bcx < zx2 and zy1 < bcy < zy2:
                        zone_count += 1
            
            color = (0, 0, 255) if zone_count > CROWD_LIMIT else (0, 255, 0)
            if zone_count > CROWD_LIMIT:
                system_stats["alert"] = True
                
            cv2.rectangle(frame, (zx1, zy1), (zx2, zy2), color, 2)
            cv2.putText(frame, f"{zone['name']}: {zone_count}", (zx1, zy1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        system_stats["current_total"] = current_frame_total
        cv2.line(frame, (0, line_y), (w, line_y), (0, 255, 255), 2)

        # Convert frame to byte format for web streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# --- API & Web Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def get_stats():
    # Frontend polls this to update numbers without refreshing the page
    return jsonify(system_stats)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)