from flask import Flask, jsonify, render_template, request, redirect, url_for, Response
import os
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort

# Create Flask application instance
app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Vehicle weights for density calculation
VEHICLE_WEIGHTS = {
    'car': 1.0,
    'motorbike': 0.5,
    'bicycle': 0.5,
    'bus': 2.5,
    'truck': 2.5
}

# Global variables
processing_status = {
    'current_file': None,
    'done': False,
    'lane1_count': 0,
    'lane2_count': 0,
    'total_count': 0,
    'car_count': 0,
    'bus_count': 0,
    'truck_count': 0,
    'motorbike_count': 0,
    'bicycle_count': 0,
    'lane1_weighted': 0.0,
    'lane2_weighted': 0.0,
    'total_weighted': 0.0
}

# Traffic history for trend prediction (last 3 cycles)
traffic_history = []  # stores dicts of past cycle results

# Global variable to store current frame for streaming
current_frame = None
frame_lock = threading.Lock()
video_processing = False

# Load YOLO model
print("Loading YOLO model...")
model = YOLO("../model/yolov8s.pt")
print("Model loaded successfully!")

# Class names for detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
              "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
              "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
              "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
              "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
              "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
              "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
              "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet",
              "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
              "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]


def calculate_trend():
    """Calculate traffic trend based on last 3 cycles"""
    if len(traffic_history) < 2:
        return "Not enough data", "⏳"
    
    current = traffic_history[-1]['total_weighted']
    previous = [h['total_weighted'] for h in traffic_history[:-1]]
    avg_prev = sum(previous) / len(previous)

    if avg_prev == 0:
        return "Not enough data", "⏳"

    change_pct = ((current - avg_prev) / avg_prev) * 100

    if change_pct > 10:
        return f"Increasing (+{change_pct:.1f}%)", "📈"
    elif change_pct < -10:
        return f"Decreasing ({change_pct:.1f}%)", "📉"
    else:
        return f"Stable ({change_pct:+.1f}%)", "➡️"


def get_congestion_level(weighted_density):
    """Return congestion label based on weighted density"""
    if weighted_density < 5:
        return "Low", "#4CAF50"
    elif weighted_density < 15:
        return "Moderate", "#FF9800"
    else:
        return "High", "#f44336"


def detect_road_area(frame):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    roi_top = int(height * 0.3)
    roi = edges[roi_top:, :]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_edges = cv2.morphologyEx(roi, cv2.MORPH_CLOSE, horizontal_kernel)
    contours, _ = cv2.findContours(horizontal_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        road_top = roi_top + y
        road_bottom = roi_top + y + h
        road_height = road_bottom - road_top
        if road_height > height * 0.2:
            return road_top, road_bottom, "HIGH"

    road_top = int(height * 0.4)
    road_bottom = int(height * 0.95)
    return road_top, road_bottom, "LOW (Using default)"


def find_optimal_lane_lines(frame):
    height, width = frame.shape[:2]
    road_top, road_bottom, confidence = detect_road_area(frame)
    road_height = road_bottom - road_top

    line1_y = road_top + int(road_height * 0.33)
    line2_y = road_top + int(road_height * 0.67)
    line1_y = max(50, min(line1_y, height - 100))
    line2_y = max(line1_y + 50, min(line2_y, height - 50))

    detection_method = f"Auto-detected (Road: {road_top}-{road_bottom}px, Confidence: {confidence})"

    print("\n" + "=" * 70)
    print("🎯 AUTOMATIC LANE DETECTION RESULTS:")
    print("=" * 70)
    print(f"   Frame size: {width}x{height}")
    print(f"   Road area: Top={road_top}px, Bottom={road_bottom}px")
    print(f"   Lane 1 position: {line1_y}px ({int((line1_y/height)*100)}% of frame)")
    print(f"   Lane 2 position: {line2_y}px ({int((line2_y/height)*100)}% of frame)")
    print(f"   Detection confidence: {confidence}")
    print("=" * 70 + "\n")

    return line1_y, line2_y, detection_method


def process_video(filepath):
    global processing_status, current_frame, video_processing, traffic_history

    video_processing = True
    processing_status['done'] = False
    processing_status['current_file'] = os.path.basename(filepath)

    # Reset counts
    processing_status['lane1_count'] = 0
    processing_status['lane2_count'] = 0
    processing_status['total_count'] = 0
    processing_status['car_count'] = 0
    processing_status['bus_count'] = 0
    processing_status['truck_count'] = 0
    processing_status['motorbike_count'] = 0
    processing_status['bicycle_count'] = 0
    processing_status['lane1_weighted'] = 0.0
    processing_status['lane2_weighted'] = 0.0
    processing_status['total_weighted'] = 0.0

    print(f"\n📹 Processing video: {filepath}")

    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    ret, first_frame = cap.read()
    if not ret:
        print("❌ Error: Could not read video")
        cap.release()
        return

    line1_y, line2_y, detection_method = find_optimal_lane_lines(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

    lane1_vehicles = set()
    lane2_vehicles = set()
    vehicle_types = {}
    counted_vehicles = set()

    # Per-lane weighted density accumulators
    lane1_weighted = 0.0
    lane2_weighted = 0.0

    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        results = model(frame, stream=True, verbose=False)
        detections = []

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = classNames[cls]

                if class_name in ["car", "motorbike", "bus", "truck", "bicycle"] and conf > 0.3:
                    detections.append([x1, y1, x2, y2, conf, class_name])

        tracked_objects = np.empty((0, 5))
        if len(detections) > 0:
            detections_np = np.array([d[:5] for d in detections])
            tracked_objects = tracker.update(detections_np)

            for i, track in enumerate(tracked_objects):
                if i < len(detections):
                    track_id = int(track[4])
                    vehicle_type = detections[i][5]
                    vehicle_types[track_id] = vehicle_type
        else:
            tracked_objects = tracker.update(np.empty((0, 5)))

        # Show lane detection method
        cv2.putText(frame, "Lanes: Auto-detected ✓", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Draw tracked vehicles
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            track_id = int(track_id)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            vehicle_type = vehicle_types.get(track_id, "vehicle")
            weight = VEHICLE_WEIGHTS.get(vehicle_type, 1.0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            label = f"ID:{track_id} {vehicle_type}(x{weight})"
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

            # Vehicle counting - Lane 1
            if line1_y - 10 < cy < line1_y + 10:
                if track_id not in lane1_vehicles:
                    lane1_vehicles.add(track_id)
                    lane1_weighted += weight
                    if track_id not in counted_vehicles:
                        counted_vehicles.add(track_id)
                        if vehicle_type == "car":
                            processing_status['car_count'] += 1
                        elif vehicle_type == "bus":
                            processing_status['bus_count'] += 1
                        elif vehicle_type == "truck":
                            processing_status['truck_count'] += 1
                        elif vehicle_type == "motorbike":
                            processing_status['motorbike_count'] += 1
                        elif vehicle_type == "bicycle":
                            processing_status['bicycle_count'] += 1

            # Vehicle counting - Lane 2
            if line2_y - 10 < cy < line2_y + 10:
                if track_id not in lane2_vehicles:
                    lane2_vehicles.add(track_id)
                    lane2_weighted += weight
                    if track_id not in counted_vehicles:
                        counted_vehicles.add(track_id)
                        if vehicle_type == "car":
                            processing_status['car_count'] += 1
                        elif vehicle_type == "bus":
                            processing_status['bus_count'] += 1
                        elif vehicle_type == "truck":
                            processing_status['truck_count'] += 1
                        elif vehicle_type == "motorbike":
                            processing_status['motorbike_count'] += 1
                        elif vehicle_type == "bicycle":
                            processing_status['bicycle_count'] += 1

        # Update counts
        processing_status['lane1_count'] = len(lane1_vehicles)
        processing_status['lane2_count'] = len(lane2_vehicles)
        processing_status['total_count'] = processing_status['lane1_count'] + processing_status['lane2_count']
        processing_status['lane1_weighted'] = round(lane1_weighted, 1)
        processing_status['lane2_weighted'] = round(lane2_weighted, 1)
        processing_status['total_weighted'] = round(lane1_weighted + lane2_weighted, 1)

        # Display counts on video (LEFT SIDE)
        y_offset = 30
        cv2.putText(frame, f"Lane 1: {processing_status['lane1_count']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        y_offset += 40
        cv2.putText(frame, f"Lane 2: {processing_status['lane2_count']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        y_offset += 40
        cv2.putText(frame, f"Total: {processing_status['total_count']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
        y_offset += 35
        cv2.putText(frame, f"Density: {processing_status['total_weighted']}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Vehicle type counts (RIGHT SIDE)
        y_offset = 30
        x_offset = width - 200
        cv2.putText(frame, f"Cars: {processing_status['car_count']}", (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Buses: {processing_status['bus_count']}", (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Trucks: {processing_status['truck_count']}", (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Bikes: {processing_status['motorbike_count']}", (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 30
        cv2.putText(frame, f"Bicycles: {processing_status['bicycle_count']}", (x_offset, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Update frame for streaming
        with frame_lock:
            current_frame = frame.copy()

        time.sleep(0.01)

    cap.release()
    video_processing = False
    processing_status['done'] = True

    # Save this cycle to traffic history (max 3)
    traffic_history.append({
        'file': os.path.basename(filepath),
        'total_count': processing_status['total_count'],
        'lane1_count': processing_status['lane1_count'],
        'lane2_count': processing_status['lane2_count'],
        'total_weighted': processing_status['total_weighted'],
        'lane1_weighted': processing_status['lane1_weighted'],
        'lane2_weighted': processing_status['lane2_weighted'],
    })
    if len(traffic_history) > 3:
        traffic_history.pop(0)

    trend_label, trend_icon = calculate_trend()

    print("\n" + "=" * 70)
    print("✅ PROCESSING COMPLETE!")
    print("=" * 70)
    print(f"   Lane 1: {processing_status['lane1_count']} vehicles (Weighted: {processing_status['lane1_weighted']})")
    print(f"   Lane 2: {processing_status['lane2_count']} vehicles (Weighted: {processing_status['lane2_weighted']})")
    print(f"   Total Weighted Density: {processing_status['total_weighted']}")
    print(f"   Traffic Trend: {trend_icon} {trend_label}")
    print(f"   Detection method: {detection_method}")
    print("=" * 70 + "\n")


def generate_frames():
    global current_frame, video_processing

    timeout = 0
    while current_frame is None and timeout < 50:
        time.sleep(0.1)
        timeout += 1

    while video_processing or not processing_status['done']:
        with frame_lock:
            if current_frame is not None:
                ret, buffer = cv2.imencode('.jpg', current_frame)
                if ret:
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.03)

        if processing_status['done'] and not video_processing:
            break

    with frame_lock:
        if current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if ret:
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def home():
    return render_template('upload.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    global current_frame, video_processing

    if 'video' not in request.files:
        return redirect(request.url)

    file = request.files['video']

    if file.filename == '':
        return redirect(request.url)

    if file:
        current_frame = None
        video_processing = False

        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)

        processing_thread = threading.Thread(target=process_video, args=(filename,))
        processing_thread.daemon = True
        processing_thread.start()

        return redirect(url_for('stream'))


@app.route('/stream')
def stream():
    return render_template('stream.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/check_status')
def check_status():
    return jsonify(processing_status)


@app.route('/results')
def results():
    return render_template('results.html')


@app.route('/get_vehicle_counts')
def get_vehicle_counts():
    return jsonify({
        'lane1Count': processing_status['lane1_count'],
        'lane2Count': processing_status['lane2_count'],
        'totalCount': processing_status['total_count'],
        'carCount': processing_status['car_count'],
        'busCount': processing_status['bus_count'],
        'truckCount': processing_status['truck_count'],
        'motorbikeCount': processing_status['motorbike_count'],
        'bicycleCount': processing_status['bicycle_count'],
        'lane1Weighted': processing_status['lane1_weighted'],
        'lane2Weighted': processing_status['lane2_weighted'],
        'totalWeighted': processing_status['total_weighted']
    })


@app.route('/get_density_trend')
def get_density_trend():
    """Return weighted density, congestion level, and trend prediction"""
    lane1_w = processing_status['lane1_weighted']
    lane2_w = processing_status['lane2_weighted']
    total_w = processing_status['total_weighted']

    lane1_congestion, lane1_color = get_congestion_level(lane1_w)
    lane2_congestion, lane2_color = get_congestion_level(lane2_w)

    if lane1_w > lane2_w:
        busier_lane = "Lane 1"
        busier_by = round(lane1_w - lane2_w, 1)
    elif lane2_w > lane1_w:
        busier_lane = "Lane 2"
        busier_by = round(lane2_w - lane1_w, 1)
    else:
        busier_lane = "Equal"
        busier_by = 0

    trend_label, trend_icon = calculate_trend()

    return jsonify({
        'lane1_weighted': lane1_w,
        'lane2_weighted': lane2_w,
        'total_weighted': total_w,
        'lane1_congestion': lane1_congestion,
        'lane2_congestion': lane2_congestion,
        'lane1_color': lane1_color,
        'lane2_color': lane2_color,
        'busier_lane': busier_lane,
        'busier_by': busier_by,
        'trend_label': trend_label,
        'trend_icon': trend_icon,
        'history_count': len(traffic_history),
        'history': traffic_history
    })


@app.route('/traffic_signal')
def traffic_signal():
    return render_template('traffic_signal.html')


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "=" * 70)
    print("🚀 TRAFFIC DETECTION SYSTEM")
    print("   ✅ Automatic Lane Detection")
    print("   ✅ Weighted Traffic Density")
    print("   ✅ Lane Congestion Comparison")
    print("   ✅ Traffic Trend Prediction")
    print("   ✅ Vehicle Counting & Classification")
    print("=" * 70)
    print(f"🌐 Server: http://127.0.0.1:{port}")
    print("=" * 70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=port)