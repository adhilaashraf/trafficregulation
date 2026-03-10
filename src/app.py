from flask import Flask, jsonify, render_template, request, redirect, url_for, Response, session
import os
import time
import threading
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from functools import wraps

app = Flask(__name__)
app.secret_key = 'traffic_secret_key_2024'

UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── USER ACCOUNTS ─────────────────────────────────────────────────────────────
USERS = {
    'admin': {'password': 'admin123', 'role': 'admin', 'name': 'Admin'},
    'police': {'password': 'police123', 'role': 'police', 'name': 'Police Department'}
}

# ── EMAIL CONFIG ──────────────────────────────────────────────────────────────
EMAIL_CONFIG = {
    'sender': 'maxpaglu3@gmail.com',
    'password': 'uymf csns oiyf wqye',
    'receiver': 'maxpaglu69@gmail.com'
}

# ── VEHICLE WEIGHTS ───────────────────────────────────────────────────────────
VEHICLE_WEIGHTS = {'car': 1.0, 'motorbike': 0.5, 'bicycle': 0.5, 'bus': 2.5, 'truck': 2.5}

# ── GLOBAL STATE ──────────────────────────────────────────────────────────────
processing_status = {
    'current_file': None, 'done': False,
    'lane1_count': 0, 'lane2_count': 0, 'total_count': 0,
    'car_count': 0, 'bus_count': 0, 'truck_count': 0,
    'motorbike_count': 0, 'bicycle_count': 0,
    'lane1_weighted': 0.0, 'lane2_weighted': 0.0, 'total_weighted': 0.0
}

traffic_history = []
alert_log = []
email_sent_this_cycle = False
current_frame = None
frame_lock = threading.Lock()
video_processing = False

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
print("Loading YOLO model...")
model = YOLO("../model/yolov8s.pt")
print("Model loaded!")

classNames = ["person","bicycle","car","motorbike","aeroplane","bus","train","truck",
              "boat","traffic light","fire hydrant","stop sign","parking meter","bench",
              "bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe",
              "backpack","umbrella","handbag","tie","suitcase","frisbee","skis","snowboard",
              "sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
              "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl",
              "banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza",
              "donut","cake","chair","sofa","pottedplant","bed","diningtable","toilet",
              "tvmonitor","laptop","mouse","remote","keyboard","cell phone","microwave",
              "oven","toaster","sink","refrigerator","book","clock","vase","scissors",
              "teddy bear","hair drier","toothbrush"]

# ── AUTH DECORATORS ───────────────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'username' not in session:
            return redirect(url_for('login'))
        if session.get('role') != 'admin':
            return redirect(url_for('police_dashboard'))
        return f(*args, **kwargs)
    return decorated

# ── EMAIL ALERT ───────────────────────────────────────────────────────────────
def send_alert_email(subject, body):
    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_CONFIG['sender']
        msg['To'] = EMAIL_CONFIG['receiver']
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_CONFIG['sender'], EMAIL_CONFIG['password'])
        server.sendmail(EMAIL_CONFIG['sender'], EMAIL_CONFIG['receiver'], msg.as_string())
        server.quit()
        print(f"✅ Alert email sent!")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False

def check_and_send_alerts():
    global email_sent_this_cycle
    if email_sent_this_cycle:
        return

    alerts = []
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lane1_w = processing_status['lane1_weighted']
    lane2_w = processing_status['lane2_weighted']

    if lane1_w >= 15 or lane2_w >= 15:
        congested_lane = "Lane 1" if lane1_w >= lane2_w else "Lane 2"
        alerts.append(f"HIGH CONGESTION on {congested_lane} (Density: {max(lane1_w, lane2_w):.1f})")

    trend_label, _ = calculate_trend()
    if "Increasing" in trend_label:
        alerts.append(f"TRAFFIC INCREASING: {trend_label}")

    if alerts:
        alert_message = "\n".join(alerts)
        body = f"""Traffic Management System Alert

Time: {timestamp}
Video: {processing_status.get('current_file', 'Unknown')}

ALERTS:
{alert_message}

Lane 1 Density: {lane1_w:.1f}
Lane 2 Density: {lane2_w:.1f}
Total Vehicles: {processing_status['total_count']}

Please take necessary action.
- Traffic Management System"""

        sent = send_alert_email("Traffic Alert - Action Required", body)
        alert_log.append({
            'time': timestamp,
            'message': alert_message,
            'email_sent': sent,
            'lane1_density': lane1_w,
            'lane2_density': lane2_w,
            'total_vehicles': processing_status['total_count']
        })
        if len(alert_log) > 20:
            alert_log.pop(0)
        email_sent_this_cycle = True

# ── HELPERS ───────────────────────────────────────────────────────────────────
def calculate_trend():
    if len(traffic_history) < 2:
        return "Not enough data", "⏳"
    current = traffic_history[-1]['total_weighted']
    avg_prev = sum(h['total_weighted'] for h in traffic_history[:-1]) / len(traffic_history[:-1])
    if avg_prev == 0:
        return "Not enough data", "⏳"
    change_pct = ((current - avg_prev) / avg_prev) * 100
    if change_pct > 10:
        return f"Increasing (+{change_pct:.1f}%)", "📈"
    elif change_pct < -10:
        return f"Decreasing ({change_pct:.1f}%)", "📉"
    else:
        return f"Stable ({change_pct:+.1f}%)", "➡️"

def get_congestion_level(w):
    if w < 5: return "Low", "#4CAF50"
    elif w < 15: return "Moderate", "#FF9800"
    else: return "High", "#f44336"

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
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        road_top, road_bottom = roi_top + y, roi_top + y + h
        if (road_bottom - road_top) > height * 0.2:
            return road_top, road_bottom, "HIGH"
    return int(height * 0.4), int(height * 0.95), "LOW"

def find_optimal_lane_lines(frame):
    height, width = frame.shape[:2]
    road_top, road_bottom, confidence = detect_road_area(frame)
    road_height = road_bottom - road_top
    line1_y = max(50, min(road_top + int(road_height * 0.33), height - 100))
    line2_y = max(line1_y + 50, min(road_top + int(road_height * 0.67), height - 50))
    return line1_y, line2_y, f"Auto-detected (Confidence: {confidence})"

# ── VIDEO PROCESSING ──────────────────────────────────────────────────────────
def process_video(filepath):
    global processing_status, current_frame, video_processing, traffic_history, email_sent_this_cycle

    video_processing = True
    email_sent_this_cycle = False
    processing_status['done'] = False
    processing_status['current_file'] = os.path.basename(filepath)

    for key in ['lane1_count','lane2_count','total_count','car_count','bus_count',
                'truck_count','motorbike_count','bicycle_count']:
        processing_status[key] = 0
    processing_status['lane1_weighted'] = 0.0
    processing_status['lane2_weighted'] = 0.0
    processing_status['total_weighted'] = 0.0

    cap = cv2.VideoCapture(filepath)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return

    line1_y, line2_y, detection_method = find_optimal_lane_lines(first_frame)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
    lane1_vehicles, lane2_vehicles = set(), set()
    vehicle_types = {}        # track_id -> final stable type
    vehicle_type_votes = {}   # track_id -> {class: count} for voting
    counted_vehicles = set()
    lane1_weighted, lane2_weighted = 0.0, 0.0
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        results = model(frame, stream=True, verbose=False)
        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0]]
                conf = float(box.conf[0])
                class_name = classNames[int(box.cls[0])]
                box_w = x2 - x1
                box_h = y2 - y1
                # Fix 2a: higher confidence + minimum size to avoid signboards/false detections
                if class_name in ["car","motorbike","bus","truck","bicycle"] and conf > 0.5:
                    if box_w > 30 and box_h > 30:  # ignore tiny/false detections
                        detections.append([x1, y1, x2, y2, conf, class_name])

        if len(detections) > 0:
            tracked_objects = tracker.update(np.array([d[:5] for d in detections]))
            for i, track in enumerate(tracked_objects):
                if i < len(detections):
                    tid = int(track[4])
                    detected_class = detections[i][5]
                    # Fix 2b: voting system - stable type, prevents truck/car switching
                    if tid not in vehicle_type_votes:
                        vehicle_type_votes[tid] = {}
                    vehicle_type_votes[tid][detected_class] = vehicle_type_votes[tid].get(detected_class, 0) + 1
                    vehicle_types[tid] = max(vehicle_type_votes[tid], key=vehicle_type_votes[tid].get)
        else:
            tracked_objects = tracker.update(np.empty((0, 5)))

        cv2.putText(frame, "Lanes: Auto-detected", (10, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        for track in tracked_objects:
            x1, y1, x2, y2, track_id = int(track[0]), int(track[1]), int(track[2]), int(track[3]), int(track[4])
            cx, cy = (x1+x2)//2, (y1+y2)//2
            vehicle_type = vehicle_types.get(track_id, "vehicle")
            weight = VEHICLE_WEIGHTS.get(vehicle_type, 1.0)

            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,255), 2)
            cv2.circle(frame, (cx,cy), 4, (0,255,0), -1)
            cv2.putText(frame, f"ID:{track_id} {vehicle_type}(x{weight})",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 2)

            if line1_y - 10 < cy < line1_y + 10 and track_id not in lane1_vehicles:
                lane1_vehicles.add(track_id)
                lane1_weighted += weight
                if track_id not in counted_vehicles:
                    counted_vehicles.add(track_id)
                    count_key = f'{vehicle_type}_count'
                    if count_key in processing_status:
                        processing_status[count_key] += 1

            if line2_y - 10 < cy < line2_y + 10 and track_id not in lane2_vehicles:
                lane2_vehicles.add(track_id)
                lane2_weighted += weight
                if track_id not in counted_vehicles:
                    counted_vehicles.add(track_id)
                    count_key = f'{vehicle_type}_count'
                    if count_key in processing_status:
                        processing_status[count_key] += 1

        processing_status['lane1_count'] = len(lane1_vehicles)
        processing_status['lane2_count'] = len(lane2_vehicles)
        processing_status['total_count'] = processing_status['lane1_count'] + processing_status['lane2_count']
        processing_status['lane1_weighted'] = round(lane1_weighted, 1)
        processing_status['lane2_weighted'] = round(lane2_weighted, 1)
        processing_status['total_weighted'] = round(lane1_weighted + lane2_weighted, 1)

        if frame_count % 30 == 0:
            threading.Thread(target=check_and_send_alerts, daemon=True).start()

        y_offset = 30
        for text, color in [(f"Lane 1: {processing_status['lane1_count']}", (0,255,0)),
                             (f"Lane 2: {processing_status['lane2_count']}", (0,0,255)),
                             (f"Total: {processing_status['total_count']}", (255,255,0))]:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            y_offset += 40
        cv2.putText(frame, f"Density: {processing_status['total_weighted']}",
                    (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        x_offset, y_offset = width - 200, 30
        for label, key in [("Cars",'car_count'),("Buses",'bus_count'),("Trucks",'truck_count'),
                            ("Bikes",'motorbike_count'),("Bicycles",'bicycle_count')]:
            cv2.putText(frame, f"{label}: {processing_status[key]}",
                        (x_offset, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
            y_offset += 30

        with frame_lock:
            current_frame = frame.copy()
        time.sleep(0.01)

    cap.release()
    video_processing = False
    processing_status['done'] = True

    traffic_history.append({
        'file': os.path.basename(filepath),
        'total_count': processing_status['total_count'],
        'lane1_count': processing_status['lane1_count'],
        'lane2_count': processing_status['lane2_count'],
        'total_weighted': processing_status['total_weighted'],
        'lane1_weighted': processing_status['lane1_weighted'],
        'lane2_weighted': processing_status['lane2_weighted'],
        'time': datetime.now().strftime("%H:%M:%S")
    })
    if len(traffic_history) > 3:
        traffic_history.pop(0)

    check_and_send_alerts()
    print("\n✅ PROCESSING COMPLETE!")

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
                    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
        time.sleep(0.03)
        if processing_status['done'] and not video_processing:
            break
    with frame_lock:
        if current_frame is not None:
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if ret:
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.route('/login', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if username in USERS and USERS[username]['password'] == password:
            session['username'] = username
            session['role'] = USERS[username]['role']
            session['name'] = USERS[username]['name']
            return redirect(url_for('home') if USERS[username]['role'] == 'admin' else url_for('police_dashboard'))
        error = 'Invalid username or password!'
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

@app.route('/')
@admin_required
def home():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
@admin_required
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
        threading.Thread(target=process_video, args=(filename,), daemon=True).start()
        return redirect(url_for('stream'))

@app.route('/stream')
@login_required
def stream():
    return render_template('stream.html')

@app.route('/video_feed')
@login_required
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/check_status')
@login_required
def check_status():
    return jsonify(processing_status)

@app.route('/results')
@login_required
def results():
    return render_template('results.html')

@app.route('/get_vehicle_counts')
@login_required
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
@login_required
def get_density_trend():
    lane1_w = processing_status['lane1_weighted']
    lane2_w = processing_status['lane2_weighted']
    lane1_congestion, lane1_color = get_congestion_level(lane1_w)
    lane2_congestion, lane2_color = get_congestion_level(lane2_w)
    busier_lane = "Lane 1" if lane1_w > lane2_w else ("Lane 2" if lane2_w > lane1_w else "Equal")
    trend_label, trend_icon = calculate_trend()
    return jsonify({
        'lane1_weighted': lane1_w, 'lane2_weighted': lane2_w,
        'total_weighted': processing_status['total_weighted'],
        'lane1_congestion': lane1_congestion, 'lane2_congestion': lane2_congestion,
        'lane1_color': lane1_color, 'lane2_color': lane2_color,
        'busier_lane': busier_lane, 'busier_by': round(abs(lane1_w - lane2_w), 1),
        'trend_label': trend_label, 'trend_icon': trend_icon,
        'history_count': len(traffic_history), 'history': traffic_history
    })

@app.route('/police_dashboard')
@login_required
def police_dashboard():
    return render_template('police_dashboard.html',
                           username=session.get('name'), role=session.get('role'))

@app.route('/get_alerts')
@login_required
def get_alerts():
    return jsonify({'alerts': alert_log, 'total_alerts': len(alert_log)})

@app.route('/traffic_signal')
@admin_required
def traffic_signal():
    return render_template('traffic_signal.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print("\n" + "="*70)
    print("🚀 TRAFFIC DETECTION SYSTEM")
    print("   ✅ Login System | ✅ Email Alerts | ✅ Police Dashboard")
    print("   ✅ Weighted Density | ✅ Trend Prediction")
    print(f"🌐 http://127.0.0.1:{port}")
    print("="*70 + "\n")
    app.run(debug=True, host='0.0.0.0', port=port)