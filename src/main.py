import cv2
import sys
import math
import numpy as np
import time
import cvzone
from ultralytics import YOLO
from sort import Sort

# Get input and output video paths from command line arguments
input_video = sys.argv[1]  # Input video file path
output_video = sys.argv[2]  # Output video file path

# Load the YOLO model
model = YOLO("../model/yolov8s.pt")

# Video Capture and Writer setup
cap = cv2.VideoCapture(input_video)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_video, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

# Class names for detection
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

# Tracking setup
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
total_countsUp = {"car": [], "bus": [], "truck": []}
total_countsDown = {"car": [], "bus": [], "truck": []}

limitsUp = [250, 400, 575, 400]
limitsDown = [700, 450, 1125, 450]

while True:
    success, img = cap.read()
    if not success:
        break

    results = model(img, stream=True)
    detections = np.empty((0, 5))

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = math.ceil((box.conf[0] * 100)) / 100
            cls = int(box.cls[0])
            currentclass = classNames[cls]

            if currentclass in ["car", "bus", "truck", "motorbike"] and conf > 0.4:
                cvzone.cornerRect(img, (x1, y1, x2 - x1, y2 - y1), l=15)
                cvzone.putTextRect(img, f'{currentclass}', (x1, max(35, y1)), scale=2, thickness=2, offset=3)
                currentarray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentarray))

    # Tracking
    resultracker = tracker.update(detections)
    for results in resultracker:
        x1, y1, x2, y2, obj_id = map(int, results)
        cx, cy = x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2
        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        
        if limitsUp[0] < cx < limitsUp[2] and limitsUp[1]-15 < cy < limitsUp[3]+15:
            total_countsUp[currentclass].append(obj_id)
        if limitsDown[0] < cx < limitsDown[2] and limitsDown[1]-15 < cy < limitsDown[3]+15:
            total_countsDown[currentclass].append(obj_id)
    
    # Display count
    y_offset = 50
    for vehicle_type in ["car", "bus", "truck"]:
        cvzone.putTextRect(img, f'{vehicle_type} up count: {len(set(total_countsUp[vehicle_type]))}', (50, y_offset), scale=2, thickness=1, offset=3, colorT=(0, 0, 0))
        cvzone.putTextRect(img, f'{vehicle_type} down count: {len(set(total_countsDown[vehicle_type]))}', (1000, y_offset), scale=2, thickness=1, offset=3, colorT=(0, 0, 0))
        y_offset += 30

    out.write(img)
    cv2.imshow("Processed Video", img)
    cv2.waitKey(1)

cap.release()
out.release()
cv2.destroyAllWindows()
