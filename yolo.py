from ultralytics import YOLO

import cv2

import csv 


# Load the YOLO model file into a model object (not a string)
model = YOLO("yolov8m-pose.pt")

results = model.predict(
    source  = "Downloads/columpioscam1-2025-01-07_16-51-50.mp4",
    tracker = "bytetrack.yaml",
    save = True,
)



