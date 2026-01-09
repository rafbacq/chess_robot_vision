# YOLOv8 detection wrapper
import ultralytics
import supervision
import torch
import cv2
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO

# Load a pretrained YOLOv8n model
model = YOLO('yolov8n.pt')

# Run inference on camera images with arguments
#model.predict()