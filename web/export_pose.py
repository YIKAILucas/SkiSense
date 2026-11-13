"""Export the pinned model for the slim CPU runtime at image build time."""
from ultralytics import YOLO


YOLO("yolo11n-pose.pt").export(format="onnx", imgsz=640, nms=True, dynamic=False, opset=17)
