"""Run the exported YOLO11n-Pose model for SkiSense's full-frame CPU path."""

from dataclasses import dataclass
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from src.skisense.config import (
    MODEL_DIR,
    POSE_VISIBILITY_THRESHOLD,
    POSE_VISIBILITY_THRESHOLD_LEGS,
    YOLO_POSE_CONFIDENCE,
)
from src.skisense.pose_analyzer import analyze_ski_pose
from src.skisense.pose_topology import COCO_17


@dataclass(frozen=True, slots=True)
class Landmark:
    x: float
    y: float
    visibility: float


class OnnxPose:
    display_name = "YOLO11n-Pose (ONNX, COCO-17)"

    def __init__(self, model_path):
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        model_input, = self.session.get_inputs()
        batch, channels, height, width = model_input.shape
        if batch != 1 or channels != 3 or height != width or not isinstance(height, int):
            raise ValueError("SkiSense requires a fixed square YOLO pose model")
        self.input_name = model_input.name
        self.size = height

    def estimate_full_frame(self, frame):
        height, width = frame.shape[:2]
        scale = self.size / max(width, height)
        resized_width = round(width * scale)
        resized_height = round(height * scale)
        left = (self.size - resized_width) // 2
        top = (self.size - resized_height) // 2
        resized = cv2.resize(frame, (resized_width, resized_height))
        tensor = np.full((1, 3, self.size, self.size), 114 / 255, dtype=np.float32)
        np.multiply(
            resized[:, :, ::-1].transpose(2, 0, 1),
            1 / 255,
            out=tensor[0, :, top:top + resized_height, left:left + resized_width],
            dtype=np.float32,
        )
        output, = self.session.run(None, {self.input_name: tensor})
        entries = []
        for row in output[0]:
            confidence = float(row[4])
            if confidence <= 0 or confidence < YOLO_POSE_CONFIDENCE:
                continue
            landmarks = [
                Landmark(
                    x=float(np.clip((x - left) / (scale * width), 0, 1)),
                    y=float(np.clip((y - top) / (scale * height), 0, 1)),
                    visibility=float(visibility),
                )
                for x, y, visibility in row[6:].reshape(17, 3)
            ]
            analysis = analyze_ski_pose(
                landmarks,
                width,
                height,
                visibility_threshold=POSE_VISIBILITY_THRESHOLD,
                visibility_threshold_legs=POSE_VISIBILITY_THRESHOLD_LEGS,
                topology=COCO_17,
            )
            if not analysis:
                continue

            x1, y1, x2, y2 = row[:4]
            x1 = int(np.clip((x1 - left) / scale, 0, width))
            y1 = int(np.clip((y1 - top) / scale, 0, height))
            x2 = int(np.clip((x2 - left) / scale, 0, width))
            y2 = int(np.clip((y2 - top) / scale, 0, height))
            detection_bbox = (x1, y1, x2 - x1, y2 - y1)
            if x2 <= x1 or y2 <= y1:
                detection_bbox = (0, 0, width, height)
            entries.append(({
                "landmarks": landmarks,
                "bbox": (0, 0, width, height),
                "detection_bbox": detection_bbox,
                "torso_center": analysis.get("torso_center"),
                "shoulder_center": analysis.get("shoulder_center"),
                "hip_center": analysis.get("hip_center"),
                "topology": COCO_17,
                "confidence": confidence,
            }, analysis))
        return entries

    def close(self):
        self.session = None


def get_backend(*, running_mode, device, use_gpu, device_str, backend):
    model_path = os.environ.get("SKISENSE_ONNX_MODEL")
    if model_path is None:
        model_path = Path(MODEL_DIR) / "yolo11n-pose.onnx"
    return OnnxPose(model_path)
