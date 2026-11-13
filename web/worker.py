"""Analyze one normalized clip with the upstream scoring and drawing pipeline."""
import importlib
import os
from pathlib import Path
import sys

import cv2

from .onnx_backend import get_backend


if __name__ == "__main__":
    video, output = map(Path, sys.argv[1:])
    os.environ["SKISENSE_ONNX_MODEL"] = str(Path(__file__).resolve().parent / "assets" / "yolo11n-pose.onnx")
    cv2.setNumThreads(1)
    upstream = importlib.import_module("src.skisense.main")
    upstream.OUTPUT_DIR = str(output)
    upstream.process_video(video_file=str(video), fast_mode=True,
                           target_mode="largest", pose_backend_name="yolo11",
                           pose_backend_factory=get_backend)
