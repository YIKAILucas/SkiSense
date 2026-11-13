"""Analyze one normalized clip with the upstream scoring and drawing pipeline."""
import importlib
import sys

import cv2
import torch


if __name__ == "__main__":
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    upstream = importlib.import_module("src.skisense.main")
    upstream.OUTPUT_DIR = sys.argv[2]
    upstream.process_video(video_file=sys.argv[1], fast_mode=True,
                           target_mode="largest", pose_backend_name="yolo11")
