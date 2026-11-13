"""Analyze one normalized clip with the upstream scoring and drawing pipeline."""
import importlib
import os
from pathlib import Path
import sys

import cv2
import torch


if __name__ == "__main__":
    video, output = map(Path, sys.argv[1:])
    os.chdir(video.parent)
    cv2.setNumThreads(1)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    from ultralytics import settings

    settings.update({"runs_dir": str(output / "yolo-runs"), "sync": False})
    upstream = importlib.import_module("src.skisense.main")
    upstream.OUTPUT_DIR = str(output)
    upstream.process_video(video_file=str(video), fast_mode=True,
                           target_mode="largest", pose_backend_name="yolo11")
