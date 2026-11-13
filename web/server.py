import json
import logging
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from flask import Flask, jsonify, request, send_file
from werkzeug.exceptions import RequestEntityTooLarge


ROOT = Path(__file__).resolve().parent.parent
MAX_VIDEO_BYTES = 8 * 1024 * 1024
MAX_SECONDS = 10
TTL_SECONDS = 30 * 60
MAX_RESULTS = 12
ALLOWED = {".mp4", ".mov", ".webm"}
FILES = tempfile.TemporaryDirectory(prefix="skisense-")
JOBS = {}
LOCK = threading.Lock()
app = Flask(__name__)
app.config.update(MAX_CONTENT_LENGTH=MAX_VIDEO_BYTES + 32 * 1024, MAX_FORM_PARTS=4)


class InvalidVideo(Exception):
    pass


def ffmpeg(*args, timeout=25):
    return subprocess.run(args, check=True, capture_output=True, timeout=timeout, text=True)


def normalize(path, output):
    # Restrict demuxers and protocols to standalone files, with no remote fetches.
    source = ("-protocol_whitelist", "file", "-format_whitelist", "mov,matroska,webm", str(path))
    try:
        data = ffmpeg("ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                      "stream=width,height,duration:format=duration", "-of", "json", *source)
        meta = json.loads(data.stdout)
        stream = meta["streams"][0]
        duration = float(meta["format"].get("duration") or stream["duration"])
        width, height = stream["width"], stream["height"]
    except (subprocess.SubprocessError, ValueError, KeyError, IndexError) as exc:
        raise InvalidVideo("无法读取影片，请换一个 MP4、MOV 或 WebM 文件。") from exc
    if not math.isfinite(duration) or duration <= 0 or not 0 < width <= 2160 or not 0 < height <= 2160:
        raise InvalidVideo("影片尺寸或时长无效，请上传最长边不超过 2160 像素的影片。")
    if duration > MAX_SECONDS + 0.05:
        raise InvalidVideo("影片超过 10 秒，请裁剪后再上传。")
    try:
        ffmpeg("ffmpeg", "-nostdin", "-v", "error", "-y", "-threads", "1", *source[:-1],
               "-i", str(path), "-map", "0:v:0", "-an", "-sn", "-dn", "-t", str(MAX_SECONDS),
               "-vf", "fps=8,scale=640:640:force_original_aspect_ratio=decrease:force_divisible_by=2",
               "-filter_threads", "1", "-threads", "1", "-c:v", "libx264", "-preset", "ultrafast",
               "-pix_fmt", "yuv420p", str(output))
    except subprocess.SubprocessError as exc:
        raise InvalidVideo("影片无法解码，请转成常见的 H.264 MP4 后再试。") from exc


def work(job_id):
    job = JOBS[job_id]
    folder, original = job["folder"], job["original"]
    with LOCK:
        job["status"] = "running"
    result = {"status": "error", "error": "分析未能完成，请试试更短的影片。"}
    try:
        normal = folder / "normalized.mp4"
        normalize(original, normal)
        original.unlink()
        env = os.environ | {
            "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
            "SKISENSE_DEVICE": "cpu", "SKISENSE_SHOW_GUI": "false",
            "SKISENSE_YOLO_POSE_MODEL": "yolo11n-pose.pt", "YOLO_CONFIG_DIR": str(folder),
        }
        analysis = folder / "analysis"
        subprocess.run([sys.executable, "-m", "web.worker", str(normal), str(analysis)],
                       cwd=ROOT, env=env, check=True, timeout=240)
        raw = next(analysis.glob("*/video_pose_fast.mp4"), None)
        if not raw or raw.stat().st_size < 512:
            raise RuntimeError("Analyzer did not produce a playable result")
        ffmpeg("ffmpeg", "-nostdin", "-v", "error", "-threads", "1", "-i", str(raw),
               "-an", "-threads", "1", "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
               "-movflags", "+faststart", str(folder / "result.mp4"))
        best = raw.parent / "best_shot.jpg"
        if best.is_file():
            best.replace(folder / "best.jpg")
        result = {"status": "done", "best": (folder / "best.jpg").is_file()}
    except InvalidVideo as exc:
        result["error"] = str(exc)
    except Exception:
        logging.exception("Analysis failed for job %s", job_id)
    finally:
        shutil.rmtree(folder / "analysis", ignore_errors=True)
        for path in (original, folder / "normalized.mp4"):
            path.unlink(missing_ok=True)
        with LOCK:
            job.update(result, finished=time.monotonic())


def prune():
    now = time.monotonic()
    completed = sorted((j["finished"], k) for k, j in JOBS.items() if "finished" in j)
    for index, (finished, key) in enumerate(completed):
        if now - finished >= TTL_SECONDS or len(completed) - index > MAX_RESULTS:
            shutil.rmtree(JOBS.pop(key)["folder"], ignore_errors=True)


def expire_results():
    while True:
        time.sleep(30)
        with LOCK:
            prune()


threading.Thread(target=expire_results, daemon=True).start()


@app.get("/")
def home():
    return send_file(ROOT / "web" / "index.html")


@app.get("/healthz")
def health():
    return jsonify(ok=True)


@app.get("/source")
def source():
    return send_file(ROOT / "dist" / "SkiSense-source.zip", as_attachment=True, download_name="SkiSense-source.zip")


@app.errorhandler(RequestEntityTooLarge)
def too_large(_exc):
    return jsonify(error="文件超过 8 MB，请压缩后再上传。"), 413


@app.post("/api/jobs")
def upload():
    with LOCK:
        if any(j["status"] in ("queued", "running") for j in JOBS.values()):
            return jsonify(error="正在分析另一段影片，请稍后再试。"), 429
    upload_file = request.files.get("video")
    suffix = Path(upload_file.filename or "").suffix.lower() if upload_file else ""
    if suffix not in ALLOWED:
        return jsonify(error="请选择 MP4、MOV 或 WebM 影片。"), 400
    with LOCK:
        prune()
        if any(j["status"] in ("queued", "running") for j in JOBS.values()):
            return jsonify(error="正在分析另一段影片，请稍后再试。"), 429
        job_id = uuid.uuid4().hex
        folder = Path(FILES.name) / job_id
        folder.mkdir()
        path = folder / ("upload" + suffix)
        try:
            upload_file.save(path)
            size = path.stat().st_size
        except OSError:
            shutil.rmtree(folder, ignore_errors=True)
            logging.exception("Failed to write uploaded video")
            return jsonify(error="暂时无法保存影片，请稍后再试。"), 503
        if not 0 < size <= MAX_VIDEO_BYTES:
            shutil.rmtree(folder, ignore_errors=True)
            return jsonify(error="请选择非空且不超过 8 MB 的影片。"), (413 if size else 400)
        JOBS[job_id] = {"status": "queued", "folder": folder, "original": path}
    threading.Thread(target=work, args=(job_id,), daemon=True).start()
    return jsonify(id=job_id), 202


@app.get("/api/jobs/<job_id>")
def status(job_id):
    with LOCK:
        prune()
        job = JOBS.get(job_id)
        if job is None:
            return jsonify(error="找不到这次分析，结果可能已过期。请重新上传。"), 404
        return jsonify({key: job[key] for key in ("status", "error", "best") if key in job})


@app.get("/api/jobs/<job_id>/<artifact>")
def artifact(job_id, artifact):
    info = {"video": ("result.mp4", "video/mp4"), "best": ("best.jpg", "image/jpeg")}.get(artifact)
    with LOCK:
        job = JOBS.get(job_id)
        if not info or not job or job["status"] != "done" or not (job["folder"] / info[0]).is_file():
            return jsonify(error="结果不可用或已过期。"), 404
        return send_file(job["folder"] / info[0], mimetype=info[1], conditional=True, max_age=0)


@app.after_request
def headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "private, no-store"
    return response
