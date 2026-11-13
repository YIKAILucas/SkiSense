FROM python:3.12-slim-bookworm AS pose-export
ENV OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /model
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir ultralytics==8.4.23 onnx==1.23.2 onnxslim==0.1.99 \
    && curl -fsSL --retry 3 https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n-pose.pt \
        -o yolo11n-pose.pt \
    && echo '869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0  yolo11n-pose.pt' | sha256sum --check
COPY web/export_pose.py .
RUN python export_pose.py

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates ffmpeg \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY web/requirements.txt /tmp/skisense-requirements.txt
RUN pip install --no-cache-dir -r /tmp/skisense-requirements.txt
COPY . .
COPY --from=pose-export /model/yolo11n-pose.onnx web/assets/yolo11n-pose.onnx
RUN python -m web.package_source && useradd --system --create-home skisense
USER skisense
EXPOSE 10000
CMD ["sh", "-c", "exec gunicorn --workers 1 --threads 3 --bind 0.0.0.0:${PORT:-10000} web.server:app"]
