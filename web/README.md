# SkiSense web deployment

Based on [kmch4n/SkiSense](https://github.com/kmch4n/SkiSense) at `8ef19827365f84a800b5302c5b323be092ed955a`.
The browser interface invokes the upstream drawing and scoring pipeline with the
CPU YOLO11n-Pose model. The 3D backend and ankle scoring are unavailable in this mode.

The service processes one MP4, MOV, or WebM clip at a time. Uploads must be at most
8 MiB and 10 seconds, with a longest side of 2160 pixels. The server resizes each
clip for CPU inference and generates a silent, annotated H.264 video and an optional
best-scoring image. It deletes uploaded clips after processing. Results are held in
temporary storage for up to 30 minutes or until the service restarts. The latest
12 completed jobs are retained at most.

Build with `docker build -t skisense .` and run with `docker run --rm -p 10000:10000 skisense`.
On Render, use the Docker runtime, `/healthz`, and one instance. The application
listens on `0.0.0.0:$PORT` (default 10000).

The upstream source remains MIT under the root `LICENSE`. The web adaptation is
licensed under [AGPL-3.0](COPYING); it combines the upstream project with Ultralytics
YOLO, distributed under AGPL-3.0. The web page links to its deployable source archive
at `/source`, the [versioned Ultralytics source](https://github.com/ultralytics/ultralytics/tree/v8.4.0),
and the [YOLO11 model release](https://github.com/ultralytics/assets/releases/tag/v8.4.0).
