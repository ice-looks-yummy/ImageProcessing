"""Local image processing lab built with Flask and OpenCV."""
from __future__ import annotations

import base64
import io
import os
import secrets
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file, session
from werkzeug.exceptions import RequestEntityTooLarge

ROOT = Path(__file__).resolve().parent
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}
MAX_UPLOAD = 12 * 1024 * 1024
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD

# Images are held in memory for this local, single-process teaching application.
IMAGES: dict[str, dict[str, object]] = {}

FILTERS = {
    "original": {"label": "Original", "equation": "g(x,y) = f(x,y)", "description": "The uploaded image, unchanged."},
    "grayscale": {"label": "Grayscale", "equation": "Gray = 0.299R + 0.587G + 0.114B", "description": "A luminance-weighted conversion from RGB to one intensity channel."},
    "gaussian": {"label": "Gaussian Blur", "equation": "G(x,y) = 1/(2πσ²) · e<sup>−(x²+y²)/(2σ²)</sup>", "description": "A weighted neighborhood average that reduces high-frequency detail and noise."},
    "median": {"label": "Median Blur", "equation": "g(x,y) = median{ f(x−i,y−j) }", "description": "Each pixel becomes the median value in its local neighborhood."},
    "sharpen": {"label": "Sharpen", "equation": "g(x,y) = Σ<sub>i,j</sub> K(i,j) · f(x−i,y−j)", "description": "Convolution with a center-weighted kernel emphasizes local detail."},
    "sobel": {"label": "Sobel Edge Detection", "equation": "G = √(Gx² + Gy²)", "description": "Horizontal and vertical intensity derivatives reveal image edges."},
    "canny": {"label": "Canny Edge Detection", "equation": "Smooth → gradient → non-maximum suppression → double threshold → hysteresis", "description": "A multi-stage edge detector that links strong edges through weak edge candidates."},
    "threshold": {"label": "Threshold", "equation": "g(x,y) = 255 if f(x,y) > T; otherwise 0", "description": "Converts a grayscale image to a binary mask using threshold T."},
    "negative": {"label": "Negative / Invert", "equation": "g(x,y) = 255 − f(x,y)", "description": "Inverts each 8-bit channel value."},
    "sepia": {"label": "Sepia", "equation": "[R′ G′ B′]ᵀ = [[.393,.769,.189],[.349,.686,.168],[.272,.534,.131]] [R G B]ᵀ", "description": "A color transform mixes RGB channels to create a warm sepia tone."},
    "brightness": {"label": "Brightness", "equation": "g(x,y) = clip(f(x,y) + β, 0, 255)", "description": "Adds an offset β to each pixel value."},
    "contrast": {"label": "Contrast", "equation": "g(x,y) = clip(α(f(x,y) − 128) + 128, 0, 255)", "description": "Scales pixel distance from mid-gray using contrast factor α."},
}


def odd_kernel(value, default=3):
    try:
        n = int(value)
    except (TypeError, ValueError):
        n = default
    return min(11, max(3, n if n % 2 else n + 1))


def apply_filter(image: np.ndarray, name: str, params: dict) -> np.ndarray:
    """Apply one named OpenCV/NumPy operation to a BGR source image."""
    if name == "original":
        return image.copy()
    if name == "grayscale":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if name == "gaussian":
        k = odd_kernel(params.get("kernel", 5))
        sigma = float(np.clip(float(params.get("sigma", 1.2)), 0.1, 5.0))
        return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)
    if name == "median":
        return cv2.medianBlur(image, odd_kernel(params.get("kernel", 5)))
    if name == "sharpen":
        strength = float(np.clip(float(params.get("strength", 1.0)), 0.0, 3.0))
        kernel = np.array([[0, -strength, 0], [-strength, 1 + 4 * strength, -strength], [0, -strength, 0]], dtype=np.float32)
        return cv2.filter2D(image, -1, kernel)
    if name == "sobel":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mode = params.get("direction", "magnitude")
        result = gx if mode == "x" else gy if mode == "y" else cv2.magnitude(gx, gy)
        return cv2.convertScaleAbs(result)
    if name == "canny":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        low = int(np.clip(int(params.get("low", 60)), 0, 254))
        high = int(np.clip(int(params.get("high", 150)), low + 1, 255))
        aperture = int(params.get("aperture", 3))
        if aperture not in (3, 5, 7):
            aperture = 3
        return cv2.Canny(gray, low, high, apertureSize=aperture)
    if name == "threshold":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        threshold = int(np.clip(int(params.get("threshold", 128)), 0, 255))
        return cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]
    if name == "negative":
        return cv2.bitwise_not(image)
    if name == "sepia":
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        matrix = np.array([[.393, .769, .189], [.349, .686, .168], [.272, .534, .131]], dtype=np.float32)
        sepia = np.clip(rgb @ matrix.T, 0, 255).astype(np.uint8)
        return cv2.cvtColor(sepia, cv2.COLOR_RGB2BGR)
    if name == "brightness":
        beta = int(np.clip(int(params.get("beta", 0)), -100, 100))
        return cv2.convertScaleAbs(image, alpha=1, beta=beta) if beta >= 0 else np.clip(image.astype(np.int16) + beta, 0, 255).astype(np.uint8)
    if name == "contrast":
        alpha = float(np.clip(float(params.get("alpha", 1.0)), 0.2, 2.5))
        return np.clip(alpha * (image.astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)
    raise ValueError("Unknown filter selected.")


def image_stats(image: np.ndarray, file_size: int | None = None) -> dict:
    h, w = image.shape[:2]
    channels = 1 if image.ndim == 2 else image.shape[2]
    values = image.astype(np.float32)
    return {"resolution": f"{w} × {h}", "width": w, "height": h, "channels": channels,
            "color_mode": "Grayscale" if channels == 1 else ("RGB" if channels == 3 else "RGBA"),
            "file_size": file_size, "mean": round(float(values.mean()), 2),
            "std_dev": round(float(values.std()), 2), "min": int(image.min()), "max": int(image.max())}


def encode_image(image: np.ndarray) -> tuple[str, bytes]:
    ext = ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError("Could not encode the processed image.")
    payload = encoded.tobytes()
    return "data:image/png;base64," + base64.b64encode(payload).decode("ascii"), payload


def current_image():
    record = IMAGES.get(session.get("image_id", ""))
    if not record:
        return None
    return record


def process_payload(name: str, params: dict) -> dict:
    record = current_image()
    if record is None:
        raise ValueError("Upload an image before applying a filter.")
    if name not in FILTERS:
        raise ValueError("Choose a supported filter.")
    result = apply_filter(record["image"], name, params)
    data_url, payload = encode_image(result)
    record["processed"] = result
    record["processed_bytes"] = payload
    record["filter"] = name
    record["params"] = params
    return {"image": data_url, "stats": image_stats(result), "filter": name, "label": FILTERS[name]["label"]}


@app.get("/")
def index():
    return render_template("index.html", filters=FILTERS)


@app.post("/api/upload")
def upload():
    file = request.files.get("image")
    if not file or not file.filename:
        return jsonify(error="Choose an image file to upload."), 400
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify(error="Unsupported format. Choose JPG, PNG, or WEBP."), 400
    raw = file.stream.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        return jsonify(error="The image is too large. Maximum upload size is 12 MB."), 413
    image = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        return jsonify(error="This image could not be decoded. Try a valid JPG, PNG, or WEBP file."), 400
    h, w = image.shape[:2]
    if w * h > 30_000_000:
        return jsonify(error="This image has too many pixels. Maximum supported resolution is 30 megapixels."), 400
    image_id = secrets.token_urlsafe(18)
    IMAGES[image_id] = {"image": image, "original_bytes": len(raw), "processed": image.copy(), "processed_bytes": raw, "filter": "original", "params": {}}
    session["image_id"] = image_id
    original_url, _ = encode_image(image)
    return jsonify(original=original_url, processed=original_url, original_stats=image_stats(image, len(raw)), processed_stats=image_stats(image), filename=Path(file.filename).name)


@app.post("/api/process")
def process():
    body = request.get_json(silent=True) or {}
    try:
        result = process_payload(body.get("filter", "original"), body.get("params", {}))
        result["equation"] = FILTERS[result["filter"]]["equation"]
        result["description"] = FILTERS[result["filter"]]["description"]
        result["kernel"] = kernel_data(result["filter"], body.get("params", {}))
        return jsonify(result)
    except (ValueError, TypeError, cv2.error) as exc:
        return jsonify(error=str(exc)), 400


def kernel_data(name: str, params: dict):
    if name == "gaussian":
        k = odd_kernel(params.get("kernel", 5))
        sigma = float(np.clip(float(params.get("sigma", 1.2)), .1, 5))
        one = cv2.getGaussianKernel(k, sigma)
        matrix = (one @ one.T)
        return {"title": f"Gaussian kernel · {k} × {k}", "matrix": np.round(matrix, 5).tolist(), "note": f"Normalized weights · σ = {sigma:.2f}"}
    if name == "median":
        k = odd_kernel(params.get("kernel", 5))
        return {"title": f"Median neighborhood · {k} × {k}", "matrix": [["f" for _ in range(k)] for _ in range(k)], "note": "Sort neighborhood values and select the middle value; entries are not convolution weights."}
    if name == "sharpen":
        s = float(np.clip(float(params.get("strength", 1)), 0, 3))
        return {"title": "Sharpen convolution kernel", "matrix": [[0, -s, 0], [-s, 1 + 4 * s, -s], [0, -s, 0]], "note": f"Strength = {s:.1f}"}
    if name == "sobel":
        return {"title": "Sobel derivative kernels", "kernels": [{"label": "Gx", "matrix": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]}, {"label": "Gy", "matrix": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]}], "note": "Applied to grayscale; combined mode uses √(Gx² + Gy²)."}
    if name == "canny":
        return {"title": "Canny edge pipeline", "steps": ["Gaussian smoothing", "Gradient calculation", "Non-maximum suppression", "Double threshold", "Hysteresis tracking"], "note": "Canny is a multi-stage detector and does not use one displayed convolution kernel."}
    return None


@app.post("/api/compare")
def compare():
    body = request.get_json(silent=True) or {}
    names = body.get("filters", [])
    if not isinstance(names, list) or len(names) > 8:
        return jsonify(error="Choose up to eight comparison filters."), 400
    record = current_image()
    if record is None:
        return jsonify(error="Upload an image before comparing filters."), 400
    try:
        results = []
        for name in names:
            if name not in FILTERS:
                continue
            params = body.get("params", {}).get(name, {})
            result = apply_filter(record["image"], name, params)
            data, _ = encode_image(result)
            results.append({"filter": name, "label": FILTERS[name]["label"], "image": data, "stats": image_stats(result)})
        return jsonify(results=results)
    except (ValueError, TypeError, cv2.error) as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/download")
def download():
    record = current_image()
    if record is None or "processed_bytes" not in record:
        return jsonify(error="There is no processed image to download."), 400
    name = FILTERS.get(record.get("filter"), FILTERS["original"])["label"].lower().replace(" / ", "-").replace(" ", "-")
    return send_file(io.BytesIO(record["processed_bytes"]), mimetype="image/png", as_attachment=True, download_name=f"{name}-result.png")


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(error="The image is too large. Maximum upload size is 12 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
