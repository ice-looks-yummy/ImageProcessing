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
    "laplacian": {"label": "Laplacian Edge Detection", "equation": "∇²f = ∂²f/∂x² + ∂²f/∂y²", "description": "The Laplacian measures second-order intensity changes to highlight edges."},
    "bilateral": {"label": "Bilateral Filter", "equation": "I(p) = (1/Wp) Σq Gσs(||p−q||) Gσr(|I(p)−I(q)|) I(q)", "description": "A non-linear, edge-preserving weighted average uses both spatial distance and color similarity."},
    "equalize": {"label": "Histogram Equalization", "equation": "s = (L − 1) × CDF(r)", "description": "Redistributes luminance values using the cumulative intensity histogram."},
    "emboss": {"label": "Emboss", "equation": "g(x,y) = clip(128 + s · (K * f)(x,y), 0, 255)", "description": "A directional convolution creates raised and recessed shading."},
    "cartoon": {"label": "Cartoon Effect", "equation": "Multi-stage image-processing operation", "description": "Color smoothing is combined with an adaptive-threshold edge mask."},
    "gamma": {"label": "Gamma Correction", "equation": "g = 255 × (f / 255)<sup>γ</sup>", "description": "A power-law transform adjusts tonal brightness non-linearly."},
    "hue_saturation": {"label": "Hue / Saturation Adjustment", "equation": "BGR → HSV; H′ = (H + Δh/2) mod 180, S′ = clip(kS, 0, 255)", "description": "Hue and saturation are adjusted in HSV while preserving value (brightness). Hue shift is specified in degrees."},
}

INSTAGRAM_FILTERS = {
    "clarendon": "Clarendon Style", "gingham": "Gingham Style", "moon": "Moon Style",
    "lark": "Lark Style", "juno": "Juno Style", "valencia": "Valencia Style",
    "xpro": "X-Pro Style", "lofi": "Lo-Fi Style", "hefe": "Hefe Style",
    "nashville": "Nashville Style", "kelvin": "Kelvin Style", "inkwell": "Inkwell Style",
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
    if name == "laplacian":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        size = int(params.get("kernel", 3))
        if size not in (1, 3, 5, 7):
            size = 3
        response = cv2.Laplacian(gray, cv2.CV_32F, ksize=size)
        return cv2.convertScaleAbs(response)
    if name == "bilateral":
        diameter = int(np.clip(int(params.get("diameter", 9)), 3, 15))
        color_sigma = float(np.clip(float(params.get("sigma_color", 75)), 1, 150))
        space_sigma = float(np.clip(float(params.get("sigma_space", 75)), 1, 150))
        return cv2.bilateralFilter(image, diameter, color_sigma, space_sigma)
    if name == "equalize":
        # Equalize luminance only, preserving chroma in the original color image.
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lab[:, :, 0] = cv2.equalizeHist(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    if name == "emboss":
        strength = float(np.clip(float(params.get("strength", 1.0)), 0, 2.5))
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        response = cv2.filter2D(image.astype(np.float32), cv2.CV_32F, kernel)
        return np.clip(128 + strength * (response - 128), 0, 255).astype(np.uint8)
    if name == "cartoon":
        smooth = cv2.bilateralFilter(image, 9, 90, 90)
        smooth = cv2.bilateralFilter(smooth, 9, 90, 90)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 7)
        return cv2.bitwise_and(smooth, smooth, mask=edges)
    if name == "gamma":
        gamma = float(np.clip(float(params.get("gamma", 1.0)), .1, 3.0))
        lut = np.clip(np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255, 0, 255).astype(np.uint8)
        return cv2.LUT(image, lut)
    if name == "hue_saturation":
        hue_degrees = float(np.clip(float(params.get("hue", 0)), -180, 180))
        saturation = float(np.clip(float(params.get("saturation", 1.0)), 0, 2.5))
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + hue_degrees / 2, 180)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
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
    return render_template("index.html", filters=FILTERS, instagram_filters=INSTAGRAM_FILTERS)


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
    if name == "laplacian":
        try:
            size = int(params.get("kernel", 1))
        except (TypeError, ValueError):
            size = 1
        if size not in (1, 3, 5, 7):
            size = 1
        if size == 1:
            return {"title": "Laplacian kernel · 3 × 3", "matrix": [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], "note": "The displayed conceptual kernel matches OpenCV's 1-pixel Laplacian stencil (shown with opposite sign; absolute response is displayed)."}
        second_x, smooth_y = cv2.getDerivKernels(2, 0, size, normalize=False, ktype=cv2.CV_32F)
        smooth_x, second_y = cv2.getDerivKernels(0, 2, size, normalize=False, ktype=cv2.CV_32F)
        matrix = second_x @ smooth_y.T + smooth_x @ second_y.T
        return {"title": f"OpenCV Laplacian kernel · {size} × {size}", "matrix": np.round(matrix).astype(int).tolist(), "note": "Actual sum of OpenCV's horizontal and vertical second-derivative kernels."}
    if name == "bilateral":
        return {"title": "Bilateral filter information", "note": "Non-linear edge-preserving filter — no single fixed kernel. Weights combine spatial distance and pixel-value similarity."}
    if name == "equalize":
        return {"title": "Luminance histogram equalization", "steps": ["BGR → CIELAB", "Equalize L channel", "Keep a, b color channels", "CIELAB → BGR"], "note": "Uses the cumulative distribution function to remap luminance while retaining chroma."}
    if name == "emboss":
        strength = float(np.clip(float(params.get("strength", 1)), 0, 2.5))
        return {"title": "Emboss convolution kernel", "matrix": [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], "note": f"Kernel response is centered at gray and scaled by strength {strength:.1f}."}
    if name == "cartoon":
        return {"title": "Cartoon processing stages", "steps": ["Bilateral color smoothing", "Grayscale conversion", "Median smoothing", "Adaptive edge mask", "Apply mask to smoothed color"], "note": "Multi-stage image-processing operation — no single fixed kernel."}
    if name == "gamma":
        gamma = float(np.clip(float(params.get("gamma", 1)), .1, 3))
        return {"title": "Power-law transform", "note": f"Current gamma γ = {gamma:.2f}; lookup table maps each 8-bit input value."}
    if name == "hue_saturation":
        return {"title": "HSV color adjustment", "steps": ["BGR → HSV", "Shift hue channel", "Scale saturation", "Preserve value", "HSV → BGR"], "note": "Hue wraps around the OpenCV 0–179 range; saturation is clipped to 0–255."}
    return None


def instagram_effect(image: np.ndarray, name: str) -> np.ndarray:
    """Create a distinct, locally computed creative look inspired by photo presets."""
    if name not in INSTAGRAM_FILTERS:
        raise ValueError("Choose a supported inspired-style filter.")
    looks = {
        "clarendon": (1.16, 1.18, 1.00, (1.02, 1.00, 1.03), 0.0, 0.0),
        "gingham": (.88, .78, 1.00, (1.04, 1.01, .96), 5.0, 0.0),
        "moon": (1.18, 0.0, 1.0, (1.03, 1.00, .96), -2.0, 0.0),
        "lark": (.96, .86, .98, (1.02, 1.00, .97), 10.0, 0.0),
        "juno": (1.06, 1.32, .98, (.98, 1.01, 1.06), 2.0, 0.0),
        "valencia": (.92, .90, .96, (1.04, 1.01, .96), 8.0, 0.0),
        "xpro": (1.28, 1.12, .94, (.98, 1.00, 1.05), 0.0, .34),
        "lofi": (1.34, 1.42, 1.0, (1.00, 1.00, 1.02), 0.0, .28),
        "hefe": (1.20, 1.27, .94, (1.04, 1.00, .94), 2.0, .22),
        "nashville": (.86, .82, 1.04, (1.05, 1.01, .94), 7.0, 0.0),
        "kelvin": (1.05, 1.16, .90, (1.10, 1.00, .88), 4.0, .08),
        "inkwell": (1.14, 0.0, 1.0, (1.00, 1.00, 1.00), 0.0, 0.0),
    }
    contrast, saturation, gamma, bgr_gain, brightness, vignette = looks[name]
    source = image.astype(np.float32)
    styled = np.clip((source - 128) * contrast + 128 + brightness, 0, 255).astype(np.uint8)
    if name in ("moon", "inkwell"):
        gray = cv2.cvtColor(styled, cv2.COLOR_BGR2GRAY)
        styled = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if name == "moon":
            styled[:, :, 0] = np.clip(styled[:, :, 0].astype(np.float32) * 1.06, 0, 255).astype(np.uint8)
    else:
        hsv = cv2.cvtColor(styled, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        styled = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    lut = np.clip(np.power(np.arange(256, dtype=np.float32) / 255.0, gamma) * 255, 0, 255).astype(np.uint8)
    styled = cv2.LUT(styled, lut)
    for channel, gain in enumerate(bgr_gain):
        styled[:, :, channel] = np.clip(styled[:, :, channel].astype(np.float32) * gain, 0, 255).astype(np.uint8)
    if name in ("juno", "hefe"):
        styled = cv2.addWeighted(styled, .86, cv2.GaussianBlur(styled, (0, 0), 2), .14, 0)
    if name in ("clarendon", "hefe"):
        styled = cv2.addWeighted(styled, 1.12, cv2.GaussianBlur(styled, (0, 0), 1.2), -.12, 0)
    if vignette:
        h, w = styled.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        radius = ((xx - (w - 1) / 2) / max(w / 2, 1)) ** 2 + ((yy - (h - 1) / 2) / max(h / 2, 1)) ** 2
        mask = np.clip(1 - vignette * radius, .45, 1).astype(np.float32)
        styled = np.clip(styled.astype(np.float32) * mask[:, :, None], 0, 255).astype(np.uint8)
    return styled


def blended_instagram(image: np.ndarray, name: str, intensity: int) -> np.ndarray:
    intensity = int(np.clip(intensity, 0, 100))
    effect = instagram_effect(image, name)
    if intensity == 0:
        return image.copy()
    return cv2.addWeighted(image, 1 - intensity / 100, effect, intensity / 100, 0)


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


@app.post("/api/instagram")
def instagram_process():
    body = request.get_json(silent=True) or {}
    record = current_image()
    if record is None:
        return jsonify(error="Upload an image before applying an inspired-style filter."), 400
    name = body.get("filter", "")
    try:
        intensity = int(np.clip(int(body.get("intensity", 100)), 0, 100))
        result = blended_instagram(record["image"], name, intensity)
        data, payload = encode_image(result)
        record["instagram_processed"] = result
        record["instagram_bytes"] = payload
        record["instagram_filter"] = name
        record["instagram_intensity"] = intensity
        return jsonify(image=data, stats=image_stats(result), filter=name,
                       label=INSTAGRAM_FILTERS[name], intensity=intensity)
    except (ValueError, TypeError, cv2.error) as exc:
        return jsonify(error=str(exc)), 400


@app.post("/api/instagram/compare")
def instagram_compare():
    body = request.get_json(silent=True) or {}
    names = body.get("filters", [])
    record = current_image()
    if record is None:
        return jsonify(error="Upload an image before comparing filters."), 400
    if not isinstance(names, list) or len(names) > 6:
        return jsonify(error="Choose up to six inspired-style filters."), 400
    try:
        intensity = int(np.clip(int(body.get("intensity", 100)), 0, 100))
        results = []
        for name in names:
            if name not in INSTAGRAM_FILTERS:
                continue
            result = blended_instagram(record["image"], name, intensity)
            data, _ = encode_image(result)
            results.append({"filter": name, "label": INSTAGRAM_FILTERS[name], "image": data})
        return jsonify(results=results)
    except (ValueError, TypeError, cv2.error) as exc:
        return jsonify(error=str(exc)), 400


@app.get("/api/download")
def download():
    record = current_image()
    if record is None:
        return jsonify(error="There is no processed image to download."), 400
    if request.args.get("source") == "instagram":
        payload = record.get("instagram_bytes")
        if payload is None:
            return jsonify(error="Apply an inspired-style filter before downloading it."), 400
        style = record.get("instagram_filter", "")
        name = INSTAGRAM_FILTERS.get(style, "Inspired Style").lower().replace(" ", "-")
    else:
        payload = record.get("processed_bytes")
        if payload is None:
            return jsonify(error="There is no processed image to download."), 400
        current_filter = record.get("filter", "original")
        name = FILTERS.get(current_filter, FILTERS["original"])["label"].lower().replace(" / ", "-").replace(" ", "-")
    return send_file(io.BytesIO(payload), mimetype="image/png", as_attachment=True, download_name=f"{name}-result.png")


@app.errorhandler(RequestEntityTooLarge)
def too_large(_error):
    return jsonify(error="The image is too large. Maximum upload size is 12 MB."), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
