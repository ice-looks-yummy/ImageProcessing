# Image Processing Studio

A local, presentation-friendly image-processing application for exploring common computer vision operations. Flask serves the interface and API, while Python, OpenCV, and NumPy perform every image transformation on the local machine.

## Features

- Upload JPG, JPEG, PNG, and WEBP images (up to 12 MB and 30 megapixels).
- View original and processed images side by side with live statistics.
- Adjust blur, edge detection, threshold, brightness, and contrast parameters without reloading.
- Inspect each filter's equation and the current kernel or processing stages.
- Compare up to eight filters against the same original image.
- Download the currently processed result as a PNG.
- Responsive layout for desktop and mobile screens.

## Technologies

Python, Flask, OpenCV (`cv2`), NumPy, HTML, CSS, and vanilla JavaScript.

## Project structure

```text
project/
├── app.py
├── requirements.txt
├── README.md
├── templates/
│   └── index.html
├── static/
│   ├── style.css
│   └── script.js
└── uploads/
```

The `uploads/` directory is created automatically. Images are decoded and kept in memory for the local application session; no uploaded path is exposed to the browser.

## Installation and run

Open a terminal in this project directory.

### Windows PowerShell

```powershell
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000** in a browser. Press Ctrl+C in the terminal to stop the server. The app has no paid API or external processing service. The optional Google Fonts import falls back to system fonts when offline.

## Filters implemented

Original, Grayscale, Gaussian Blur, Median Blur, Sharpen, Sobel Edge Detection (X, Y, or magnitude), Canny Edge Detection, Binary Threshold, Negative / Invert, Sepia, Brightness, and Contrast.

## Image-processing concepts

- **Grayscale** combines red, green, and blue according to their relative luminance contribution.
- **Gaussian blur** computes a weighted neighborhood average; larger sigma spreads the weights more broadly.
- **Median blur** substitutes the median of a local window, which can suppress impulse noise while preserving edges.
- **Sharpen** convolves with a center-positive, neighbor-negative kernel to emphasize local changes.
- **Sobel** estimates horizontal and vertical derivatives. The combined output is the gradient magnitude `sqrt(Gx² + Gy²)`.
- **Canny** smooths the image, computes gradients, thins edges, applies two thresholds, and tracks connected edges by hysteresis.
- **Threshold** assigns pixels to black or white based on whether intensity exceeds the selected threshold.
- **Negative** complements each 8-bit value. **Sepia** multiplies RGB values by a color transformation matrix.
- **Brightness** adds an offset. **Contrast** scales values around the midpoint 128. Both clip to `[0,255]`.

## Equations and kernels

Gaussian blur uses `G(x,y) = 1/(2πσ²) exp(−(x²+y²)/(2σ²))`; its normalized discrete kernel is generated from the current kernel size and sigma. Median blur is a rank operation rather than a weighted convolution. Sharpen uses `[[0,−s,0],[−s,1+4s,−s],[0,−s,0]]`, where `s` is the strength. Sobel uses the standard 3×3 X and Y derivative kernels and combines them as `sqrt(Gx² + Gy²)`. Canny is a pipeline rather than a single convolution kernel. The displayed controls and equations reflect the operations used by the backend.
