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

## Additional technical filters

The original twelve filters remain available in the same filter selector. Seven additional OpenCV filters use the same live processing, statistics, equation, and kernel panels:

- **Laplacian Edge Detection** computes the sum of second derivatives. The 3 x 3 four-neighbor stencil is shown for the matching OpenCV aperture; expanded apertures are identified in the information panel.
- **Bilateral Filter** uses OpenCV's edge-preserving color and spatial weights. Diameter, sigma color, and sigma space update live. It has no single fixed convolution kernel.
- **Histogram Equalization** equalizes the L channel in CIELAB, retaining the source chroma. Its cumulative-distribution mapping is `s = (L - 1) * CDF(r)`.
- **Emboss** convolves with `[[-2,-1,0],[-1,1,1],[0,1,2]]` and offers an adjustable strength.
- **Cartoon Effect** combines bilateral color smoothing, median-smoothed grayscale, an adaptive threshold edge mask, and masked color.
- **Gamma Correction** applies `g = 255 * (f / 255)^gamma` with a live gamma control from 0.1 to 3.0.
- **Hue / Saturation Adjustment** edits HSV hue and saturation while preserving value (brightness).

## Inspired-style creative presets

The bottom **Instagram-Style Filters** section includes Clarendon, Gingham, Moon, Lark, Juno, Valencia, X-Pro, Lo-Fi, Hefe, Nashville, Kelvin, and Inkwell styles. These are locally computed creative effects inspired by familiar social-media photo treatments; they do not reproduce or claim to reproduce Instagram's proprietary algorithms. Each effect combines image operations such as tone, contrast, saturation, channel grading, grayscale, gamma, vignette, or smoothing. The live intensity slider blends the original and effect from 0% (original) to 100% (full effect). The section includes its own original/result previews, result statistics, download action, and comparison grid for up to six presets. Its comparison does not replace the technical-filter comparison above.

## Usage

1. Upload a JPG, PNG, or WEBP image in the workspace.
2. Choose a technical filter in the existing **Filter controls** selector. Its controls appear in place; slider changes update the preview and statistics without a reload.
3. Read the matching equation and kernel or process explanation in the reference card. Use **Filter comparison** to compare technical filters on the same source image.
4. Use **Download result** to save the current processed image.
5. At the bottom of the page, select an inspired-style preset and adjust its intensity. Compare up to six creative styles or download the selected style result.

All processing is performed locally by the Flask application using OpenCV and NumPy. The technical and inspired-style download buttons share the same download endpoint, with a source parameter so each downloads the result shown in its own section.
