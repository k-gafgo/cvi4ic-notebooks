# CVI4IC - Computer Vision Notebooks

**Summer Semester 2026 - FH Upper Austria**

Jupyter notebooks with exercises and code examples for the Computer Vision (CVI4IC) course.

## Sessions

| # | Topic | Notebook |
|---|-------|----------|
| 01 | Introduction to Computer Vision | [01-introduction.ipynb](01-introduction.ipynb) |
| 02 | Image Representation & Color Spaces | [02-image-representation.ipynb](02-image-representation.ipynb) |
| 03 | Image Filtering & Convolution | [03-image-filtering.ipynb](03-image-filtering.ipynb) |
| 04 | Edge Detection & Gradients | [04-edge-detection.ipynb](04-edge-detection.ipynb) |
| 05 | Feature Detection & Description | [05-feature-detection.ipynb](05-feature-detection.ipynb) |
| 06 | Image Segmentation | [06-image-segmentation.ipynb](06-image-segmentation.ipynb) |
| 07 | Morphological Operations & Contours | [07-morphology-contours.ipynb](07-morphology-contours.ipynb) |
| 08 | Camera Models & Calibration | [08-camera-models.ipynb](08-camera-models.ipynb) |
| 09 | Introduction to Deep Learning for CV | [09-intro-deep-learning.ipynb](09-intro-deep-learning.ipynb) |
| 10 | Convolutional Neural Networks | [10-cnns.ipynb](10-cnns.ipynb) |
| 11 | Object Detection | [11-object-detection.ipynb](11-object-detection.ipynb) |
| 12 | Semantic Segmentation | [12-semantic-segmentation.ipynb](12-semantic-segmentation.ipynb) |

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate    # Linux/macOS
venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook
```

## Requirements

- Python 3.10+
- OpenCV, NumPy, Matplotlib (all sessions)
- PyTorch & torchvision (sessions 09-12)
- Ultralytics (session 11, optional)
