# CVI4IC - Computer Vision Notebooks

**Summer Semester 2026 - FH Upper Austria**

Jupyter notebooks with exercises and code examples for the Computer Vision (CVI4IC) course.

## Sessions

| # | Topic | Colab |
|---|-------|-------|
| 01 | Introduction to Computer Vision | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/schedldave/cvi4ic-notebooks/blob/main/01-introduction.ipynb) |
| 02 | Image Representation & Color Spaces | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/schedldave/cvi4ic-notebooks/blob/main/02-image-representation.ipynb) |
| 03 | Image Filtering & Frequency Analysis | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/schedldave/cvi4ic-notebooks/blob/main/03-image-filtering.ipynb) |
| 04 | Edge Detection & Feature Detection | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/schedldave/cvi4ic-notebooks/blob/main/04-edge-feature-detection.ipynb) |
| 05 | Image Stitching & Homographies | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/schedldave/cvi4ic-notebooks/blob/main/05-image-stitching.ipynb) |

> More sessions will be added as the course progresses.

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
- PyTorch & torchvision (sessions 07-12)
- Ultralytics (session 09, optional)
