"""
sam3_seg_utils.py
Utility functions for SAM3-based instance segmentation on Fruits360.
"""

import pathlib
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import torch
from PIL import Image
from transformers import (
    Sam3Model, Sam3Processor,
    Sam3TrackerModel, Sam3TrackerProcessor,
)

MODEL_ID = "facebook/sam3"

# Distinct colours for up to 10 simultaneous instances
INSTANCE_COLORS = [
    "#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6",
    "#1abc9c", "#e67e22", "#34495e", "#e91e63", "#00bcd4",
]


# ── Internal: score extraction ────────────────────────────────────────────────

def _tracker_score(outputs, idx=(0, 0)):
    """
    Extract the IoU/confidence score from a Sam3TrackerModel output object.
    The attribute name varies across transformers versions; try all known names
    and fall back to 1.0 so callers always get a float.
    """
    for attr in ("iou_predictions", "iou_scores", "pred_iou_scores", "scores"):
        tensor = getattr(outputs, attr, None)
        if tensor is not None:
            return float(tensor.cpu()[idx[0], idx[1]])
    return 1.0   # attribute not found — return neutral score


# ── Model loading ──────────────────────────────────────────────────────────────

def load_sam3_model(model_id=MODEL_ID, device=None):
    """
    Load Sam3Model + Sam3Processor.
    Use this model for text-prompted and combined prompts.
    Returns (model, processor).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Sam3Model from {model_id} on {device} ...")
    processor = Sam3Processor.from_pretrained(model_id)
    model = Sam3Model.from_pretrained(model_id).to(device).eval()
    print("Sam3Model ready.")
    return model, processor


def load_tracker_model(model_id=MODEL_ID, device=None):
    """
    Load Sam3TrackerModel + Sam3TrackerProcessor.
    Use this model for visual prompts: points and bounding boxes (SAM-style).
    Returns (model, processor).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading Sam3TrackerModel from {model_id} on {device} ...")
    processor = Sam3TrackerProcessor.from_pretrained(model_id)
    model = Sam3TrackerModel.from_pretrained(model_id).to(device).eval()
    print("Sam3TrackerModel ready.")
    return model, processor


# ── Segmentation ───────────────────────────────────────────────────────────────

@torch.no_grad()
def segment_by_text(image_path, model, processor, text, device,
                    threshold=0.1, mask_threshold=0.5):
    """
    Segment all instances matching `text` using Sam3Model.

    A low default threshold (0.1) returns all detections so the caller can
    filter interactively; for most scene images 0.4–0.6 is a good working value.

    Returns (detections, PIL.Image).
    detections: list of dicts sorted by score descending:
        {"mask": np.bool_ (H, W), "score": float, "box": np.ndarray [x1,y1,x2,y2]}
    """
    img = Image.open(image_path).convert("RGB")
    # TODO

    detections = []
    # TODO
    detections.sort(key=lambda d: d["score"], reverse=True)
    return detections, img


@torch.no_grad()
def segment_by_bbox(image_path, model, processor, box_xyxy, device):
    """
    Segment the object inside `box_xyxy = [x1, y1, x2, y2]` using Sam3TrackerModel.

    Returns (result, PIL.Image).
    result: {"mask": np.bool_ (H, W), "score": float}
    """
    img = Image.open(image_path).convert("RGB")
    inputs = None # TODO
    outputs = None # TODO
    masks  = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]                                              # (num_obj, num_masks, H, W)
    return {"mask": masks[0, 0].numpy().astype(bool), "score": _tracker_score(outputs)}, img


@torch.no_grad()
def segment_by_point(image_path, model, processor, point_xy, device, label=1):
    """
    Segment the object at pixel `point_xy = (x, y)` using Sam3TrackerModel.
    label=1 → positive click; label=0 → negative (exclude).

    Returns (result, PIL.Image).
    result: {"mask": np.bool_ (H, W), "score": float}
    """
    img = Image.open(image_path).convert("RGB")
    inputs = None # TODO
    outputs = model(**inputs, multimask_output=False)
    masks  = processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"]
    )[0]
    return {"mask": masks[0, 0].numpy().astype(bool), "score": _tracker_score(outputs)}, img


# ── Dataset helpers ────────────────────────────────────────────────────────────

def get_image_paths(train_root, class_name, n=10):
    """Return up to n sorted image paths for a class inside train_root."""
    root  = pathlib.Path(train_root)
    paths = sorted((root / class_name).glob("*.jpg"))[:n]
    if not paths:
        paths = sorted((root / class_name).glob("*.png"))[:n]
    assert paths, f"No images found for '{class_name}' under {root}"
    return paths


def find_class(name, classes):
    """Case-insensitive prefix match against a list of class names."""
    for c in classes:
        if c.lower().startswith(name.lower()):
            return c
    raise ValueError(f"'{name}' not found. Sample classes: {classes[:8]}")


# ── Internal helpers ───────────────────────────────────────────────────────────

def _fast_contour(ax, mask, color, linewidth=0.9, max_dim=512):
    """
    Draw a contour for `mask` without paying the full marching-squares cost.

    plt.contour on a multi-megapixel boolean mask is O(H*W) and extremely slow
    when called many times (e.g. N classes × M instances). This helper subsamples
    the mask to at most `max_dim` in each dimension, then uses `extent` to map the
    subsampled coordinates back to the original image space — producing visually
    identical results at a fraction of the cost.
    """
    h, w = mask.shape
    step = max(1, max(h, w) // max_dim)
    small = mask[::step, ::step].astype(float)
    ax.contour(small, levels=[0.5],
               extent=[0, w, 0, h],
               colors=[color], linewidths=linewidth)


def _color_overlay(scene_arr, detections, alpha=0.45):
    """Return a copy of scene_arr with coloured instance masks blended in."""
    composite = scene_arr.astype(float).copy()
    for i, det in enumerate(detections):
        rgb = np.array(mcolors.to_rgb(INSTANCE_COLORS[i % len(INSTANCE_COLORS)]))
        composite[det["mask"]] = (
            composite[det["mask"]] * (1 - alpha) + rgb * 255 * alpha
        )
    return composite.clip(0, 255).astype(np.uint8)


# ── Visualisation ──────────────────────────────────────────────────────────────

def show_text_segmentation(scene_arr, detections, text_prompt, save_path=None):
    """
    Three-panel figure: original | coloured instance overlay | per-instance score bar.
    """
    n = len(detections)
    if n == 0:
        print("No detections to show.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 0 — original
    axes[0].imshow(scene_arr)
    axes[0].set_title(f"Original scene\nText prompt: '{text_prompt}'", fontsize=12)
    axes[0].axis("off")

    # Panel 1 — coloured overlay + contours
    axes[1].imshow(_color_overlay(scene_arr, detections))
    for i, det in enumerate(detections):
        _fast_contour(axes[1], det["mask"],
                      color=INSTANCE_COLORS[i % len(INSTANCE_COLORS)], linewidth=1.0)
    axes[1].set_title(f"SAM3 instance segmentation  ({n} objects)", fontsize=12)
    axes[1].axis("off")

    # Panel 2 — horizontal score bar
    colors = [INSTANCE_COLORS[i % len(INSTANCE_COLORS)] for i in range(n)]
    axes[2].barh(range(n), [d["score"] for d in detections], color=colors)
    axes[2].set_yticks(range(n))
    axes[2].set_yticklabels(
        [f"#{i + 1}  {d['score']:.3f}" for i, d in enumerate(detections)], fontsize=9)
    axes[2].set_xlabel("IoU / confidence score")
    axes[2].set_xlim(0, 1.05)
    axes[2].set_title("Instance scores", fontsize=12)
    axes[2].invert_yaxis()

    plt.suptitle(f"SAM3 text-prompted segmentation — '{text_prompt}'",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def show_visual_segmentation(scene_arr, result, query_mode,
                              query_box=None, query_point=None, save_path=None):
    """
    Three-panel figure: original + query indicator | tinted mask | dimmed background.
    result: {"mask": bool array (H,W), "score": float}
    """
    mask  = result["mask"]
    score = result["score"]

    tinted = scene_arr.astype(float).copy()
    rgb    = np.array(mcolors.to_rgb(INSTANCE_COLORS[0]))
    tinted[mask] = tinted[mask] * 0.55 + rgb * 255 * 0.45

    dimmed = scene_arr.astype(float).copy()
    dimmed[~mask] *= 0.20
    dimmed = dimmed.clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 0 — original + prompt indicator
    axes[0].imshow(scene_arr)
    if query_box is not None:
        x1, y1, x2, y2 = query_box
        axes[0].add_patch(mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor="#e74c3c", facecolor="none", linestyle="--"))
    if query_point is not None:
        axes[0].plot(query_point[0], query_point[1],
                     "r+", markersize=14, markeredgewidth=2.5)
    axes[0].set_title("Original scene", fontsize=12)
    axes[0].axis("off")

    # Panel 1 — tinted mask
    axes[1].imshow(tinted.clip(0, 255).astype(np.uint8))
    _fast_contour(axes[1], mask, color="#00ff88", linewidth=1.2)
    axes[1].set_title(
        f"SAM3 mask  (mode: '{query_mode}')\nIoU score: {score:.3f}", fontsize=12)
    axes[1].axis("off")

    # Panel 2 — dimmed background
    axes[2].imshow(dimmed)
    _fast_contour(axes[2], mask, color="#00ff88", linewidth=1.2)
    axes[2].set_title("Segmented region  (background dimmed)", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(
        f"SAM3 visual-prompted segmentation — mode: '{query_mode}'",
        fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def show_instance_grid(scene_arr, detections, n_cols=4, save_path=None):
    """
    Grid of individual instances, each cropped to its bounding box with the
    background dimmed so the segmented object is the visual focus.
    """
    n = len(detections)
    if n == 0:
        print("No detections.")
        return
    n_rows = max(1, (n + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(n_cols * 3.2, n_rows * 3.2))
    axes = np.array(axes).reshape(-1)

    for i, det in enumerate(detections):
        mask = det["mask"]
        rows, cols = np.where(mask)
        if rows.size == 0:
            axes[i].axis("off")
            continue
        pad  = 20
        r0   = max(0, rows.min() - pad)
        r1   = min(scene_arr.shape[0], rows.max() + pad)
        c0   = max(0, cols.min() - pad)
        c1   = min(scene_arr.shape[1], cols.max() + pad)

        crop      = scene_arr[r0:r1, c0:c1].astype(float)
        crop_mask = mask[r0:r1, c0:c1]
        crop[~crop_mask] *= 0.15
        axes[i].imshow(crop.clip(0, 255).astype(np.uint8))
        axes[i].set_title(
            f"#{i + 1}  score={det['score']:.3f}",
            fontsize=8, color=INSTANCE_COLORS[i % len(INSTANCE_COLORS)])
        axes[i].axis("off")

    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.suptitle("Individual detected instances  (background dimmed)", fontsize=10)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def show_score_sweep(scene_arr, all_detections, thresholds, text_prompt, save_path=None):
    """
    Side-by-side panels showing how the number and selection of detected instances
    changes as the score threshold varies.

    all_detections: full detection list obtained at a low inference threshold (e.g. 0.1).
    thresholds: list of score thresholds to visualise.
    """
    n = len(thresholds)
    fig, axes = plt.subplots(1, n, figsize=(4.8 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, thr in zip(axes, thresholds):
        dets = [d for d in all_detections if d["score"] >= thr]
        ax.imshow(_color_overlay(scene_arr, dets))
        for i, det in enumerate(dets):
            _fast_contour(ax, det["mask"],
                          color=INSTANCE_COLORS[i % len(INSTANCE_COLORS)],
                          linewidth=0.8)
        ax.set_title(f"threshold ≥ {thr}\n{len(dets)} object(s)", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"Score threshold sensitivity — '{text_prompt}'", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def show_gallery_reference(ref_paths, class_name, n_show=6, save_path=None):
    """Show a strip of gallery reference images for a Fruits360 class."""
    n = min(n_show, len(ref_paths))
    fig, axes = plt.subplots(1, n, figsize=(n * 2.3, 2.6))
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, ref_paths[:n]):
        ax.imshow(Image.open(p))
        ax.set_title(p.stem, fontsize=6)
        ax.axis("off")
    plt.suptitle(f"Gallery reference images: {class_name}", fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def show_multiclass_segmentation(scene_arr, detections_per_class, save_path=None):
    """
    One panel per text class showing its detected instances overlaid on the scene.
    detections_per_class: dict {class_name: detections_list}
    """
    classes = list(detections_per_class.keys())
    n = len(classes)
    fig, axes = plt.subplots(1, n + 1, figsize=(5.5 * (n + 1), 5.5))

    axes[0].imshow(scene_arr)
    axes[0].set_title("Original scene", fontsize=11)
    axes[0].axis("off")

    for i, cls in enumerate(classes):
        dets = detections_per_class[cls]
        axes[i + 1].imshow(_color_overlay(scene_arr, dets))
        for j, det in enumerate(dets):
            _fast_contour(axes[i + 1], det["mask"],
                          color=INSTANCE_COLORS[j % len(INSTANCE_COLORS)],
                          linewidth=0.9)
        axes[i + 1].set_title(
            f"'{cls}'\n{len(dets)} instance(s)", fontsize=11)
        axes[i + 1].axis("off")

    plt.suptitle("SAM3 open-vocabulary multi-class segmentation", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
