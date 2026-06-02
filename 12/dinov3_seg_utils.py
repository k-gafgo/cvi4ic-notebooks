"""
dinov3_seg_utils.py
Utility functions for DINOv3 patch-similarity segmentation.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from transformers import AutoModel, AutoImageProcessor

# ── Constants ──────────────────────────────────────────────────────────────
PATCH_SIZE = 16
IMG_SIZE   = 224
N_SIDE     = IMG_SIZE // PATCH_SIZE   # 14
N_PATCH    = N_SIDE ** 2              # 196


# ── Model loading ──────────────────────────────────────────────────────────

def load_model(model_id="facebook/dinov3-vitb16-pretrain-lvd1689m", device=None):
    """
    Load DINOv3 model and processor.
    Returns (model, processor, num_reg).
    num_reg: number of register tokens inserted between CLS and patch tokens.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device} ...")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id).to(device).eval()
    num_reg = getattr(model.config, "num_register_tokens", 0)
    print(f"Ready — register tokens: {num_reg}")
    return model, processor, num_reg


# ── Feature extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def get_patch_tokens(img_path, model, processor, device, num_reg):
    """
    Returns (N_PATCH, 768) L2-normalised patch tokens for one image.
    Skips the CLS token and any register tokens in last_hidden_state.
    """
    img    = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out    = model(**inputs)
    tokens = out.last_hidden_state[0, 1 + num_reg:, :].cpu().float()
    assert tokens.shape[0] == N_PATCH, \
        f"Expected {N_PATCH} patch tokens, got {tokens.shape[0]}"
    return F.normalize(tokens, dim=-1)


def patch_sim_map(query_tokens, query_rc, target_tokens):
    """
    Cosine similarity of one query patch to every patch in a target image.

    query_tokens : (N_PATCH, d) tensor
    query_rc     : (row, col) index in the N_SIDE×N_SIDE grid
    target_tokens: (N_PATCH, d) tensor
    Returns      : (N_SIDE, N_SIDE) numpy array
    """
    r, c = query_rc
    q    = query_tokens[r * N_SIDE + c].unsqueeze(0)   # (1, d)
    sims = (q @ target_tokens.T).squeeze(0).numpy()    # (N_PATCH,)
    return sims.reshape(N_SIDE, N_SIDE)


# ── Dataset helpers ────────────────────────────────────────────────────────

def get_image_paths(train_root, class_name, n=10):
    """Return up to n sorted image paths for a class inside train_root."""
    import pathlib
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


# ── Prototype builders ─────────────────────────────────────────────────────

def build_gallery_prototype(ref_paths, model, processor, device, num_reg):
    """
    Build a prototype from the foreground patches of reference images.

    Fruits360 images have a plain white background, so PC1 of the patch
    tokens cleanly separates fruit-surface patches from background patches.
    Foreground = patches whose PC1 score is above the per-image mean.

    Returns (prototype, n_fg_patches).
    """
    print(f"Extracting patch tokens for {len(ref_paths)} reference images ...")
    stacked = np.stack([
        get_patch_tokens(p, model, processor, device, num_reg).numpy()
        for p in ref_paths
    ])  # (N_REF, N_PATCH, d)

    flat    = stacked.reshape(-1, stacked.shape[-1]).astype(np.float32)
    pc1     = PCA(n_components=1, random_state=42).fit_transform(flat)[:, 0]
    pc1_2d  = pc1.reshape(len(ref_paths), N_PATCH)
    fg_mask = None # TODO
    fg_vecs = np.vstack([stacked[i][fg_mask[i]] for i in range(len(ref_paths))])

    prototype = None # TODO
    prototype /= None # TODO
    print(f"Prototype from {len(fg_vecs)} foreground patches.")
    return prototype, len(fg_vecs)


def build_bbox_prototype(scene_tokens, query_box, orig_w, orig_h):
    """
    Build a prototype from the patches whose receptive fields overlap a
    bounding box drawn on the scene image.

    query_box : (x1, y1, x2, y2) in original image pixels
    Returns (prototype, n_patches).
    """
    x1, y1, x2, y2 = query_box
    assert 0 <= x1 < x2 <= orig_w and 0 <= y1 < y2 <= orig_h, \
        f"Box {query_box} is out of image bounds ({orig_w}×{orig_h})"
    sx, sy = IMG_SIZE / orig_w, IMG_SIZE / orig_h
    pc_lo  = None # TODO
    pc_hi  = None # TODO
    pr_lo  = None # TODO
    pr_hi  = None # TODO
    idx    = [r * N_SIDE + c for r in range(pr_lo, pr_hi) for c in range(pc_lo, pc_hi)]
    assert idx, "Box maps to zero patches — make the box larger."

    prototype = scene_tokens[idx].numpy().mean(axis=0)
    prototype /= np.linalg.norm(prototype) + 1e-8
    print(f"[bbox] {len(idx)} patches  "
          f"(rows {pr_lo}–{pr_hi - 1}, cols {pc_lo}–{pc_hi - 1})")
    return prototype, len(idx)


def build_point_prototype(scene_tokens, query_point, orig_w, orig_h):
    """
    Build a prototype from the single patch at a given pixel coordinate.

    query_point : (x, y) in original image pixels
    Returns (prototype, patch_row, patch_col).
    """
    px, py = query_point
    assert 0 <= px < orig_w and 0 <= py < orig_h, \
        f"Point {query_point} is outside image bounds ({orig_w}×{orig_h})"
    sx, sy = IMG_SIZE / orig_w, IMG_SIZE / orig_h
    q_col  = None # TODO
    q_row  = None # TODO
    prototype = scene_tokens[q_row * N_SIDE + q_col].numpy()
    print(f"[point] ({px}, {py}) → patch row {q_row}, col {q_col}")
    return prototype, q_row, q_col


# ── Detection ──────────────────────────────────────────────────────────────

def compute_sim_mask(prototype, scene_tokens, sim_percentile, orig_w, orig_h):
    """
    Compute cosine similarity between a prototype and every scene patch,
    upsample to original resolution, then threshold.

    Returns (sim_upsampled, binary_mask, threshold).
    sim_upsampled : (orig_h, orig_w) float array
    binary_mask   : (orig_h, orig_w) bool array
    threshold     : float — the sim_percentile-th percentile value
    """
    proto_t       = torch.tensor(prototype, dtype=torch.float32).unsqueeze(0)
    sims          = None # TODO
    sim_grid      = sims.reshape(N_SIDE, N_SIDE)                    # (14, 14)
    sim_upsampled = np.array(
        Image.fromarray(sim_grid.astype(np.float32), mode="F")
             .resize((orig_w, orig_h), Image.BILINEAR)
    )
    threshold   = np.percentile(sim_upsampled, sim_percentile)
    binary_mask = None # TODO
    return sim_upsampled, binary_mask, threshold


# ── Visualisation ──────────────────────────────────────────────────────────

def show_patch_sim_grid(query_path, query_tokens, query_patches, patch_colors,
                        same_paths, same_tokens, cross_paths, cross_tokens,
                        class_a_name, class_b_name, save_path=None):
    """
    Grid figure showing dense patch cosine similarity.
    Rows = query patches; columns = [query | same-class × N | gap | cross-class × N].
    Bright = high similarity, dark = low similarity.
    """
    n_targets = min(len(same_paths), len(cross_paths))
    n_rows    = len(query_patches)
    query_arr = np.array(Image.open(query_path).resize((IMG_SIZE, IMG_SIZE)))

    fig = plt.figure(
        figsize=(2.6 * (1 + n_targets * 2 + 0.3), 2.8 * n_rows + 0.6))
    gs  = gridspec.GridSpec(
        n_rows, 1 + n_targets + 1 + n_targets,
        figure=fig, wspace=0.04, hspace=0.10,
        width_ratios=[1] + [1] * n_targets + [0.06] + [1] * n_targets,
    )

    for row_i, (qrc, color) in enumerate(zip(query_patches, patch_colors)):
        r, c = qrc

        # Query image with highlighted patch
        ax_q = fig.add_subplot(gs[row_i, 0])
        ax_q.imshow(query_arr)
        ax_q.add_patch(mpatches.FancyBboxPatch(
            (c * PATCH_SIZE, r * PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
            boxstyle="square,pad=0", linewidth=2.5,
            edgecolor=color, facecolor=color, alpha=0.50))
        ax_q.axis("off")
        if row_i == 0:
            ax_q.set_title("Query", fontsize=8, fontweight="bold")

        # Same-class heatmaps
        for t_i in range(n_targets):
            ax    = fig.add_subplot(gs[row_i, 1 + t_i])
            t_arr = np.array(Image.open(same_paths[t_i]).resize((IMG_SIZE, IMG_SIZE)))
            smap  = patch_sim_map(query_tokens, qrc, same_tokens[t_i])
            ax.imshow(t_arr, alpha=0.40)
            ax.imshow(np.kron(smap, np.ones((PATCH_SIZE, PATCH_SIZE))),
                      cmap="inferno", alpha=0.75, vmin=smap.min(), vmax=smap.max())
            ax.axis("off")
            if row_i == 0:
                ax.set_title(f"Same\n({class_a_name.split()[0]})",
                             fontsize=7, color="steelblue", fontweight="bold")

        # Spacer column
        fig.add_subplot(gs[row_i, 1 + n_targets]).axis("off")

        # Cross-class heatmaps
        for t_i in range(n_targets):
            ax    = fig.add_subplot(gs[row_i, 2 + n_targets + t_i])
            t_arr = np.array(Image.open(cross_paths[t_i]).resize((IMG_SIZE, IMG_SIZE)))
            smap  = patch_sim_map(query_tokens, qrc, cross_tokens[t_i])
            ax.imshow(t_arr, alpha=0.40)
            ax.imshow(np.kron(smap, np.ones((PATCH_SIZE, PATCH_SIZE))),
                      cmap="inferno", alpha=0.75, vmin=smap.min(), vmax=smap.max())
            ax.axis("off")
            if row_i == 0:
                ax.set_title(f"Cross\n({class_b_name.split()[0]})",
                             fontsize=7, color="tomato", fontweight="bold")

        # Coloured dot in left margin
        fig.text(0.005, 1 - (row_i + 0.5) / n_rows, "●",
                 color=color, fontsize=14, va="center",
                 transform=fig.transFigure)

    # Vertical divider
    div_x = (1 + n_targets + 0.5) / (1 + n_targets * 2 + 0.3 + 1)
    fig.add_artist(plt.Line2D(
        [div_x, div_x], [0.03, 0.95],
        transform=fig.transFigure, color="black", linewidth=1.0))

    fig.suptitle(
        f"DINOv3 dense patch cosine similarity — query: {class_a_name}\n"
        "Each row = one query patch  ·  bright = similar  ·  dark = dissimilar",
        fontsize=10, y=1.01)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def show_detection(scene_arr, sim_upsampled, binary_mask, query_mode,
                   threshold, sim_percentile,
                   query_box=None, query_point=None, save_path=None):
    """
    Three-panel figure: original scene | similarity heatmap | thresholded mask.

    query_box   : (x1, y1, x2, y2) drawn as a dashed rectangle on panel 0
    query_point : (x, y)  drawn as a red cross on panel 0
    """
    dimmed               = scene_arr.astype(float).copy()
    dimmed[~binary_mask] *= 0.20
    dimmed               = dimmed.clip(0, 255).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # Panel 0 — original + query reference overlay
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

    # Panel 1 — similarity heatmap
    axes[1].imshow(scene_arr)
    hm = axes[1].imshow(
        sim_upsampled, cmap="inferno", alpha=0.55,
        vmin=np.percentile(sim_upsampled, 10), vmax=sim_upsampled.max())
    plt.colorbar(hm, ax=axes[1], fraction=0.03, pad=0.02, label="cosine sim")
    axes[1].set_title(f"Similarity heatmap  (mode: '{query_mode}')", fontsize=12)
    axes[1].axis("off")

    # Panel 2 — thresholded mask
    axes[2].imshow(dimmed)
    axes[2].contour(binary_mask.astype(float), levels=[0.5],
                    colors=["#00ff88"], linewidths=1.2)
    axes[2].set_title(
        f"Detected regions  (top {100 - sim_percentile}%,  "
        f"threshold = {threshold:.3f})", fontsize=12)
    axes[2].axis("off")

    plt.suptitle(
        f"DINOv3 patch similarity detection — mode: '{query_mode}'",
        fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def show_threshold_sweep(scene_arr, sim_upsampled, percentiles, query_mode,
                         save_path=None):
    """
    Show detection masks side-by-side for a range of percentile thresholds.
    Illustrates the recall/precision trade-off of the threshold hyperparameter.
    """
    fig, axes = plt.subplots(1, len(percentiles),
                              figsize=(4 * len(percentiles), 4.5))
    for ax, pct in zip(axes, percentiles):
        thr  = np.percentile(sim_upsampled, pct)
        mask = sim_upsampled >= thr
        dim  = scene_arr.astype(float).copy()
        dim[~mask] *= 0.20
        dim  = dim.clip(0, 255).astype(np.uint8)
        ax.imshow(dim)
        ax.contour(mask.astype(float), levels=[0.5],
                   colors=["#00ff88"], linewidths=0.8)
        ax.set_title(f"top {100 - pct}%\n(thr = {thr:.3f})", fontsize=9)
        ax.axis("off")

    plt.suptitle(f"Threshold sensitivity — mode: '{query_mode}'", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()


def show_patch_grid_overlay(scene_arr, orig_w, orig_h, save_path=None):
    """
    Overlay the 14×14 DINOv3 patch grid on the scene to visualise the
    coarse spatial resolution of the similarity map.
    """
    patch_w = PATCH_SIZE / (IMG_SIZE / orig_w)
    patch_h = PATCH_SIZE / (IMG_SIZE / orig_h)

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.imshow(scene_arr)
    for r in range(N_SIDE + 1):
        ax.axhline(r * patch_h, color="cyan", linewidth=0.5, alpha=0.7)
    for c in range(N_SIDE + 1):
        ax.axvline(c * patch_w, color="cyan", linewidth=0.5, alpha=0.7)
    ax.set_title(
        f"14×14 DINOv3 patch grid on {orig_w}×{orig_h} scene\n"
        f"Each cell ≈ {patch_w:.0f} × {patch_h:.0f} px  —  "
        "minimum segmentation boundary precision",
        fontsize=10)
    ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    plt.show()
