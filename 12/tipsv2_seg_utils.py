"""
tipsv2_seg_utils.py
Utility functions for TIPSv2 feature visualisation, zero-shot segmentation,
depth/normals estimation, and supervised segmentation.

Based on the official TIPSv2 Hugging Face Space demo:
  https://huggingface.co/spaces/google/TIPSv2
"""

import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from torchvision import transforms
from transformers import AutoModel

# ── Constants ──────────────────────────────────────────────────────────────
PATCH_SIZE         = 14
DEFAULT_RESOLUTION = 448    # → 32×32 patch grid
MAX_LEN            = 64

# Template ensemble: averaging over 9 phrasings reduces sensitivity to wording
TCL_PROMPTS = [
    "itap of a {}.",
    "a bad photo of a {}.",
    "a origami {}.",
    "a photo of the large {}.",
    "a {} in a video game.",
    "art of the {}.",
    "a photo of the small {}.",
    "a photo of many {}.",
    "a photo of {}s.",
]

# ── ADE20K palette (150 classes, for supervised segmentation) ──────────────
NUM_ADE20K_CLASSES = 150
ADE20K_PALETTE = np.zeros((NUM_ADE20K_CLASSES + 1, 3), dtype=np.uint8)
for _i in range(1, NUM_ADE20K_CLASSES + 1):
    _h = (_i * 0.618033988749895) % 1.0
    _s = 0.65 + 0.35 * ((_i * 7) % 5) / 4.0
    _v = 0.70 + 0.30 * ((_i * 11) % 3) / 2.0
    _r, _g, _b = colorsys.hsv_to_rgb(_h, _s, _v)
    ADE20K_PALETTE[_i] = [int(_r * 255), int(_g * 255), int(_b * 255)]

ADE20K_CLASSES = (
    "wall", "building", "sky", "floor", "tree", "ceiling", "road", "bed",
    "windowpane", "grass", "cabinet", "sidewalk", "person", "earth", "door",
    "table", "mountain", "plant", "curtain", "chair", "car", "water",
    "painting", "sofa", "shelf", "house", "sea", "mirror", "rug", "field",
    "armchair", "seat", "fence", "desk", "rock", "wardrobe", "lamp",
    "bathtub", "railing", "cushion", "base", "box", "column", "signboard",
    "chest_of_drawers", "counter", "sand", "sink", "skyscraper", "fireplace",
    "refrigerator", "grandstand", "path", "stairs", "runway", "case",
    "pool_table", "pillow", "screen_door", "stairway", "river", "bridge",
    "bookcase", "blind", "coffee_table", "toilet", "flower", "book", "hill",
    "bench", "countertop", "stove", "palm", "kitchen_island", "computer",
    "swivel_chair", "boat", "bar", "arcade_machine", "hovel", "bus", "towel",
    "light", "truck", "tower", "chandelier", "awning", "streetlight", "booth",
    "television", "airplane", "dirt_track", "apparel", "pole", "land",
    "bannister", "escalator", "ottoman", "bottle", "buffet", "poster",
    "stage", "van", "ship", "fountain", "conveyer_belt", "canopy", "washer",
    "plaything", "swimming_pool", "stool", "barrel", "basket", "waterfall",
    "tent", "bag", "minibike", "cradle", "oven", "ball", "food", "step",
    "tank", "trade_name", "microwave", "pot", "animal", "bicycle", "lake",
    "dishwasher", "screen", "blanket", "sculpture", "hood", "sconce", "vase",
    "traffic_light", "tray", "ashcan", "fan", "pier", "crt_screen", "plate",
    "monitor", "bulletin_board", "shower", "radiator", "glass", "clock", "flag",
)


# ── Model state ────────────────────────────────────────────────────────────

class TIPSv2State:
    """Container for all loaded TIPSv2 model components."""
    __slots__ = ("dpt", "vision_encoder", "text_encoder",
                 "tokenizer", "temperature", "device")

    def __init__(self, dpt, vision_encoder, text_encoder,
                 tokenizer, temperature, device):
        self.dpt            = dpt
        self.vision_encoder = vision_encoder
        self.text_encoder   = text_encoder
        self.tokenizer      = tokenizer
        self.temperature    = temperature
        self.device         = device


def load_model(model_id="google/tipsv2-b14-dpt", device=None):
    """
    Load a TIPSv2 DPT model variant and extract backbone components.

    Use the '-dpt' variant (e.g. 'google/tipsv2-b14-dpt') — it bundles:
      - TIPS backbone  (vision encoder + text encoder)
      - DPT depth head       (trained on NYU Depth V2)
      - DPT surface-normals head
      - DPT segmentation head (ADE20K, 150 classes)

    Returns a TIPSv2State with all components already on `device`.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_id} on {device} ...")
    dpt = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    dpt.eval()
    dpt._get_backbone()
    backbone = dpt._backbone

    state = TIPSv2State(
        dpt            = dpt.to(device),
        vision_encoder = backbone.vision_encoder.to(device),
        text_encoder   = backbone.text_encoder.to(device),
        tokenizer      = backbone._load_tokenizer(),
        temperature    = backbone.config.temperature,
        device         = device,
    )
    sp = DEFAULT_RESOLUTION // PATCH_SIZE
    print(f"Ready — patch grid: {sp}×{sp} = {sp**2} tokens  |  device: {device}")
    return state


# ── Preprocessing & array helpers ──────────────────────────────────────────

def make_transform(resolution=DEFAULT_RESOLUTION):
    """TIPSv2 image transform — resize + ToTensor, NO ImageNet normalisation."""
    return transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
    ])


def l2_normalize(x, axis=-1):
    return x / np.linalg.norm(x, ord=2, axis=axis, keepdims=True).clip(min=1e-3)


def upsample(arr, h, w, mode="bilinear"):
    """Upsample (sp, sp, C) or (sp, sp) numpy array to (h, w[, C])."""
    t = torch.from_numpy(arr.astype(np.float32))
    if t.ndim == 2:
        t = t.unsqueeze(-1)
    t  = t.permute(2, 0, 1).unsqueeze(0)
    kw = dict(align_corners=False) if mode == "bilinear" else {}
    up = None # TODO
    return up[0].permute(1, 2, 0).numpy()


def to_uint8(x):
    return (np.asarray(x) * 255).clip(0, 255).astype(np.uint8)


def _as_array(img_path_or_arr):
    if isinstance(img_path_or_arr, np.ndarray):
        return img_path_or_arr
    return np.array(Image.open(img_path_or_arr).convert("RGB"))


def _to_tensor(img_path_or_arr, resolution, device):
    if isinstance(img_path_or_arr, np.ndarray):
        img = Image.fromarray(img_path_or_arr).convert("RGB")
    else:
        img = Image.open(img_path_or_arr).convert("RGB")
    return make_transform(resolution)(img).unsqueeze(0).to(device)


# ── Feature extraction ─────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(img_path_or_arr, state, resolution=DEFAULT_RESOLUTION):
    """
    Standard patch feature extraction.
    Runs the full vision encoder forward pass.
    Returns (sp, sp, D) spatial features where sp = resolution // 14.
    """
    tensor              = _to_tensor(img_path_or_arr, resolution, state.device)
    # TODO

@torch.no_grad()
def extract_features_value_attention(img_path_or_arr, state,
                                     resolution=DEFAULT_RESOLUTION):
    """
    Value-attention features for higher-quality zero-shot segmentation.

    Runs all-but-last ViT blocks normally, then extracts only the V (value)
    stream from the last block's QKV projection. This skips the query/key
    attention mixing that tends to blur spatial boundaries in the final layer.

    Returns (sp, sp, D) spatial features.
    """
    tensor = _to_tensor(img_path_or_arr, resolution, state.device)
    enc    = state.vision_encoder

    x = enc.prepare_tokens_with_masks(tensor)
    for blk in enc.blocks[:-1]:
        x = blk(x)

    blk     = enc.blocks[-1]
    num_reg = getattr(enc, "num_register_tokens", 1)
    B, N, C = x.shape
    H       = blk.attn.num_heads

    qkv   = blk.attn.qkv(blk.norm1(x))
    qkv   = qkv.reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
    v     = qkv[2]                              # (B, H, N, D_head)
    v_out = v.transpose(1, 2).reshape(B, N, C)
    v_out = blk.ls1(blk.attn.proj(v_out))
    x_val = v_out + x
    x_val = enc.norm(x_val + blk.ls2(blk.mlp(blk.norm2(x_val))))

    patch_tokens = x_val[:, 1 + num_reg:, :]
    sp = resolution // PATCH_SIZE
    return patch_tokens.cpu().reshape(sp, sp, -1).numpy()


# ── Text encoding ──────────────────────────────────────────────────────────

@torch.no_grad()
def encode_text_classes(classes, state):
    """
    Encode class names using the TCL template ensemble (9 prompt templates).
    Averaging over templates reduces sensitivity to exact phrasing.
    Returns (N, D) L2-normalised text embeddings.
    """
    all_embs = []
    for template in TCL_PROMPTS:
        prompts   = [template.format(c) for c in classes]
        ids, pads = state.tokenizer.tokenize(prompts, max_len=MAX_LEN)
        embs      = state.text_encoder(
            torch.from_numpy(ids).to(state.device),
            torch.from_numpy(pads).to(state.device),
        )
        all_embs.append(embs.cpu().numpy())
    return l2_normalize(np.mean(all_embs, axis=0))


# ── Feature visualisations ─────────────────────────────────────────────────

def vis_pca(spatial):
    """3-component PCA → (sp, sp, 3) uint8 via sigmoid mapping."""
    h, w  = spatial.shape[:2]
    feat  = spatial.reshape(-1, spatial.shape[-1]).astype(np.float32)
    rgb   = PCA(n_components=3, whiten=True).fit_transform(feat).reshape(h, w, 3)
    return to_uint8(1.0 / (1.0 + np.exp(-2.0 * rgb)))


def vis_depth_pca(spatial):
    """1st PCA component → (sp, sp, 3) uint8 with inferno colormap."""
    h, w  = spatial.shape[:2]
    feat  = spatial.reshape(-1, spatial.shape[-1]).astype(np.float32)
    d     = PCA(n_components=1).fit_transform(feat).reshape(h, w)
    d     = (d - d.min()) / (d.max() - d.min() + 1e-8)
    return to_uint8(cm.inferno(d)[:, :, :3])


def vis_kmeans(spatial, h, w, n_clusters=6):
    """
    K-means clustering of L2-normalised patch features → (h, w, 3) uint8.
    Labels are upsampled from the patch grid to (h, w) with nearest-neighbour.
    """
    sp_h, sp_w = spatial.shape[:2]
    feat   = l2_normalize(spatial.reshape(-1, spatial.shape[-1]).astype(np.float32))
    labels = None # TODO
    labels_up = None # TODO
    palette = (plt.cm.tab20(np.linspace(0, 1, max(n_clusters, 2)))
               [:n_clusters, :3] * 255).astype(np.uint8)
    return palette[labels_up]


# ── Zero-shot segmentation ─────────────────────────────────────────────────

def zeroseg(spatial, orig_arr, classes, class_embs):
    """
    Zero-shot semantic segmentation via patch–text cosine similarity.

    spatial    : (sp, sp, D) L2-normalised spatial features
    orig_arr   : (H, W, 3) uint8 original image
    classes    : list of N class name strings
    class_embs : (N, D) L2-normalised text embeddings from encode_text_classes()

    Returns (blend_arr, mask_arr, label_map).
    blend_arr  : (H, W, 3) uint8  — colour map blended onto original
    mask_arr   : (H, W, 3) uint8  — flat colour map
    label_map  : (H, W) int       — per-pixel class index
    """
    h, w       = orig_arr.shape[:2]
    sp_h, sp_w = spatial.shape[:2]
    n          = len(classes)

    feat    = l2_normalize(spatial.reshape(-1, spatial.shape[-1]))
    sim = None # TODO
    sim_up  = None # TODO
    labels  = None # TODO

    palette  = (plt.cm.tab20(np.linspace(0, 1, max(n, 2)))[:n, :3] * 255).astype(np.uint8)
    seg_rgb  = palette[labels].astype(np.float32) / 255.0
    mask_arr = to_uint8(seg_rgb)
    blend    = to_uint8(0.15 * orig_arr.astype(np.float32) / 255.0 + 0.85 * seg_rgb)
    return blend, mask_arr, labels


# ── DPT dense prediction ───────────────────────────────────────────────────

@torch.no_grad()
def run_depth(img_path_or_arr, state, resolution=DEFAULT_RESOLUTION):
    """Run DPT depth head. Returns (H', W') float numpy array."""
    tensor = _to_tensor(img_path_or_arr, resolution, state.device)
    return # TODO


@torch.no_grad()
def run_normals(img_path_or_arr, state, resolution=DEFAULT_RESOLUTION):
    """Run DPT surface-normals head. Returns (3, H', W') float numpy in [-1, 1]."""
    tensor = _to_tensor(img_path_or_arr, resolution, state.device)
    return # TODO


@torch.no_grad()
def run_segmentation(img_path_or_arr, state, resolution=DEFAULT_RESOLUTION):
    """Run DPT ADE20K head. Returns (150, H', W') logits numpy array."""
    tensor = _to_tensor(img_path_or_arr, resolution, state.device)
    return # TODO


# ── DPT result visualisations ──────────────────────────────────────────────

def vis_depth_dpt(depth_arr, h, w):
    """Colour (H', W') depth array → (h, w, 3) uint8 with turbo colormap."""
    d = (depth_arr - depth_arr.min()) / (depth_arr.max() - depth_arr.min() + 1e-8)
    return to_uint8(upsample(cm.turbo(d)[:, :, :3].astype(np.float32), h, w))


def vis_normals_dpt(normals_arr, h, w):
    """Map (3, H', W') normals [-1, 1] → [0, 1] → (h, w, 3) uint8."""
    n = ((normals_arr + 1.0) / 2.0).transpose(1, 2, 0)   # (H', W', 3)
    return to_uint8(upsample(n, h, w))


def vis_segmentation_dpt(logits_arr, orig_arr):
    """
    Colour (150, H', W') ADE20K logits → (H, W, 3) uint8 blend.
    Returns (blend_arr, label_map).
    """
    h, w      = orig_arr.shape[:2]
    logits_up = upsample(logits_arr.transpose(1, 2, 0), h, w)  # (H, W, 150)
    pred      = logits_up.argmax(axis=-1)
    seg_rgb   = ADE20K_PALETTE[pred.astype(np.int32) + 1].astype(np.float32) / 255.0
    blend     = to_uint8(0.15 * orig_arr.astype(np.float32) / 255.0 + 0.85 * seg_rgb)
    return blend, pred


# ── Combined show functions ────────────────────────────────────────────────

def show_pca_panel(img_path_or_arr, state, resolution=DEFAULT_RESOLUTION,
                   title="", save_path=None):
    """
    Three-panel: original | PCA→RGB | 1st-PC pseudo-depth.
    Upsamples from the patch grid (NEAREST) to preserve the block structure.
    """
    orig = _as_array(img_path_or_arr)
    h, w = orig.shape[:2]
    sp   = resolution // PATCH_SIZE
    spatial  = extract_features(img_path_or_arr, state, resolution)
    pca_img  = np.array(Image.fromarray(vis_pca(spatial)).resize((w, h), Image.NEAREST))
    depth_img = np.array(Image.fromarray(vis_depth_pca(spatial)).resize((w, h), Image.NEAREST))

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].imshow(orig);      axes[0].set_title("Original", fontsize=12)
    axes[1].imshow(pca_img);   axes[1].set_title(f"PCA → RGB  ({sp}×{sp} grid)", fontsize=12)
    axes[2].imshow(depth_img); axes[2].set_title("1st PC  (pseudo-depth, inferno)", fontsize=12)
    for ax in axes:
        ax.axis("off")
    suptitle = f"TIPSv2 patch features — PCA  ·  {resolution}px input"
    if title:
        suptitle += f"  ·  {title}"
    plt.suptitle(suptitle, fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"Saved → {save_path}")
    plt.show()


def show_pca_gallery(ref_paths, state, resolution=DEFAULT_RESOLUTION,
                     class_name="", save_path=None):
    """
    Two-row strip: original (top) | PCA→RGB (bottom) for each reference image.
    """
    n   = len(ref_paths)
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 6.5))
    axes = np.array(axes)

    for i, p in enumerate(ref_paths):
        img     = np.array(Image.open(p).convert("RGB"))
        spatial = extract_features(p, state, resolution)
        pca_img = np.array(Image.fromarray(vis_pca(spatial)).resize(
            (img.shape[1], img.shape[0]), Image.NEAREST))
        axes[0, i].imshow(img);     axes[0, i].axis("off")
        axes[1, i].imshow(pca_img); axes[1, i].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("PCA → RGB", fontsize=9)
    plt.suptitle(f"TIPSv2 patch features — {class_name}", fontsize=11)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"Saved → {save_path}")
    plt.show()


def show_kmeans_sweep(img_path_or_arr, state, cluster_counts=(3, 5, 8, 12),
                      resolution=DEFAULT_RESOLUTION, save_path=None):
    """
    Show K-means clustering for a range of k values side by side.
    """
    orig = _as_array(img_path_or_arr)
    h, w = orig.shape[:2]
    spatial = extract_features(img_path_or_arr, state, resolution)

    n = len(cluster_counts)
    fig, axes = plt.subplots(1, n + 1, figsize=(5 * (n + 1), 5))
    axes[0].imshow(orig); axes[0].set_title("Original", fontsize=11); axes[0].axis("off")

    for ax, k in zip(axes[1:], cluster_counts):
        ax.imshow(vis_kmeans(spatial, h, w, k))
        ax.set_title(f"k = {k}", fontsize=11)
        ax.axis("off")

    plt.suptitle("TIPSv2 K-means feature clustering", fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"Saved → {save_path}")
    plt.show()


def show_zeroseg(img_path_or_arr, classes, state,
                 use_value_attention=True,
                 resolution=DEFAULT_RESOLUTION, save_path=None):
    """
    Three-panel zero-shot segmentation: original | colour overlay | area bar chart.
    """
    orig = _as_array(img_path_or_arr)

    extract_fn = (extract_features_value_attention
                  if use_value_attention else extract_features)
    spatial    = extract_fn(img_path_or_arr, state, resolution)
    class_embs = encode_text_classes(classes, state)
    blend, _, label_map = zeroseg(spatial, orig, classes, class_embs)

    n       = len(classes)
    palette = (plt.cm.tab20(np.linspace(0, 1, max(n, 2)))[:n, :3] * 255).astype(np.uint8)
    uids, counts = np.unique(label_map, return_counts=True)
    order        = np.argsort(-counts)
    uids, counts = uids[order], counts[order]
    total        = counts.sum()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    axes[0].imshow(orig)
    axes[0].set_title("Original scene", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(blend)
    axes[1].legend(
        handles=[mpatches.Patch(color=palette[i] / 255.0, label=classes[i])
                 for i in range(n)],
        loc="lower right", fontsize=8, framealpha=0.85)
    mode_str = "value-attention" if use_value_attention else "standard"
    axes[1].set_title(f"Zero-shot segmentation  ({mode_str} features)", fontsize=12)
    axes[1].axis("off")

    axes[2].barh(
        range(len(uids)),
        counts / total * 100,
        color=[palette[i] / 255.0 for i in uids])
    axes[2].set_yticks(range(len(uids)))
    axes[2].set_yticklabels(
        [f"{classes[i]}  ({counts[j] / total * 100:.1f}%)"
         for j, i in enumerate(uids)], fontsize=9)
    axes[2].set_xlabel("Coverage (%)")
    axes[2].set_title("Class area distribution", fontsize=12)
    axes[2].invert_yaxis()

    feat_label = "TCL ensemble + value-attention" if use_value_attention \
                 else "TCL ensemble + standard features"
    plt.suptitle(f"TIPSv2 zero-shot segmentation  ·  {feat_label}",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"Saved → {save_path}")
    plt.show()


def show_zeroseg_comparison(img_path_or_arr, classes, state,
                             resolution=DEFAULT_RESOLUTION, save_path=None):
    """
    Side-by-side comparison: standard features vs value-attention features.
    Shows why the value-attention trick sharpens segment boundaries.
    """
    orig = _as_array(img_path_or_arr)
    class_embs = encode_text_classes(classes, state)

    spatial_std = extract_features(img_path_or_arr, state, resolution)
    spatial_val = extract_features_value_attention(img_path_or_arr, state, resolution)

    blend_std, _, _ = zeroseg(spatial_std, orig, classes, class_embs)
    blend_val, _, _ = zeroseg(spatial_val, orig, classes, class_embs)

    n       = len(classes)
    palette = (plt.cm.tab20(np.linspace(0, 1, max(n, 2)))[:n, :3] * 255).astype(np.uint8)
    patches = [mpatches.Patch(color=palette[i] / 255.0, label=classes[i])
               for i in range(n)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].imshow(orig);       axes[0].set_title("Original", fontsize=12)
    axes[1].imshow(blend_std);  axes[1].set_title("Standard features", fontsize=12)
    axes[2].imshow(blend_val);  axes[2].set_title("Value-attention features", fontsize=12)
    for ax in axes:
        ax.axis("off")
    axes[2].legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.85)
    plt.suptitle("Zero-shot segmentation: standard vs value-attention features",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"Saved → {save_path}")
    plt.show()


def show_depth_normals(img_path_or_arr, state,
                       resolution=DEFAULT_RESOLUTION, save_path=None):
    """
    Three-panel: original | DPT depth (turbo) | DPT surface normals (XYZ→RGB).
    """
    orig = _as_array(img_path_or_arr)
    h, w = orig.shape[:2]

    depth_img   = vis_depth_dpt(run_depth(img_path_or_arr, state, resolution), h, w)
    normals_img = vis_normals_dpt(run_normals(img_path_or_arr, state, resolution), h, w)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    axes[0].imshow(orig);        axes[0].set_title("Original scene", fontsize=12)
    axes[1].imshow(depth_img);   axes[1].set_title("DPT Depth  (turbo)", fontsize=12)
    axes[2].imshow(normals_img); axes[2].set_title("DPT Surface Normals  (XYZ → RGB)", fontsize=12)
    for ax in axes:
        ax.axis("off")
    plt.suptitle("TIPSv2 DPT — monocular depth & surface normals  (NYU Depth V2)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"Saved → {save_path}")
    plt.show()


def show_supervised_seg(img_path_or_arr, state,
                        resolution=DEFAULT_RESOLUTION, save_path=None):
    """
    Two-panel: original | DPT ADE20K segmentation with top-class legend.
    Only classes covering ≥ 2% of the image appear in the legend.
    """
    orig = _as_array(img_path_or_arr)
    blend, pred = vis_segmentation_dpt(run_segmentation(img_path_or_arr, state, resolution), orig)

    uids, counts = np.unique(pred, return_counts=True)
    order        = np.argsort(-counts)
    uids, counts = uids[order], counts[order]
    pcts         = counts / counts.sum() * 100
    leg_mask     = pcts >= 2.0

    patches = [
        mpatches.Patch(
            color=ADE20K_PALETTE[cid + 1] / 255.0,
            label=f"{ADE20K_CLASSES[cid] if cid < len(ADE20K_CLASSES) else f'class_{cid}'}  "
                  f"{pct:.1f}%"
        )
        for cid, pct in zip(uids[leg_mask], pcts[leg_mask])
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(orig);  axes[0].set_title("Original scene", fontsize=12); axes[0].axis("off")
    axes[1].imshow(blend); axes[1].set_title("DPT Supervised Segmentation  (ADE20K 150 classes)",
                                             fontsize=12); axes[1].axis("off")
    axes[1].legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.85)
    plt.suptitle("TIPSv2 DPT supervised segmentation  (frozen backbone, ADE20K)",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight"); print(f"Saved → {save_path}")
    plt.show()


# ── Fruits360 dataset helpers ──────────────────────────────────────────────

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
