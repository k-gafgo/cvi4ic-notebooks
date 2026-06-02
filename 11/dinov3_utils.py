"""
dinov3_utils.py
===============
Shared utilities for the DINOv3 × Fruits-360 lecture notebooks.

Covers:
  - Dataset helpers  (class resolution, image-path sampling)
  - Model loading    (frozen DINOv3 backbone + processor)
  - Feature extraction (CLS tokens, patch tokens, batched embeddings)
  - Feature caching  (load-or-compute, keyed to a cache directory)
  - PCA helpers      (per-image RGB map from patch tokens)
  - Similarity helpers (patch sim-maps, top-pair mining, pair-grid display)
  - Triplet learning  (TripletNet, IdentityNet, TripletDataset, eval metrics)
"""

# ── Standard library ──────────────────────────────────────────────────────
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ── Numeric / ML ──────────────────────────────────────────────────────────
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA as _PCA
from torch.utils.data import Dataset

# ── Visualization ─────────────────────────────────────────────────────────
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────
# Global constants  (DINOv3 ViT-B/16, 224 px input)
# ─────────────────────────────────────────────────────────────────────────
PATCH_SIZE: int = 16
IMG_SIZE:   int = 224
N_SIDE:     int = IMG_SIZE // PATCH_SIZE   # 14
N_PATCH:    int = N_SIDE ** 2             # 196


# ═════════════════════════════════════════════════════════════════════════
# Dataset helpers
# ═════════════════════════════════════════════════════════════════════════

def resolve_class(name: str, available: list) -> str:
    """
    Case-insensitive *prefix* match of *name* against *available* class names.

    >>> resolve_class("pine", ["Pineapple", "Plum"])
    'Pineapple'
    """
    for c in available:
        if c.lower().startswith(name.lower()):
            return c
    raise ValueError(
        f"Class '{name}' not found.\n"
        f"Available (first 20): {available[:20]}"
    )


def get_paths(class_name: str, n: int, train_root: Path) -> list:
    """
    Return up to *n* randomly-sampled ``pathlib.Path`` objects for *class_name*.

    Searches for ``.jpg`` and ``.png`` files under ``train_root / class_name``.
    """
    folder = train_root / class_name
    paths  = sorted(folder.glob("*.jpg")) + sorted(folder.glob("*.png"))
    if not paths:
        raise FileNotFoundError(f"No images found in {folder}")
    return random.sample(paths, min(n, len(paths)))


# ═════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════

def load_backbone(
    model_id: str = "facebook/dinov3-vitb16-pretrain-lvd1689m",
    device: Optional[str] = None,
) -> tuple:
    """
    Load a frozen DINOv3 backbone and its image processor.

    Returns
    -------
    processor : ``AutoImageProcessor``
    model     : ``AutoModel`` (frozen, eval mode)
    device    : ``str``  – 'cuda' or 'cpu'
    num_reg   : ``int``  – number of register tokens inserted between CLS and
                           patch tokens in ``last_hidden_state``
    """
    from transformers import AutoModel, AutoImageProcessor

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading {model_id} on {device} …")
    processor = AutoImageProcessor.from_pretrained(model_id)
    model     = AutoModel.from_pretrained(model_id).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    num_reg = getattr(model.config, "num_register_tokens", 0)
    print(
        f"Backbone ready  │  register tokens: {num_reg}"
        f"  │  patch grid: {N_SIDE}×{N_SIDE}"
    )
    return processor, model, device, num_reg


# ═════════════════════════════════════════════════════════════════════════
# Feature extraction
# ═════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def extract_features(
    paths_by_class: dict,
    processor,
    model,
    device: str,
    num_reg: int,
    batch_size: int = 16,
) -> tuple:
    """
    Extract **CLS tokens** and **patch tokens** for every image.

    ``last_hidden_state`` layout for DINOv3:
      [CLS, reg_0 … reg_k, patch_0 … patch_195]

    Parameters
    ----------
    paths_by_class : ``{class_name: [Path, …], …}``

    Returns
    -------
    cls_tokens   : ``np.ndarray`` (N, 768)
    patch_tokens : ``np.ndarray`` (N, 196, 768)
    labels       : ``list[str]`` of length N
    """
    flat_paths, flat_labels = [], []
    for cls, ps in paths_by_class.items():
        flat_paths.extend(ps)
        flat_labels.extend([cls] * len(ps))

    all_cls, all_patch = [], []
    for i in range(0, len(flat_paths), batch_size):
        batch  = [Image.open(p).convert("RGB") for p in flat_paths[i : i + batch_size]]
        inputs = None #TODO
        inputs = None #TODO
        out    = None #TODO

        all_cls.append(out.pooler_output.cpu().float().numpy())

        patches = None #TODO
        assert patches.shape[1] == N_PATCH, (
            f"Expected {N_PATCH} patches, got {patches.shape[1]}"
        )
        all_patch.append(patches)
        print(f"  {min(i + batch_size, len(flat_paths))}/{len(flat_paths)}", end="\r")

    print()
    return np.vstack(all_cls), np.vstack(all_patch), flat_labels


def load_or_compute_features(
    cache_dir: Path,
    paths_by_class: dict,
    processor,
    model,
    device: str,
    num_reg: int,
) -> tuple:
    """
    Load CLS + patch features from *cache_dir*, or compute and cache them.

    Cache files:
      ``<cache_dir>/cls_tokens.npy``
      ``<cache_dir>/patch_tokens.npy``
      ``<cache_dir>/labels.json``

    Returns
    -------
    cls_tokens, patch_tokens, labels  — same as :func:`extract_features`
    """
    cls_file   = cache_dir / "cls_tokens.npy"
    patch_file = cache_dir / "patch_tokens.npy"
    meta_file  = cache_dir / "labels.json"

    if cls_file.exists() and patch_file.exists() and meta_file.exists():
        print("Loading cached features …")
        cls_tokens   = np.load(cls_file)
        patch_tokens = np.load(patch_file)
        labels       = json.load(open(meta_file))
    else:
        print("Extracting features (runs once) …")
        cache_dir.mkdir(exist_ok=True)
        cls_tokens, patch_tokens, labels = extract_features(
            paths_by_class, processor, model, device, num_reg
        )
        np.save(cls_file,   cls_tokens)
        np.save(patch_file, patch_tokens)
        json.dump(labels, open(meta_file, "w"))

    print(f"CLS tokens  : {cls_tokens.shape}")
    print(f"Patch tokens: {patch_tokens.shape}")
    return cls_tokens, patch_tokens, labels


@torch.no_grad()
def embed_images(paths, processor, model, device) -> torch.Tensor:
    """
    Return **L2-normalised CLS embeddings** of shape ``(N, 768)``.

    Processes images one-by-one (suitable for small N).
    For large batches, prefer :func:`extract_features`.
    """
    vecs = []
    for p in paths:
        img    = Image.open(p).convert("RGB")
        inputs = processor(images=img, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        out    = model(**inputs)
        vecs.append(out.pooler_output.squeeze(0).cpu())
    emb = torch.stack(vecs)          # (N, 768)
    return F.normalize(emb, dim=-1)  # L2-normalise


@torch.no_grad()
def compute_embeddings(
    paths_by_class: dict,
    processor,
    model,
    device: str,
    batch_size: int = 32,
) -> tuple:
    """
    Batched CLS-only embedding for all classes.

    Returns
    -------
    embeddings : ``np.ndarray`` (N, 768)
    labels     : ``list[str]`` of length N
    """
    flat_paths, flat_labels = [], []
    for cls, ps in paths_by_class.items():
        flat_paths.extend(ps)
        flat_labels.extend([cls] * len(ps))

    all_emb = []
    for i in range(0, len(flat_paths), batch_size):
        batch  = [Image.open(p).convert("RGB") for p in flat_paths[i : i + batch_size]]
        inputs = processor(images=batch, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            all_emb.append(model(**inputs).pooler_output.cpu().float().numpy())
        print(f"  {min(i + batch_size, len(flat_paths))}/{len(flat_paths)}", end="\r")

    print()
    return np.vstack(all_emb), flat_labels


def load_or_compute_cls_embeddings(
    cache_dir: Path,
    paths_by_class: dict,
    processor,
    model,
    device: str,
) -> tuple:
    """
    Load CLS-only embeddings from *cache_dir*, or compute and cache them.

    Cache files:
      ``<cache_dir>/embeddings.npy``
      ``<cache_dir>/metadata.json``

    Returns
    -------
    embeddings : ``np.ndarray`` (N, 768)
    labels     : ``list[str]`` of length N
    """
    emb_file  = cache_dir / "embeddings.npy"
    meta_file = cache_dir / "metadata.json"

    if emb_file.exists() and meta_file.exists():
        print("Loading cached embeddings …")
        embeddings = np.load(emb_file)
        labels     = json.load(open(meta_file))
    else:
        print("Computing embeddings (runs once) …")
        cache_dir.mkdir(exist_ok=True)
        embeddings, labels = compute_embeddings(paths_by_class, processor, model, device)
        np.save(emb_file, embeddings)
        json.dump(labels, open(meta_file, "w"))

    print(f"Embeddings: {embeddings.shape}  │  classes: {list(dict.fromkeys(labels))}")
    return embeddings, labels


@torch.no_grad()
def extract_patch_tokens(img_path, processor, model, device, num_reg) -> torch.Tensor:
    """
    Return ``(N_PATCH, 768)`` **L2-normalised** patch tokens for one image,
    skipping the CLS and register tokens.
    """
    img    = Image.open(img_path).convert("RGB")
    inputs = processor(images=img, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out    = model(**inputs)
    tokens = out.last_hidden_state[0, 1 + num_reg :, :].cpu().float()
    assert tokens.shape[0] == N_PATCH, (
        f"Expected {N_PATCH} patch tokens, got {tokens.shape[0]}."
    )
    return F.normalize(tokens, dim=-1)


# ═════════════════════════════════════════════════════════════════════════
# PCA helpers
# ═════════════════════════════════════════════════════════════════════════

def pca_rgb_map(patch_vec: np.ndarray) -> tuple:
    """
    Fit PCA on a single image's ``(196, 768)`` patch token matrix.

    Each of the three components is min-max normalised to ``[0, 1]``
    and mapped to an R/G/B channel, producing an intuitive semantic colour map:

    | Component | Typical content                              |
    |-----------|----------------------------------------------|
    | PC1 → R   | Foreground vs. background                   |
    | PC2 → G   | Spatial gradient (left/right or top/bottom) |
    | PC3 → B   | Fine-grained texture regions                |

    Returns
    -------
    rgb_map : ``np.ndarray`` of shape ``(14, 14, 3)``, values in ``[0, 1]``
    pca     : fitted ``sklearn.decomposition.PCA`` object
    """
    pca  = _PCA(n_components=3)
    feat = pca.fit_transform(patch_vec.astype(np.float32))   # (196, 3)
    for k in range(3):
        mn, mx     = feat[:, k].min(), feat[:, k].max()
        feat[:, k] = (feat[:, k] - mn) / (mx - mn + 1e-8)
    return feat.reshape(N_SIDE, N_SIDE, 3), pca


# ═════════════════════════════════════════════════════════════════════════
# Similarity helpers
# ═════════════════════════════════════════════════════════════════════════

def patch_sim_map(
    query_tok: torch.Tensor,
    query_rc: tuple,
    target_tok: torch.Tensor,
) -> np.ndarray:
    """
    Cosine similarity of **one query patch** to all patches of a target image.

    Parameters
    ----------
    query_tok  : ``(N_PATCH, D)`` L2-normalised patch tokens of the query image
    query_rc   : ``(row, col)`` of the selected patch in the 14×14 grid
    target_tok : ``(N_PATCH, D)`` L2-normalised patch tokens of the target image

    Returns
    -------
    ``np.ndarray`` of shape ``(N_SIDE, N_SIDE)`` with values in ``[-1, 1]``
    """
    r, c  = query_rc
    q     = query_tok[r * N_SIDE + c].unsqueeze(0)
    sims  = (q @ target_tok.T).squeeze(0).numpy()
    return sims.reshape(N_SIDE, N_SIDE)


def top_pairs(
    sim_block: np.ndarray,
    top_k: int = 3,
    highest: bool = True,
    exclude_diagonal: bool = False,
) -> list:
    """
    Return the *top-k* ``(row_idx, col_idx, similarity)`` entries from *sim_block*.

    Parameters
    ----------
    sim_block        : 2-D similarity array
    top_k            : number of pairs to return
    highest          : ``True`` → highest similarities; ``False`` → lowest
    exclude_diagonal : if ``True`` skip entries where ``row == col``
    """
    flat  = None # TODO
    order = None # TODO
    pairs = []
    for idx in order:
        r, c = None, None # TODO
        if exclude_diagonal and r == c:
            continue
        pairs.append((r, c, float(sim_block[r, c])))
        if len(pairs) == top_k:
            break
    return pairs


def show_pair_grid(
    pairs: list,
    row_paths: list,
    col_paths: list,
    title: str,
    color: str,
) -> None:
    """
    Display image pairs (one per row) with their cosine similarity score.

    Parameters
    ----------
    pairs     : list of ``(row_idx, col_idx, sim)`` from :func:`top_pairs`
    row_paths : image paths indexable by ``row_idx``
    col_paths : image paths indexable by ``col_idx``
    title     : suptitle text
    color     : title / score colour (e.g. ``"steelblue"``, ``"crimson"``)
    """
    fig, axes = plt.subplots(len(pairs), 2, figsize=(5, 2.5 * len(pairs)))
    if len(pairs) == 1:
        axes = axes.reshape(1, 2)
    for row, (r, c, sim) in enumerate(pairs):
        for j, (idx, pool) in enumerate([(r, row_paths), (c, col_paths)]):
            axes[row, j].imshow(Image.open(pool[idx]))
            axes[row, j].axis("off")
        axes[row, 0].set_title(f"sim = {sim:.3f}", fontsize=9, color=color)
    plt.suptitle(title, fontsize=11, color=color, fontweight="bold")
    plt.tight_layout()
    plt.show()


# ═════════════════════════════════════════════════════════════════════════
# Triplet metric learning — model definitions
# ═════════════════════════════════════════════════════════════════════════

class TripletNet(nn.Module):
    """
    Lightweight MLP head trained on top of frozen DINOv3 features.

    Architecture::

        768 → Linear(512) → ReLU → Dropout(0.2)
            → Linear(512) → ReLU → Dropout(0.2)
            → Linear(embedding_dim)
            → L2-normalise

    The output is always unit-norm, so cosine similarity equals dot product.
    """

    def __init__(self, input_dim: int = 768, embedding_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=1)


class IdentityNet(nn.Module):
    """
    Fixed random **orthogonal** projection  768 → *embedding_dim*.

    Used as an *untrained* baseline so that before/after comparisons are
    fair (same output dimensionality, no learned structure).
    """

    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(768, embedding_dim, bias=False)
        nn.init.orthogonal_(self.proj.weight)
        for p in self.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), p=2, dim=1)


class TripletDataset(Dataset):
    """
    On-the-fly random **(anchor, positive, negative)** triplets from
    pre-computed numpy embeddings.

    Each ``__getitem__`` call:
      1. Draws a random anchor index.
      2. Samples a **positive** (same class, different index) at random.
      3. Samples a **negative** (any other class) at random.

    No image I/O during training — works entirely on cached float32 arrays.
    """

    def __init__(self, embeddings_by_class: dict):
        self.class_names = list(embeddings_by_class.keys())
        all_emb, all_lbl = [], []
        for i, (_, embs) in enumerate(embeddings_by_class.items()):
            all_emb.extend(embs)
            all_lbl.extend([i] * len(embs))
        self.emb    = np.array(all_emb, dtype=np.float32)
        self.labels = np.array(all_lbl)

    def __len__(self) -> int:
        return len(self.emb) * 4   # expose more random triplets per epoch

    def __getitem__(self, _) -> tuple:
        a_idx    = random.randint(0, len(self.emb) - 1)
        a_lbl    = self.labels[a_idx]
        pos_pool = np.where(self.labels == a_lbl)[0]
        neg_pool = np.where(self.labels != a_lbl)[0]
        p_idx    = random.choice(pos_pool)
        n_idx    = random.choice(neg_pool)
        t        = lambda i: torch.tensor(self.emb[i])
        return t(a_idx), t(p_idx), t(n_idx)


# ═════════════════════════════════════════════════════════════════════════
# Triplet metric learning — evaluation utilities
# ═════════════════════════════════════════════════════════════════════════

def transform(net: nn.Module, emb_np: np.ndarray) -> np.ndarray:
    """
    Pass *emb_np* through *net* in eval mode.

    Returns an L2-normalised ``np.ndarray`` of shape ``(N, embedding_dim)``.
    """
    net.eval()
    with torch.no_grad():
        return net(torch.tensor(emb_np, dtype=torch.float32)).numpy()


def build_centroids(net: nn.Module, by_class: dict) -> dict:
    """
    Compute the mean projected embedding per class.

    Returns
    -------
    ``{class_name: centroid_vector}`` (raw, not re-normalised)
    """
    return {
        cls: transform(net, embs).mean(axis=0)
        for cls, embs in by_class.items()
    }


def evaluate(
    net: nn.Module,
    emb: np.ndarray,
    lbl_str: list,
    classes: list,
    centroids: Optional[dict] = None,
) -> dict:
    """
    Evaluate *net* on held-out embeddings *emb*.

    Metrics
    -------
    **1-NN accuracy**
        Leave-one-out nearest-neighbour in projected space (cosine similarity).

    **centroid accuracy** *(only if centroids are provided)*
        Nearest-centroid classifier (cosine similarity to per-class centroids).

    Returns
    -------
    ``dict``  e.g. ``{"1-NN accuracy": 0.87, "centroid accuracy": 0.85}``
    """
    X = transform(net, emb)
    y = np.array([classes.index(l) for l in lbl_str])

    sim = X @ X.T
    np.fill_diagonal(sim, -np.inf)
    nn_acc = float((y[sim.argmax(axis=1)] == y).mean())
    result = {"1-NN accuracy": nn_acc}

    if centroids is not None:
        class_order  = list(centroids.keys())
        cmat         = np.stack([centroids[c] for c in class_order])
        cmat        /= np.linalg.norm(cmat, axis=1, keepdims=True) + 1e-8
        pred_names   = [class_order[i] for i in (X @ cmat.T).argmax(axis=1)]
        result["centroid accuracy"] = float(
            np.mean([p == g for p, g in zip(pred_names, lbl_str)])
        )

    return result


def separation_score(net: nn.Module, by_class: dict) -> float:
    """
    **Inter-centroid distance / mean intra-class distance** ratio.

    Higher is better: clusters are far apart relative to their internal spread.
    """
    centroids, intra = {}, []
    for cls, embs in by_class.items():
        trans          = transform(net, embs)
        c              = trans.mean(axis=0)
        centroids[cls] = c
        intra.append(np.linalg.norm(trans - c, axis=1).mean())

    names = list(centroids.keys())
    inter = [
        np.linalg.norm(centroids[names[i]] - centroids[names[j]])
        for i in range(len(names))
        for j in range(i + 1, len(names))
    ]
    return float(np.mean(inter) / (np.mean(intra) + 1e-8))


def centroid_predictions(
    net: nn.Module,
    emb: np.ndarray,
    centroids_dict: dict,
) -> list:
    """
    Predict class labels for *emb* using nearest-centroid (cosine distance).

    Returns
    -------
    ``list[str]`` of predicted class names
    """
    return # TODO
