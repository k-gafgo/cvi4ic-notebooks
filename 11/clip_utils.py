"""
clip_utils.py — Utility functions for the CLIP Fruits 360 lecture notebook.

Covers:
  - Dataset discovery and image loading
  - Text / image encoding helpers
  - Prompt ensembling
  - Zero-shot classification & accuracy evaluation
  - Gallery building and retrieval (text-to-image, image-to-image)

All functions accept `model`, `preprocess`, and `device` explicitly so they
work with any CLIP checkpoint — no module-level state.
"""

from __future__ import annotations

import pathlib
from typing import Iterator

import clip
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


def discover_classes(dataset_root: pathlib.Path) -> list[str]:
    """Return sorted list of class folder names found under *dataset_root*."""
    return sorted(d.name for d in dataset_root.iterdir() if d.is_dir())


def _normalize(s: str) -> str:
    """Lowercase and collapse underscores, hyphens, and spaces to a single space."""
    return " ".join(s.lower().replace("_", " ").replace("-", " ").split())


def find_folder(prefix: str, all_classes: list[str],
                dataset_root: pathlib.Path) -> str | None:
    """
    Resolve a folder name from a human-readable prefix.

    Matching is case-insensitive and treats underscores, hyphens, and spaces
    as equivalent, so ``"apple_golden_1"`` matches ``"Apple Golden 1"``.

    Resolution order:
    1. Exact path exists on disk.
    2. Normalised exact match against *all_classes*.
    3. First normalised prefix match against *all_classes*.

    Returns the folder name, or ``None`` if nothing matches.
    """
    if (dataset_root / prefix).is_dir():
        return prefix
    norm_prefix = _normalize(prefix)
    for c in all_classes:
        if _normalize(c) == norm_prefix:
            return c
    for c in all_classes:
        if _normalize(c).startswith(norm_prefix):
            return c
    return None


def load_class_images(folder: str, dataset_root: pathlib.Path,
                      n: int = 20) -> list[Image.Image]:
    """
    Load up to *n* JPEG images from *dataset_root / folder*, sorted by name.

    Returns a list of RGB PIL Images.
    """
    paths = sorted((dataset_root / folder).glob("*.jpg"))[:n]
    return [Image.open(p).convert("RGB") for p in paths]


def build_fruit_map(fruit_keywords: list[tuple[str, str]],
                    all_classes: list[str],
                    dataset_root: pathlib.Path) -> dict[str, str]:
    """
    Resolve a list of ``(folder_prefix, clean_label)`` pairs to a mapping of
    ``{resolved_folder_name: clean_label}``.

    Prints a warning for any prefix that cannot be resolved.
    """
    fruit_map: dict[str, str] = {}
    for prefix, label in fruit_keywords:
        folder = find_folder(prefix, all_classes, dataset_root)
        if folder:
            fruit_map[folder] = label
        else:
            print(f"[WARNING] no folder found for prefix '{prefix}'")
    return fruit_map


def suggest_folders(prefix: str, all_classes: list[str], n: int = 5) -> None:
    """
    Print the *n* closest class folder names to *prefix* (by normalised prefix).

    Call this when ``build_fruit_map`` warns that a prefix cannot be resolved,
    to see what the dataset actually contains.

    Example::

        suggest_folders("apple", all_classes)
        # Apple Golden 1
        # Apple Braeburn 1
        # ...
    """
    norm = _normalize(prefix)
    ranked = sorted(all_classes, key=lambda c: (
        0 if _normalize(c).startswith(norm) else
        1 if norm in _normalize(c) else 2
    ))
    print(f"Closest matches for '{prefix}':")
    for c in ranked[:n]:
        print(f"  {c}")


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def encode_text_prompts(
    prompts: list[str],
    model: clip.model.CLIP,
    device: str,
) -> torch.Tensor:
    """
    Tokenise and encode a list of text *prompts* with CLIP.

    Returns a normalised float32 tensor of shape ``(len(prompts), D)``.
    """
    with torch.no_grad():
        tokens = None  # TODO
        feats = None  # TODO
    return F.normalize(feats, dim=-1)


def encode_images_batched(
    images: list[Image.Image],
    model: clip.model.CLIP,
    preprocess,
    device: str,
    batch_size: int = 64,
    show_progress: bool = False,
) -> torch.Tensor:
    """
    Encode a list of PIL *images* with the CLIP image encoder in mini-batches.

    Returns a normalised float32 tensor of shape ``(len(images), D)`` on CPU.
    """
    parts: list[torch.Tensor] = []
    batches: Iterator = range(0, len(images), batch_size)
    if show_progress:
        batches = tqdm(batches, desc="Encoding images")
    with torch.no_grad():
        for i in batches:
            batch = images[i : i + batch_size]
            inputs = torch.stack([preprocess(img) for img in batch]).to(device)
            feats = None  # TODO
            parts.append(feats.cpu())
    return torch.cat(parts, dim=0)


def encode_template(
    template: str,
    labels: list[str],
    model: clip.model.CLIP,
    device: str,
) -> torch.Tensor:
    """
    Fill *template* (must contain ``{}``) with each label and encode the result.

    Returns a normalised float32 tensor of shape ``(len(labels), D)``.
    """
    prompts = [template.format(lbl) for lbl in labels]
    return encode_text_prompts(prompts, model, device)


def build_ensemble(
    templates: dict[str, str],
    labels: list[str],
    model: clip.model.CLIP,
    device: str,
) -> torch.Tensor:
    """
    Average text embeddings over all *templates* and re-normalise.

    This is the prompt-ensembling technique from the CLIP paper (+3.5% on
    ImageNet zero-shot over the single best template).

    Returns a normalised float32 tensor of shape ``(len(labels), D)``.
    """
    stack: list[torch.Tensor] = []
    with torch.no_grad():
        for tmpl in templates.values():
            feats = encode_template(tmpl, labels, model, device)
            stack.append(feats)
    return F.normalize(torch.stack(stack).mean(dim=0), dim=-1)


# ---------------------------------------------------------------------------
# Zero-shot classification
# ---------------------------------------------------------------------------


def classify_images(
    images: list[Image.Image],
    text_feats: torch.Tensor,
    model: clip.model.CLIP,
    preprocess,
    device: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Classify *images* against pre-computed text features.

    Parameters
    ----------
    images:
        List of PIL images to classify.
    text_feats:
        Normalised text embeddings, shape ``(N_classes, D)``.

    Returns
    -------
    np.ndarray
        Softmax probability matrix of shape ``(len(images), N_classes)``.
    """
    image_feats = encode_images_batched(images, model, preprocess, device, batch_size)
    probs = None  # TODO
    return probs.cpu().numpy()


def gallery_accuracy(
    class_images: dict[str, list[Image.Image]],
    class_folders: list[str],
    text_feats: torch.Tensor,
    model: clip.model.CLIP,
    preprocess,
    device: str,
) -> float:
    """
    Compute overall zero-shot accuracy over all classes in *class_images*.

    Each folder in *class_folders* is treated as the ground-truth class at the
    corresponding index.

    Returns
    -------
    float
        Fraction of correctly classified images across all classes.
    """
    correct = total = 0
    for true_idx, folder in enumerate(class_folders):
        imgs = class_images[folder]
        preds = None  # TODO
        correct += 0  # TODO
        total += len(imgs)
    return correct / total


# ---------------------------------------------------------------------------
# Gallery building & retrieval
# ---------------------------------------------------------------------------


def build_gallery(
    class_images: dict[str, list[Image.Image]],
    fruit_map: dict[str, str],
    model: clip.model.CLIP,
    preprocess,
    device: str,
    batch_size: int = 64,
) -> tuple[list[Image.Image], list[str], torch.Tensor]:
    """
    Flatten all class images into a single gallery and encode them.

    Returns
    -------
    gallery_imgs:
        Flat list of all PIL images.
    gallery_labels:
        Corresponding list of clean label strings.
    gallery_feats:
        Normalised image feature tensor, shape ``(N_gallery, D)``, on CPU.
    """
    gallery_imgs: list[Image.Image] = []
    gallery_labels: list[str] = []
    for folder, label in fruit_map.items():
        gallery_imgs.extend(class_images[folder])
        gallery_labels.extend([label] * len(class_images[folder]))

    gallery_feats = encode_images_batched(
        gallery_imgs, model, preprocess, device,
        batch_size=batch_size, show_progress=True,
    )
    return gallery_imgs, gallery_labels, gallery_feats


def text_retrieve(
    query: str,
    gallery_feats: torch.Tensor,
    model: clip.model.CLIP,
    device: str,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve gallery images most similar to a text *query*.

    Returns
    -------
    top_idx:
        Indices into the gallery, sorted by descending similarity.
    top_sims:
        Corresponding cosine similarities.
    """
    query_feat = encode_text_prompts([query], model, device)  # (1, D)
    sims = (query_feat.cpu() @ gallery_feats.T).squeeze().numpy()
    top_idx = sims.argsort()[::-1][:top_k]
    return top_idx, sims[top_idx]


def image_retrieve(
    query_img: Image.Image,
    gallery_feats: torch.Tensor,
    model: clip.model.CLIP,
    preprocess,
    device: str,
    top_k: int = 6,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Retrieve gallery images most similar to a query *image* (no text needed).

    Returns
    -------
    top_idx:
        Indices into the gallery, sorted by descending similarity.
    top_sims:
        Corresponding cosine similarities.
    """
    query_feat = encode_images_batched([query_img], model, preprocess, device)  # (1, D)
    sims = None  # TODO
    top_idx = None  # TODO
    return top_idx, sims[top_idx]
