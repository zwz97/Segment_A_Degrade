#!/usr/bin/env python3
"""Prepare a degradations dataset for SAM2 Analysis.

Steps performed:
1. Sample `n` images from a COCO annotation file that contain **at least one**
   segmentation mask.
2. Download the images and their SINGLE annotation JSON into `data/images/` and
   `data/image_annotations/`.
3. Copy & rename the images sequentially into `data/pic/` (1.jpg, 2.jpg, …)
   together with their annotation files (`1_annotations.json`, …).
4. Generate degraded versions of each image under
   `data/pic_degraded/<degradation_type>/` according to a fixed parameter grid.

Run:
    python data/data_scripts/code_degradation.py --coco-annotations path/to/instances_train2017.json --samples 100

"""
from __future__ import annotations
import argparse
import logging
import random
import json
from pathlib import Path
from typing import List, Dict

import requests
from pycocotools.coco import COCO
from PIL import Image, ImageEnhance
import numpy as np
import cv2

# -----------------------------------------------------------------------------
# Project directories (always relative to repo root)
# -----------------------------------------------------------------------------
# Define project directories relative to the repository root
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
ANNO_DIR = DATA_DIR / "image_annotations"
PIC_DIR = DATA_DIR / "pic"
PIC_DEG_DIR = DATA_DIR / "pic_degraded"

# Create directories if they do not exist
for d in (IMAGES_DIR, ANNO_DIR, PIC_DIR, PIC_DEG_DIR):
    d.mkdir(parents=True, exist_ok=True)

# -----------------------------------------------------------------------------
# Degradation parameter grid
# -----------------------------------------------------------------------------
# Define the degradation parameter grid with degradation types and their parameters
PARAM_GRID: Dict[str, Dict[str, List[float | int]]] = {
    "gaussian_blur": {"kernel_size": [3, 5, 11, 21, 31]},  # odd ints only
    "motion_blur": {"kernel_size": [5, 15, 25, 35, 45]},
    "jpeg_compression": {"quality": [100, 80, 60, 40, 20]},
    "low_contrast": {"factor": [1.0, 0.8, 0.6, 0.4, 0.2]},
}

# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def sample_image_ids(coco: COCO, k: int, rng: random.Random) -> List[int]:
    """Return up to *k* image IDs that have a non-empty segmentation mask."""
    # Initialize a list to store valid image IDs
    valid_ids: List[int] = []
    
    # Iterate through all image IDs in the COCO dataset
    for img_id in coco.getImgIds():
        # Get annotation IDs for the current image
        ann_ids = coco.getAnnIds(imgIds=[img_id], iscrowd=None)
        
        # Skip images without annotations
        if not ann_ids:
            continue
        
        # Load annotations for the current image
        anns = coco.loadAnns(ann_ids)
        
        # Check if any annotation has a non-empty segmentation mask
        if any(a.get("segmentation") for a in anns):
            # Add the image ID to the list of valid IDs
            valid_ids.append(img_id)
    
    # Shuffle the list of valid IDs and return up to k IDs
    rng.shuffle(valid_ids)
    return valid_ids[:k]


def download_images_and_annotations(coco: COCO, img_ids: List[int]) -> None:
    """Download images from COCO URLs and save the first annotation for each."""
    # Log the number of images to download
    logging.info("Downloading %d images …", len(img_ids))
    
    # Load image metadata from the COCO dataset
    imgs = coco.loadImgs(img_ids)
    
    # Iterate through the images and download them
    for img in imgs:
        # Download the image from the COCO URL
        img_bytes = requests.get(img["coco_url"], timeout=10).content
        
        # Save the image to the images directory
        img_path = IMAGES_DIR / img["file_name"]
        img_path.write_bytes(img_bytes)

        # Get the first annotation ID for the current image
        ann_id = coco.getAnnIds(imgIds=[img["id"]], iscrowd=None)[0]
        
        # Load the annotation for the current image
        ann = coco.loadAnns([ann_id])[0]
        
        # Save the annotation to the image annotations directory
        anno_path = ANNO_DIR / (img["file_name"].rsplit(".", 1)[0] + ".json")
        anno_path.write_text(json.dumps(ann, indent=2))
        
        # Log the saved image
        logging.debug("Saved %s", img_path.name)


def rename_sequential(rng: random.Random) -> None:
    """Copy images & annos from images/ & image_annotations/ into pic/ sequentially numbered."""
    # List source images, sort for consistency before shuffling
    files = sorted([p for p in IMAGES_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}])
    
    # Shuffle the list of files
    rng.shuffle(files)
    
    # Iterate through the files and rename them sequentially
    for new_idx, img_path in enumerate(files, start=1):
        # Construct the new image path
        new_img_path = PIC_DIR / f"{new_idx}.jpg"
        
        # Convert to RGB (drop alpha) and save as JPEG
        Image.open(img_path).convert("RGB").save(new_img_path, "JPEG")

        # Copy corresponding annotation file if it exists
        anno_src = ANNO_DIR / (img_path.stem + ".json")
        anno_dst = PIC_DIR / f"{new_idx}_annotations.json"
        if anno_src.exists():
            anno_dst.write_text(anno_src.read_text())
        
        # Log the renamed image
        logging.debug("Renamed %s -> %s", img_path.name, new_img_path.name)


# Degradation application functions (using OpenCV and PIL)
def _apply_gaussian_blur(img: np.ndarray, k: int) -> np.ndarray:
    """Apply Gaussian blur with kernel size k."""
    # Apply Gaussian blur to the image
    return cv2.GaussianBlur(img, (k, k), 0)


def _apply_motion_blur(img: np.ndarray, k: int) -> np.ndarray:
    """Apply horizontal motion blur with kernel size k."""
    # Create a kernel for horizontal motion blur
    kernel = np.zeros((k, k), np.float32)
    kernel[k // 2, :] = 1.0 / k
    
    # Apply motion blur to the image
    return cv2.filter2D(img, -1, kernel)


def _apply_jpeg_compression(img: np.ndarray, q: int) -> np.ndarray:
    """Apply JPEG compression with quality level q."""
    # Encode the image as JPEG with the specified quality level
    _, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), q])
    
    # Decode the JPEG image
    return cv2.imdecode(enc, 1)


def _apply_low_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """Reduce image contrast by factor."""
    # Convert the image to PIL format
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Enhance the contrast of the image
    enhanced = ImageEnhance.Contrast(pil).enhance(factor)
    
    # Convert the image back to OpenCV format
    return cv2.cvtColor(np.asarray(enhanced), cv2.COLOR_RGB2BGR)


# Dictionary mapping degradation type names to their implementation functions
_DISPATCH = {
    "gaussian_blur": _apply_gaussian_blur,
    "motion_blur": _apply_motion_blur,
    "jpeg_compression": _apply_jpeg_compression,
    "low_contrast": _apply_low_contrast,
}


def generate_degradations() -> None:
    """Apply all defined degradations to all images in pic/."""
    # Ensure output subdirectories exist for each degradation type
    for name, params in PARAM_GRID.items():
        (PIC_DEG_DIR / name).mkdir(parents=True, exist_ok=True)

    # Process each sequentially named image from pic/
    for img_path in PIC_DIR.glob("*.jpg"):
        # Load the image
        img = cv2.imread(str(img_path))
        
        # Iterate through degradation types and their parameters
        for d_type, param_dict in PARAM_GRID.items():
            for param_name, values in param_dict.items():
                for v in values:
                    # Construct output path like: pic_degraded/gaussian_blur/1_gaussian_blur_3.jpg
                    out_path = PIC_DEG_DIR / d_type / f"{img_path.stem}_{d_type}_{v}.jpg"
                    
                    # Skip if the output file already exists
                    if out_path.exists():
                        continue
                    
                    # Apply the degradation to the image
                    processed = _DISPATCH[d_type](img.copy(), v)
                    
                    # Save the degraded image
                    cv2.imwrite(str(out_path), processed)
        
        # Log the degraded image
        logging.debug("Degraded %s", img_path.name)

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    # Create an argument parser
    p = argparse.ArgumentParser(description="Download COCO subset and generate degraded images.")
    
    # Add command line arguments
    p.add_argument("--coco-annotations", type=Path, required=True, help="Path to COCO instances JSON (e.g. train2017)")
    p.add_argument("--samples", type=int, default=100, help="Number of images to sample (default: 100)")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--verbose", action="store_true", help="Enable debug logging")
    
    # Parse the command line arguments
    return p.parse_args()


def main() -> None:
    """Main script execution logic."""
    # Parse the command line arguments
    args = parse_args()
    
    # Set up logging level based on verbosity flag
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    # Initialize random number generator for reproducibility
    rng = random.Random(args.seed)

    # Load COCO API
    coco = COCO(str(args.coco_annotations))

    # Sample image IDs from the COCO dataset
    img_ids = sample_image_ids(coco, args.samples, rng)
    
    # Download images and annotations
    download_images_and_annotations(coco, img_ids)
    
    # Rename images and annotations sequentially
    rename_sequential(rng)
    
    # Generate degraded images
    generate_degradations()
    
    # Log the completion of the script
    logging.info("Dataset preparation complete.\nOriginals: %d\nDegradations per image: %d", len(img_ids), sum(len(v) for v in PARAM_GRID.values()))


if __name__ == "__main__":
    main()
