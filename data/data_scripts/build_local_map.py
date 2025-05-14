"""Generate 'degradation_map.json' from local image/annotation structure.

Scans the following directories relative to the project root:
- data/images/gt_img/        (for <id>.jpg and <id>_annotations.json)
- data/images/img_degraded/  (for <degradation_type>/<id>_*_<level>.jpg)

Produces 'data/degradation_map.json' with the structure expected by
`sam2_eval_pipeline.py`.

Run directly from the project root directory:
    python data/data_scripts/build_local_map.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List
import numpy as np

from PIL import Image
from pycocotools import mask as mask_utils

# -----------------------------------------------------------------------------
# Configuration (Adjust if needed)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]  # Project root
DATA_DIR = ROOT_DIR / "data"
GT_IMG_DIR = DATA_DIR / "images" / "gt_img"         # Originals + Annotations
DEG_IMG_DIR = DATA_DIR / "images" / "img_degraded" # Degraded images sub-folders
OUTPUT_JSON = DATA_DIR / "degradation_map.json"   # Output file location

# Regex to extract ID and Level from degraded filenames (adjust if needed)
# Assumes format like: <id>_<typename>_<level>.<ext>
# Example: 1_gaussian_blur_5.jpg -> id=1, level=5
# Example: 2_jpeg_70.jpg -> id=2, level=70
# Captures: 1: id, 2: level (numeric part)
FILENAME_PATTERN = re.compile(r"^(\d+)_.*?_(\d+(?:\.\d+)?)\.(?:jpg|png|jpeg)$", re.IGNORECASE)

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def get_mask_from_annotation(anno_path: Path, height: int, width: int) -> Dict | None:
    """
    Loads annotation JSON and converts segmentation (polygon or RLE) to a single RLE mask.
    Handles cases where an object consists of multiple polygon pieces by combining them.
    
    Args:
        anno_path: Path to the annotation JSON file
        height: Image height
        width: Image width
        
    Returns:
        RLE encoded mask dictionary with 'size' and 'counts', or None if failed
    """
    if not anno_path.is_file():
        print(f"Warning: Annotation file not found: {anno_path}")
        return None
    
    try:
        with open(anno_path, "r") as f:
            data = json.load(f)

        # Normalize to list[annotation]
        anns = data if isinstance(data, list) else [data]
        if isinstance(data, dict) and "annotations" in data:
            anns = data["annotations"]
        if not isinstance(anns, list):
            anns = [anns]  # Handle case where it's a single dict not in a list

        
        for ann in anns:
            seg = ann.get("segmentation")
            if not seg:
                continue
                
            # If it's already RLE, return it directly (ensuring size is correct)
            if isinstance(seg, dict) and "counts" in seg and "size" in seg:
                rle = seg.copy()
                # Ensure size is [height, width]
                rle["size"] = [height, width]
                # Ensure counts is string for JSON serialization
                if isinstance(rle["counts"], bytes):
                    rle["counts"] = rle["counts"].decode("utf-8")
                return rle
                
            # Handle polygon segmentation (list of polygons or single polygon)
            if isinstance(seg, list):
                # Create a blank binary mask
                mask = np.zeros((height, width), dtype=np.uint8)
                
                # Determine if seg is a list of polygons or a single polygon
                if seg and isinstance(seg[0], list) and all(isinstance(p, (int, float)) for p in seg[0]):
                    # seg is a list of polygons
                    polygons = seg
                else:
                    # seg is a single polygon
                    polygons = [seg]
                
                # Convert each polygon to a mask and combine them
                for polygon in polygons:
                    if len(polygon) >= 6 and len(polygon) % 2 == 0:  # Valid polygon (at least 3 points)
                        # Convert polygon to RLE using pycocotools
                        rle = mask_utils.frPyObjects([polygon], height, width)
                        # Convert RLE to mask
                        poly_mask = mask_utils.decode(rle)
                        # Squeeze the mask to remove the trailing dimension (H, W, 1) -> (H, W)
                        poly_mask = np.squeeze(poly_mask, axis=-1)
                        # Combine with existing mask (logical OR)
                        mask = np.logical_or(mask, poly_mask).astype(np.uint8)
                    else:
                        print(f"Warning: Invalid polygon format in {anno_path.name}: {polygon}")
                
                # If we have a valid mask (at least one polygon was processed)
                if np.any(mask):
                    # Convert the combined mask to RLE
                    rle = mask_utils.encode(np.asfortranarray(mask))
                    # Ensure counts is string for JSON serialization
                    if isinstance(rle["counts"], bytes):
                        rle["counts"] = rle["counts"].decode("utf-8")
                    # Ensure size is [height, width]
                    rle["size"] = [height, width]
                    return rle
                
        print(f"Warning: No valid segmentation found in {anno_path}")
        return None
        
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {anno_path}")
    except Exception as e:
        print(f"Error processing annotation {anno_path}: {e}")
    return None

# -----------------------------------------------------------------------------
# Main Logic
# -----------------------------------------------------------------------------

def build_degradation_map() -> Dict[str, Dict]:
    """Scans directories and builds the degradation map."""
    degradation_map = {}
    print(f"Scanning ground truth images and annotations in: {GT_IMG_DIR}")

    processed_ids = set()

    # Iterate through potential annotation files first
    for anno_file in GT_IMG_DIR.glob("*_annotations.json"):
        if not anno_file.is_file():
            continue

        img_id_match = re.match(r"^(\d+)_annotations\.json$", anno_file.name)
        if not img_id_match:
            print(f"Warning: Skipping file with unexpected name format: {anno_file.name}")
            continue

        img_id = img_id_match.group(1)
        if img_id in processed_ids:
             print(f"Warning: Duplicate annotation file found for ID {img_id}, skipping {anno_file.name}")
             continue

        # --- Find corresponding image --- 
        img_file = GT_IMG_DIR / f"{img_id}.jpg"
        if not img_file.is_file():
            # Try other extensions if needed
            img_file_png = GT_IMG_DIR / f"{img_id}.png"
            if img_file_png.is_file():
                img_file = img_file_png
            else:
                 print(f"Warning: Original image for annotation {anno_file.name} (ID: {img_id}) not found. Searched for {img_id}.jpg/.png. Skipping this ID.")
                 continue

        print(f"Processing ID: {img_id} (Image: {img_file.name}, Annotation: {anno_file.name})")

        # --- Get Image Dimensions --- 
        try:
            with Image.open(img_file) as im:
                width, height = im.size # PIL gives W, H
        except Exception as e:
            print(f"Error opening image {img_file}: {e}. Skipping ID {img_id}.")
            continue

        # --- Get Ground Truth Mask as RLE --- 
        gt_rle = get_mask_from_annotation(anno_file, height, width)
        if not gt_rle:
            print(f"Warning: Could not get valid mask from {anno_file.name}. Skipping ID {img_id}.")
            continue

        # --- Initialize entry for this image_id --- 
        degradation_map[img_id] = {
            "ground_truth_rle": gt_rle,
            "versions": {},
        }

        # --- Add Original Version --- 
        # Path relative to DATA_DIR (which is likely the image_base_dir in config)
        original_rel_path = Path("images") / "gt_img" / img_file.name
        degradation_map[img_id]["versions"]["original"] = {
            "filepath": str(original_rel_path),
            "level": 0,
            "degradation_type": "original",
        }

        # --- Scan for Degraded Versions --- 
        if not DEG_IMG_DIR.is_dir():
             print(f"Warning: Degraded images directory not found: {DEG_IMG_DIR}")
             continue # Move to next image ID if no degraded dir exists

        for deg_type_dir in DEG_IMG_DIR.iterdir():
            if not deg_type_dir.is_dir():
                continue
            deg_type = deg_type_dir.name

            # Add dict for this degradation type if not present
            if deg_type not in degradation_map[img_id]["versions"]:
                degradation_map[img_id]["versions"][deg_type] = {}

            for deg_file in deg_type_dir.glob(f"{img_id}_*.jpg"):
                 match = FILENAME_PATTERN.match(deg_file.name)
                 if match and match.group(1) == img_id:
                     level_str = match.group(2)
                     try:
                         level_val = float(level_str) if '.' in level_str else int(level_str)
                     except ValueError:
                          print(f"Warning: Could not parse level '{level_str}' from filename {deg_file.name}. Skipping this file.")
                          continue

                     # Path relative to DATA_DIR
                     deg_rel_path = Path("images") / "img_degraded" / deg_type / deg_file.name
                     degradation_map[img_id]["versions"][deg_type][level_str] = {
                         "filepath": str(deg_rel_path),
                         "level": level_val,
                         "degradation_type": deg_type,
                     }
                 
        processed_ids.add(img_id)

    if not degradation_map:
        print("Error: No valid image/annotation pairs found. Please check directories:")
        print(f"- Ground Truth/Annotations: {GT_IMG_DIR}")
        print(f"- Degraded Images: {DEG_IMG_DIR}")

    return degradation_map


def main() -> None:
    """Script entry point: build map and save to JSON."""
    print("Starting degradation map generation...")
    degradation_map = build_degradation_map()

    if degradation_map:
        OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

        print(f"Saving degradation map to: {OUTPUT_JSON}")
        try:
            with open(OUTPUT_JSON, "w") as fo:
                json.dump(degradation_map, fo, indent=2)
            print(f"Successfully wrote map with {len(degradation_map)} image entries.")
        except Exception as e:
            print(f"Error writing JSON file: {e}")
    else:
        print("No data processed, skipping JSON file generation.")

if __name__ == "__main__":
    main()
