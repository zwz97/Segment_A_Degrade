"""Generate `degradation_map.json` for SAM2 pipeline.

Scans the `data/images/` directory plus sibling degradation folders
(`data/pic_degraded/<degradation_type>/`) and produces a JSON file with the
structure expected by `sam2_eval_pipeline.py`:

{
  "<image_id>": {
    "ground_truth_rle": {"size": [H, W], "counts": "..."},
    "versions": {
      "original":           {"filepath": "images/<id>.jpg", "level": 0, "degradation_type": "original"},
      "gaussian_blur": {
        "5": {"filepath": "pic_degraded/gaussian_blur/<id>_gaussian_blur_5.jpg", "level": 5,  "degradation_type": "gaussian_blur"},
        ...
      },
      "jpeg_compression": {
        "80": {"filepath": "pic_degraded/jpeg_compression/<id>_jpeg_compression_80.jpg", "level": 80,  "degradation_type": "jpeg_compression"}
      }
    }
  }
}

Run directly:
    python data/data_scripts/code_json.py

Requirements:
    pip install pycocotools pillow
"""
from __future__ import annotations
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any

from PIL import Image
from pycocotools import mask as mask_utils

# -----------------------------------------------------------------------------
# Configuration (relative to project root)
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]  # project root
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"  # originals (renamed 1.jpg, 2.jpg, ...)
DEG_DIR = DATA_DIR / "pic_degraded"  # sub-dirs per degradation type
ANNO_DIR = DATA_DIR / "image_annotations"  # *_annotations.json from COCO
OUTPUT_JSON = DATA_DIR / "degradation_map.json"

# Degradation types we expect as sub-directories inside DEG_DIR
DEGRADATIONS = [d.name for d in DEG_DIR.iterdir() if d.is_dir()]
print("Detected degradation types:", DEGRADATIONS)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def load_first_annotation_rle(anno_path: Path) -> Dict[str, Any]:
    """Return first polygon list `[x1,y1,x2,y2,...]` from annotation JSON."""
    with open(anno_path, "r") as f:
        data = json.load(f)

    # Normalise to list[annotation]
    anns = (
        data["annotations"] if isinstance(data, dict) and "annotations" in data else data
    )
    if not isinstance(anns, list):
        anns = [anns]

    polygon: List[float] | None = None
    rle_obj: Dict[str, Any] | None = None

    # Take first annotation with segmentation
    for ann in anns:
        seg = ann.get("segmentation")
        if not seg:
            continue
        if isinstance(seg, dict):  # Already RLE
            rle_obj = seg
        else:
            # COCO polygon: list[list[float]] or list[float]
            polygon = seg[0] if isinstance(seg[0], list) else seg
        break

    if rle_obj:
        if isinstance(rle_obj.get("counts"), bytes):
            rle_obj["counts"] = rle_obj["counts"].decode("ascii")
        return rle_obj

    if polygon:
        return coco_poly_to_rle(polygon, anno_path)

    return {}


def coco_poly_to_rle(segmentation: List, anno_path: Path) -> Dict:
    """Convert COCO polygon segmentation format to Run-Length Encoding (RLE).

    Args:
        segmentation: List of polygon coordinates [x1, y1, x2, y2, ...].
        anno_path: Path to annotation JSON file.

    Returns:
        RLE dictionary {'counts': [...], 'size': [h, w]}.
    """
    if not segmentation:
        return {}
    with Image.open(anno_path.parent / anno_path.stem.replace("_annotations", ".jpg")) as im:
        w, h = im.size  # PIL gives (width, height)
    rle_list = mask_utils.frPyObjects([segmentation], h, w)
    rle = mask_utils.merge(rle_list) if isinstance(rle_list, list) else rle_list
    # `counts` may be bytes; convert to str for JSON serialisation
    if isinstance(rle["counts"], bytes):
        rle["counts"] = rle["counts"].decode("ascii")
    return {"size": [h, w], "counts": rle["counts"]}


def build_degradation_map() -> Dict[str, Dict]:
    """Construct the main degradation map dictionary.

    Iterates through sequentially numbered images in `pic/`, finds their
    original path, ground truth annotation (converted to RLE), and all
    corresponding degraded versions.

    Returns:
        A dictionary where keys are base image identifiers (e.g., '1') and
        values are dictionaries containing 'image_path', 'gt_mask_rle', and
        a nested 'versions' dictionary.
    """
    degradation_map = {}

    # Iterate through original, sequentially named images (e.g., 1.jpg, 2.jpg)
    for img_path in sorted(IMAGES_DIR.iterdir()):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img_id = img_path.stem  # assumes numeric or unique string

        # Load image size
        with Image.open(img_path) as im:
            w, h = im.size  # PIL gives (width, height)

        # --- Ground Truth Mask --- Check if annotation file exists
        anno_path = ANNO_DIR / f"{img_id}_annotations.json"
        if not anno_path.exists():
            print(f"[Warning] Annotation missing for {img_id}, skipping.")
            continue

        # ---- Ground Truth RLE ----
        gt_rle = load_first_annotation_rle(anno_path)

        if not gt_rle:
            print(f"[Warning] Could not derive GT mask for {img_id}, skipping.")
            continue

        # --- Degraded Versions --- Build nested structure
        versions: Dict[str, Any] = {
            "original": {
                "filepath": str(Path("images") / img_path.name),
                "level": 0,
                "degradation_type": "original",
            }
        }

        # Scan degradation folders to populate nested dicts
        for deg_type in DEGRADATIONS:
            folder = DEG_DIR / deg_type
            pattern = f"{img_id}_{deg_type}_*.jpg"
            for file in folder.glob(pattern):
                # Extract level/parameter from filename (everything after last underscore)
                level_str = file.stem.rsplit("_", 1)[1]
                try:
                    level_val: Any = float(level_str) if "." in level_str else int(level_str)
                except ValueError:
                    print(f"[Warning] Could not parse level from {file.name}")
                    continue

                # Ensure the degradation type dict exists
                if deg_type not in versions:
                    versions[deg_type] = {}

                versions[deg_type][level_str] = {
                    "filepath": str(Path("pic_degraded") / deg_type / file.name),
                    "level": level_val,
                    "degradation_type": deg_type,
                }

        # --- Assemble Entry --- Add the complete entry for this base image ID
        degradation_map[img_id] = {
            "ground_truth_rle": gt_rle,
            "versions": versions,
        }

    return degradation_map


def main() -> None:
    """Script entry point: build map and save to JSON."""
    # Build degradation map
    degradation_map = build_degradation_map()

    # Define the output path for the JSON map
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)

    # Save the map to the JSON file
    with open(OUTPUT_JSON, "w") as fo:
        json.dump(degradation_map, fo, indent=2)
    print(f"Written {OUTPUT_JSON.relative_to(ROOT_DIR)} with {len(degradation_map)} images.")


if __name__ == "__main__":
    main()
