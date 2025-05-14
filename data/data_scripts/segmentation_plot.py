import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pycocotools import mask as mask_utils
import argparse
from pathlib import Path

def plot_single_coco_segmentation(image_path: Path, ann_file: Path, category_ids=None, save_path: Path | None = None):
    """
    Overlays COCO-style segmentations for one image, when your JSON is either:
      - a single annotation dict
      - a list of annotation dicts, or
      - a dict with key 'annotations' → list of dicts.
    """
    # 1) Load annotations from JSON file
    with open(ann_file, 'r') as f:
        data = json.load(f)

    # 2) Figure out if JSON is single annotation dict or list of annotations
    # Normalize the data to a list of annotation dicts
    if isinstance(data, dict):
        if 'annotations' in data:
            anns = data['annotations']
        else:
            
            anns = [data]
    elif isinstance(data, list):
        anns = data
    else:
        raise ValueError(
            f"Unsupported JSON root type {type(data)}; "
            "expected dict-with-annotations, list, or single-annotation dict."
        )

    # 3) Optionally filter annotations by category ID
    if category_ids is not None:
        anns = [ann for ann in anns if ann.get('category_id') in category_ids]

    # 4) Load image and get its size
    img = np.array(Image.open(str(image_path)))
    height, width = img.shape[:2]

    # 5) Plot the image
    plt.figure(figsize=(10,10))
    plt.imshow(img)
    ax = plt.gca()

    # 6) Iterate over annotations and plot each mask/polygon
    for ann in anns:
        seg = ann.get('segmentation')

        # --- Handle different segmentation formats --- 
        # RLE (Run-Length Encoding)
        if isinstance(seg, dict):
            rle = seg

        # Uncompressed RLE: list-of-dicts
        elif isinstance(seg, list) and seg and isinstance(seg[0], dict):
            rle = mask_utils.frPyObjects(seg, height, width)

        # Polygon: list-of-lists-of-coordinates
        elif isinstance(seg, list) and seg and isinstance(seg[0], (list, tuple)):
            # Draw polygon outlines directly
            for poly in seg:
                pts = np.array(poly).reshape(-1, 2)
                ax.plot(pts[:, 0], pts[:, 1], linewidth=2)
            continue

        else:
            raise ValueError(f"Unsupported segmentation format: {type(seg)}")

        # Decode RLE and overlay mask
        # Create a binary mask from RLE
        m = mask_utils.decode(rle)
        
        # Overlay the mask with transparency
        ax.imshow(np.ma.masked_where(m == 0, m), alpha=0.4)

    # Remove axes for cleaner look
    ax.axis('off')
    plt.tight_layout()

    # 7) Save or show the plot
    if save_path:
        plt.savefig(str(save_path), bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Visualise COCO-style segmentations on an image.")
    parser.add_argument("image", type=Path, help="Path to the image file (e.g. data/images/1.jpg)")
    parser.add_argument("annotation", type=Path, help="Path to the corresponding COCO annotation JSON")
    parser.add_argument("--save", type=Path, default=None, help="Optional output path to save the visualisation instead of showing it")
    parser.add_argument("--categories", type=int, nargs="*", default=None, help="Optional list of category_ids to plot (default: all)")

    args = parser.parse_args()
    
    # Call the plotting function with parsed arguments
    plot_single_coco_segmentation(args.image, args.annotation, category_ids=args.categories, save_path=args.save)


if __name__ == "__main__":
    main()
