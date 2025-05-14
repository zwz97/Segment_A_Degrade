# pipeline_utils.py
import os
import cv2
import numpy as np
import json 
import torch 
from pycocotools import mask as mask_utils 

from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except ImportError:
    print("Error: Could not import from the 'sam2' library.")
    print("Please ensure you have cloned the repo and installed it following the official README:")
    print("  git clone https://github.com/facebookresearch/sam2.git")
    print("  cd sam2")
    print("  pip install -e .")
    print("  cd ..")
    # Set to None so checks later will fail on purposw
    SAM2ImagePredictor = None
    SAM2AutomaticMaskGenerator = None

# --- Globals / Caching --- 
# Simple cache to avoid reloading the model and generator repeatedly if 
# the same model_hf_id and config are used across different pipeline runs 
# within the same script execution (less relevant if main.py is run per pipeline).
_cached_model = None
_cached_generator_config = None
_cached_predictor = None
_cached_generator = None

# --- Model Loading --- 
def load_sam2_predictor_and_generator(model_hf_id: str, generator_config: dict) -> tuple[SAM2ImagePredictor | None, SAM2AutomaticMaskGenerator | None]:
    """Loads the SAM2 model predictor and mask generator.

    Uses the SAM2 library's from_pretrained method, which handles downloading 
    from Hugging Face Hub (if model_hf_id is an ID) or loading from a local 
    checkpoint (.pt) specified by the ID.

    Uses a simple global cache to avoid reloading if called again with the 
    same model_hf_id and generator_config.

    Args:
        model_hf_id: The identifier for the SAM2 model. Can be a Hugging Face 
                     model ID (e.g., 'facebook/sam2-hiera-large') or potentially
                     a path to a local checkpoint recognized by the library.
        generator_config: A dictionary containing parameters for the 
                          SAM2AutomaticMaskGenerator (points_per_side, etc.).

    Returns:
        A tuple containing (SAM2ImagePredictor, SAM2AutomaticMaskGenerator), or (None, None)
        if loading fails.
    """
    global _cached_model, _cached_generator_config, _cached_predictor, _cached_generator
    
    # Check cache
    if _cached_model == model_hf_id and _cached_generator_config == generator_config and _cached_predictor and _cached_generator:
        print(f"Using cached SAM2 predictor and generator for {model_hf_id}")
        return _cached_predictor, _cached_generator

    print(f"Loading SAM2 model and generator for {model_hf_id}...")
    try:
        # Load the predictor using the sam2 library's method
        # This handles loading from HF Hub or local checkpoints internally
        predictor = SAM2ImagePredictor.from_pretrained(model_hf_id)
        
        # Create the mask generator using the model from the loaded predictor
        # Ensure the predictor and its model loaded successfully
        if predictor is None or predictor.model is None:
             raise ValueError("Predictor or predictor.model is None after loading.")
             
        generator = SAM2AutomaticMaskGenerator(predictor.model, **generator_config)
        
        # Update cache
        _cached_model = model_hf_id
        _cached_generator_config = generator_config
        _cached_predictor = predictor
        _cached_generator = generator
        
        print("SAM2 model and generator loaded successfully.")
        return predictor, generator
    except Exception as e:
        print(f"Error loading SAM2 model or generator {model_hf_id}: {e}")
        # Reset cache on error
        _cached_model = None
        _cached_generator_config = None
        _cached_predictor = None
        _cached_generator = None
        return None, None

# --- Prediction --- 
def predict_auto_mask(predictor: SAM2ImagePredictor, generator: SAM2AutomaticMaskGenerator, image_rgb: np.ndarray, image_path_for_logging: str = "") -> list | None:
    """Generates masks for an entire image using SAM2AutomaticMaskGenerator.

    Args:
        predictor: The initialized SAM2ImagePredictor.
        generator: The initialized SAM2AutomaticMaskGenerator.
        image_rgb: The input image as a NumPy array in RGB format.
        image_path_for_logging: Optional image path string for error messages.

    Returns:
        A list of dictionaries, where each dict represents a predicted mask 
        (containing 'segmentation', 'area', etc.), or None if an error occurs.
    """
    if predictor is None or generator is None:
        print("Error: Predictor or Generator not initialized.")
        return None
    try:
        # Set the image for the predictor (required step)
        predictor.set_image(image_rgb)
        # Generate masks using the automatic generator
        masks = generator.generate(image_rgb)
        return masks
    except Exception as e:
        print(f"Error during mask generation for image {image_path_for_logging}: {e}")
        return None

# --- Data Loading & Preprocessing --- 
def decode_coco_rle(rle_obj: dict) -> np.ndarray | None:
    """Decodes a COCO RLE (Run-Length Encoding) object into a binary mask.

    Args:
        rle_obj: The RLE object, typically a dictionary with 'size' and 'counts'.

    Returns:
        A NumPy array representing the binary mask (0s and 1s), or None if decoding fails.
    """
    try:
        if isinstance(rle_obj.get("counts"), str):
            rle_obj = {
                "size": rle_obj["size"],
                "counts": rle_obj["counts"].encode("utf-8"),
            }

        mask = mask_utils.decode(rle_obj)

        # pycocotools returns (H, W, 1). squeeze to (H, W) for easier comparison
        if mask.ndim == 3 and mask.shape[2] == 1:
            mask = np.squeeze(mask, axis=2)

        # Ensure binary (0/1 uint8)
        mask = (mask > 0).astype(np.uint8)
        return mask
    except Exception as e:
        print(f"Error decoding RLE object: {e}. RLE: {rle_obj}")
        return None

def load_sam2_evaluation_data(data_path: str, image_base_dir: str) -> list[dict] | None:
    """Loads the evaluation data map JSON, decodes ground truth masks, and prepares items.

    Expects a JSON structure like (nested `versions`):
    {
        "image_id_1": {
            "ground_truth_rle": { "size": [h, w], "counts": "..." },
            "versions": {
                "original": { "filepath": "images/1.jpg", "level": 0, "degradation_type": "original" },
                "gaussian_blur": {
                    "3": { "filepath": "pic_degraded/gaussian_blur/1_gaussian_blur_3.jpg", "level": 3, "degradation_type": "gaussian_blur" },
                    "5": { "filepath": "pic_degraded/gaussian_blur/1_gaussian_blur_5.jpg", "level": 5, "degradation_type": "gaussian_blur" }
                },
                "jpeg_compression": {
                    "80": { "filepath": "pic_degraded/jpeg_compression/1_jpeg_compression_80.jpg", "level": 80, "degradation_type": "jpeg_compression" }
                }
            }
        }
    }

    Args:
        data_path: Path to the JSON data map file (e.g., 'data/degradation_map.json').
        image_base_dir: Base directory where image files are stored (e.g., 'data/images/').
                        The relative paths in the JSON will be joined with this base.

    Returns:
        A list of dictionaries, where each dictionary represents one item to evaluate,
        containing keys like 'image_id', 'version_key', 'level', 'image_filepath' 
        (absolute path), and 'gt_mask' (decoded NumPy array). Returns None if loading fails.
    """
    print(f"Loading evaluation data from: {data_path}")
    try:
        with open(data_path, 'r') as f:
            data_map = json.load(f)
    except FileNotFoundError:
        print(f"Error: Data map file not found at {data_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in data map file {data_path}: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while loading data map: {e}")
        return None
    
    processed_items = []
    print("Processing data map entries...")
    # Iterate through each image entry in the JSON map
    for image_id, image_data in data_map.items():
        # --- 1. Decode Ground Truth Mask --- 
        if 'ground_truth_rle' not in image_data:
            print(f"Warning: Missing 'ground_truth_rle' for image_id {image_id}. Skipping this image.")
            continue
            
        gt_rle = image_data['ground_truth_rle']
        gt_mask = decode_coco_rle(gt_rle)
        if gt_mask is None:
            print(f"Warning: Failed to decode ground truth RLE for image_id {image_id}. Skipping this image.")
            continue
            
        # --- 2. Process Image Versions --- 
        if 'versions' not in image_data or not isinstance(image_data['versions'], dict):
            print(f"Warning: Missing or invalid 'versions' dictionary for image_id {image_id}. Skipping this image.")
            continue
            
        # Iterate through each version (original or degraded) for the current image.
        # Handle nested dictionaries (deg_type -> level -> data)
        versions_dict = image_data['versions']

        for v_key, v_value in versions_dict.items():
            # Case 1: Leaf dictionary (e.g., 'original')
            if isinstance(v_value, dict) and 'filepath' in v_value:
                relative_path = v_value['filepath']
                level = v_value.get('level', 0)
                absolute_image_path = os.path.abspath(os.path.join(image_base_dir, relative_path))
                processed_items.append({
                    'image_id': image_id,
                    'version_key': v_key,
                    'level': level,
                    'image_filepath': absolute_image_path,
                    'gt_mask': gt_mask,
                })
                continue

            # Case 2: Nested dict of levels under a degradation type
            if isinstance(v_value, dict):
                deg_type = v_key
                for lvl_key, lvl_data in v_value.items():
                    if not isinstance(lvl_data, dict) or 'filepath' not in lvl_data or 'level' not in lvl_data:
                        print(f"Warning: Invalid nested version data for image_id {image_id}, deg_type {deg_type}, level {lvl_key}. Skipping.")
                        continue
                    relative_path = lvl_data['filepath']
                    level = lvl_data['level']
                    absolute_image_path = os.path.abspath(os.path.join(image_base_dir, relative_path))
                    # Compose a unique version key similar to previous flat format, e.g., 'gaussian_blur_3'
                    version_key = f"{deg_type}_{lvl_key}"
                    processed_items.append({
                        'image_id': image_id,
                        'version_key': version_key,
                        'level': level,
                        'image_filepath': absolute_image_path,
                        'gt_mask': gt_mask,
                    })
            else:
                print(f"Warning: Unrecognized version format for image_id {image_id}, key {v_key}. Skipping.")
                
    if not processed_items:
        print("Warning: No valid evaluation items were processed from the data map.")
        return None 
        
    print(f"Successfully processed {len(processed_items)} evaluation items.")
    return processed_items

if __name__ == "__main__":
    """Light-weight sanity checks that run with `python pipeline_utils.py`.
    They avoid heavy model downloads and only use in-memory dummy data/files."""
    print("Running pipeline_utils self-tests …")

    import pycocotools.mask as mask_util
    import tempfile, json, cv2, os

    # --- 1. decode_coco_rle round-trip ---
    base_mask = np.zeros((4, 4), dtype=np.uint8)
    base_mask[1, 1] = 1  # single pixel
    # RLE encode needs Fortran-contiguous array (for OpenCV) - idk gemini told me this lol
    rle = mask_util.encode(np.asfortranarray(base_mask.copy()))
    # JSON requires str, not bytes
    rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]
    decoded = decode_coco_rle(rle)
    assert decoded is not None and np.array_equal(decoded, base_mask), "decode_coco_rle failed"
    print("  ✓ decode_coco_rle basic round-trip passed")

    # --- 2. load_sam2_evaluation_data minimal flow ---
    with tempfile.TemporaryDirectory() as tmpdir:
        # Dummy image file (pipeline doesn’t read it at this stage)
        dummy_img_path = os.path.join(tmpdir, "dummy.jpg")
        cv2.imwrite(dummy_img_path, np.zeros((4, 4, 3), dtype=np.uint8))

        data_map = {
            "img1": {
                "ground_truth_rle": rle,
                "versions": {
                    "v1": {"filepath": os.path.basename(dummy_img_path), "level": 0}
                },
            }
        }
        data_json_path = os.path.join(tmpdir, "map.json")
        with open(data_json_path, "w") as f:
            json.dump(data_map, f)

        items = load_sam2_evaluation_data(data_json_path, tmpdir)
        assert items and len(items) == 1, "load_sam2_evaluation_data failed"
        assert os.path.isabs(items[0]["image_filepath"]), "image_filepath not absolute"
        print("  ✓ load_sam2_evaluation_data minimal flow passed")

    print("pipeline_utils self-tests completed successfully.")
