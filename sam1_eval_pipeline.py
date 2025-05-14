import os
import cv2 
import numpy as np
import pandas as pd
from tqdm import tqdm 
from datetime import datetime 
import glob 
from metrics import calculate_miou, calculate_boundary_f1
from sam1_pipeline_utils import (
    load_sam1_predictor_and_generator,
    predict_auto_mask, 
    load_sam1_evaluation_data 
)

# --- Main Pipeline --- 

def run_evaluation_pipeline(config: dict):
    """
    Runs the full evaluation pipeline based on the provided configuration dictionary,
    using the official sam1 library loaded via Hugging Face ID.
    Loads pre-processed data using load_sam1_evaluation_data.
    Evaluates the best matching SAM1 prediction against the single GT mask for each image version.

    Args:
        config (dict): A dictionary containing configuration parameters:
            - model_hf_id (str): Hugging Face ID of the SAM2 model.
            - data_path (str): Path to the JSON file containing the evaluation data map.
            - image_base_dir (str): Base directory containing input images.
            - output_path (str): Path template to save the resulting CSV file (will be timestamped).
            - bf1_tolerance (int, optional): Pixel tolerance for BF1 calc. Defaults to 2.
            - generator_config (dict, optional): Configuration for SamAutomaticMaskGenerator.
    """
    print("--- Starting SAM1 Evaluation Pipeline --- ")

    # --- 1. Parameter Extraction --- 
    # Get necessary paths, model ID, and settings from the config 
    data_path = config.get('data_path')
    image_base_dir = config.get('image_base_dir')
    model_type = config.get('model_type')
    checkpoint_path = config.get('checkpoint_path')
    output_path_template = config.get("output_path", "output/default_results.csv") 
    bf1_tolerance = config.get("bf1_tolerance", 2)
    generator_config = config.get("generator_config", {})

    # Validate essential config paths
    if not all([model_type, checkpoint_path, data_path, image_base_dir]):
        print("Error: Missing required configuration: 'model_type', 'checkpoint_path', 'data_path', or 'image_base_dir'")
        return

    # --- 2. Load Model and Data --- 
    try:
        # Load the SAM1 predictor and mask generator
        predictor, generator = load_sam1_predictor_and_generator(model_type, checkpoint_path, generator_config)
        if predictor is None or generator is None:
             raise ValueError("Model/Generator loading failed.") 
        print("Model loaded successfully.")
        
        # Load and preprocess the evaluation data map 
        evaluation_items = load_sam1_evaluation_data(data_path, image_base_dir)
        if evaluation_items is None:
            print("Fatal Error: Failed to load or process evaluation data. Exiting.")
            return
        print(f"Loaded {len(evaluation_items)} items from {data_path}") 
    
    except Exception as e:
        print(f"Error during model or data loading: {e}")
        return # Stop pipeline if essential components fail to load

    # --- 3. Main Evaluation Loop --- 
    results_data = [] 
    print("Processing evaluation items...")
    for item in tqdm(evaluation_items, desc="Evaluating Images"): 
        image_id = item['image_id']
        version_key = item['version_key']
        level = item['level']
        image_path = item['image_filepath'] 
        gt_mask = item['gt_mask'] # Ground truth mask (already decoded numpy array)
        
        # Store basic info 
        base_result = {
            'image_id': image_id,
            'version_key': version_key,
            'level': level,
            'filepath': image_path, 
            'iou': np.nan,
            'bf1': np.nan,
            'sam2_score': np.nan,
            'status': 'Success' # Default status
        }

        try:
            # Check if image file exists
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            # Load image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image: {image_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # --- 3a. Run SAM2 Inference --- 
            # Generate masks for the entire image
            predictions = predict_auto_mask(predictor, generator, image_rgb, image_path) 
            
            # Check if any masks were generated
            if predictions is None or not predictions: 
                status = 'Error: Mask Generation Failed' if predictions is None else 'No Masks Generated'
                base_result['status'] = status
                results_data.append(base_result)
                continue

            # --- 3b. Compare Predictions to Ground Truth --- 
            best_iou = -1
            best_bf1 = -1
            best_sam2_score = np.nan 
            
            for pred_mask_data in predictions:
                pred_mask = pred_mask_data['segmentation'] # Predicted mask (numpy array)
                if pred_mask.shape != gt_mask.shape[:2]:
                    print(f"Warning: Shape mismatch GT {gt_mask.shape[:2]} vs Pred {pred_mask.shape} for {image_path}. Skipping this prediction.")
                    continue 
                
                current_iou = calculate_miou(pred_mask, gt_mask)
                current_bf1 = calculate_boundary_f1(pred_mask, gt_mask, tolerance_px=bf1_tolerance)

                if current_iou > best_iou:
                    best_iou = current_iou
                    best_bf1 = current_bf1
                    best_sam2_score = pred_mask_data.get('predicted_iou', np.nan)

            # --- 3c. Record Best Result for Image --- 
            base_result['iou'] = best_iou if best_iou > -1 else np.nan 
            base_result['bf1'] = best_bf1 if best_iou > -1 else np.nan
            base_result['sam2_score'] = best_sam2_score
            results_data.append(base_result)

        except FileNotFoundError as fnf_err:
            print(f"Error processing item {image_id} ({version_key}): {fnf_err}")
            base_result['status'] = 'Error: Image File Missing'
            results_data.append(base_result)
        except ValueError as val_err:
            print(f"Error processing item {image_id} ({version_key}): {val_err}")
            base_result['status'] = 'Error: Image Load Error'
            results_data.append(base_result)
        except Exception as e:
            # Catch-all for other unexpected errors during processing of a single item
            print(f"Unexpected error processing item {image_id} ({version_key}): {e}")
            base_result['status'] = f'Error: {e}' 
            results_data.append(base_result)

    # --- 4. Process and Save Results --- 
    print("Processing complete. Saving results...")
    if not results_data:
        print("Warning: No results were generated.")
        return
        
    # Convert results list to a Pandas DataFrame for easier handling and saving
    results_df = pd.DataFrame(results_data)
    
    # Generate timestamped output filename
    original_output_path = output_path_template
    output_dir = os.path.dirname(original_output_path)
    if not output_dir:
        output_dir = "."
    base_filename = os.path.splitext(os.path.basename(original_output_path))[0] 
    file_extension = os.path.splitext(original_output_path)[1] or ".csv" 

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"{base_filename}_{timestamp}{file_extension}"
    output_path = os.path.join(output_dir, timestamped_filename)

    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Ensured output directory exists: {output_dir}")
        
        # Save DataFrame to CSV
        results_df.to_csv(output_path, index=False)
        print(f" Evaluated {len(results_data)} images (saved → {output_path})") 
        
        # --- 5. Cleanup Old Results (Optional) --- 
        # Keep only the latest N results files as a FIFO cache
        keep_latest_n = 5 
        all_csv_files = sorted(
            glob.glob(os.path.join(output_dir, f"{base_filename}_*{file_extension}")),
            key=os.path.getmtime,
            reverse=True
        )
        
        if len(all_csv_files) > keep_latest_n:
            files_to_delete = all_csv_files[keep_latest_n:]
            print(f"Cleaning up old results (keeping latest {keep_latest_n}). Deleting {len(files_to_delete)} file(s)...")
            for f_del in files_to_delete:
                try:
                    os.remove(f_del)
                    print(f"  Deleted old result file: {f_del}")
                except OSError as oe:
                    print(f"Warning: Could not delete old results file {f_del}: {oe}")
                    
    except Exception as e:
        print(f"Error during results saving or cleanup: {e}")

    print("--- SAM2 Evaluation Pipeline Finished --- ")


# ---------------------------------------------------------------------------
# Lightweight Smoke Test (runs when executing this file directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    """Run a fast, dependency-light smoke test that exercises the full
    run_evaluation_pipeline wiring without downloading a SAM1 model. It
    replaces heavyweight functions with in-memory stubs so the test
    finishes in <1 s and verifies that a CSV result file is produced."""
    import tempfile, os, json, numpy as np, cv2
    import pycocotools.mask as mask_util

    # --- 1. Stub heavy functions ------------------------------------------------
    def _stub_load_sam1_predictor_and_generator(model_type: str, checkpoint_path: str, generator_config: dict):
        class _Stub:  # empty placeholder object
            pass
        return _Stub(), _Stub()

    def _stub_predict_auto_mask(predictor, generator, image_rgb, image_path):
        # Return a single all-zero mask the same H×W as the input image
        h, w, _ = image_rgb.shape
        return [{
            "segmentation": np.zeros((h, w), dtype=np.uint8),
            "predicted_iou": 1.0,
        }]

    # Monkey-patch the functions inside this module’s namespace
    load_sam1_predictor_and_generator = _stub_load_sam1_predictor_and_generator  # type: ignore
    predict_auto_mask = _stub_predict_auto_mask  # type: ignore

    # --- 2. Build minimal dataset in a temporary directory ----------------------
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a tiny 2×2 black JPEG image
        dummy_img_path = os.path.join(tmpdir, "img.jpg")
        cv2.imwrite(dummy_img_path, np.zeros((2, 2, 3), dtype=np.uint8))

        # Ground-truth: 2×2 all-zero mask encoded as RLE
        gt_mask = np.zeros((2, 2), dtype=np.uint8)
        rle = mask_util.encode(np.asfortranarray(gt_mask))
        # Ensure counts is a str for JSON serialization
        rle["counts"] = rle["counts"].decode("ascii") if isinstance(rle["counts"], bytes) else rle["counts"]

        data_map = {
            "img1": {
                "ground_truth_rle": rle,
                "versions": {
                    "v1": {
                        "filepath": os.path.basename(dummy_img_path),
                        "level": 0,
                    }
                },
            }
        }
        data_json_path = os.path.join(tmpdir, "map.json")
        with open(data_json_path, "w") as f:
            json.dump(data_map, f)

        cfg = {
            "model_type": "stub-model",  # arbitrary placeholder
            "checkpoint_path": "stub-checkpoint",  # arbitrary placeholder
            "data_path": data_json_path,
            "image_base_dir": tmpdir,
            "output_path": os.path.join(tmpdir, "results.csv"),
            "bf1_tolerance": 1,
            "generator_config": {},
        }

        print("\n>>> Running sam1_eval_pipeline smoke test …")
        run_evaluation_pipeline(cfg)

        # Assert that at least one CSV file was created in tmpdir
        created_csvs = [f for f in os.listdir(tmpdir) if f.endswith(".csv")]
        assert created_csvs, "Smoke test failed: no CSV output found"
        print(f"✓ Smoke test succeeded   (CSV → {created_csvs[0]})")
