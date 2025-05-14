import json
from pathlib import Path
import argparse
import numpy as np
import shutil
from typing import List, Dict

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow is required for visualization. Run 'pip install Pillow'")
    Image = None

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: Matplotlib is required for visualization. Run 'pip install matplotlib'")
    plt = None

try:
    import pycocotools.mask as maskUtils
except ImportError:
    print("Error: pycocotools is required for RLE decoding. Run 'pip install pycocotools'")
    maskUtils = None

# --- Define the expected degradation parameters ---
PARAM_GRID: Dict[str, Dict[str, List[float | int]]] = {
    "gaussian_blur": {"kernel_size": [3, 5, 11, 21, 31]},
    "motion_blur": {"kernel_size": [5, 15, 25, 35, 45]},
    "jpeg_compression": {"quality": [100, 80, 60, 40, 20]},
    "low_contrast": {"factor": [1.0, 0.8, 0.6, 0.4, 0.2]},
}
EXPECTED_DEGRADATION_TYPES = set(PARAM_GRID.keys())

def count_images(base_data_path: Path):
    """Counts images in gt_img and img_degraded directories."""
    gt_img_path = base_data_path / "data" / "images" / "gt_img"
    degraded_img_path = base_data_path / "data" / "images" / "img_degraded"
    
    gt_count = 0
    if gt_img_path.is_dir():
        # Count only common image file extensions
        gt_count = len([f for f in gt_img_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        print(f"Found {gt_count} ground truth images in {gt_img_path}")
    else:
        print(f"Warning: Ground truth directory not found: {gt_img_path}")

    degraded_counts = {}
    total_degraded = 0
    if degraded_img_path.is_dir():
        print(f"Scanning degraded images in {degraded_img_path}...")
        for deg_type_path in degraded_img_path.iterdir():
            if deg_type_path.is_dir():
                # Count only common image file extensions
                count = len([f for f in deg_type_path.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
                degraded_counts[deg_type_path.name] = count
                total_degraded += count
                print(f" - Found {count} images for degradation type: {deg_type_path.name}")
        print(f"Found a total of {total_degraded} degraded images across {len(degraded_counts)} types.")
    else:
         print(f"Warning: Degraded images directory not found: {degraded_img_path}")

    return {"ground_truth": gt_count, "degraded": degraded_counts, "total_degraded": total_degraded}

def validate_degradation_map(map_path: Path, base_data_path: Path) -> bool:
    """Validates the structure and file existence for the degradation map."""
    if not map_path.is_file():
        print(f"Error: Degradation map not found at {map_path}")
        return False

    try:
        with open(map_path, 'r') as f:
            data = json.load(f)
        print(f"Successfully loaded degradation map: {map_path}")
    except Exception as e:
        print(f"Error: Could not read or parse {map_path}: {e}")
        return False

    if not isinstance(data, dict) or not data:
        print("Error: Degradation map is empty or not a JSON object (dictionary).")
        return False
        
    map_image_ids = set(data.keys())
    print(f"Found {len(map_image_ids)} unique image IDs in the map.")

    structural_errors = 0
    file_not_found_errors = 0
    completeness_errors = 0
    is_structurally_valid = True
    original_image_dir = None 

    print("\nValidating structure, file paths, and completeness for each map entry...")
    for image_id, entry in data.items():
        valid_entry = True
        if not isinstance(entry, dict):
            print(f"Error [ID: {image_id}]: Entry is not a dictionary.")
            structural_errors += 1
            valid_entry = False
            continue # Skip file checks if structure is wrong

        # Check ground truth RLE structure
        if "ground_truth_rle" not in entry:
            print(f"Error [ID: {image_id}]: Missing 'ground_truth_rle'.")
            structural_errors += 1
            valid_entry = False
        elif not isinstance(entry["ground_truth_rle"], dict) or \
             'size' not in entry["ground_truth_rle"] or \
             'counts' not in entry["ground_truth_rle"]:
            print(f"Error [ID: {image_id}]: Invalid 'ground_truth_rle' format (must be dict with 'size' and 'counts').")
            structural_errors += 1
            valid_entry = False

        # Check versions structure
        if "versions" not in entry:
            print(f"Error [ID: {image_id}]: Missing 'versions' dictionary.")
            structural_errors += 1
            valid_entry = False
        elif not isinstance(entry.get("versions"), dict):
             print(f"Error [ID: {image_id}]: 'versions' is not a dictionary.")
             structural_errors += 1
             valid_entry = False
        else:
            # Check original version
            if "original" not in entry["versions"]:
                 print(f"Error [ID: {image_id}]: Missing 'original' entry in 'versions'.")
                 structural_errors += 1
                 valid_entry = False
            elif not isinstance(entry["versions"].get("original"), dict) or \
                 "filepath" not in entry["versions"].get("original", {}):
                 print(f"Error [ID: {image_id}]: Invalid 'original' version format (must be dict with 'filepath').")
                 structural_errors += 1
                 valid_entry = False
            else:
                # Check original file path
                original_filepath_rel = entry["versions"]["original"]["filepath"]
                original_filepath_abs = base_data_path / "data" / original_filepath_rel 
                if not original_image_dir:
                    original_image_dir = (base_data_path / "data" / Path(original_filepath_rel)).parent
                if not original_filepath_abs.is_file():
                     print(f"Error [ID: {image_id}]: Original image file not found: {original_filepath_abs}")
                     file_not_found_errors += 1
                     valid_entry = False
                     
            # Check degraded versions 
            for deg_type, deg_entry in entry["versions"].items():
                if deg_type == "original": continue # Already checked
                if not isinstance(deg_entry, dict):
                    print(f"Error [ID: {image_id}, Type: {deg_type}]: Entry in 'versions' is not a dictionary.")
                    structural_errors += 1
                    valid_entry = False
                    continue
                    
                # Check each degradation level within the type
                for level, level_data in deg_entry.items():
                    if not isinstance(level_data, dict) or "filepath" not in level_data:
                         print(f"Error [ID: {image_id}, Type: {deg_type}, Level: {level}]: Invalid format (must be dict with 'filepath').")
                         structural_errors += 1
                         valid_entry = False
                    else:
                        filepath_rel = level_data["filepath"]
                        filepath_abs = base_data_path / "data" / filepath_rel
                        if not filepath_abs.is_file():
                             print(f"Error [ID: {image_id}, Type: {deg_type}, Level: {level}]: Degraded file not found: {filepath_abs}")
                             file_not_found_errors += 1
                             valid_entry = False
                             
        # --- Start: New Completeness Check ---
        actual_degradation_types = set(entry["versions"].keys()) - {"original"}

        # Check degradation types
        missing_types = EXPECTED_DEGRADATION_TYPES - actual_degradation_types
        unexpected_types = actual_degradation_types - EXPECTED_DEGRADATION_TYPES

        if missing_types:
            print(f"Warning [ID: {image_id}]: Missing expected degradation types: {sorted(list(missing_types))}")
            completeness_errors += len(missing_types)
            valid_entry = False 
        if unexpected_types:
            # Log unexpected types 
            print(f"Info [ID: {image_id}]: Found unexpected degradation types (not in PARAM_GRID): {sorted(list(unexpected_types))}")

        # Check levels/parameters for each *expected* type found
        for deg_type in EXPECTED_DEGRADATION_TYPES.intersection(actual_degradation_types):
            if not isinstance(entry["versions"].get(deg_type), dict):
                continue

            param_name = list(PARAM_GRID[deg_type].keys())[0] # e.g., 'kernel_size'
            
            expected_levels = set(map(str, PARAM_GRID[deg_type][param_name]))
            actual_levels = set(entry["versions"][deg_type].keys())

            missing_levels = expected_levels - actual_levels
            unexpected_levels = actual_levels - expected_levels

            if missing_levels:
                print(f"Warning [ID: {image_id}, Type: {deg_type}]: Missing expected levels/params ('{param_name}'): {sorted(list(missing_levels))}")
                completeness_errors += len(missing_levels)
                valid_entry = False
            if unexpected_levels:
                print(f"Info [ID: {image_id}, Type: {deg_type}]: Found unexpected levels/params ('{param_name}'): {sorted(list(unexpected_levels))}")

        # --- End: New Completeness Check ---
        
        if not valid_entry:
            is_structurally_valid = False
            
    if structural_errors == 0 and file_not_found_errors == 0 and completeness_errors == 0:
        print("\nAll map entries are structurally valid, complete according to PARAM_GRID, and all referenced files exist.")
    else:
        print(f"\nValidation Summary: Structural Errors = {structural_errors}, Files Not Found = {file_not_found_errors}, Completeness Errors = {completeness_errors}")
        
    # --- Cross-validation with Original Images Directory --- 
    print("\nPerforming cross-validation with original images directory...")
    disk_image_ids = set()
    if original_image_dir and original_image_dir.is_dir():
        print(f"Scanning directory for original images: {original_image_dir}")
        image_extensions = {'.jpg', '.jpeg', '.png'} # Add more if needed
        found_files = 0
        for item in original_image_dir.iterdir():
            if item.is_file() and item.suffix.lower() in image_extensions:
                disk_image_ids.add(item.stem) # item.stem gets filename without extension
                found_files += 1
        print(f"Found {found_files} image files in the directory.")
        print(f"Found {len(disk_image_ids)} unique image IDs (basenames) on disk.")
        
        # Compare sets
        map_ids_not_on_disk = map_image_ids - disk_image_ids
        disk_ids_not_in_map = disk_image_ids - map_image_ids
        
        if not map_ids_not_on_disk and not disk_ids_not_in_map:
            if len(map_image_ids) == len(disk_image_ids):
                 print("Cross-validation PASSED: Map IDs perfectly match image files found on disk.")
            else:
                 # Should not happen if sets match, but as a safeguard
                 print("Warning: Set comparison indicates match, but counts differ slightly. Check for duplicate IDs or file naming issues.")
                 print(f"Map IDs: {len(map_image_ids)}, Disk IDs: {len(disk_image_ids)}")
                 is_structurally_valid = False 
        else:
            is_structurally_valid = False # Mismatch found
            print("Cross-validation FAILED: Discrepancies found between map IDs and disk images.")
            if map_ids_not_on_disk:
                print(f"  - IDs in map but MISSING image file on disk ({len(map_ids_not_on_disk)}): {sorted(list(map_ids_not_on_disk))}")
            if disk_ids_not_in_map:
                print(f"  - Image files on disk but MISSING ID in map ({len(disk_ids_not_in_map)}): {sorted(list(disk_ids_not_in_map))}")
                
    elif original_image_dir:
        print(f"Error: Determined original image directory does not exist: {original_image_dir}")
        print("Skipping disk cross-validation.")
        is_structurally_valid = False 
    else:
        print("Warning: Could not determine original image directory from map entries (map might be empty or missing original versions).")
        print("Skipping disk cross-validation.")

    overall_valid = structural_errors == 0 and file_not_found_errors == 0 and completeness_errors == 0 and is_structurally_valid
    return overall_valid


def visualize_sample(image_id: str, map_path: Path, base_data_path: Path, save_path: Path = None):
    """Loads and visualizes the original image and its GT mask for a given ID.
    
    Args:
        image_id: The ID of the image to visualize.
        map_path: Path to the degradation_map.json file.
        base_data_path: Path to the base project directory.
        save_path: Optional path to save the plot image. If None, displays interactively.
    """
    if not all([Image, plt, maskUtils]):
        print("Missing required visualization dependencies (Pillow, Matplotlib, pycocotools).")
        return False

    if not map_path.is_file():
        print(f"Error: Degradation map not found at {map_path}")
        return False
        
    try:
        with open(map_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Could not read or parse {map_path}: {e}")
        return False

    if image_id not in data:
        print(f"Error: Image ID '{image_id}' not found in the degradation map.")
        return False
        
    entry = data[image_id]

    # Get original image path
    try:
        original_filepath_rel = entry["versions"]["original"]["filepath"]
        original_filepath_abs = base_data_path / "data" / original_filepath_rel 
    except KeyError:
        print(f"Error: Could not find 'original' version filepath for ID '{image_id}'.")
        return False

    if not original_filepath_abs.is_file():
        print(f"Error: Original image file not found: {original_filepath_abs}")
        return False
        
    # Get RLE mask
    try:
        rle = entry["ground_truth_rle"]
        if not isinstance(rle, dict) or 'size' not in rle or 'counts' not in rle:
            raise ValueError("Invalid RLE format")
    except (KeyError, ValueError) as e:
        print(f"Error: Could not find or parse valid 'ground_truth_rle' for ID '{image_id}': {e}")
        return False
        
    # Load image
    try:
        img = Image.open(original_filepath_abs).convert('RGB')
    except Exception as e:
        print(f"Error loading image {original_filepath_abs}: {e}")
        return False

    # Decode RLE mask
    try:
        mask = maskUtils.decode(rle)
        # Ensure mask is boolean for visualization overlay
        mask = mask.astype(bool)
        # Compare mask shape (H, W) tuple with RLE size [H, W] list (converted to tuple)
        if mask.shape[:2] != tuple(rle['size']): 
             print(f"Warning [ID: {image_id}]: Decoded mask shape {mask.shape[:2]} differs from RLE size {rle['size']}. Check RLE encoding.")
             
    except Exception as e:
        print(f"Error decoding RLE mask for ID '{image_id}': {e}")
        return False
        
    # --- Visualization ---
    try:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        ax.imshow(img)
        
        # Create slightly transparent overlay for the mask
        
        colored_mask = np.zeros((*mask.shape[:2], 4)) # RGBA, ensure shape matches image dims
        colormap = plt.colormaps['autumn']
        colored_mask[mask, :3] = colormap(0.5)[:3] # Example: Red color (using index 0.5 from the colormap)
        colored_mask[mask, 3] = 0.4 
        
        ax.imshow(colored_mask)
        ax.set_title(f"Image ID: {image_id} with Ground Truth Mask")
        ax.axis('off')
        plt.tight_layout()

        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight', pad_inches=0.1)
            print(f"Saved plot to {save_path}")
            plt.close(fig) 
        else:
            plt.show() # Display interactively if not saving
        return True
        
    except Exception as e:
        print(f"Error during plotting/saving for ID '{image_id}': {e}")
        if 'fig' in locals() and plt.fignum_exists(fig.number):
             plt.close(fig) 
        return False


def parse_image_ids(id_specs: list[str], all_available_ids: set[str]) -> list[str]:
    """
    Parses a list of ID specifications (single numbers or ranges like '10-20')
    into a sorted list of valid, unique image IDs.

    Args:
        id_specs: List of strings from the command line (e.g., ['10', '15-17', '5']).
                If None or empty, returns all available IDs.
        all_available_ids: A set of all valid image IDs from the map file.

    Returns:
        A sorted list of unique, valid image IDs.
    """
    selected_ids = set()
    if not id_specs: 
        return sorted(list(all_available_ids))

    for spec in id_specs:
        spec = spec.strip()
        if not spec:
            continue
        if '-' in spec:
            try:
                start_str, end_str = spec.split('-', 1)
                start = int(start_str)
                end = int(end_str)
                if start > end:
                    print(f"Warning: Invalid range '{spec}', start > end. Skipping.")
                    continue
                if start < 0 or end < 0:
                    print(f"Warning: Invalid range '{spec}', negative numbers not allowed. Skipping.")
                    continue
                for i in range(start, end + 1):
                    id_str = str(i)
                    if id_str in all_available_ids:
                        selected_ids.add(id_str)
                    else:
                         pass 
            except ValueError:
                print(f"Warning: Invalid range format '{spec}'. Skipping.")
        else:
            try:
                # Check if it's a valid non-negative integer ID string
                num = int(spec)
                if num < 0:
                    print(f"Warning: Invalid ID format '{spec}', negative numbers not allowed. Skipping.")
                    continue
                id_str = str(num)
                if id_str in all_available_ids:
                    selected_ids.add(id_str)
                else:
                    print(f"Warning: Specified ID '{id_str}' not found in map. Skipping.")
            except ValueError:
                 print(f"Warning: Invalid ID format '{spec}'. Skipping.")

    final_ids = sorted(list(selected_ids))
    if id_specs and not final_ids:
         print("Warning: No valid image IDs selected after filtering.")
    elif not final_ids:
         print("Warning: No image IDs found to process.")
         
    return final_ids

def plot_all_gt_masks(map_path: Path, base_data_path: Path, output_dir: Path, id_specs: list[str] = None, clean_output: bool = False):
    """Generates and saves plots for specified or all ground truth masks."""
    if not map_path.is_file():
        print(f"Error: Degradation map not found at {map_path}")
        return

    try:
        with open(map_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error: Could not read or parse {map_path}: {e}")
        return
        
    all_ids_in_map = set(data.keys())
    ids_to_plot = parse_image_ids(id_specs, all_ids_in_map)
    
    if not ids_to_plot:
        print("No images selected for plotting.")
        return
        
    # Clean output directory if requested and it exists
    if clean_output:
        if output_dir.exists():
            print(f"Warning: Cleaning existing output directory: {output_dir}")
            try:
                shutil.rmtree(output_dir)
                print(f"Successfully removed {output_dir}")
            except OSError as e:
                print(f"Error: Could not remove directory {output_dir}: {e}")
                print("Proceeding without cleaning.")
        else:
            print(f"Output directory {output_dir} does not exist, no cleaning needed.")
            
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
         print(f"Error: Could not create output directory {output_dir}: {e}")
         print("Cannot save plots.")
         return
         
    print(f"Saving GT mask plots to: {output_dir}")
    print(f"Attempting to plot {len(ids_to_plot)} image(s)...")
    
    success_count = 0
    fail_count = 0
    total_items = len(ids_to_plot)

    for i, image_id in enumerate(ids_to_plot):
        # print(f"Processing {image_id} ({i+1}/{total_items})...") # Can be verbose
        save_path = output_dir / f"gt_mask_{image_id}.png"
        if visualize_sample(image_id, map_path, base_data_path, save_path=save_path):
            success_count += 1
        else:
            fail_count += 1
            # visualize_sample already prints error details
            # print(f"Failed to generate plot for ID: {image_id}") 

    print("-" * 20)
    print(f"Batch plotting complete. Plotted: {success_count}, Failed: {fail_count}")

def main():
    parser = argparse.ArgumentParser(
        description="Data utilities for image dataset and degradation map.",
        formatter_class=argparse.RawTextHelpFormatter 
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=".", # Assume script is run from SAM2_analysis root
        help="Path to the base project directory (containing the 'data' folder). Defaults to current directory.",
    )
    parser.add_argument(
        "--map_file",
        type=str,
        default="data/degradation_map.json",
        help="Path to the degradation map JSON file relative to data_dir.",
    )
    parser.add_argument(
        "--action",
        type=str,
        choices=["count", "validate", "visualize", "plot_all", "all"],
        required=True, 
        help="Action to perform:\n"
             "  count      : Count images referenced in the map.\n"
             "  validate   : Validate map structure and file existence.\n"
             "  visualize  : Show plot for specific image(s) (use --ids).\n"
             "  plot_all   : Save plots for specific/all image(s) (use --ids).\n"
             "  all        : Perform 'count' and 'validate'."
    )
    parser.add_argument(
        "--ids",
        type=str,
        nargs='*',
        default=[], 
        help="Image ID(s) to process for 'visualize' or 'plot_all' actions.\n"
             "Examples:\n"
             "  --ids 55          (single ID)\n"
             "  --ids 10 25 99    (multiple IDs)\n"
             "  --ids 10-20       (range, inclusive)\n"
             "  --ids 5 10-15 99  (mixed)\n"
             "If omitted for 'plot_all', all images are processed.\n"
             "If omitted for 'visualize', ID '1' is used."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/gt_plots",
        help="Directory to save plots for the 'plot_all' action, relative to data_dir.",
    )
    parser.add_argument(
        "--clean",
        action='store_true', 
        help="If specified with --action plot_all, cleans (deletes) the output directory before plotting."
    )

    args = parser.parse_args()

    # Resolve paths
    base_path = Path(args.data_dir).resolve() 
    map_path = base_path / args.map_file 
    output_dir_abs = base_path / args.output_dir

    print(f"Using base project directory: {base_path}")
    print(f"Using degradation map: {map_path}")
    if args.action == 'plot_all':
        print(f"Plot output directory: {output_dir_abs}")

    # --- Execute Actions ---
    if args.action in ["count", "all"]:
        print("\n--- Counting Images ---")
        count_images(base_path)

    if args.action in ["validate", "all"]:
        print("\n--- Validating Degradation Map ---")
        validate_degradation_map(map_path, base_path)
        
    if args.action == "visualize":
        # Get all available IDs for validation
        try:
            with open(map_path, 'r') as f:
                all_ids_in_map = set(json.load(f).keys())
        except Exception as e:
            print(f"Error reading map file for ID validation: {e}")
            all_ids_in_map = set()
            
        ids_to_visualize = parse_image_ids(args.ids, all_ids_in_map)
        
        if not ids_to_visualize:
            # If user specified IDs but none were valid, or if they specified none
            if args.ids: 
                print("No valid IDs provided for visualization.")
            else:
                # Default to ID '1' if no IDs specified and '1' is valid
                if '1' in all_ids_in_map:
                    print("No IDs specified, defaulting to visualize ID '1'.")
                    ids_to_visualize = ['1']
                else:
                    print("No IDs specified, and default ID '1' not found in map.")
        
        if ids_to_visualize:
             print(f"\n--- Visualizing Sample(s): {', '.join(ids_to_visualize)} ---")
             for vis_id in ids_to_visualize:
                 print(f"Displaying plot for ID: {vis_id}")
                 # Call visualize without save_path to display interactively
                 visualize_sample(vis_id, map_path, base_path, save_path=None) 

    if args.action == "plot_all":
        print(f"\n--- Plotting Ground Truth Masks ---")
        # Pass the clean flag to the function
        plot_all_gt_masks(map_path, base_path, output_dir_abs, args.ids, args.clean)

if __name__ == "__main__":
    main()
