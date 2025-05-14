import numpy as np
import cv2 

def calculate_miou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU) between a predicted binary mask
    and a ground truth binary mask. Robust to None inputs and shape mismatches.

    Args:
        pred_mask: Binary mask from model prediction (HxW, dtype=bool or uint8). Can be None.
        gt_mask: Binary ground truth mask (HxW, dtype=bool or uint8). Can be None.

    Returns:
        IoU score (float), or 0.0 if inputs are invalid/mismatched.
    """
    if pred_mask is None or gt_mask is None:
        # print("Warning: calculate_miou received None mask.")
        return 0.0

    # Ensure boolean
    pred_mask = pred_mask.astype(bool)
    gt_mask = gt_mask.astype(bool)

    if pred_mask.shape != gt_mask.shape:
        print(f"Warning: Shape mismatch in calculate_miou: Pred {pred_mask.shape}, GT {gt_mask.shape}. Returning 0.")
        return 0.0 # Shape mismatch is a critical problem

    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()

    if union == 0:
        # Both masks are empty. IoU is 1 if GT is also empty (implies intersection is 0), 0 otherwise
        return 1.0 if intersection == 0 else 0.0

    iou = float(intersection) / float(union)
    return iou


def calculate_boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 2) -> float:
    """
    Calculate boundary F1 score using OpenCV Canny and Distance Transform.
    Robust to None inputs and shape mismatches.

    Args:
        pred_mask: Binary prediction mask (HxW, dtype=bool or uint8). Can be None.
        gt_mask: Binary ground truth mask (HxW, dtype=bool or uint8). Can be None.
        tolerance_px: Pixel tolerance for boundary matching (default: 2).

    Returns:
        Boundary F1 score (float), or 0.0 if inputs are invalid.
    """
    if pred_mask is None or gt_mask is None:
        # print("Warning: calculate_boundary_f1 received None mask.")
        return 0.0

    # Added shape check for robustness
    if pred_mask.shape != gt_mask.shape:
        print(f"Warning: Shape mismatch in calculate_boundary_f1: Pred {pred_mask.shape}, GT {gt_mask.shape}. Returning 0.")
        return 0.0

    # Ensure uint8 for OpenCV functions
    pred_mask_u8 = pred_mask.astype(np.uint8) * 255
    gt_mask_u8 = gt_mask.astype(np.uint8) * 255

    # Detect boundaries using Canny edge detection
    pred_boundary = cv2.Canny(pred_mask_u8, 100, 200) # Thresholds might need tuning
    gt_boundary = cv2.Canny(gt_mask_u8, 100, 200)

    # Count boundary pixels
    pred_sum = np.count_nonzero(pred_boundary)
    gt_sum = np.count_nonzero(gt_boundary)

    if gt_sum == 0 and pred_sum == 0:
        return 1.0  # Both empty, perfect match
    if gt_sum == 0 or pred_sum == 0:
        return 0.0  # One empty, the other not, zero score

    # Create distance transforms - distance to the nearest boundary point
    # Invert boundary map so distance is 0 on the boundary
    gt_dist_map = cv2.distanceTransform(cv2.bitwise_not(gt_boundary), cv2.DIST_L2, 3)
    pred_boundary_pixels = pred_boundary > 0
    pred_matched_count = np.sum(gt_dist_map[pred_boundary_pixels] <= tolerance_px)

    # Calculate distance from Pred boundary to GT boundary pixels
    pred_dist_map = cv2.distanceTransform(1 - (pred_boundary // 255), cv2.DIST_L2, 3)
    gt_boundary_pixels = np.where(gt_boundary == 255)
    gt_matched_count = np.sum(pred_dist_map[gt_boundary_pixels] <= tolerance_px)

    # Calculate Precision, Recall, and F1 Score
    precision = float(pred_matched_count) / float(pred_sum) if pred_sum > 0 else 0
    recall = float(gt_matched_count) / float(gt_sum) if gt_sum > 0 else 0

    if precision + recall == 0:
        return 0.0

    f1_score = 2.0 * (precision * recall) / (precision + recall)

    return f1_score


def _create_test_circle_mask(shape: tuple, center: tuple, radius: int) -> np.ndarray:
    """Helper to create a binary mask with a filled circle."""
    mask = np.zeros(shape, dtype=np.uint8)
    Y, X = np.ogrid[:shape[0], :shape[1]]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)
    mask[dist_from_center <= radius] = 1
    return mask


# --- Simple Self-Tests (runnable via python metrics.py) --- 
if __name__ == "__main__":
    print("Running simple metric self-tests...")

    # --- Test Data ---
    mask_full = np.ones((10, 10), dtype=np.uint8)
    mask_empty = np.zeros((10, 10), dtype=np.uint8)
    mask_half = np.zeros((10, 10), dtype=np.uint8)
    mask_half[5:, :] = 1

    # --- mIoU Tests ---
    print("\nTesting mIoU...")
    try:
        iou_perfect = calculate_miou(mask_full, mask_full)
        print(f"  Perfect match (expect 1.0): {iou_perfect}")
        assert iou_perfect == 1.0

        iou_empty = calculate_miou(mask_empty, mask_empty)
        print(f"  Empty match (expect 1.0): {iou_empty}")
        assert iou_empty == 1.0

        iou_half = calculate_miou(mask_half, mask_full)
        print(f"  Half match (expect 0.5): {iou_half}")
        assert iou_half == 0.5

        iou_none1 = calculate_miou(None, mask_full)
        iou_none2 = calculate_miou(mask_full, None)
        print(f"  None inputs (expect 0.0): {iou_none1}, {iou_none2}")
        assert iou_none1 == 0.0 and iou_none2 == 0.0
        print("  mIoU tests PASSED")
    except AssertionError as e:
        print(f"  mIoU test FAILED: {e}")
    except Exception as e:
        print(f"  mIoU test FAILED with unexpected error: {e}")

    # --- BF1 Tests ---
    print("\nTesting Boundary F1...")
    try:
        bf1_perfect = calculate_boundary_f1(mask_full, mask_full, tolerance_px=1)
        print(f"  Perfect match (expect 1.0): {bf1_perfect}")
        assert bf1_perfect == 1.0

        bf1_empty = calculate_boundary_f1(mask_empty, mask_empty, tolerance_px=1)
        print(f"  Empty match (expect 1.0): {bf1_empty}")
        assert bf1_empty == 1.0

        bf1_none1 = calculate_boundary_f1(None, mask_full)
        bf1_none2 = calculate_boundary_f1(mask_full, None)
        print(f"  None inputs (expect 0.0): {bf1_none1}, {bf1_none2}")
        assert bf1_none1 == 0.0 and bf1_none2 == 0.0
        print("  BF1 tests PASSED")
    except AssertionError as e:
        print(f"  BF1 test FAILED: {e}")
    except Exception as e:
        print(f"  BF1 test FAILED with unexpected error: {e}")

    print("\n--- Running Basic Self-Tests ---")
    try:
        mask_full = np.ones((10, 10), dtype=np.uint8)
        mask_empty = np.zeros((10, 10), dtype=np.uint8)
        mask_half = np.zeros((10, 10), dtype=np.uint8)
        mask_half[5:, :] = 1

        print("\nTesting mIoU (Basic)...")
        assert calculate_miou(mask_full, mask_full) == 1.0, "Basic mIoU perfect fail"
        assert calculate_miou(mask_empty, mask_empty) == 1.0, "Basic mIoU empty fail"
        assert calculate_miou(mask_half, mask_full) == 0.5, "Basic mIoU half fail"
        assert calculate_miou(None, mask_full) == 0.0, "Basic mIoU none fail 1"
        assert calculate_miou(mask_full, None) == 0.0, "Basic mIoU none fail 2"
        print("  Basic mIoU tests PASSED")

        print("\nTesting Boundary F1 (Basic)...")
        assert calculate_boundary_f1(mask_full, mask_full, tolerance_px=1) == 1.0, "Basic BF1 perfect fail"
        assert calculate_boundary_f1(mask_empty, mask_empty, tolerance_px=1) == 1.0, "Basic BF1 empty fail"
        assert calculate_boundary_f1(None, mask_full) == 0.0, "Basic BF1 none fail 1"
        assert calculate_boundary_f1(mask_full, None) == 0.0, "Basic BF1 none fail 2"
        print("  Basic BF1 tests PASSED")
        print("\nBasic self-tests PASSED.")
    except AssertionError as e:
        print(f"\nBasic self-test FAILED: {e}")
    except Exception as e:
        print(f"\nBasic self-test FAILED with unexpected error: {e}")

    print("\nSelf-tests completed.")
