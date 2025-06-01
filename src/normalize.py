import numpy as np
from typing import List
from src.utils import get_logger

logger = get_logger(__name__)


def normalize_landmarks(landmarks: List[dict], rotate: bool = True) -> List[dict]:
    """
    Normalize pose landmarks to:
    - Center on the midpoint of the hips
    - Scale based on torso length (hip-to-shoulder distance)
    - Optionally rotate to make torso vertical

    Parameters
    ----------
    landmarks : List[dict]
        List of dicts with x, y, z fields (normalized coordinates)
    rotate : bool
        If True, rotates the pose so the shoulder-hip vector is vertical

    Returns
    -------
    norm_landmarks : List[dict]
        Normalized landmark list (same format, centered/scaled/rotated)
    """
    if len(landmarks) < 25:
        logger.error("Insufficient landmarks for torso normalization.")
        raise ValueError("Expected at least 25 landmarks for torso normalization.")

    # Center around midpoint between left and right hips
    left_hip = np.array([landmarks[23]["x"], landmarks[23]["y"]])
    right_hip = np.array([landmarks[24]["x"], landmarks[24]["y"]])
    mid_hip = (left_hip + right_hip) / 2
    logger.debug(f"Mid-hip center: {mid_hip}")

    # Calculate torso vector from mid-hip to midpoint of shoulders
    left_shoulder = np.array([landmarks[11]["x"], landmarks[11]["y"]])
    right_shoulder = np.array([landmarks[12]["x"], landmarks[12]["y"]])
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    torso_vec = mid_shoulder - mid_hip
    torso_length = np.linalg.norm(torso_vec)
    logger.debug(f"Torso vector: {torso_vec}, length: {torso_length:.4f}")

    if torso_length < 1e-6:
        logger.error("Torso length is too small to normalize reliably.")
        raise ValueError("Torso length is too small to normalize reliably.")

    # Compute rotation matrix to align torso with the vertical axis if needed
    if rotate:
        target_vec = np.array([0, -1])  # y-axis up
        angle = np.arctan2(torso_vec[1], torso_vec[0]) - np.arctan2(target_vec[1], target_vec[0])
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        logger.debug(f"Rotation angle (radians): {angle:.4f}")

    # Apply centering, scaling, and optional rotation to each landmark
    norm_landmarks = []
    for idx, lm in enumerate(landmarks):
        pt = np.array([lm["x"], lm["y"]])
        pt_centered = pt - mid_hip
        pt_scaled = pt_centered / torso_length
        pt_rotated = rotation_matrix @ pt_scaled if rotate else pt_scaled

        norm_landmarks.append({
            "x": float(pt_rotated[0]),
            "y": float(pt_rotated[1]),
            "z": lm["z"] / torso_length,  # Scale Z dimension as well
            "visibility": lm.get("visibility", 1.0),
        })

    logger.info(f"Normalized {len(norm_landmarks)} landmarks with rotate={rotate}")
    return norm_landmarks
