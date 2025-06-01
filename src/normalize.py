import numpy as np
from typing import List


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
        raise ValueError("Expected at least 25 landmarks for torso normalization.")

    # Step 1: center around mid-hip
    left_hip = np.array([landmarks[23]["x"], landmarks[23]["y"]])
    right_hip = np.array([landmarks[24]["x"], landmarks[24]["y"]])
    mid_hip = (left_hip + right_hip) / 2

    # Step 2: scale by torso length (mid-hip to mid-shoulder)
    left_shoulder = np.array([landmarks[11]["x"], landmarks[11]["y"]])
    right_shoulder = np.array([landmarks[12]["x"], landmarks[12]["y"]])
    mid_shoulder = (left_shoulder + right_shoulder) / 2
    torso_vec = mid_shoulder - mid_hip
    torso_length = np.linalg.norm(torso_vec)

    if torso_length < 1e-6:
        raise ValueError("Torso length is too small to normalize reliably.")

    # Step 3: compute rotation if needed
    if rotate:
        # Rotate so torso_vec points straight up (aligned to y-axis)
        target_vec = np.array([0, -1])
        angle = np.arctan2(torso_vec[1], torso_vec[0]) - np.arctan2(target_vec[1], target_vec[0])
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Step 4: normalize each point
    norm_landmarks = []
    for lm in landmarks:
        pt = np.array([lm["x"], lm["y"]])
        pt_centered = pt - mid_hip
        pt_scaled = pt_centered / torso_length
        pt_rotated = rotation_matrix @ pt_scaled if rotate else pt_scaled

        norm_landmarks.append({
            "x": float(pt_rotated[0]),
            "y": float(pt_rotated[1]),
            "z": lm["z"] / torso_length,  # Z scaled too for consistency
            "visibility": lm.get("visibility", 1.0),
        })

    return norm_landmarks
