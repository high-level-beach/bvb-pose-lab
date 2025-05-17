import cv2
import numpy as np
from typing import Optional

from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def get_landmarks(image_path: Path, model_path: Path) -> dict:
    """
    Uses MediaPipe's PoseLandmarker (Tasks API) to extract pose landmarks and optional segmentation mask.

    Parameters
    ----------
    image_path : Path
        Path to the static image (e.g., .jpg or .png).
    model_path : Path
        Path to the downloaded `pose_landmarker.task` model file.

    Returns
    -------
    output : dict
        Contains coordinates of the landmarks (person and world) and optionally the segmentation mask.

    Raises
    ------
    FileNotFoundError
        If image or model path doesn't exist.
    ValueError
        If detection fails.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Load image
    image = mp.Image.create_from_file(str(image_path))

    # Setup PoseLandmarker
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    # Run detection
    result = detector.detect(image)

    if len(result.pose_landmarks) == 0:
        raise ValueError(f"No pose detected in image: {image_path}")

    # Only supporting 1 person (index 0)
    lm = result.pose_landmarks[0]
    world_lm = result.pose_world_landmarks[0]

    landmarks = [
        {"x": l.x, "y": l.y, "z": l.z, "visibility": l.visibility}
        for l in lm
    ]
    world = [
        {"x": l.x, "y": l.y, "z": l.z}
        for l in world_lm
    ]

    output = {
        "landmarks": landmarks,
        "world_landmarks": world,
    }

    # Grab the segmentation mask
    if result.segmentation_masks:
        mask = result.segmentation_masks[0].numpy_view()
        output["segmentation_mask"] = mask  # You can visualize or post-process this later

    return output

def visualize_pose(
    image_input: Path | np.ndarray,
    landmarks: Optional[list[dict]] = None,
    segmentation_mask: Optional[np.ndarray] = None,
    show: bool = False,
    alpha_mask: float = 0.6,
    landmark_color: tuple = (0, 255, 0),
    connection_color: tuple = (255, 0, 0),
) -> np.ndarray:
    """
    Overlays pose landmarks and/or segmentation mask on an image.

    Parameters
    ----------
    image_input : Path or np.ndarray
        Path to image or already-loaded BGR image array.
    landmarks : list of dict, optional
        List of 33 pose landmarks with keys x, y, (optional z, visibility).
        Normalized (0-1) coordinates.
    segmentation_mask : np.ndarray, optional
        Float32 mask same shape as image (HxW), values in [0, 1].
    show : bool, default False
        If True, opens a cv2 window to preview.
    alpha_mask : float, default 0.6
        Transparency of mask overlay.
    landmark_color : tuple, default (0, 255, 0) - Green
        BGR color for landmarks.
    connection_color : tuple, default (255, 0, 0) - Red
        BGR color for landmark lines.

    Returns
    -------
    image : np.ndarray
        Image with overlays (BGR).
    """
    # Load image
    if isinstance(image_input, Path):
        image = cv2.imread(str(image_input))
    else:
        image = image_input.copy()

    height, width = image.shape[:2]

    # Overlay segmentation mask
    if segmentation_mask is not None:
        if segmentation_mask.shape != (height, width):
            raise ValueError("Segmentation mask size does not match image size.")
        colored_mask = cv2.applyColorMap((segmentation_mask * 255).astype(np.uint8), cv2.COLORMAP_JET)
        image = cv2.addWeighted(colored_mask, alpha_mask, image, 1 - alpha_mask, 0)

    # Draw landmarks
    if landmarks is not None:
        for lm in landmarks:
            x_px = int(lm["x"] * width)
            y_px = int(lm["y"] * height)
            cv2.circle(image, (x_px, y_px), 4, landmark_color, -1)

        # Draw connections (using simplified MediaPipe style)
        mp_connections = [
            # Face & head
            (0, 1), (1, 2), (2, 3),  # Nose to left eye
            (0, 4), (4, 5), (5, 6),  # Nose to right eye
            (2, 7), (5, 8),  # Eyes to ears

            # Arms
            (11, 13), (13, 15),  # Left shoulder → elbow → wrist
            (15, 17), (15, 19), (15, 21),  # Left wrist to thumb, pinky, index
            (12, 14), (14, 16),  # Right shoulder → elbow → wrist
            (16, 18), (16, 20), (16, 22),  # Right wrist to thumb, pinky, index

            # Torso
            (11, 12),  # Shoulders
            (23, 24),  # Hips
            (11, 23), (12, 24),  # Shoulder to hip

            # Legs
            (23, 25), (25, 27),  # Left hip → knee → ankle
            (27, 29), (29, 31),  # Left ankle → heel → toe
            (24, 26), (26, 28),  # Right hip → knee → ankle
            (28, 30), (30, 32),  # Right ankle → heel → toe
        ]
        for i, j in mp_connections:
            if i < len(landmarks) and j < len(landmarks):
                pt1 = (int(landmarks[i]["x"] * width), int(landmarks[i]["y"] * height))
                pt2 = (int(landmarks[j]["x"] * width), int(landmarks[j]["y"] * height))
                cv2.line(image, pt1, pt2, connection_color, 2)

        # Nose to midpoint of shoulders (pseudo-neck)
        if landmarks and len(landmarks) > 12:
            nose = (int(landmarks[0]["x"] * width), int(landmarks[0]["y"] * height))
            left_shoulder = (int(landmarks[11]["x"] * width), int(landmarks[11]["y"] * height))
            right_shoulder = (int(landmarks[12]["x"] * width), int(landmarks[12]["y"] * height))
            neck = (
                int((left_shoulder[0] + right_shoulder[0]) / 2),
                int((left_shoulder[1] + right_shoulder[1]) / 2),
            )
            cv2.line(image, nose, neck, connection_color, 2)

    if show:
        cv2.imshow("Pose Overlay", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image