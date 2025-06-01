import cv2
import numpy as np

from PIL import Image
from pathlib import Path
from typing import Optional, List


def draw_overlay_on_image(
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
    # Load image (without applying EXIF orientation)
    if isinstance(image_input, Path):
        pil_image = Image.open(image_input)
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    else:
        image = image_input.copy()

    height, width = image.shape[:2]

    # Overlay segmentation mask
    if segmentation_mask is not None:
        if segmentation_mask.shape != (height, width):
            raise ValueError(f"Segmentation mask size {segmentation_mask.shape} does not match image size {image.shape[:2]}.")
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

def draw_normalized_pose_on_blank(
    landmarks: List[dict],
    canvas_size: tuple = (500, 500),
    scale: float = 100.0,
    landmark_color: tuple = (0, 255, 0),
    connection_color: tuple = (255, 0, 0),
    show: bool = False,
) -> np.ndarray:
    """
    Draws a normalized pose on a blank white canvas.

    This function takes a list of normalized 2D pose landmarks (where (0,0) is the center),
    maps them onto a blank image canvas, and draws the pose using circles for landmarks
    and lines for connections. This is helpful for visualizing pose landmarks after normalization
    (e.g., centered and rotated poses).

    Parameters
    ----------
    landmarks : List[dict]
        List of landmark dictionaries with at least "x" and "y" keys (normalized values).
    canvas_size : tuple, default (500, 500)
        Size of the output image in pixels (width, height).
    scale : float, default 100.0
        Multiplier to scale normalized coordinates to pixels.
    landmark_color : tuple, default (0, 255, 0)
        BGR color used to draw each landmark dot.
    connection_color : tuple, default (255, 0, 0)
        BGR color used to draw the lines between connected landmarks.
    show : bool, default False
        If True, displays the rendered image in a window.

    Returns
    -------
    image : np.ndarray
        The final pose drawing as a BGR image.
    """
    width, height = canvas_size
    image = np.ones((height, width, 3), dtype=np.uint8) * 255  # start with a white canvas

    def to_pixel_coords(x: float, y: float):
        """Convert normalized coordinates to pixel coordinates with canvas center as origin."""
        return (
            int(width / 2 + x * scale),       # move x from center to the right
            int(height / 2 - y * scale),      # flip y to make +y up
        )

    # Draw landmarks as circles
    for lm in landmarks:
        x_px, y_px = to_pixel_coords(lm["x"], lm["y"])
        cv2.circle(image, (x_px, y_px), 4, landmark_color, -1)

    # MediaPipe-style pose connections
    mp_connections = [
        (0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6),
        (2, 7), (5, 8), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21),
        (12, 14), (14, 16), (16, 18), (16, 20), (16, 22),
        (11, 12), (23, 24), (11, 23), (12, 24),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32),
    ]
    # Draw lines connecting keypoints
    for i, j in mp_connections:
        if i < len(landmarks) and j < len(landmarks):
            pt1 = to_pixel_coords(landmarks[i]["x"], landmarks[i]["y"])
            pt2 = to_pixel_coords(landmarks[j]["x"], landmarks[j]["y"])
            cv2.line(image, pt1, pt2, connection_color, 2)

    # Draw nose to midpoint of shoulders to approximate the neck line
    if len(landmarks) > 12:
        nose = to_pixel_coords(landmarks[0]["x"], landmarks[0]["y"])
        l_sh = to_pixel_coords(landmarks[11]["x"], landmarks[11]["y"])
        r_sh = to_pixel_coords(landmarks[12]["x"], landmarks[12]["y"])
        neck = ((l_sh[0] + r_sh[0]) // 2, (l_sh[1] + r_sh[1]) // 2)
        cv2.line(image, nose, neck, connection_color, 2)

    if show:
        cv2.imshow("Normalized Pose", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return image
