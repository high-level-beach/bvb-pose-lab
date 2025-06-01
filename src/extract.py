import mediapipe as mp

from pathlib import Path
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
    