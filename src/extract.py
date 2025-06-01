import mediapipe as mp

from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from src.utils import get_logger

logger = get_logger(__name__)


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
        logger.error(f"Image file not found: {image_path}")
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    logger.info(f"Loading image from {image_path}")
    image = mp.Image.create_from_file(str(image_path))

    logger.debug("Initializing MediaPipe PoseLandmarker...")
    base_options = python.BaseOptions(model_asset_path=str(model_path))
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        output_segmentation_masks=True,
        running_mode=vision.RunningMode.IMAGE,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    logger.debug("Running pose detection...")
    result = detector.detect(image)

    if len(result.pose_landmarks) == 0:
        logger.warning(f"No pose detected in image: {image_path}")
        raise ValueError(f"No pose detected in image: {image_path}")

    logger.info(f"Pose landmarks detected for 1 person in {image_path}")
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

    if result.segmentation_masks:
        logger.debug("Segmentation mask detected and included in output.")
        mask = result.segmentation_masks[0].numpy_view()
        output["segmentation_mask"] = mask

    logger.debug(f"Returning {len(landmarks)} 2D landmarks and {len(world)} world landmarks.")
    return output
