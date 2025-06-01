import argparse
import cv2

from src.extract import get_landmarks
from src.visualize import draw_overlay_on_image, draw_normalized_pose_on_blank
from src.normalize import normalize_landmarks
from src.config import PATH_TO_ROOT
from src.utils import get_logger

logger = get_logger(__name__)


def main(args) -> None:
    model_path = PATH_TO_ROOT / "models/pose_landmarker_full.task"
    filename = args.image_path.split(".")[0]

    if args.extract:
        logger.info(f"Extracting landmarks from {args.image_path}")
        image_path = PATH_TO_ROOT / args.image_path
        output = get_landmarks(
            image_path=image_path,
            model_path=model_path,
        )
        logger.info(f"Extracted {len(output['landmarks'])} landmarks from {args.image_path}")

        # Visualize pose on image
        image_overlay = draw_overlay_on_image(
            image_input=image_path,
            landmarks=output["landmarks"],
            segmentation_mask=output.get("segmentation_mask"),
            show=True,
        )
        output_path = f"{PATH_TO_ROOT}/data/processed/{filename}_overlay.jpg"
        cv2.imwrite(str(output_path), image_overlay)
        logger.info(f"Saved image overlay to {output_path}")

        # Normalize landmarks
        logger.info(f"Normalizing landmarks for {args.image_path}")
        normalized_landmarks = normalize_landmarks(output["landmarks"])
        # Visualize normalized pose on blank canvas (for debugging)
        norm_overlay = draw_normalized_pose_on_blank(
            landmarks=normalized_landmarks,
            show=True
        )
        output_path = f"{PATH_TO_ROOT}/data/processed/{filename}_norm.jpg"
        cv2.imwrite(str(output_path), norm_overlay)
        logger.info(f"Saved normalized image overlay to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--image-path", type=str, default="data/raw/test_serve_receive.jpg")

    main(parser.parse_args())