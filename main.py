import argparse
import cv2

from src.extract import get_landmarks, visualize_pose
from src.config import PATH_TO_ROOT

def main(args) -> None:
    model_path = PATH_TO_ROOT / "models/pose_landmarker_full.task"

    if args.extract:
        image_path = PATH_TO_ROOT / args.image_path
        output = get_landmarks(
            image_path=image_path,
            model_path=model_path,
        )
    
        overlay = visualize_pose(
            image_input=image_path,
            landmarks=output["landmarks"],
            segmentation_mask=output.get("segmentation_mask"),
            show=True,
        )
        output_path = PATH_TO_ROOT / "data/processed/test-image-pose.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), overlay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--image-path", type=str, default="data/raw/test-image.jpg")

    main(parser.parse_args())