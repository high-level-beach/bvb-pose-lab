import numpy as np
import pytest
from unittest import mock
from src.extract import get_landmarks


@pytest.fixture
def fake_image_path(tmp_path):
    path = tmp_path / "image.jpg"
    path.write_bytes(b"fake image content")
    return path


@pytest.fixture
def fake_model_path(tmp_path):
    path = tmp_path / "model.task"
    path.write_bytes(b"fake model content")
    return path


@mock.patch("src.extract.mp.Image.create_from_file")
@mock.patch("src.extract.vision.PoseLandmarker.create_from_options")
def test_successful_landmark_extraction(mock_create_from_options, mock_create_from_file, fake_image_path, fake_model_path):
    # Mock image
    mock_create_from_file.return_value = mock.Mock()

    # Create fake detection result
    mock_result = mock.Mock()
    mock_result.pose_landmarks = [[mock.Mock(x=0.1, y=0.2, z=0.3, visibility=0.9)] * 33]
    mock_result.pose_world_landmarks = [[mock.Mock(x=1.1, y=1.2, z=1.3)] * 33]
    mock_result.segmentation_masks = [mock.Mock(numpy_view=lambda: np.zeros((256, 256), dtype=np.float32))]

    # Mock detector
    mock_detector = mock.Mock()
    mock_detector.detect.return_value = mock_result
    mock_create_from_options.return_value = mock_detector

    output = get_landmarks(fake_image_path, fake_model_path)

    assert "landmarks" in output
    assert "world_landmarks" in output
    assert "segmentation_mask" in output
    assert len(output["landmarks"]) == 33
    assert len(output["world_landmarks"]) == 33
    assert isinstance(output["segmentation_mask"], np.ndarray)


def test_missing_image_file(tmp_path, fake_model_path):
    bad_path = tmp_path / "does_not_exist.jpg"
    with pytest.raises(FileNotFoundError, match="Image file not found"):
        get_landmarks(bad_path, fake_model_path)


def test_missing_model_file(tmp_path, fake_image_path):
    bad_path = tmp_path / "missing_model.task"
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        get_landmarks(fake_image_path, bad_path)


@mock.patch("src.extract.mp.Image.create_from_file")
@mock.patch("src.extract.vision.PoseLandmarker.create_from_options")
def test_no_pose_detected(mock_create_from_options, mock_create_from_file, fake_image_path, fake_model_path):
    mock_create_from_file.return_value = mock.Mock()
    mock_detector = mock.Mock()
    mock_detector.detect.return_value = mock.Mock(pose_landmarks=[], pose_world_landmarks=[])
    mock_create_from_options.return_value = mock_detector

    with pytest.raises(ValueError, match="No pose detected in image"):
        get_landmarks(fake_image_path, fake_model_path)
