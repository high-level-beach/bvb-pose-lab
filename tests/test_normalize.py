import pytest
import numpy as np
from src.normalize import normalize_landmarks


# ---- Fixtures ----

def generate_mock_landmarks(shift=(0.0, 0.0), scale=1.0):
    """Generate a simplified landmark list with proper torso joints and dummy points."""
    lmk = [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in range(33)]

    # Set hips and shoulders with some scale and shift
    lmk[23]["x"], lmk[23]["y"] = -0.1 * scale + shift[0], 0.0 * scale + shift[1]  # left hip
    lmk[24]["x"], lmk[24]["y"] =  0.1 * scale + shift[0], 0.0 * scale + shift[1]  # right hip
    lmk[11]["x"], lmk[11]["y"] = -0.1 * scale + shift[0], -0.5 * scale + shift[1]  # left shoulder
    lmk[12]["x"], lmk[12]["y"] =  0.1 * scale + shift[0], -0.5 * scale + shift[1]  # right shoulder

    # Add some extra points for stability
    lmk[0]["x"], lmk[0]["y"] = 0.0 + shift[0], -0.8 + shift[1]  # nose
    return lmk


# ---- Tests ----

def test_basic_centering_and_scaling():
    landmarks = generate_mock_landmarks()
    normalized = normalize_landmarks(landmarks, rotate=False)

    # Check that mid-hip is centered at (0, 0)
    mid_hip_x = (landmarks[23]["x"] + landmarks[24]["x"]) / 2
    mid_hip_y = (landmarks[23]["y"] + landmarks[24]["y"]) / 2
    centered_x = normalized[23]["x"]
    centered_y = normalized[23]["y"]

    assert np.isclose(centered_x, -0.2, atol=1e-4)
    assert np.isclose(centered_y, 0.0, atol=1e-4)

    # Check that torso length is normalized to 1.0
    l_sh = np.array([normalized[11]["x"], normalized[11]["y"]])
    r_sh = np.array([normalized[12]["x"], normalized[12]["y"]])
    mid_sh = (l_sh + r_sh) / 2
    mid_hip = np.array([0.0, 0.0])
    torso_vec = mid_sh - mid_hip
    torso_length = np.linalg.norm(torso_vec)
    assert np.isclose(torso_length, 1.0, atol=1e-4)


def test_rotation_enabled_aligns_vertical():
    landmarks = generate_mock_landmarks()
    normalized = normalize_landmarks(landmarks, rotate=True)

    l_sh = np.array([normalized[11]["x"], normalized[11]["y"]])
    r_sh = np.array([normalized[12]["x"], normalized[12]["y"]])
    mid_sh = (l_sh + r_sh) / 2
    torso_vec = mid_sh - np.array([0.0, 0.0])

    # Should be nearly vertical (angle ≈ 90° or -90° from horizontal)
    angle = np.arctan2(torso_vec[1], torso_vec[0])
    assert np.isclose(np.abs(angle), np.pi / 2, atol=1e-2)


def test_invalid_landmark_count_raises():
    with pytest.raises(ValueError):
        normalize_landmarks([{"x": 0.0, "y": 0.0, "z": 0.0}] * 10)
