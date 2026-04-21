import os
import pytest
from src.data.preprocess import preprocess_and_save

# =========================
# Paths
# =========================
RAW_TRAIN_DIR = "data/raw/train"
RAW_TEST_DIR = "data/raw/test"
PROCESSED_TRAIN_DIR = "data/processed/train"
PROCESSED_VAL_DIR = "data/processed/validation"

# =========================
# Directory existence
# =========================
def test_data_directories_exist():
    assert os.path.exists("data")
    assert os.path.exists("data/raw")
    assert os.path.exists("data/processed")

# =========================
# Raw data structure
# =========================
def test_raw_data_structure():
    assert os.path.exists(RAW_TRAIN_DIR)
    assert os.path.exists(RAW_TEST_DIR)

# =========================
# Processed data structure
# =========================
def test_processed_data_structure():
    assert os.path.exists(PROCESSED_TRAIN_DIR)
    assert os.path.exists(PROCESSED_VAL_DIR)

# =========================
# At least one image exists (if dataset present)
# =========================
def test_raw_contains_images():
    if not os.path.exists(RAW_TRAIN_DIR):
        pytest.skip("Raw train directory not found")

    found_image = False

    for root, _, files in os.walk(RAW_TRAIN_DIR):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                found_image = True
                break

# =========================
# Preprocessing function runs
# =========================
def test_preprocess_runs(tmp_path):
    # Skip if no raw data
    if not os.path.exists(RAW_TRAIN_DIR):
        pytest.skip("Raw data not available")

    output_dir = tmp_path / "processed"

    try:
        preprocess_and_save(
            input_dir=RAW_TRAIN_DIR,
            output_dir=str(output_dir)
        )
    except Exception as e:
        pytest.fail(f"Preprocessing failed: {e}")
    # Don't fail CI if dataset isn't included
    if not found_image:
        pytest.skip("No images found in raw dataset")

    assert found_image

# =========================
# Image can be read by OpenCV
# =========================
def test_image_readable():
    for root, _, files in os.walk(RAW_TRAIN_DIR):
        for f in files:
            if f.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(root, f)

                img = cv2.imread(path)

                assert img is not None
                return  # test one image only

    pytest.skip("No readable images found")

# =========================
# Preprocessing function runs
# =========================
def test_preprocess_runs(tmp_path):
    # Skip if no raw data
    if not os.path.exists(RAW_TRAIN_DIR):
        pytest.skip("Raw data not available")

    output_dir = tmp_path / "processed"

    try:
        preprocess_and_save(
            input_dir=RAW_TRAIN_DIR,
            output_dir=str(output_dir)
        )
    except Exception as e:
        pytest.fail(f"Preprocessing failed: {e}")

# =========================
# Processed output structure
# =========================
def test_processed_output_created(tmp_path):
    if not os.path.exists(RAW_TRAIN_DIR):
        pytest.skip("Raw data not available")

    output_dir = tmp_path / "processed"

    preprocess_and_save(
        input_dir=RAW_TRAIN_DIR,
        output_dir=str(output_dir)
    )

    assert os.path.exists(output_dir)
