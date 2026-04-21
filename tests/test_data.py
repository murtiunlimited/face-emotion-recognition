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
