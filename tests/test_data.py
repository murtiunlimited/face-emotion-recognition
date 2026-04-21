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
