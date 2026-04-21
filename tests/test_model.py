import numpy as np
import pytest

from src.models.model import build_light_model
from src.inference.predict import IMG_SIZE, CLASS_NAMES


# =========================
# Model builds correctly
# =========================
def test_model_build():
    model = build_light_model()

    assert model is not None
    assert hasattr(model, "summary")
    assert hasattr(model, "input_shape")
    assert hasattr(model, "output_shape")

# =========================
# Input / Output contract
# =========================
def test_model_shapes():
    model = build_light_model()

    # Input should be (None, H, W, C)
    assert len(model.input_shape) == 4
    
    # Output should match number of classes
    assert model.output_shape[-1] == len(CLASS_NAMES)

# =========================
# Forward pass (single sample)
# =========================
def test_forward_pass_single():
    model = build_light_model()

    x = np.random.rand(1, IMG_SIZE, IMG_SIZE, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, len(CLASS_NAMES))


# =========================
# Forward pass (batch)
# =========================
def test_forward_pass_batch():
    model = build_light_model()

    batch_size = 4
    x = np.random.rand(batch_size, IMG_SIZE, IMG_SIZE, 1).astype("float32")

    y = model(x)

    assert y.shape == (batch_size, len(CLASS_NAMES))
