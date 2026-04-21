import numpy as np
import pytest

from src.models.model import build_light_model
from src.inference.predict import IMG_SIZE, CLASS_NAMES


# 🔧 FIX: handle IMG_SIZE being int OR tuple
if isinstance(IMG_SIZE, tuple):
    H, W = IMG_SIZE
else:
    H, W = IMG_SIZE, IMG_SIZE


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

    x = np.random.rand(1, H, W, 1).astype("float32")
    y = model(x)

    assert y.shape == (1, len(CLASS_NAMES))


# =========================
# Forward pass (batch)
# =========================
def test_forward_pass_batch():
    model = build_light_model()

    batch_size = 4
    x = np.random.rand(batch_size, H, W, 1).astype("float32")

    y = model(x)

    assert y.shape == (batch_size, len(CLASS_NAMES))


# =========================
# Output behaves like probabilities
# =========================
def test_output_probabilities():
    model = build_light_model()

    x = np.random.rand(1, H, W, 1).astype("float32")
    y = model(x).numpy()

    # Values should be between 0 and 1
    assert np.all(y >= 0)
    assert np.all(y <= 1)

    # Probabilities should sum to ~1 (softmax)
    total = np.sum(y)
    assert np.isclose(total, 1.0, atol=1e-3)


# =========================
# Model can compile and train one step
# =========================
def test_train_step():
    model = build_light_model()

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy"
    )

    x = np.random.rand(2, H, W, 1).astype("float32")
    y = np.zeros((2, len(CLASS_NAMES)))
    y[:, 0] = 1  # fake one-hot labels

    history = model.fit(x, y, epochs=1, verbose=0)

    assert history is not None


# =========================
# Invalid input should fail
# =========================
def test_invalid_input_shape():
    model = build_light_model()

    bad_input = np.random.rand(1, 10, 10, 1).astype("float32")

    with pytest.raises(Exception):
        model(bad_input)
