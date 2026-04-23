# =========================
# Helper
# =========================
def create_dummy_face():
    """Creates a valid grayscale face image"""
    return np.zeros((48, 48), dtype=np.uint8)


# =========================
# Model loading
# =========================
def test_get_model_lazy_loading(monkeypatch):
    """Ensure model loads only when needed"""

    import src.inference.predict as p

    # reset state
    p.model = None

    def fake_load(_):
        class FakeModel:
            def predict(self, x, verbose=0):
                return np.array([[0.1] * len(CLASS_NAMES)])
        return FakeModel()
    
    monkeypatch.setattr("tensorflow.keras.models.load_model", fake_load)

    model1 = p.get_model()
    model2 = p.get_model()

    assert model1 is model2  # same cached model


# =========================
# Preprocessing
# =========================
def test_preprocess_face_shape():
    img = create_dummy_face()

    processed = preprocess_face(img)

    assert processed.shape == (1, 48, 48, 1)
    assert processed.dtype in [np.float32, np.float64]

def test_preprocess_normalization():
    img = np.ones((48, 48), dtype=np.uint8) * 255

    processed = preprocess_face(img)

    assert np.max(processed) <= 1.0
    assert np.min(processed) >= 0.0

# =========================
# Prediction - mocked model
# =========================
def test_predict_emotion(monkeypatch):
    """Test full pipeline with mocked model"""

    class FakeModel:
        def predict(self, x, verbose=0):
            # always return class 3 (Happy)
            out = np.zeros((1, len(CLASS_NAMES)))
            out[0][3] = 1.0
            return out

    monkeypatch.setattr("src.inference.predict.get_model", lambda: FakeModel())

    img = create_dummy_face()

    result = predict_emotion(img)

    assert result == "Happy"

# =========================
# Prediction output validity
# =========================
def test_prediction_is_valid_class(monkeypatch):
    class FakeModel:
        def predict(self, x, verbose=0):
            return np.random.rand(1, len(CLASS_NAMES))

    monkeypatch.setattr("src.inference.predict.get_model", lambda: FakeModel())

    img = create_dummy_face()

    result = predict_emotion(img)

    assert result in CLASS_NAMES

# =========================
# Deterministic behavior (same input -> same output)
# =========================
def test_prediction_deterministic(monkeypatch):
    class FakeModel:
        def predict(self, x, verbose=0):
            out = np.zeros((1, len(CLASS_NAMES)))
            out[0][6] = 1.0
            return out

    monkeypatch.setattr("src.inference.predict.get_model", lambda: FakeModel())

    img = create_dummy_face()

    r1 = predict_emotion(img)
    r2 = predict_emotion(img)

    assert r1 == r2
    assert r1 == "Neutral"
