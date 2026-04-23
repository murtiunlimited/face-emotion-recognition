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
