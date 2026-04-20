from src.models.model import build_light_model

def test_model_creation():
    model = build_light_model()

    assert model is not None
    assert len(model.layers) > 0
