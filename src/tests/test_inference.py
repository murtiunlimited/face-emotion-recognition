from src.inference.predict import load_model, predict

def test_inference():
    model = load_model("models_artifacts/best_emotion_model.keras")

    dummy_input = [[[0.0] * 48 for _ in range(48)]]

    output = predict(model, dummy_input)

    assert output is not None
