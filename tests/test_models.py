import torch
from models.model_factory import get_available_models

def test_model_forward():
    models = get_available_models()
    dummy_input = torch.randn(1, 3, 32, 32)  # CIFAR-10 shape

    for model_fn in models:
        model = model_fn()
        out = model(dummy_input)
        assert out.shape[-1] == 10, f"Output shape mismatch: {out.shape}"
        print(f"{model.__class__.__name__} passed.")

if __name__ == "__main__":
    test_model_forward()
