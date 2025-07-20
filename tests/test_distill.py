import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from data.transforms import default_transform
from models.model_factory import get_available_models
from distillation.distill import confidence_weighted_distillation
from config.config import config

def test_distillation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=default_transform)
    public_data = random_split(dataset, [int(len(dataset) * 0.1), len(dataset) - int(len(dataset) * 0.1)])[0]
    public_loader = DataLoader(public_data, batch_size=16)

    models = [get_available_models()[0]() for _ in range(3)]
    global_model = get_available_models()[0]()

    loss = confidence_weighted_distillation(global_model, models, public_loader, device)
    assert isinstance(loss, float), "Loss is not a float."
    print("Distillation test passed.")

if __name__ == "__main__":
    test_distillation()
