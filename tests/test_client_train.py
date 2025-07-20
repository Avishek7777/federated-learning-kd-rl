import torch
from torchvision import datasets
from torch.utils.data import Subset, DataLoader
from data.transforms import default_transform
from data.prepare_data import split_dataset_uneven
from clients.train import train_client
from models.model_factory import get_available_models
from config.config import config

def test_client_train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=default_transform)
    client_indices = split_dataset_uneven(dataset, config['num_clients'])
    subset = Subset(dataset, client_indices[0][:128])
    loader = DataLoader(subset, batch_size=32)

    model = get_available_models()[0]()
    model = train_client(model, loader, device, max_epochs=1)
    print("Client training test passed.")

if __name__ == "__main__":
    test_client_train()
