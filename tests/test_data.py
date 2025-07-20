from torchvision import datasets
from torch.utils.data import Subset
from data.prepare_data import split_dataset_uneven
from data.transforms import default_transform
from config.config import config

def test_data_split():
    dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=default_transform)
    client_indices = split_dataset_uneven(dataset, config['num_clients'])
    total = sum(len(indices) for indices in client_indices.values())
    assert total == len(dataset), "Split data size mismatch."
    print("Data split successful.")

if __name__ == "__main__":
    test_data_split()
