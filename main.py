import torch
import logging
import random
import numpy as np
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets

from config.config import config
from models.model_factory import initialize_client_models
from clients.clustering import ClientClusterer
from clients.selection_rl import ClientSelectionAgent
from clients.train import train_client
from data.prepare_data import split_dataset_uneven, plot_data_distribution
from distillation.finetune import finetune_global_model
from distillation.evaluate import evaluate_global_model
from distillation.distill import enhanced_distillation
from utils.memory import MemoryManager
from data.transforms import default_transform
from utils.logging_utils import setup_logging


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main():
    setup_logging("training.log")
    set_seed(config['seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # === Load and split data
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=default_transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=default_transform)

    client_indices = split_dataset_uneven(full_train, config['num_clients'])
    plot_data_distribution(full_train, client_indices)

    client_loaders = [
        DataLoader(
            Subset(full_train, client_indices[i]),
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=torch.cuda.is_available()
        ) for i in range(config['num_clients'])
    ]

    public_data = random_split(full_train, [
        len(full_train) - int(len(full_train) * config['public_data_ratio']),
        int(len(full_train) * config['public_data_ratio'])
    ])[1]

    public_loader = DataLoader(public_data, batch_size=config['batch_size'], shuffle=True,
                               num_workers=config['num_workers'], pin_memory=torch.cuda.is_available())

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], num_workers=config['num_workers'],
                             pin_memory=torch.cuda.is_available())

    # === Initialize
    clusterer = ClientClusterer(num_clusters=3)
    rl_agent = ClientSelectionAgent(config['num_clients'], num_clusters=3)
    client_models = initialize_client_models(config['num_clients'])

    # Global model setup (DenseNet)
    global_model = torch.hub.load('pytorch/vision:v0.14.0', 'densenet121', weights=None)
    global_model.classifier = torch.nn.Linear(global_model.classifier.in_features, config['num_classes'])

    # === Initial fine-tuning and eval
    global_model = finetune_global_model(global_model, public_loader, device)
    prev_accuracy = evaluate_global_model(global_model, test_loader, device)
    logging.info(f"Initial Accuracy after fine-tuning: {prev_accuracy * 100:.2f}%")
    client_performances = [prev_accuracy] * config['num_clients']

    # === Training loop
    for round_num in range(config['num_rounds']):
        logging.info(f"\n=== Round {round_num+1}/{config['num_rounds']} ===")
        logging.info(f"Memory: {MemoryManager.get_memory_usage():.2f} GB")

        for i in range(config['num_clients']):
            logging.info(f"Client {i+1} Training")
            client_models[i] = train_client(client_models[i], client_loaders[i], device,
                                            config['max_epochs'], config['early_stop_patience'])

        clustered_clients = clusterer.cluster_clients(client_indices, client_models, full_train, public_loader, device)

        selected_clients, state = rl_agent.select_clients_for_round(clustered_clients, round_num, client_performances)

        for epoch in range(config['distill_epochs']):
            loss = enhanced_distillation(global_model, selected_clients, client_models, public_loader, device)
            logging.info(f"Enhanced Distillation Epoch {epoch+1}: Loss = {loss:.4f}")

        curr_accuracy = evaluate_global_model(global_model, test_loader, device)
        reward = rl_agent.compute_reward(prev_accuracy, curr_accuracy, selected_clients)
        rl_agent.update_q_network(reward)

        logging.info(f"Round {round_num+1} Accuracy: {curr_accuracy*100:.2f}% | Reward: {reward:.4f}")
        prev_accuracy = curr_accuracy
        client_performances = [curr_accuracy] * config['num_clients']

    # === Final fine-tune and evaluation
    global_model = finetune_global_model(global_model, public_loader, device)
    final_accuracy = evaluate_global_model(global_model, test_loader, device)
    logging.info(f"\nFinal Accuracy: {final_accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
