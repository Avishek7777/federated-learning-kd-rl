import torchvision.models as models
from models.cnn import SimpleCNN
from models.vit import ViT

from config.config import config
import random

def get_available_models():
    """
    Returns a list of available model names.
    """
    return [
        # lambda: ViT(config['num_classes']),  # Uncomment when needed
        lambda: SimpleCNN(config['num_classes']),
        lambda: models.resnet18(weights=None, num_classes=config['num_classes']),
        lambda: models.resnet50(weights=None, num_classes=config['num_classes']),
        lambda: models.mobilenet_v2(weights=None, num_classes=config['num_classes']),
        lambda: models.vgg16(weights=None, num_classes=config['num_classes']),
        lambda: models.vgg19(weights=None, num_classes=config['num_classes']),
        lambda: models.efficientnet_b0(weights=None, num_classes=config['num_classes']),
        lambda: models.efficientnet_b1(weights=None, num_classes=config['num_classes']),
        lambda: models.efficientnet_b2(weights=None, num_classes=config['num_classes']),
    ]

def initialize_client_models(num_clients):
    model_fns = get_available_models()
    if config['model_selection_strategy'] == 'diverse':
        client_models = []
        for i in range(num_clients):
            model_fn = model_fns[i % len(model_fns)]
            client_models.append(model_fn())
        return client_models
    else:
        # Randomly select a model for each client
        return [random.choice(model_fns)() for _ in range(num_clients)]
