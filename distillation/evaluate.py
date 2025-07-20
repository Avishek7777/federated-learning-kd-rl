import torch
from tqdm import tqdm
from utils.memory import MemoryManager


def evaluate_global_model(global_model, test_loader, device):
    global_model = MemoryManager.optimize_model_loading(global_model, device)
    global_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)

            if torch.cuda.is_available():
                with torch.amp.autocast('cuda'):
                    outputs = global_model(images)
            else:
                outputs = global_model(images)

            predicted = outputs.argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    global_model.cpu()
    MemoryManager.clear_cache()

    return accuracy
