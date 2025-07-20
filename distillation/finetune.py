import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from config.config import config
from utils.memory import MemoryManager


def finetune_global_model(global_model, public_loader, device, epochs=5):
    print(f"Fine-tuning global model for {epochs} epochs...")
    global_model = global_model.to(device)
    global_model.train()

    optimizer = optim.AdamW(global_model.parameters(), lr=config['lr'] * 0.1)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in tqdm(public_loader, desc=f'Fine-tuning Epoch {epoch+1}/{epochs}'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            predicted = outputs.argmax(1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = correct / total
        print(f"Fine-tuning Epoch {epoch+1}: Loss={total_loss/len(public_loader):.4f}, Acc={accuracy*100:.2f}%")

    global_model = global_model.cpu()
    torch.cuda.empty_cache()
    return global_model
