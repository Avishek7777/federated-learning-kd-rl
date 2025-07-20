import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda import amp
from tqdm import tqdm

from config.config import config
from utils.memory import MemoryManager


def train_client(model, train_loader, device, max_epochs, patience=3):
    model = MemoryManager.optimize_model_loading(model, device)
    model.train()

    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    criterion = nn.CrossEntropyLoss()

    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(max_epochs):
        correct, total, running_loss = 0, 0, 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}")

        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), config['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

        acc = correct / total
        scheduler.step()

        if acc > best_acc + 0.001:
            best_acc = acc
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            break

    model.cpu()
    MemoryManager.clear_cache()
    return model
