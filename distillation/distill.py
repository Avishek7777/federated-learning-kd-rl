import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from config.config import config
from utils.memory import MemoryManager

def select_best_cluster_per_metric(selected_clients_by_metric, client_models, public_loader, device):
    """Select the best performing cluster from each metric"""
    best_clusters = {}
    
    for metric, cluster_selections in selected_clients_by_metric.items():
        best_cluster_info = None
        best_avg_confidence = 0
        
        print(f"Evaluating clusters for {metric}...")
        
        for cluster_id, selected_clients in cluster_selections.items():
            if len(selected_clients) == 0:
                continue
                
            # Quick evaluation on a small batch to get cluster quality
            total_confidence = 0
            num_samples = 0
            
            for images, _ in list(public_loader)[:2]:  # Just first 2 batches for speed
                images = images.to(device)
                batch_confidences = []
                
                for client_id in selected_clients:
                    model = client_models[client_id].to(device)
                    model.eval()
                    with torch.no_grad():
                        logits = model(images)
                        probs = F.softmax(logits, dim=1)
                        confidence = probs.max(dim=1)[0].mean().item()
                        batch_confidences.append(confidence)
                    model.cpu()
                    torch.cuda.empty_cache()
                
                if batch_confidences:
                    avg_confidence = sum(batch_confidences) / len(batch_confidences)
                    total_confidence += avg_confidence
                    num_samples += 1
            
            if num_samples > 0:
                cluster_avg_confidence = total_confidence / num_samples
                
                if cluster_avg_confidence > best_avg_confidence:
                    best_avg_confidence = cluster_avg_confidence
                    best_cluster_info = (cluster_id, selected_clients)
        
        if best_cluster_info:
            best_clusters[metric] = best_cluster_info
            print(f"  Best cluster for {metric}: Cluster {best_cluster_info[0]} with clients {best_cluster_info[1]} (confidence: {best_avg_confidence:.4f})")
    
    return best_clusters

def confidence_weighted_distillation(global_model, client_models, public_loader, device):
    global_model = global_model.to(device)
    global_model.train()
    optimizer = optim.AdamW(global_model.parameters(), lr=config['lr'])
    kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
    T = config['kd_temperature']

    for images, _ in tqdm(public_loader, desc='Distilling'):
        images = images.to(device)
        optimizer.zero_grad()
        all_logits, confidences = [], []

        for model in client_models:
            model = model.to(device)
            model.eval()
            with torch.no_grad():
                logits = model(images)
                probs = F.softmax(logits, dim=1)
                confidence = probs.max(dim=1)[0].mean()
                confidences.append(confidence.item())
                all_logits.append(logits / T)
            model = model.cpu()
            torch.cuda.empty_cache()

        weights = F.softmax(torch.tensor(confidences), dim=0)
        avg_logits = sum(w * l for w, l in zip(weights, all_logits))
        teacher_probs = F.softmax(avg_logits, dim=1)

        student_logits = global_model(images)
        student_log_probs = F.log_softmax(student_logits / T, dim=1)
        loss = kd_loss_fn(student_log_probs, teacher_probs) * (T ** 2)
        loss.backward()
        optimizer.step()

    global_model = global_model.cpu()
    torch.cuda.empty_cache()
    return loss.item()

def enhanced_distillation(global_model, selected_clients_by_metric, client_models, public_loader, device):
    """Perform distillation with best cluster from each metric"""
    
    # Select best cluster from each metric (using existing function)
    best_clusters = select_best_cluster_per_metric(selected_clients_by_metric, client_models, public_loader, device)
    
    global_model = MemoryManager.optimize_model_loading(global_model, device)
    global_model.train()
    
    # Use AdamW with lower learning rate for distillation
    optimizer = optim.AdamW(global_model.parameters(), lr=config['lr'] * 0.1, weight_decay=0.01)
    kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
    T = config['kd_temperature']
    
    # Mixed precision for distillation
    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
    
    total_loss = 0
    num_clusters_processed = 0
    
    print(f"\nDistilling from {len(best_clusters)} best clusters...")
    
    # Process clusters in batches for memory efficiency
    for metric, (cluster_id, selected_clients) in best_clusters.items():
        print(f"  Processing {metric}: Cluster {cluster_id} with {len(selected_clients)} clients")
        
        cluster_loss = 0
        num_batches = 0
        
        # Batch processing for memory efficiency
        for batch_idx, (images, _) in enumerate(tqdm(public_loader, desc=f'Distilling {metric}')):
            images = images.to(device)
            
            # Process clients in smaller groups to save memory
            client_groups = [selected_clients[i:i+2] for i in range(0, len(selected_clients), 2)]
            
            for group in client_groups:
                optimizer.zero_grad()
                
                # Get teacher logits from current group
                teacher_logits = []
                confidences = []
                
                for client_id in group:
                    model = MemoryManager.optimize_model_loading(client_models[client_id], device)
                    model.eval()
                    
                    with torch.no_grad():
                        if scaler is not None:
                            with torch.amp.autocast('cuda'):
                                logits = model(images)
                        else:
                            logits = model(images)
                        
                        probs = F.softmax(logits, dim=1)
                        confidence = probs.max(dim=1)[0].mean()
                        confidences.append(confidence.item())
                        teacher_logits.append(logits / T)
                    
                    model.cpu()
                    MemoryManager.clear_cache()
                
                # Compute ensemble teacher output
                if len(teacher_logits) > 1:
                    weights = F.softmax(torch.tensor(confidences), dim=0)
                    avg_logits = sum(w * l for w, l in zip(weights, teacher_logits))
                else:
                    avg_logits = teacher_logits[0]
                
                teacher_probs = F.softmax(avg_logits, dim=1)
                
                # Student forward pass and loss computation
                if scaler is not None:
                    with torch.amp.autocast('cuda'):
                        student_logits = global_model(images)
                        student_log_probs = F.log_softmax(student_logits / T, dim=1)
                        loss = kd_loss_fn(student_log_probs, teacher_probs) * (T ** 2)
                    
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    student_logits = global_model(images)
                    student_log_probs = F.log_softmax(student_logits / T, dim=1)
                    loss = kd_loss_fn(student_log_probs, teacher_probs) * (T ** 2)
                    
                    loss.backward()
                    optimizer.step()
                
                cluster_loss += loss.item()
                num_batches += 1
            
            # Limit batches for efficiency
            #if batch_idx >= 10:  # Process only first 10 batches per cluster
                #break
        
        avg_cluster_loss = cluster_loss / num_batches if num_batches > 0 else 0
        total_loss += avg_cluster_loss
        num_clusters_processed += 1
        
        print(f"    {metric} cluster {cluster_id} average loss: {avg_cluster_loss:.4f}")
    
    global_model.cpu()
    MemoryManager.clear_cache()
    
    return total_loss / num_clusters_processed if num_clusters_processed > 0 else 0