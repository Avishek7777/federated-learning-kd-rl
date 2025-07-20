from collections import defaultdict
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn

from config.config import config
from utils.memory import MemoryManager


class ClientClusterer:
    def __init__(self, num_clusters=3):
        self.num_clusters = num_clusters
        self.client_clusters = {}
        self.similarity_cache = {}  # NEW: Cache for similarity computations
        self.similarity_metrics = ['data_distribution', 'gradient_similarity', 'performance_similarity']
    
    def _get_cache_key(self, client_ids, metric_type):
        """Generate cache key for similarity metrics"""
        return f"{metric_type}_{hash(tuple(sorted(client_ids)))}"
    
    def compute_data_distribution_similarity(self, client_indices, dataset):
        """OPTIMIZED: Compute similarity with caching"""
        cache_key = self._get_cache_key(client_indices.keys(), 'data_dist')
        
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        client_distributions = []
        for client_id, indices in client_indices.items():
            # Use numpy for faster computation
            labels = np.array([dataset[idx][1] for idx in indices])
            class_counts = np.bincount(labels, minlength=config['num_classes'])
            # Normalize to get distribution
            class_counts = class_counts / (class_counts.sum() + 1e-8)  # Added epsilon for stability
            client_distributions.append(class_counts)
        
        result = np.array(client_distributions)
        self.similarity_cache[cache_key] = result
        return result
    
    def compute_gradient_similarity(self, client_models, public_loader, device):
        """OPTIMIZED: More efficient gradient similarity computation"""
        client_gradients = []
        criterion = nn.CrossEntropyLoss()
        
        # Use only a subset of data for efficiency
        limited_loader = list(public_loader)[:2]  # Only first 2 batches
        
        for idx, model in enumerate(client_models):
            model = MemoryManager.optimize_model_loading(model, device)
            model.train()
            
            # Collect gradients more efficiently
            grad_vectors = []
            for batch_idx, (images, labels) in enumerate(limited_loader):
                if batch_idx >= 1:  # Only use first batch
                    break
                    
                images, labels = images.to(device), labels.to(device)
                model.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                
                # More efficient gradient flattening
                grad_vec = []
                for param in model.parameters():
                    if param.grad is not None:
                        grad_vec.append(param.grad.flatten())
                
                if grad_vec:
                    grad_vector = torch.cat(grad_vec).cpu().numpy()
                    grad_vectors.append(grad_vector)
            
            # Average gradients across batches
            if grad_vectors:
                avg_grad = np.mean(grad_vectors, axis=0)
                client_gradients.append(avg_grad)
            else:
                # Fallback
                client_gradients.append(np.random.randn(1000))
            
            model.cpu()
            MemoryManager.clear_cache()
        
        if not client_gradients:
            return np.random.rand(len(client_models), 1000)
        
        # Normalize all gradients to same size more efficiently
        min_size = min(len(grad) for grad in client_gradients)
        normalized_gradients = np.array([grad[:min_size] for grad in client_gradients])
        
        # L2 normalize
        norms = np.linalg.norm(normalized_gradients, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized_gradients = normalized_gradients / norms
        
        return normalized_gradients
    
    def compute_performance_similarity(self, client_models, public_loader, device):
        """Compute similarity based on performance on public data"""
        client_performances = []
        
        for model in client_models:
            model = model.to(device)
            model.eval()
            
            correct, total = 0, 0
            with torch.no_grad():
                for images, labels in public_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    predicted = outputs.argmax(1)
                    correct += (predicted == labels).sum().item()
                    total += labels.size(0)
            
            accuracy = correct / total
            client_performances.append([accuracy])
            
            model = model.cpu()
            torch.cuda.empty_cache()
        
        return np.array(client_performances)
    
    def cluster_clients(self, client_indices, client_models, dataset, public_loader, device):
        """Perform clustering based on multiple similarity metrics"""

        print("Now Clustering Clients")
        num_clients = len(client_models)
        
        # Compute similarity matrices
        print("Getting data similarity")
        data_sim = self.compute_data_distribution_similarity(client_indices, dataset)
        print("Getting distribution similarity")
        grad_sim = self.compute_gradient_similarity(client_models, public_loader, device)
        print("Getting performance similarity")
        perf_sim = self.compute_performance_similarity(client_models, public_loader, device)
        
        # Perform clustering for each metric
        clusters = {}
        
        # Data distribution clustering
        kmeans_data = KMeans(n_clusters=self.num_clusters, random_state=config['seed'], n_init=10)
        clusters['data_distribution'] = kmeans_data.fit_predict(data_sim)
        
        # Gradient similarity clustering
        if len(grad_sim) > 0:
            kmeans_grad = KMeans(n_clusters=self.num_clusters, random_state=config['seed'], n_init=10)
            clusters['gradient_similarity'] = kmeans_grad.fit_predict(grad_sim)
        else:
            clusters['gradient_similarity'] = np.random.randint(0, self.num_clusters, num_clients)
        
        # Performance similarity clustering
        kmeans_perf = KMeans(n_clusters=self.num_clusters, random_state=config['seed'], n_init=10)
        clusters['performance_similarity'] = kmeans_perf.fit_predict(perf_sim)
        
        # Organize clusters by metric
        self.client_clusters = {}
        for metric in self.similarity_metrics:
            self.client_clusters[metric] = defaultdict(list)
            for client_id, cluster_id in enumerate(clusters[metric]):
                self.client_clusters[metric][cluster_id].append(client_id)
        
        return self.client_clusters