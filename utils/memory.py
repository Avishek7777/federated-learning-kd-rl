import torch

class MemoryManager:
    @staticmethod
    def clear_cache():
        """Clear CUDA cache if available"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def get_memory_usage():
        """Get current memory usage"""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024**3  # in GB
        return 0

    @staticmethod
    def optimize_model_loading(model, device):
        """Move model to device"""
        return model.to(device)

    @staticmethod
    def cleanup_model(model):
        """Offload model and clear memory"""
        model.cpu()
        del model
        MemoryManager.clear_cache()
