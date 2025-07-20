# Federated Learning with Knowledge Distillation and RL-Based Client Selection

This project implements an advanced Federated Learning (FL) pipeline using:

- ✅ **Client Clustering** based on data, gradient, and performance similarity
- ✅ **RL-based Client Selection** using a Q-network to select the most valuable clients
- ✅ **Knowledge Distillation** from the best clusters to a global model
- ✅ Support for **Diverse Model Architectures** per client (CNN, ResNet, EfficientNet, etc.)
- ✅ Memory-optimized training and modular design for scalability

---

## 📁 Project Structure

FL-KD-RL-Project/
│
├── config/ # Global configuration dictionary
├── models/ # All model definitions and wrappers
├── data/ # Dataset loading, splitting, transforms
├── clients/ # Client training, clustering, RL agent
├── distillation/ # Global fine-tuning, distillation, evaluation
├── utils/ # Memory management, logging setup
├── tests/ # Unit tests for core modules
│
├── main.py # Main federated training loop
├── requirements.txt # Python dependencies
└── README.md # You're here!

---

## 🚀 How to Run

### 1. Install dependencies (Python 3.8+)

```bash
pip install -r requirements.txt
```
