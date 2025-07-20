from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim

from config.config import config


class ClientSelectionAgent:
    def __init__(self, num_clients, num_clusters=3, lr=0.01):
        self.num_clients = num_clients
        self.num_clusters = num_clusters
        self.lr = lr
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.1
        
        # Q-network for client selection
        self.q_network = nn.Sequential(
            nn.Linear(num_clients + 1, 128),  # +1 for round number
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_clients)  # Output: selection probability for each client
        )
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.memory = deque(maxlen=1000)
        self.reward_history = []
        
    def get_state(self, round_num, client_performances):
        """Create state representation"""
        state = list(client_performances) + [round_num / config['num_rounds']]
        return torch.FloatTensor(state)
    
    def select_clients_from_cluster(self, cluster_clients, state):
        """Select clients from a specific cluster"""
        if len(cluster_clients) == 0:
            return []
        
        # Epsilon-greedy selection
        if random.random() < self.epsilon:
            # Random selection
            num_select = random.randint(1, min(len(cluster_clients), 5))
            selected = random.sample(cluster_clients, num_select)
        else:
            # Q-network based selection
            with torch.no_grad():
                q_values = self.q_network(state)
                
            # Filter Q-values for clients in this cluster
            cluster_q_values = [(q_values[client_id].item(), client_id) for client_id in cluster_clients]
            cluster_q_values.sort(reverse=True)
            
            # Select top clients based on Q-values
            num_select = random.randint(1, min(len(cluster_clients), 5))
            selected = [client_id for _, client_id in cluster_q_values[:num_select]]
        
        return selected
    
    def select_clients_for_round(self, clustered_clients, round_num, client_performances):
        """Select clients from each cluster for the current round"""
        state = self.get_state(round_num, client_performances)
        selected_clients = {}
        
        for metric in clustered_clients:
            selected_clients[metric] = {}
            for cluster_id, cluster_clients in clustered_clients[metric].items():
                selected = self.select_clients_from_cluster(cluster_clients, state)
                selected_clients[metric][cluster_id] = selected
        
        return selected_clients, state
    
    def compute_reward(self, prev_accuracy, curr_accuracy, selected_clients):
        """Compute reward based on accuracy improvement"""
        accuracy_improvement = curr_accuracy - prev_accuracy
        
        # Base reward from accuracy improvement
        reward = accuracy_improvement * 100
        
        # Penalty for selecting too many clients (efficiency)
        total_selected = sum(len(clients) for metric_clients in selected_clients.values() 
                           for clients in metric_clients.values())
        efficiency_penalty = -0.01 * total_selected
        
        return reward + efficiency_penalty
    
    def update_q_network(self, reward):
        """Update Q-network based on reward"""
        self.reward_history.append(reward)
        
        if len(self.memory) < 32:  # Wait for enough samples
            return
        
        # Sample from memory
        batch = random.sample(self.memory, min(32, len(self.memory)))
        states, actions, rewards, next_states = zip(*batch)
        
        states = torch.stack(states)
        rewards = torch.FloatTensor(rewards)
        
        # Compute Q-values
        q_values = self.q_network(states)
        
        # Simple policy gradient-like update
        loss = -torch.mean(q_values.sum(dim=1) * rewards)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def store_experience(self, state, action, reward, next_state):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state))
