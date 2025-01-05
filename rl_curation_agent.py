import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import pandas as pd
from typing import Dict, List, Tuple, Optional
from preprocessing import SignalPreprocessor

# Experience replay tuple
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])

class DuelingDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int, hidden_size: int = 128):
        """
        Dueling DQN architecture with separate value and advantage streams.
        
        Args:
            state_size: Dimension of state space (APR changes, query trends, etc.)
            action_size: Number of possible actions (discretized allocation choices)
            hidden_size: Size of hidden layers
        """
        super(DuelingDQN, self).__init__()
        
        # Shared feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)  # Single value for state
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size)  # Advantage for each action
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            state: Current state tensor
            
        Returns:
            Q-values for each action
        """
        features = self.feature_layer(state)
        
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        
        # Combine value and advantages using dueling architecture
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        """
        Experience replay buffer for storing transitions.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def add(self, state: np.ndarray, action: int, reward: float, 
            next_state: np.ndarray, done: bool):
        """Add experience to buffer."""
        experience = Experience(state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Tuple:
        """Sample random batch of experiences."""
        experiences = random.sample(self.buffer, batch_size)
        
        # Convert to torch tensors
        states = torch.FloatTensor([e.state for e in experiences])
        actions = torch.LongTensor([e.action for e in experiences])
        rewards = torch.FloatTensor([e.reward for e in experiences])
        next_states = torch.FloatTensor([e.next_state for e in experiences])
        dones = torch.FloatTensor([e.done for e in experiences])
        
        return (states, actions, rewards, next_states, dones)
    
    def __len__(self) -> int:
        return len(self.buffer)

class RLCurationAgent:
    def __init__(self, state_size: int, action_size: int, 
                 learning_rate: float = 1e-4, gamma: float = 0.99,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995, buffer_size: int = 10000,
                 batch_size: int = 64, target_update: int = 10,
                 history_window: int = 10):
        """
        RL agent for optimizing curation signal allocation.
        
        Args:
            state_size: Size of state space
            action_size: Size of action space
            learning_rate: Learning rate for optimizer
            gamma: Discount factor
            epsilon_start: Starting exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for exploration
            buffer_size: Size of replay buffer
            batch_size: Size of training batches
            target_update: Frequency of target network updates
        """
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # Replay buffer
        self.memory = ReplayBuffer(buffer_size)
        
        # Training tracking
        self.steps = 0
        
        # Preprocessor
        self.preprocessor = SignalPreprocessor(window_size=history_window)
        self.historical_data = []
    
    def update_historical_data(self, subgraphs_data: Dict):
        """Update historical data for preprocessing."""
        df = pd.DataFrame(subgraphs_data['new_signals'])
        self.historical_data.append(df)
        
        # Keep only recent history
        if len(self.historical_data) > self.preprocessor.window_size:
            self.historical_data.pop(0)
    
    def get_state(self, subgraphs_data: Dict) -> np.ndarray:
        """
        Convert subgraph data into state representation following the paper's approach.
        The state space captures recent trends in APR movements and signal allocations.
        
        Args:
            subgraphs_data: Dictionary containing subgraph metrics
            
        Returns:
            State vector combining temporal features and current metrics
        """
        df = pd.DataFrame(subgraphs_data['new_signals'])
        self.update_historical_data(subgraphs_data)
        
        # Get preprocessed features for current state
        current_features = self.preprocessor.transform(df)
        
        # Calculate temporal features from historical data
        if len(self.historical_data) >= 2:
            # Get APR changes over time
            apr_history = pd.concat([d[['apr']] for d in self.historical_data], axis=1)
            apr_changes = apr_history.diff(axis=1).fillna(0)
            temporal_features = apr_changes.values
            
            # Combine current and temporal features
            features = np.hstack([
                current_features,
                temporal_features.reshape(len(df), -1)
            ])
        else:
            features = current_features
        
        # Ensure features are 2D
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Ensure we have the correct number of features
        n_subgraphs = len(df)
        features_per_subgraph = self.state_size // n_subgraphs
        
        # If we don't have enough features, pad with zeros
        if features.shape[1] < features_per_subgraph:
            padding = np.zeros((features.shape[0], features_per_subgraph - features.shape[1]))
            features = np.hstack([features, padding])
        
        return features.flatten().astype(np.float32)
    
    def select_action(self, state: np.ndarray) -> int:
        """
        Select action using epsilon-greedy policy.
        
        Args:
            state: Current state vector
            
        Returns:
            Selected action index
        """
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        else:
            return random.randrange(self.action_size)
    
    def optimize_model(self):
        """Perform one step of optimization on the policy network."""
        if len(self.memory) < self.batch_size:
            return False  # Return False if no optimization was performed
        
        # Scale rewards for better gradient flow
        reward_scale = 1.0  # Increased scale factor for stronger gradients
        
        # Sample experiences
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Compute next Q values using target network (Double DQN)
        with torch.no_grad():
            # Get actions from policy network
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            # Get Q-values from target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target_q_values = reward_scale * rewards.unsqueeze(1) + self.gamma * next_q_values * (1 - dones.unsqueeze(1))
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values, target_q_values)
        
        # Ensure loss is non-zero
        if loss.item() == 0:
            # Add small noise to prevent zero gradients
            noise = torch.randn_like(current_q_values) * 1e-4
            current_q_values = current_q_values + noise
            loss = nn.MSELoss()(current_q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients for stability
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        
        self.optimizer.step()
        
        # Update target network
        if self.steps % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        self.steps += 1
    
    def calculate_reward(self, old_state: np.ndarray, new_state: np.ndarray) -> float:
        """
        Calculate reward based on change in portfolio value.
        
        Args:
            old_state: Previous state
            new_state: Current state
            
        Returns:
            Reward value
        """
        # Extract APRs from states (first feature for each subgraph)
        n_features = 7  # 6 base features + 1 combined signal
        old_aprs = old_state[::n_features]  # Every n_features value starting from 0
        new_aprs = new_state[::n_features]
        
        # Calculate reward based on APR improvement
        apr_changes = new_aprs - old_aprs
        
        # Reward function with balanced positive and negative rewards
        rewards = []
        for change in apr_changes:
            if change > 0:
                rewards.append(5 * change)  # Positive reward
            else:
                rewards.append(10 * change)  # Negative reward
        
        # Add smaller bonus for APR spread
        apr_bonus = (new_aprs.max() - new_aprs.min())
        reward = np.mean(rewards) + apr_bonus
        
        return float(reward)

    def optimize_signals(self, subgraphs_data: Dict, total_new_signal: float = 1000.0,
                        min_allocation: float = 10.0, training: bool = True) -> pd.DataFrame:
        """
        Optimize signal allocation using the trained RL policy.
        
        Args:
            subgraphs_data: Dictionary containing subgraph data
            total_new_signal: Total amount of signal to allocate
            min_allocation: Minimum allocation per subgraph
            training: Whether to update the policy during optimization
            
        Returns:
            DataFrame containing optimized allocations and metrics
        """
        current_state = self.get_state(subgraphs_data)
        action = self.select_action(current_state)
        
        # Convert action to allocation percentages using a more sophisticated mapping
        n_subgraphs = len(subgraphs_data['new_signals'])
        base_allocation = min_allocation * n_subgraphs
        remaining_signal = total_new_signal - base_allocation
        
        # Initialize with minimum allocation
        allocations = np.ones(n_subgraphs) * min_allocation
        
        # Action space: Each action represents a different allocation strategy
        # We divide actions into n_subgraphs * n_subgraphs possibilities
        # to allow for more nuanced allocation combinations
        action_matrix = action % (n_subgraphs * n_subgraphs)
        primary_idx = action_matrix // n_subgraphs
        secondary_idx = action_matrix % n_subgraphs
        
        if remaining_signal > 0:
            # Allocate 70% to primary subgraph and 30% to secondary
            allocations[primary_idx] += remaining_signal * 0.7
            allocations[secondary_idx] += remaining_signal * 0.3
        
        # Process results
        df = pd.DataFrame(subgraphs_data['new_signals'])
        results = []
        
        for idx, row in df.iterrows():
            new_signal = allocations[idx]
            
            # Calculate effective APR
            dilution_factor = row['signal_amount'] / (row['signal_amount'] + new_signal)
            effective_apr = row['apr'] * dilution_factor
            
            results.append({
                'ipfs_hash': row['ipfs_hash'],
                'signal_amount': row['signal_amount'],
                'new_signal': new_signal,
                'allocation_percentage': (new_signal / total_new_signal) * 100,
                'apr': effective_apr,
                'original_apr': row['apr'],
                'weekly_queries': row['weekly_queries'],
                'total_earnings': row['total_earnings']  # Include total_earnings in results
            })
        
        result_df = pd.DataFrame(results)
        
        if training:
            # Calculate new state and reward
            new_state = self.get_state({'new_signals': results})
            reward = self.calculate_reward(current_state, new_state)
            done = False  # In this case, each optimization is a single step
            
            # Store experience
            self.memory.add(current_state, action, reward, new_state, done)
            
            # Optimize model if we have enough experiences
            if len(self.memory) >= self.batch_size:
                self.optimize_model()
                
                # Force some learning in early stages
                if self.steps < 100:  # Extra optimization steps early on
                    for _ in range(4):  # Multiple updates per step
                        self.optimize_model()
        
        return result_df

# Example usage
if __name__ == "__main__":
    # Example data (same as in original implementation)
    data = {'new_signals': [{'ipfs_hash': 'QmfQ9PWRhQZc2XeFL2hKMkyen33e3RSXxYzcTyhLGVTeGn', 'signal_amount': 148.32804053050063, 'signalled_tokens': 166.9279939015239, 'annual_queries': 50778624, 'total_earnings': 2031.14496, 'curator_share': 203.11449600000003, 'estimated_earnings': 180.48246127485018, 'apr': 601.0175641493462, 'weekly_queries': 976512}, {'ipfs_hash': 'QmYYmaFvdLjAYDpQwhPq4FNkpiyNLiedLNgtcvEDipcGd3', 'signal_amount': 159.1006748320156, 'signalled_tokens': 604.8178153692892, 'annual_queries': 74746412, 'total_earnings': 2989.85648, 'curator_share': 298.985648, 'estimated_earnings': 78.64983000350767, 'apr': 244.17499371785297, 'weekly_queries': 1437431}]}
    
    # Initialize agent
    n_subgraphs = len(data['new_signals'])
    features_per_subgraph = 7  # 6 base features + 1 combined signal
    state_size = n_subgraphs * features_per_subgraph
    action_size = n_subgraphs  # One action per subgraph for simplified allocation
    
    agent = RLCurationAgent(state_size=state_size, action_size=action_size)
    
    # Run optimization
    result_df = agent.optimize_signals(data, total_new_signal=25000.0, min_allocation=100.0)
    
    # Display results
    pd.set_option('display.float_format', lambda x: '%.2f' % x)
    print("\nOptimal Signal Allocations:")
    result_display = result_df.sort_values('apr', ascending=False)
    print(result_display.to_string())
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Total Allocated Signal: {result_df['new_signal'].sum():,.2f}")
    print(f"Average New APR: {result_df['apr'].mean():.2f}%")
    print(f"Max APR: {result_df['apr'].max():.2f}%")
    print(f"Min APR: {result_df['apr'].min():.2f}%")
    print(f"APR Standard Deviation: {result_df['apr'].std():.2f}%")
