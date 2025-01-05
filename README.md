# RL-Based Curation Agent

This repository implements a reinforcement learning-based curation agent for The Graph protocol, based on the approach outlined in "Advanced Statistical Arbitrage with Reinforcement Learning" by Boming Ning and Kiseop Lee.

## Architecture Overview

### State Space Design

Following the paper's state space formulation:
```math
S_t = [d_{t-l+1}, d_{t-l+2}, ..., d_t]
```
where each `d_i` represents APR changes and signal allocation trends. Our implementation extends this with:

1. **Temporal Features**:
   - APR movement trends over time
   - Moving averages and standard deviations
   - Momentum indicators for both APR and signal amounts

2. **Mean Reversion Metrics**:
   - Empirical mean reversion time calculation
   - Optimal coefficient search for feature combination
   - Normalized feature representation

### Q-Learning Implementation

We implement the paper's Q-learning update rule:
```math
Q_{new}(S_t, A_t) = (1 - α)Q(S_t, A_t) + α · R_{t+1} + α · γ · \max_a Q(S_{t+1}, a)
```

Key components:
1. **Dueling Double DQN Architecture**:
   - Separate value and advantage streams
   - Target network for stable Q-value estimation
   - Double Q-learning to prevent overestimation

2. **Experience Replay**:
   - Prioritized sampling of experiences
   - Batch processing for efficient learning
   - Memory buffer with configurable size

### Action Space

The action space is designed to provide fine-grained control over signal allocation:
- Actions mapped to `n_subgraphs * n_subgraphs` possibilities
- Primary (70%) and secondary (30%) allocation strategy
- Minimum allocation constraints for stability

### Training Methodology

Following the paper's approach:
1. **Epsilon-Greedy Strategy**:
   - Exploration during training (ε > 0)
   - Pure exploitation during testing (ε = 0)

2. **Reward Function**:
   - Rewards based on APR improvements
   - Penalties for APR degradation
   - Bonus for optimal spread allocation

## Usage

### Installation
```bash
pip install -r requirements.txt
```

### Basic Usage
```python
from rl_curation_agent import RLCurationAgent

# Initialize agent
agent = RLCurationAgent(
    state_size=n_subgraphs * features_per_subgraph,
    action_size=n_subgraphs * n_subgraphs,
    history_window=10  # For temporal features
)

# Optimize signal allocation
result_df = agent.optimize_signals(
    subgraphs_data,
    total_new_signal=amount,
    min_allocation=min_amount
)
```

### Configuration Parameters

1. **State Space**:
   - `state_size`: Total feature dimensions
   - `history_window`: Temporal window size
   - `features_per_subgraph`: Number of features per subgraph

2. **Training**:
   - `learning_rate`: Q-learning update rate (α)
   - `gamma`: Discount factor (γ)
   - `epsilon_start`: Initial exploration rate
   - `epsilon_decay`: Exploration decay rate
   - `target_update`: Target network update frequency

## Implementation Details

### Preprocessing Pipeline (`preprocessing.py`)
```python
preprocessor = SignalPreprocessor(window_size=10)
features = preprocessor.fit_transform(data)
```

Key features:
1. Mean reversion time calculation
2. Optimal coefficient search
3. Trend feature generation
4. Feature normalization

### RL Agent (`rl_curation_agent.py`)
```python
class RLCurationAgent:
    def optimize_signals(self, data, total_signal, min_allocation):
        # State representation
        current_state = self.get_state(data)
        
        # Action selection (ε-greedy)
        action = self.select_action(current_state)
        
        # Signal allocation
        allocations = self.compute_allocations(action, total_signal)
        
        # Update Q-values
        self.optimize_model()
```

### Neural Network Architecture
```python
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size):
        # Feature layer
        self.feature_layer = nn.Sequential(...)
        
        # Value stream
        self.value_stream = nn.Sequential(...)
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(...)
```

## Testing

Comprehensive test suite covering:
```bash
python -m pytest test_preprocessing.py test_rl_curation_agent.py -v
```

1. State representation
2. Action selection
3. Reward calculation
4. End-to-end optimization
5. Training loop validation

## References

Implementation based on:
- Ning, B., & Lee, K. "Advanced Statistical Arbitrage with Reinforcement Learning"
  - State space design (Section 1)
  - Q-learning implementation (Section 2)
  - Action selection strategy (Section 3)
  - Training methodology (Section 6)


# Cline auto-generation video (start to finish)

https://github.com/user-attachments/assets/957c5a79-b4c2-477d-a7ae-b51581932bbc



