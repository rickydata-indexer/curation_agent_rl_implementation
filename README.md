# RL-Based Curation Agent

This repository implements a reinforcement learning-based curation agent for The Graph protocol, based on the approach outlined in "Advanced Statistical Arbitrage with Reinforcement Learning" by Boming Ning and Kiseop Lee.

## Architecture Overview

### Mean Reversion Time (Section 3)

Following the paper's empirical mean reversion calculation:

1. **Local Extrema Identification**:
   ```math
   τ_n = inf{u ∈ [τ_{n-1}, T] : X_u is a local maximum}
   ```
   For odd-numbered moments, and mean crossings:
   ```math
   τ_n = inf{u ∈ [τ_{n-1}, T] : X_u = θ̂}
   ```
   For even-numbered moments.

2. **Time Sequence Construction**:
   - Alternating sequence of maxima and mean crossings
   - Empirical mean reversion time calculation:
   ```math
   r = \frac{2}{N} \sum_{i=2, \, i \text{ even}}^{N} (τ_n - τ_{n-1})
   ```

3. **Coefficient Optimization**:
   - Grid search over coefficient space
   - Constraint: a₁ = 1, aᵢ ∈ [-3.00, 3.00]
   - Objective: minimize mean reversion time

### State Space Design (Section 4.1)

State representation capturing recent price movements:
```math
S_t = [d_{t-l+1}, d_{t-l+2}, ..., d_t]
```

Implementation features:
1. **APR Movement Trends**:
   - Recent APR changes
   - Moving averages and volatility
   - Momentum indicators

2. **Signal Metrics**:
   - Current allocation levels
   - Historical allocation changes
   - Query volume trends

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

### RL Model Design (Section 4.2)

1. **Reward Function**:
   ```math
   G_t = \sum_{s=t+1}^{T} e^{-r \cdot (s-t)} R_s + I_T \cdot X_T
   ```
   where:
   - r: Interest rate for reward discounting
   - R_s: Immediate rewards from APR changes
   - I_T: Terminal position value
   - X_T: Final portfolio state

2. **Training Strategy**:
   - Positive rewards: 5x multiplier for APR improvements
   - Negative rewards: 10x multiplier for APR degradation
   - Exponential decay factor for time value
   - Terminal value based on portfolio spread

2. **Position Sizing & Risk Management**:
   - Primary/secondary allocation strategy (70%/30%)
   - Minimum allocation constraints to prevent extreme positions
   - Dynamic position sizing based on APR spreads
   - Automatic rebalancing through Q-value optimization

3. **Performance Metrics**:
   - Total APR improvement tracking
   - Portfolio diversification metrics
   - Signal allocation efficiency
   - Risk-adjusted returns through APR stability

4. **Hyperparameter Adaptation**:
   - Model-free approach minimizing fixed parameters
   - Dynamic exploration rate adjustment
   - Automatic feature coefficient optimization
   - Adaptive learning rate based on loss stability

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

1. **State Space Configuration**:
   - `state_size`: Total feature dimensions (n_subgraphs * features_per_subgraph)
   - `history_window`: Temporal window for trend calculation (default: 10)
   - `features_per_subgraph`: Base features (6) + derived features (1)

2. **Training Parameters**:
   ```python
   agent = RLCurationAgent(
       learning_rate=1e-4,    # Q-learning update rate (α)
       gamma=0.99,           # Future reward discount (γ)
       epsilon_start=1.0,    # Initial exploration rate
       epsilon_end=0.01,     # Minimum exploration rate
       epsilon_decay=0.995,  # Exploration decay rate
       buffer_size=10000,    # Experience replay capacity
       batch_size=64,        # Training batch size
       target_update=10      # Target network update frequency
   )
   ```

3. **Risk Management Settings**:
   ```python
   result_df = agent.optimize_signals(
       data,
       total_new_signal=1000.0,  # Total allocation amount
       min_allocation=10.0,      # Minimum per subgraph
       training=True             # Enable policy updates
   )
   ```

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

## Testing and Validation

### Unit Tests
```bash
python -m pytest test_preprocessing.py test_rl_curation_agent.py -v
```

1. **Feature Engineering Tests**:
   - Mean reversion time calculation
   - Coefficient optimization
   - Trend feature generation
   - Feature normalization

2. **Agent Tests**:
   - State representation
   - Action selection
   - Reward calculation
   - End-to-end optimization
   - Training loop validation

### Performance Monitoring

Monitor key metrics during training:
```python
# Training loop with metrics
for episode in range(n_episodes):
    result_df = agent.optimize_signals(data, training=True)
    
    # Performance metrics
    apr_improvement = (result_df['apr'] - result_df['original_apr']).mean()
    allocation_efficiency = result_df['allocation_percentage'].std()  # Lower is better
    portfolio_diversity = len(result_df[result_df['new_signal'] > min_allocation])
    
    print(f"Episode {episode}:")
    print(f"  APR Improvement: {apr_improvement:.2f}%")
    print(f"  Allocation Efficiency: {allocation_efficiency:.2f}")
    print(f"  Portfolio Diversity: {portfolio_diversity}")
```

## Limitations and Future Work

### Current Limitations

1. **Parameter Assumptions**:
   - Mean reversion and coefficient estimations assume relative stability
   - Historical data dependency for initial feature optimization
   - Fixed window size for temporal feature calculation

2. **Market Dynamics**:
   - APR changes may not follow strict mean reversion patterns
   - Signal dilution effects might vary across market conditions
   - Limited handling of extreme market events

3. **Computational Considerations**:
   - Training requires significant historical data
   - Real-time optimization with many subgraphs may be computationally intensive
   - Memory requirements grow with experience replay buffer size

### Future Improvements

1. **Enhanced RL Techniques**:
   - Integration of more sophisticated deep RL algorithms
   - Exploration of different network architectures
   - Implementation of prioritized experience replay

2. **Dynamic Optimization**:
   - Adaptive window sizes for temporal features
   - Dynamic reward scaling based on market conditions
   - Automatic hyperparameter tuning

3. **Risk Management**:
   - More sophisticated position sizing strategies
   - Integration of volatility-based risk measures
   - Advanced portfolio diversification techniques

### Practical Considerations

When implementing this agent in production:

1. **Data Requirements**:
   ```python
   # Minimum historical data needed
   min_history_window = 10  # For temporal features
   min_training_episodes = 100  # For initial policy learning
   ```

2. **Performance Optimization**:
   ```python
   # Batch processing for efficiency
   batch_size = min(64, len(subgraphs))  # Adjust based on number of subgraphs
   update_frequency = max(1, len(subgraphs) // 10)  # Regular policy updates
   ```

3. **Resource Management**:
   ```python
   # Memory management
   max_buffer_size = 10000  # Limit experience replay buffer
   cleanup_frequency = 1000  # Regular memory cleanup
   ```

## References

Implementation based on:
- Ning, B., & Lee, K. "Advanced Statistical Arbitrage with Reinforcement Learning"
  - State space design (Section 1)
  - Q-learning implementation (Section 2)
  - Action selection strategy (Section 3)
  - Training methodology (Section 6)
