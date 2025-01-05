import pytest
import numpy as np
import torch
from rl_curation_agent import DuelingDQN, ReplayBuffer, RLCurationAgent

def test_dueling_dqn_architecture():
    """Test the Dueling DQN network architecture."""
    n_subgraphs = 2
    features_per_subgraph = 7  # 6 base features + 1 combined signal
    state_size = n_subgraphs * features_per_subgraph
    action_size = 2
    network = DuelingDQN(state_size, action_size)
    
    # Test forward pass
    batch_size = 4
    state = torch.randn(batch_size, state_size)
    output = network(state)
    
    # Check output shape
    assert output.shape == (batch_size, action_size)
    
    # Check value and advantage streams
    features = network.feature_layer(state)
    value = network.value_stream(features)
    advantages = network.advantage_stream(features)
    
    assert value.shape == (batch_size, 1)
    assert advantages.shape == (batch_size, action_size)

def test_replay_buffer():
    """Test the experience replay buffer."""
    buffer = ReplayBuffer(capacity=100)
    
    # Test adding experiences
    state = np.array([1, 2, 3, 4])
    next_state = np.array([2, 3, 4, 5])
    action = 1
    reward = 0.5
    done = False
    
    buffer.add(state, action, reward, next_state, done)
    assert len(buffer) == 1
    
    # Test sampling
    for _ in range(9):  # Add 9 more experiences
        buffer.add(state, action, reward, next_state, done)
    
    batch_size = 4
    states, actions, rewards, next_states, dones = buffer.sample(batch_size)
    
    assert states.shape == (batch_size, len(state))
    assert actions.shape == (batch_size,)
    assert rewards.shape == (batch_size,)
    assert next_states.shape == (batch_size, len(next_state))
    assert dones.shape == (batch_size,)

def test_state_representation():
    """Test state representation generation."""
    n_subgraphs = 2
    features_per_subgraph = 7  # 6 base features + 1 combined signal
    agent = RLCurationAgent(
        state_size=n_subgraphs * features_per_subgraph,
        action_size=2,
        batch_size=4,
        history_window=2  # Small window for testing
    )
    
    # Test data
    data = {
        'new_signals': [
            {
                'ipfs_hash': 'hash1',
                'signal_amount': 100.0,
                'apr': 10.0,
                'weekly_queries': 1000,
                'total_earnings': 50.0
            },
            {
                'ipfs_hash': 'hash2',
                'signal_amount': 200.0,
                'apr': 20.0,
                'weekly_queries': 2000,
                'total_earnings': 100.0
            }
        ]
    }
    
    state = agent.get_state(data)
    
    # Check state properties
    assert isinstance(state, np.ndarray)
    assert state.dtype == np.float32
    assert len(state.shape) == 1  # Flattened state vector
    
    # With preprocessing, exact values will be normalized
    # Just check the shape is correct (7 features per subgraph: 6 base + 1 combined)
    expected_features = n_subgraphs * features_per_subgraph
    assert state.shape == (expected_features,)

def test_action_selection():
    """Test action selection mechanism."""
    n_subgraphs = 2
    features_per_subgraph = 7
    agent = RLCurationAgent(
        state_size=n_subgraphs * features_per_subgraph,
        action_size=2,
        batch_size=4,
        learning_rate=1e-3,
        epsilon_decay=0.9  # Faster exploration decay for testing
    )
    state = np.zeros(n_subgraphs * features_per_subgraph)
    
    # Test epsilon-greedy behavior
    agent.epsilon = 0.0  # Force exploitation
    action1 = agent.select_action(state)
    assert isinstance(action1, int)
    assert 0 <= action1 < agent.action_size
    
    agent.epsilon = 1.0  # Force exploration
    action2 = agent.select_action(state)
    assert isinstance(action2, int)
    assert 0 <= action2 < agent.action_size

def test_reward_calculation():
    """Test reward calculation."""
    n_subgraphs = 2
    features_per_subgraph = 7
    agent = RLCurationAgent(
        state_size=n_subgraphs * features_per_subgraph,
        action_size=2,
        batch_size=4,
        learning_rate=1e-3,
        epsilon_decay=0.9
    )
    
    # Test states with known APR changes and temporal features
    old_state = np.array([
        10.0, 100.0, 1000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,  # First subgraph features + temporal
        20.0, 200.0, 2000, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0   # Second subgraph features + temporal
    ])
    new_state = np.array([
        15.0, 100.0, 1000, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0,  # First subgraph features + temporal
        25.0, 200.0, 2000, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0   # Second subgraph features + temporal
    ])
    
    reward = agent.calculate_reward(old_state, new_state)
    
    # Expected reward calculation:
    # APR changes: (15-10) = 5, (25-20) = 5
    # Rewards: (5 * 5) + (5 * 5) = 50
    # APR bonus: max(15, 25) - min(15, 25) = 10
    # Total reward: mean([25, 25]) + 10 ≈ 31.67
    assert reward == pytest.approx(31.67, rel=1e-2)

def test_end_to_end_optimization():
    """Test end-to-end signal optimization."""
    # Test data
    data = {
        'new_signals': [
            {
                'ipfs_hash': 'hash1',
                'signal_amount': 100.0,
                'apr': 10.0,
                'weekly_queries': 1000,
                'total_earnings': 50.0
            },
            {
                'ipfs_hash': 'hash2',
                'signal_amount': 200.0,
                'apr': 20.0,
                'weekly_queries': 2000,
                'total_earnings': 100.0
            }
        ]
    }
    
    n_subgraphs = 2
    features_per_subgraph = 7
    agent = RLCurationAgent(
        state_size=n_subgraphs * features_per_subgraph,
        action_size=2,
        batch_size=4,
        learning_rate=1e-3,
        epsilon_decay=0.9,
        history_window=2
    )
    
    # Test optimization with expanded action space
    result_df = agent.optimize_signals(
        data,
        total_new_signal=1000.0,
        min_allocation=10.0,
        training=True
    )
    
    # Additional checks for new action space behavior
    total_allocated = result_df['new_signal'].sum()
    assert total_allocated == pytest.approx(1000.0, rel=1e-5)
    
    # Check that at least one subgraph gets significant allocation (>30%)
    assert any(result_df['allocation_percentage'] > 30.0)
    
    # Verify results
    assert len(result_df) == 2
    assert result_df['new_signal'].sum() == pytest.approx(1000.0, rel=1e-5)
    assert all(result_df['new_signal'] >= 10.0)  # Check minimum allocation
    assert all(result_df['allocation_percentage'] >= 1.0)  # Check percentage constraints
    assert all(result_df['apr'] <= result_df['original_apr'])  # Check APR dilution

def test_training_loop():
    """Test training loop and model updates."""
    n_subgraphs = 2
    features_per_subgraph = 7
    agent = RLCurationAgent(
        state_size=n_subgraphs * features_per_subgraph,
        action_size=2,
        batch_size=4,
        learning_rate=1e-3,
        epsilon_decay=0.9,
        target_update=5,  # More frequent target network updates
        history_window=2
    )
    
    # Initial network state
    initial_params = [param.clone().detach() for param in agent.policy_net.parameters()]
    
    # Training data with larger APR differences to generate meaningful rewards
    data = {
        'new_signals': [
            {
                'ipfs_hash': 'hash1',
                'signal_amount': 100.0,
                'apr': 100.0,  # Higher APR
                'weekly_queries': 1000,
                'total_earnings': 50.0
            },
            {
                'ipfs_hash': 'hash2',
                'signal_amount': 200.0,
                'apr': 20.0,  # Lower APR
                'weekly_queries': 2000,
                'total_earnings': 100.0
            }
        ]
    }
    
    # Run more optimization steps with smaller batches
    for _ in range(50):  # More iterations to ensure learning
        agent.optimize_signals(data, total_new_signal=1000.0, min_allocation=10.0, training=True)
    
    # Check if network parameters have been updated
    current_params = [param.clone().detach() for param in agent.policy_net.parameters()]
    
    # Verify that at least some parameters have changed
    params_changed = False
    for initial, current in zip(initial_params, current_params):
        if not torch.allclose(initial, current):
            params_changed = True
            break
    
    assert params_changed, "Network parameters should update during training"

if __name__ == "__main__":
    pytest.main([__file__])
