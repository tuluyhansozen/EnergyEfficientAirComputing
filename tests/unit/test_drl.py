import numpy as np
import pytest
import torch

from aircompsim.drl.actor_critic import ActorCriticAgent
from aircompsim.drl.ddqn import DDQNAgent
from aircompsim.drl.dqn import DQNAgent
from aircompsim.drl.replay_buffer import ReplayBuffer


class TestDRLComponents:
    @pytest.fixture
    def state_size(self):
        return 10

    @pytest.fixture
    def action_size(self):
        return 5

    @pytest.fixture
    def device(self):
        return "cpu"

    def test_replay_buffer(self):
        buffer = ReplayBuffer(capacity=100)
        state = np.zeros(10)
        next_state = np.zeros(10)

        # Test push
        buffer.push(state, 1, 1.0, next_state, False)
        assert len(buffer) == 1

        # Test sample
        states, actions, _, _, _ = buffer.sample(1)
        assert states.shape == (1, 10)
        assert len(actions) == 1

        # Test clear
        buffer.clear()
        assert len(buffer) == 0

    def test_dqn_agent_initialization(self, state_size, action_size, device):
        agent = DQNAgent(state_size, action_size, device=device)
        assert agent.state_size == state_size
        assert agent.action_size == action_size
        assert agent.epsilon == 1.0

    def test_dqn_agent_select_action(self, state_size, action_size, device):
        agent = DQNAgent(state_size, action_size, device=device)
        state = np.zeros(state_size)

        # Test random action (exploration)
        agent.epsilon = 1.0
        action = agent.get_action_exploration(state)
        assert 0 <= action < action_size

        # Test greedy action (exploitation)
        agent.epsilon = 0.0
        action = agent.get_action_exploration(state)
        assert 0 <= action < action_size

    def test_dqn_agent_learning(self, state_size, action_size, device):
        agent = DQNAgent(state_size, action_size, device=device)
        state = np.zeros(state_size)
        next_state = np.zeros(state_size)

        # Fill buffer enough to sample
        for _ in range(agent.batch_size + 5):
            agent.replay_buffer.push(state, 1, 1.0, next_state, False)

        loss = agent.learn(state, 1, 1.0, next_state, False)
        assert loss is not None
        assert isinstance(loss, float)

    def test_dqn_save_load(self, state_size, action_size, tmp_path, device):
        agent = DQNAgent(state_size, action_size, device=device)
        save_path = tmp_path / "dqn_agent.pth"

        agent.save(str(save_path))
        assert save_path.exists()

        agent_new = DQNAgent(state_size, action_size, device=device)
        agent_new.load(str(save_path))

        # Check if weights loaded (basic check)
        for p1, p2 in zip(agent.network.parameters(), agent_new.network.parameters()):
            assert torch.equal(p1, p2)

    def test_ddqn_agent_lifecycle(self, state_size, action_size, device):
        agent = DDQNAgent(state_size, action_size, device=device)
        state = np.zeros(state_size)
        next_state = np.zeros(state_size)

        # Select action
        action = agent.select_action(state)
        assert 0 <= action < action_size

        # Fill buffer
        for _ in range(agent.batch_size + 5):
            agent.replay_buffer.push(state, 1, 1.0, next_state, False)

        # Learn
        loss = agent.learn(state, 1, 1.0, next_state, False)
        assert loss is not None

    def test_actor_critic_lifecycle(self, state_size, action_size, device):
        agent = ActorCriticAgent(state_size, action_size, device=device)
        state = np.zeros(state_size)
        next_state = np.zeros(state_size)

        # Select action
        action = agent.select_action(state)
        assert 0 <= action < action_size

        # Learn (AC learns every step usually)
        loss = agent.learn(state, action, 1.0, next_state, False)
        assert loss is not None
