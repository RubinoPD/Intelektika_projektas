import numpy as np

class QLearnAgent:
    """
    Q-learning agent that learns to navigate the grid world.

    Attributes:
        num_states (int): Number of states in the environment
        num_actions (int): Number of actions available
        q_table (np.array): Q-value table
        learning_rate (float): Learning rate alpha
        discount_factor (float): Discount factor gamma
        exploration_rate (float): Exploration rate epsilon
        min_exploration_rate (float): Minimum exploration rate
        exploration_decay (float): Decay rate for exploration
    """

    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        """
        Initialize the Q-learning agent.
        """
        self.num_states = num_states
        self.num_actions = num_actions

        # Initialize Q-table with zeros
        self.q_table = np.zeros((num_states, num_actions))

        # Learning parameters
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.min_exploration_rate = min_exploration_rate
        self.exploration_decay = exploration_decay

        # Training statistics
        self.rewards_per_episode = []
        self.steps_per_episode = []