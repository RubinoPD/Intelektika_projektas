import numpy as np


class QLearningAgent:
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

    def __init__(self, num_states, num_actions, learning_rate=0.1, discount_factor=0.9,
                 exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995):
        """Initialize the Q-learning agent."""
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

    def choose_action(self, state):
        """
        Choose an action using epsilon-greedy policy.

        Args:
            state (int): Current state

        Returns:
            int: Chosen action
        """
        try:
            # Konvertuojame state į tinkamą formatą
            if isinstance(state, dict) and 'agent_pos' in state:
                row, col = state['agent_pos']
                width = state['width']
                state_idx = int(row * width + col)
            else:
                state_idx = int(state)

            # Exploration: choose a random action
            if np.random.random() < self.exploration_rate:
                return np.random.randint(0, self.num_actions)
            # Exploitation: choose the best action
            else:
                return np.argmax(self.q_table[state_idx])
        except Exception as e:
            print(f"Klaida pasirenkant veiksmą: {e}")
            # Saugumo sumetimais grąžiname atsitiktinį veiksmą
            return np.random.randint(0, self.num_actions)

    def learn(self, state, action, reward, next_state, done):
        """
        Update Q-value for the given state-action pair.

        Args:
            state (int): Current state
            action (int): Action taken
            reward (float): Reward received
            next_state (int): Next state
            done (bool): Whether episode is finished
        """
        try:
            # Apdorojame state
            if isinstance(state, dict) and 'agent_pos' in state:
                row, col = state['agent_pos']
                width = state['width']
                state_idx = int(row * width + col)
            else:
                state_idx = int(state)

            # Apdorojame next_state - PATAISYTA DALIS
            if isinstance(next_state, dict) and 'agent_pos' in next_state:
                next_row, next_col = next_state['agent_pos']
                next_width = next_state['width']
                next_state_idx = int(next_row * next_width + next_col)
            else:
                next_state_idx = int(next_state)

            action = int(action)

            # Get current Q-value
            current_q = self.q_table[state_idx, action]

            # Get max Q-value for next state
            if done:
                max_next_q = 0
            else:
                max_next_q = np.max(self.q_table[next_state_idx])

            # Calculate new Q-value using the Q-learning formula
            new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)

            # Update Q-table
            self.q_table[state_idx, action] = new_q

        except Exception as e:
            print(f"Klaida atnaujinant Q-lentelę: {e}")
            print(f"State: {state}")
            print(f"Next state: {next_state}")
            print(f"Action: {action}")

    def decay_exploration(self):
        """Decay the exploration rate."""
        self.exploration_rate = max(self.min_exploration_rate, self.exploration_rate * self.exploration_decay)

    def save_episode_result(self, episode_reward, episode_steps):
        """Save the results of an episode."""
        self.rewards_per_episode.append(episode_reward)
        self.steps_per_episode.append(episode_steps)

    def save_q_table(self, filename="q_table.npy"):
        """Save the Q-table to a file."""
        np.save(filename, self.q_table)

    def load_q_table(self, filename="q_table.npy"):
        """Load the Q-table from a file."""
        self.q_table = np.load(filename)