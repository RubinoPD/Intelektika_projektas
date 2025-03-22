import numpy as np


class GridWorld:
    """
    A grid-based environment with obstacles, rewards, and penalties.

    Attributes:
        width (int): Width of the grid
        height (int): Height of the grid
        grid (np.array): 2D array representing the grid environment
        start_pos (tuple): Starting position (row, col)
        goal_pos (tuple): Goal position (row, col)
        agent_pos (tuple): Current agent position (row, col)
        done (bool): Whether episode is finished
        max_steps (int): Maximum steps per episode
        step_count (int): Current step count
    """

    # Cell types
    EMPTY = 0
    OBSTACLE = 1
    GOAL = 2
    PENALTY = 3

    # Rewards
    STEP_REWARD = -0.1 # pakeisti is -0.1 i -0.01 greitesniam
    GOAL_REWARD = 10.0 # padidinti atlygi is 1.0 iki 10.0 jei norisi greitesnio mokymosi
    OBSTACLE_PENALTY = -1.0
    PENALTY_REWARD = -0.5

    def __init__(self, width=10, height=10, obstacle_density=0.2, penalty_density=0.1, max_steps=100):
        """Initialize the grid world environment."""
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.grid = np.zeros((height, width), dtype=int)
        self.step_count = 0

        # Generate random obstacles
        num_obstacles = int(width * height * obstacle_density)
        print(f"Kuriama {num_obstacles} kliūčių")
        obstacle_count = 0
        for _ in range(num_obstacles):
            row, col = np.random.randint(0, height), np.random.randint(0, width)
            if (row, col) != (0, 0) and (row, col) != (height - 1, width - 1):  # Nestatome kliūčių pradžioje ir tiksle
                self.grid[row, col] = self.OBSTACLE
                obstacle_count += 1
        print(f"Sukurta {obstacle_count} kliūčių")

        # Generate random penalties
        num_penalties = int(width * height * penalty_density)
        print(f"Kuriama {num_penalties} baudų")
        penalty_count = 0
        for _ in range(num_penalties):
            row, col = np.random.randint(0, height), np.random.randint(0, width)
            if self.grid[row, col] == self.EMPTY and (row, col) != (0, 0) and (row, col) != (height - 1, width - 1):
                self.grid[row, col] = self.PENALTY
                penalty_count += 1
        print(f"Sukurta {penalty_count} baudų")

        # Set start position (top-left corner)
        self.start_pos = (0, 0)
        self.grid[self.start_pos] = self.EMPTY

        # Set goal position (bottom-right corner)
        self.goal_pos = (height - 1, width - 1)
        self.grid[self.goal_pos] = self.GOAL

        # Initialize agent position
        self.agent_pos = self.start_pos
        self.done = False

    def reset(self):
        """Reset the environment for a new episode."""
        self.agent_pos = self.start_pos
        self.done = False
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        """Convert agent position to state representation."""
        return int(self.agent_pos[0] * self.width + self.agent_pos[1])

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): 0: up, 1: right, 2: down, 3: left

        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Increase step count
        self.step_count += 1

        # Get current position
        row, col = self.agent_pos

        # Determine next position based on action
        if action == 0:  # Up
            next_row, next_col = max(0, row - 1), col
        elif action == 1:  # Right
            next_row, next_col = row, min(self.width - 1, col + 1)
        elif action == 2:  # Down
            next_row, next_col = min(self.height - 1, row + 1), col
        elif action == 3:  # Left
            next_row, next_col = row, max(0, col - 1)
        else:
            raise ValueError(f"Invalid action: {action}")

        # Check if hitting an obstacle
        if self.grid[next_row, next_col] == self.OBSTACLE:
            reward = self.OBSTACLE_PENALTY
            next_row, next_col = row, col  # Stay in place
        # Check if reaching the goal
        elif self.grid[next_row, next_col] == self.GOAL:
            reward = self.GOAL_REWARD
            self.done = True
        # Check if stepping on a penalty
        elif self.grid[next_row, next_col] == self.PENALTY:
            reward = self.PENALTY_REWARD
        # Empty cell
        else:
            reward = self.STEP_REWARD

        # Update agent position
        self.agent_pos = (next_row, next_col)

        # Check if maximum steps reached
        if self.step_count >= self.max_steps:
            self.done = True

        # Return next state, reward, done, and info
        return self._get_state(), reward, self.done, {}

    def get_num_states(self):
        """Get the total number of states in the environment."""
        return self.width * self.height

    def get_num_actions(self):
        """Get the total number of actions in the environment."""
        return 4  # Up, Right, Down, Left