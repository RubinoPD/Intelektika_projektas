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
    STEP_REWARD = -0.1
    GOAL_REWARD = 1.0
    OBSTACLE_PENALTY = -1.0
    RENALTY_REWARD = -0.5

    def __init__(self, width=10, height=10, obstacle_density=0.2, penalty_density=0.1, max_steps=100):
        """ Initialize the grid world environment """
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.grid = np.zeros((height, width), dtype=int)
        self.step_count = 0

        # Generate random obstacles
        num_penalties = int(width * height * penalty_density)
        for _ in range(num_penalties):
            row, col = np.random.randint(0, height), np.random.randint(0, width)
            if self.grid[row, col] == self.EMPTY:
                self.grid[row, col] == self.PENALTY

        # Set start positionn (top-left corner)
        self.start_pos = (0, 0)
        self.grid[self.start_pos] = self.EMPTY

        # Set goal position (bottom-right corner)
        self.goal_pos = (height - 1, width - 1)
        self.grid[self.goal_pos] = self.GOAL

        # Initialize agent position
        self.agent_pos = self.start_pos
        self.done = False


    def reset(self):

