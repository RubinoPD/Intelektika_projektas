import pygame
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg

class GridWorldVisualizer:
    """
    Visualizer for the grid world environment using Pygame.

    Attributes:
        env (GridWorld): Grid world environment
        agent (QLearningAgent): Q-learning agent
        cell_size (int): Size of each cell in pixels
        width (int): Width of the window in pixels
        height (int): Height of the window in pixels
        screen (pygame.Surface): Pygame screen
        clock (pygame.time.Clock): Pygame clock
        font (pygame.font.Font): Pygame font
        colors (dict): Dictionary of colors
    """

    def __init__(self, env, agent, cell_size):
        """
        Initialize the visualiser
        """
        self.env = env
        self.agent = agent
        self.cell_size = cell_size

        # Calculate window dimensions
        self.width = env.width * cell_size
        self.height = env.height * cell_size + 200 # Extra space for stats

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        # Define colors
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 0, 0),
            'green': (0, 255, 0),
            'blue': (0, 0, 255),
            'yellow': (255, 255, 0),
            'gray': (128, 128, 128),
            'darkgray': (64, 64, 64),
            'lightblue': (173, 216, 230)
        }


