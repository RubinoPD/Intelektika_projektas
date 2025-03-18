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

    def draw_grid(self):
        """
        Draw the grid world
        """
        # Clear the screen
        self.screen.fill(self.colors['white'])

        # Draw grid cells
        for row in range(self.env.height):
            for col in range(self.env.width):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size)

                # Draw cell based on type
                if(row, col) == self.env.agent_pos:
                    pygame.draw.rect(self.screen, self.colors['blue'], rect)
                elif (row, col) == self.env.goal_pos:
                    pygame.draw.rect(self.screen, self.colors['green'], rect)
                elif self.env.grid[row, col] == self.env.OBSTACLE:
                    pygame.draw.rect(self.screen, self.colors['black'], rect)
                elif self.env.grid[row, col] == self.env.PENALTY:
                    pygame.draw.rect(self.screen, self.colors['red'], rect)
                else:
                    pygame.draw.rect(self.screen, self.colors['lightblue'], rect)

                # Draw cell border
                pygame.draw.rect(self.screen, self.colors['gray'], rect)

    def draw_stats(self, episode, total_reward, exploration_rate):
        """
        Draw the statistics
        """
        # Draw stats background
        stats_rect = pygame.Rect(0, self.env.height * self.cell_size, self.width, 200)
        pygame.draw.rect(self.screen, self.colors['white'], stats_rect)

        # Draw episode info
        episode_text = self.font.render(f"Episode: {episode}", True, self.colors['black'])
        self.screen.blit(episode_text, (10, self.env.height * self.cell_size + 10))

        # Draw reward info
        reward_text = self.font.render(f"Total reward: {total_reward:.2f}", True, self.colors['black'])
        self.screen.blit(reward_text, (10, self.env.height * self.cell_size + 40))

        # Draw exploration rate
        epsilon_text = self.font.render(f"Exploration rate: {exploration_rate:.4f}", True, self.colors['black'])
        self.screen.blit(epsilon_text, (10, self.env.height * self.cell_size + 70))

        # Draw learning curve if enough episodes
        if len(self.agent.rewards_per_episode) > 1:
            self.draw_learning_curve()