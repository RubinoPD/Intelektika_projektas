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

    def __init__(self, env, agent, cell_size=80):
        """Initialize the visualizer."""
        self.env = env
        self.agent = agent
        self.cell_size = cell_size

        # Calculate window dimensions
        self.width = env.width * cell_size
        self.height = env.height * cell_size + 200  # Extra space for stats

        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Q-Learning Grid World")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont(None, 24)

        # Define colors - naudojame ryškesnes spalvas
        self.colors = {
            'white': (255, 255, 255),
            'black': (0, 0, 0),
            'red': (255, 50, 50),  # Ryškesnė raudona
            'green': (50, 255, 50),  # Ryškesnė žalia
            'blue': (50, 50, 255),  # Ryškesnė mėlyna
            'yellow': (255, 255, 0),
            'gray': (150, 150, 150),
            'darkgray': (64, 64, 64),
            'lightblue': (200, 230, 255)  # Šviesesnė mėlyna
        }

    def draw_grid(self):
        """Draw the grid world."""
        # Clear the screen
        self.screen.fill(self.colors['white'])

        # Draw grid cells
        for row in range(self.env.height):
            for col in range(self.env.width):
                rect = pygame.Rect(col * self.cell_size, row * self.cell_size,
                                   self.cell_size, self.cell_size)

                # Draw cell based on type
                if (row, col) == self.env.agent_pos:
                    pygame.draw.rect(self.screen, self.colors['blue'], rect)
                elif (row, col) == self.env.goal_pos:
                    pygame.draw.rect(self.screen, self.colors['green'], rect)
                elif self.env.grid[row, col] == self.env.OBSTACLE:
                    pygame.draw.rect(self.screen, self.colors['black'], rect)
                elif self.env.grid[row, col] == self.env.PENALTY:
                    pygame.draw.rect(self.screen, self.colors['red'], rect)
                else:
                    pygame.draw.rect(self.screen, self.colors['lightblue'], rect)

                # Draw cell border - SVARBU: piešiame tik kraštą, ne visą stačiakampį
                pygame.draw.rect(self.screen, self.colors['gray'], rect, 1)

    def draw_stats(self, episode, total_reward, exploration_rate):
        """Draw statistics."""
        # Draw stats background
        stats_rect = pygame.Rect(0, self.env.height * self.cell_size,
                                 self.width, 200)
        pygame.draw.rect(self.screen, self.colors['white'], stats_rect)

        # Draw episode info
        episode_text = self.font.render(f"Episode: {episode}", True, self.colors['black'])
        self.screen.blit(episode_text, (10, self.env.height * self.cell_size + 10))

        # Draw reward info
        reward_text = self.font.render(f"Total Reward: {total_reward:.2f}", True, self.colors['black'])
        self.screen.blit(reward_text, (10, self.env.height * self.cell_size + 40))

        # Draw exploration rate
        epsilon_text = self.font.render(f"Exploration Rate: {exploration_rate:.4f}", True, self.colors['black'])
        self.screen.blit(epsilon_text, (10, self.env.height * self.cell_size + 70))

        # Draw agent position
        pos_text = self.font.render(f"Agent Position: {self.env.agent_pos}", True, self.colors['black'])
        self.screen.blit(pos_text, (10, self.env.height * self.cell_size + 100))

        # Draw learning curve if enough episodes
        if len(self.agent.rewards_per_episode) > 1:
            self.draw_learning_curve()

    def draw_learning_curve(self):
        """Draw the learning curve using matplotlib."""
        try:
            # Create figure and plot
            fig, ax = plt.subplots(figsize=(4, 2), dpi=80)

            # Plottuojame visus atlygius arba paskutinius 100, jei jų daugiau
            rewards_to_plot = self.agent.rewards_per_episode[-min(100, len(self.agent.rewards_per_episode)):]
            ax.plot(rewards_to_plot)
            ax.set_title('Rewards (Last 100 Episodes)')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Total Reward')

            # Convert plot to pygame surface
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            renderer = canvas.get_renderer()
            raw_data = renderer.tostring_rgb()
            size = canvas.get_width_height()

            # Create pygame surface from raw data
            surf = pygame.image.fromstring(raw_data, size, "RGB")
            self.screen.blit(surf, (self.width // 2, self.env.height * self.cell_size + 10))

            # Close figure to prevent memory leak
            plt.close(fig)
        except Exception as e:
            print(f"Klaida piešiant mokymosi kreivę: {e}")

    def update(self, episode, total_reward):
        """Update the visualization."""
        print(f"Atnaujinama vizualizacija. Agento pozicija: {self.env.agent_pos}")
        self.draw_grid()
        self.draw_stats(episode, total_reward, self.agent.exploration_rate)
        pygame.display.flip()
        self.clock.tick(5)  # Sumažiname kadrų dažnį, kad geriau matytųsi

    def check_events(self):
        """Check for Pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def close(self):
        """Close the visualizer."""
        pygame.quit()