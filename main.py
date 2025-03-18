import time
import numpy as np
import pygame.time

from environment import GridWorld
from agent import QLearningAgent
from visualization import GridWorldVisualizer

def train(env, agent, visualizer=None, num_episodes=1000, render=True, render_freq=1):
    """
    Train the agent on the environment.

    Args:
        env (GridWorld): Environment
        agent (QLearningAgent): Agent
        visualizer (GridWorldVisualizer): Visualizer
        num_episodes (int): Number of episodes to train
        render (bool): Whether to render the environment
        render_freq (int): How often to render (in episodes)
    """

    for episode in range(num_episodes):
        # Reset environment and get initial state
        state = env.reset()
        done = False
        total_reward = 0
        step_count = 0

        print(f"Epizodas {episode}: Pradžios būsena: {state}")

        # Run episode
        while not done:
            # Choose action
            action = agent.choose_action(state)
            print(f"  Žingsnis {step_count}: Būsena {state}, Veiksmas {action}")

            # Take action
            next_state, reward, done, _ = env.step(action)
            print(f"  Rezultatas: Nauja būsena {next_state}, Atlygis {reward}, Baigta {done}")

            # Learn from experience
            agent.learn(state, action, reward, next_state, done)

            # Update state and total reward
            state = next_state
            total_reward += reward
            step_count =+ 1

            # Render if needed
            if render and visualizer and episode % render_freq == 0:
                visualizer.update(episode, total_reward)
                pygame.time.delay(200) # 200ms pauze
                if not visualizer.check_events():
                    return # Exit if window closed

        # Update exploration rate
        agent.decay_exploration()

        # Save episode results
        agent.save_episode_result(total_reward, env.step_count)


        # Print episode info
        if episode % 10 == 0:
            recent_rewards = agent.rewards_per_episode[-min(10, len(agent.rewards_per_episode)):]
            avg_reward = np.mean(recent_rewards) if recent_rewards else 0
            print(f"Episode: {episode}, Average Reward (Last 10): {avg_reward:.2f}, Exploration Rate: {agent.exploration_rate:.4f}")

    # Save Q-table after training
    agent.save_q_table()


def main():
    """
    Main function
    """
    # Create environment and agent
    env = GridWorld(width=10, height=10, obstacle_density=0.2, penalty_density=0.1, max_steps=100)
    agent = QLearningAgent(num_states=env.get_num_states(), num_actions=env.get_num_actions(), learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, min_exploration_rate=0.01, exploration_decay=0.995)

    # Create visualizer
    visualizer = GridWorldVisualizer(env, agent, cell_size=100)

    # Train agent
    train(env, agent, visualizer, num_episodes=500, render=True, render_freq=1)

    # Close visualizer
    visualizer.close()

if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
