import time
import numpy as np
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

        # Run episode
        while not done:
            # Choose action
            action = agent.choose_action(state)

            # Take action
            next_state, reward, done, _ = env.step(action)

            # Learn from experience
            agent.learn(state, action, reward, next_state, done)

            # Update state and total reward
            state = next_state
            total_reward += reward

            # Render if needed
            if render and visualizer and episode % render_freq == 0:
                visualizer.update(episode, total_reward)
                if not visualizer.check_events():
                    return # Exit if window closed

        # Update exploration rate
        agent.decay_exploration()

        # Save episode results
        agent.save_episode_result(total_reward, env.step_count)


        # Print episode info
        if episode % 10 == 0:
            avg_reward = np.mean(agent.rewards_per_episode[-10])
            print(f"Episode: {episode}, Average Reward (Last 10): {avg_reward:.2f}, Exploration Rate: {agent.exploration_rate:.4f}")

    # Save Q-table after training
    agent.save_q_table()




if __name__ == '__main__':
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
