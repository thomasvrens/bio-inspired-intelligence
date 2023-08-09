import os
import contextlib
import time

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from Agent import DDQNAgent

# Disable GPU
tf.config.set_visible_devices([], 'GPU')
# Disable Tensorflow logging for cleaner output
tf.keras.utils.disable_interactive_logging()

# HYPERPARAMETERS
EPISODES = 500
SHOW_STATS_EVERY = 10
EPISODE_TIME_WINDOW = 10
TRAIN_EVERY = 4
MAX_STEPS = 500

env = gym.make('LunarLander-v2')

agent = DDQNAgent(env.observation_space.shape, env.action_space.n)
reward_list = []
episode_times = []

# If interupted early by user, model is still saved
try:
    for episode in range(EPISODES):
        episode_reward = 0
        start_time = time.time()

        cur_state, _ = env.reset()
        step_count = 1
        
        done = False
        while not done:
            
            action = agent.act(cur_state)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            
            agent.add_memory((cur_state, action, reward, new_state, done))
            
            if step_count % TRAIN_EVERY == 0:
                agent.train()

            cur_state = new_state

            step_count += 1
            if step_count > MAX_STEPS:
                done = True
        
        end_time = time.time()
        episode_time = end_time - start_time
        episode_times.append(episode_time)
        avg_time = np.average(episode_times[-EPISODE_TIME_WINDOW:])
        print(f'Episode: {episode}/{EPISODES}, reward: {episode_reward:.0f}, time: {episode_time:.1f} [s], steps: {step_count}, time per step: {(episode_time / step_count):.3f} [s]')
        print(f'Average episode time ({EPISODE_TIME_WINDOW}eps): {avg_time:.1f} [s], ETA: {avg_time * (EPISODES - episode) / 60:.2f} [min]')

        reward_list.append(episode_reward)

        # Bring back epsilon
        if episode % agent.reset_epsilon_every == 0 and episode != 0:
            agent.reset_epsilon()
        
        agent.increase_target_model_counter()
        agent.decrease_epsilon()

        if episode % SHOW_STATS_EVERY == 0:
            avg_reward = np.average(reward_list[-SHOW_STATS_EVERY:])
            max_reward = np.max(reward_list[-SHOW_STATS_EVERY:])
            min_reward = np.min(reward_list[-SHOW_STATS_EVERY:])
            print(f'\nEpisode: {episode}, avg: {avg_reward:.0f}, max: {max_reward:.0f}, min: {min_reward:.0f}, epsilon: {agent.epsilon:.4f}\n')

except KeyboardInterrupt:
    print('Interrupted by user')
    pass

# save model
agent.save_model()
# TODO: save hyperparameters and learning curve


window_size = 20
reward_series = pd.Series(reward_list)
rol_average = reward_series.rolling(window_size).mean()
plt.plot(reward_list)
plt.plot(rol_average)
plt.show()