import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf


from Agent import DDQNAgent

tf.config.set_visible_devices([], 'GPU')

# HYPERPARAMETERS
EPISODES = 150
SHOW_RENDER_EVERY = 20000
SHOW_STATS_EVERY = 10

env = gym.make('LunarLander-v2')

agent = DDQNAgent(env.observation_space.shape, env.action_space.n)
reward_list = []

for episode in range(EPISODES):
    episode_reward = 0

    print(f'Episode: {episode}')

    cur_state, _ = env.reset()
    
    done = False
    while not done:
        action = agent.act(cur_state)
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        episode_reward += reward

        agent.add_memory((cur_state, action, reward, new_state, done))
        agent.train()

        cur_state = new_state
    
    reward_list.append(episode_reward)

    agent.decrease_epsilon()
    agent.increase_target_model_counter()

    if episode % SHOW_STATS_EVERY == 0:
        avg_reward = np.average(reward_list[-SHOW_STATS_EVERY:])
        max_reward = np.max(reward_list[-SHOW_STATS_EVERY:])
        min_reward = np.min(reward_list[-SHOW_STATS_EVERY:])
        print(f'Episode: {episode}, avg: {avg_reward}, max: {max_reward}, min: {min_reward}, epsilon: {agent.epsilon}')

window_size = 20
reward_series = pd.Series(reward_list)
rol_average = reward_series.rolling(window_size).mean()
plt.plot(reward_list)
plt.plot(rol_average)
plt.show()