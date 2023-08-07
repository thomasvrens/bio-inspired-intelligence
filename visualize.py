import gymnasium as gym
import tensorflow as tf

from Agent import DDQNAgent

env = gym.make('LunarLander-v2', render_mode='human')

agent = DDQNAgent(env.observation_space.shape, env.action_space.n)

# Load model and set epsilon to 0
model_name = 'models/1691423099.model'
agent.load_model(model_name)

# Disable Tensorflow logging for cleaner output
tf.keras.utils.disable_interactive_logging()

SHOW_EPISODES = 20

for episode in range(SHOW_EPISODES):

    episode_reward = 0

    print(f'Episode: {episode}')

    cur_state, _ = env.reset()
    
    done = False
    while not done:
        action = agent.act(cur_state)
        new_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        episode_reward += reward
        cur_state = new_state

    
    print(f'Reward: {episode_reward}')