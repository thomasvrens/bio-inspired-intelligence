import gymnasium as gym
import tensorflow as tf

from Agent import DQNAgent
from keras.optimizers import Adam

env = gym.make('LunarLander-v2', render_mode='human')

agent = DQNAgent(env.observation_space.shape, env.action_space.n)

# Load model and set epsilon to 0
model_name = 'models/SOLVED_LR_0.001_DF_0.99_ED_0.99_RE_1000/1692967006_EN_351.h5'
agent.load_model(model_name)
agent.epsilon = 0

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