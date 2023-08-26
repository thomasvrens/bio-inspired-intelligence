import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Load timesteps
with open('episode_steps/LR_0.001_DF_0.99_ED_0.99_RE_1000/1692959546_EN_267.pickle', 'rb') as f:
    episode_steps1 = pickle.load(f)

# Load rewards
with open('rewards/LR_0.001_DF_0.99_ED_0.99_RE_1000/1692959546_EN_267.pickle', 'rb') as f:
    reward_list1 = pickle.load(f)

window_size = 10
rol_average1 = pd.Series(reward_list1).rolling(window_size).mean()

# plot rewards and episode steps on different scales
fig, ax1 = plt.subplots()

ax1.plot(episode_steps1, color='tab:blue')
ax1.set_xlabel('Episode')
ax1.set_ylabel('Timesteps', color='tab:blue')
ax1.tick_params(axis='y', labelcolor='tab:blue')

ax2 = ax1.twinx()

ax2.plot(rol_average1, color='tab:orange')
ax2.set_ylabel('Rewards', color='tab:orange')
ax2.tick_params(axis='y', labelcolor='tab:orange')

plt.grid()
plt.show()
