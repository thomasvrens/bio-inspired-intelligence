'''
    Script for plotting the sensitivity analysis.

    Author: Thomas
'''

import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Load rewards
with open('rewards/LR_0.001_DF_0.99_ED_0.99_RE_1000/1691858365_EN_283.pickle', 'rb') as f:
    reward_list = pickle.load(f)

window_size = 10
rol_average = pd.Series(reward_list).rolling(window_size).mean()

# Plot rewards
plt.plot(reward_list)
plt.plot(rol_average)
plt.legend(['Reward', '10-episode average'])
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.grid()
plt.show()