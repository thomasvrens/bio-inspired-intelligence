'''
    Script for plotting the sensitivity analysis.

    Author: Thomas
'''

import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Load rewards
with open('rewards/LR_0.0001_DF_0.99_ED_0.99_RE_1000/1691920052_EN_499.pickle', 'rb') as f:
    reward_list1 = pickle.load(f)
with open('rewards/LR_0.001_DF_0.99_ED_0.99_RE_1000/1691858365_EN_283.pickle', 'rb') as f:
    reward_list2 = pickle.load(f)
with open('rewards/LR_0.01_DF_0.99_ED_0.99_RE_1000/1692786992_EN_398.pickle', 'rb') as f:
   reward_list3 = pickle.load(f)

window_size = 10
rol_average1 = pd.Series(reward_list1).rolling(window_size).mean()
rol_average2 = pd.Series(reward_list2).rolling(window_size).mean()
rol_average3 = pd.Series(reward_list3).rolling(window_size).mean()

# Plot rewards
plt.plot(rol_average1)
plt.plot(rol_average2)
plt.plot(rol_average3)
plt.grid()
plt.legend(['$\\alpha = 0.0001$', '$\\alpha = 0.001$', '$\\alpha = 0.01$'])
plt.xlabel('Episode')
plt.ylabel('Average reward (10 episodes)')
plt.show()