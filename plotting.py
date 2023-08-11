import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Load rewards
with open('rewards/LR_0.001_DF_0.9_ED_0.99_RE_1000/1691677056_EN_499.pickle', 'rb') as f:
    reward_list1 = pickle.load(f)
with open('rewards/LR_0.001_DF_0.99_ED_0.99_RE_1000/1691681022_EN_312.pickle', 'rb') as f:
    reward_list2 = pickle.load(f)
with open('rewards/LR_0.001_DF_0.999_ED_0.99_RE_1000/1691725153_EN_488.pickle', 'rb') as f:
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
plt.legend(['DF=0.9', 'DF=0.99', 'DF=0.999'])
plt.show()