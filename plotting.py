import matplotlib.pyplot as plt
import pickle

# Load rewards
with open('rewards/LR_0.001_DF_0.9925_ED_0.99_RE_250/1691585660_EN_195.pickle', 'rb') as f:
    reward_list = pickle.load(f)

# Plot rewards
plt.plot(reward_list)
plt.show()