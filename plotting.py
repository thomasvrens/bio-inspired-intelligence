import matplotlib.pyplot as plt
import pickle

# Load rewards
with open('sims/1691423099/1620994500.pkl', 'rb') as f:
    reward_list = pickle.load(f)

# Plot rewards
plt.plot(reward_list)