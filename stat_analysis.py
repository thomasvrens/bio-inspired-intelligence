'''
Script for performing statistical analysis.
Plots some learning runs and calculates statistics used in the report.

Author: Thomas
(Written largely by Github Copilot)
'''

import os
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Set the directory path
dir_path = 'rewards/LR_0.001_DF_0.99_ED_0.99_RE_1000'

# Get a list of all pickle files in the directory
files = os.listdir(dir_path)
pickle_files = [f for f in files if f.endswith('.pickle')]

# Load the data from the pickle files
data = []
for file in pickle_files:
    with open(os.path.join(dir_path, file), 'rb') as f:
        rewards = pickle.load(f)
        data.append(rewards)

# Calculate the 10-episode rolling average for each series
rol_average = [np.convolve(rewards, np.ones(10)/10, mode='valid') for rewards in data]

# Calculate the x-values for the plot
x = np.arange(len(rol_average[0]))

# Plot the first 5 10-episode rolling average reward series
for i in range(5):
    plt.plot(rol_average[i])

# Add labels and title
plt.xlabel('Episode')
plt.ylabel('Average Reward (10 episodes)')

# Show the plot
plt.grid()
plt.show()

# Calculate the length statistics of all series
lengths = [len(rewards) for rewards in data]
avg_length = np.mean(lengths)
min_length = np.min(lengths)
max_length = np.max(lengths)
std_length = np.std(lengths)

# Add labels and title
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.title('Histogram of Lengths')

# Print the length statistics
print(f'Average length: {avg_length}')
print(f'Minimum length: {min_length}')
print(f'Maximum length: {max_length}')
print(f'Standard deviation: {std_length}')