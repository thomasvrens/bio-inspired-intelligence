import random
import time
import pickle

import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.activations import leaky_relu, relu, linear
from keras.optimizers import Adam
from collections import deque


# AGENT HYPERPARAMETERS
LEARNING_RATE = 0.001
START_EPSILON = 1.0
RESET_EPSILON = 0.8
EPSILON_DECAY_FACTOR = 0.995
DISCOUNT_FACTOR = 0.9925
RESET_EPSILON_EVERY = 1_000 # Episodes
MAX_MEMORY_SIZE = 10_000
MIN_MEMORY_SIZE = 1_000
BATCH_SIZE = 64
UPDATE_TARGET_MODEL_EVERY = 5  # Episodes


class DDQNAgent:
    def __init__(self, state_shape, action_size):
        self.state_size = state_shape[0]
        self.action_size = action_size
        self.state_shape = state_shape

        print(f'State size: {self.state_size}')
        print(f'Action size: {self.action_size}')
        
        self.model = self.make_model()
        
        self.target_model = self.make_model()
        self.target_model.set_weights(self.model.get_weights())

        # Circular buffer for faster memory access
        self.memory = [None] * MAX_MEMORY_SIZE
        self.memory_index = 0
        self.actual_memory_size = 0

        self.target_counter = 0

        self.epsilon = START_EPSILON
        self.reset_epsilon_every = RESET_EPSILON_EVERY

        self.save_str = f'LR_{LEARNING_RATE}_DF_{DISCOUNT_FACTOR}_ED_{EPSILON_DECAY_FACTOR}_RE_{RESET_EPSILON_EVERY}'
    
    def make_model(self):
        model = Sequential()

        # Add layers
        model.add(Dense(64, input_shape=self.state_shape))
        model.add(Activation(relu))
        model.add(Dense(64))
        model.add(Activation(relu))
        model.add(Dense(32))
        model.add(Activation(relu))
        
        # Output layer
        model.add(Dense(self.action_size))
        model.add(Activation(linear))

        model.compile(loss=tf.keras.losses.Huber(), optimizer=Adam(learning_rate=LEARNING_RATE), metrics=['accuracy'])

        return model

    def act(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        else:
            state = np.reshape(state, [1, self.state_size])
            return np.argmax(self.model.predict(state)[0])
    
    def add_memory(self, step_data):
        if self.actual_memory_size < MAX_MEMORY_SIZE:
            self.actual_memory_size += 1
        self.memory[self.memory_index] = step_data
        self.memory_index = (self.memory_index + 1) % MAX_MEMORY_SIZE
    
    def train(self):
        if self.actual_memory_size < MIN_MEMORY_SIZE:
            return
        
        indices = np.random.choice(self.actual_memory_size, BATCH_SIZE, replace=False)
        batch = [self.memory[i] for i in indices]

        cur_states = np.array([step_data[0] for step_data in batch])
        cur_qs = self.model.predict(cur_states)

        new_states = np.array([step_data[3] for step_data in batch])
        future_qs = self.target_model.predict(new_states)

        X = []
        y = []

        for index, (cur_state, action, reward, new_state, done) in enumerate(batch):
            if done:
                new_q = reward
            else:
                new_q = reward + DISCOUNT_FACTOR * np.max(future_qs[index])
            
            cur_qs[index][action] = new_q

            X.append(cur_state)
            y.append(cur_qs[index])

        self.model.fit(np.array(X), np.array(y), batch_size=BATCH_SIZE, verbose=0, shuffle=False)
    
    def increase_target_model_counter(self):
        self.target_counter += 1
        if self.target_counter >= UPDATE_TARGET_MODEL_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_counter = 0

    def decrease_epsilon(self):
        self.epsilon *= EPSILON_DECAY_FACTOR

    def reset_epsilon(self):
        self.epsilon = RESET_EPSILON
    
    def save_model(self, solved):
        if solved:
            save_string = f'models/SOLVED_{self.save_str}/{int(time.time())}.model'
        else:
            save_string = f'models/{self.save_str}/{int(time.time())}.model'
        self.model.save(save_string)

    def save_rewards(self, reward_list):
        file_path = f'rewards/{self.save_str}/{int(time.time())}.pkl'
        with open(file_path, 'wb') as f:
            pickle.dump(reward_list, f)

    def save_solution(self):
        save_string = f'solutions/{self.save_str}/{int(time.time())}.model'
        self.model.save(save_string)
    
    def load_model(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.target_model.set_weights(self.model.get_weights())


