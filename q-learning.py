"""
This file uses the grid world functions to train a Q-learning agent to play the game.
The Q function is built as a neural network based on the keras Sequential API running on TensorFlow
"""
import gridworld as g
import numpy as np
import random
import matplotlib.pyplot as plt
import time as t
import json
import sys
import keyboard

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.optimizers import RMSprop
from IPython.display import clear_output


def model_init(height):
    input_size = height * 5 * 4
    nn = Sequential()
    nn.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(input_size,)))
    nn.add(Activation('relu'))
    # nn.add(Dropout(0.2))

    nn.add(Dense(150, kernel_initializer='lecun_uniform'))
    nn.add(Activation('relu'))
    # nn.add(Dropout(0.2))

    nn.add(Dense(5, kernel_initializer='lecun_uniform'))
    nn.add(Activation('linear'))

    nn.compile(loss='mse', optimizer=RMSprop())
    return nn


def training_easy(grid, model, n, height, num_of_steps):
    episodes = n * 1000
    gamma = 0.9
    epsilon = 1
    input_size = height * 5 * 4
    indices = []
    fidelities = []
    for i in range(episodes):
        for j in range(num_of_steps):
            # In state S, run Q function on S to get Q values for all possible actions
            pre_state = grid.state
            q_value = model.predict(pre_state.reshape(1, input_size), batch_size=1)
            if np.random.uniform(0, 1) < epsilon:  # choose random action
                action = np.random.randint(0, 5)
            else:  # choose best action from Q(s,a) values
                action = (np.argmax(q_value))
            # Take action, observe new state S'
            grid.agent_move(action)
            new_state = grid.state
            # Observe reward
            reward = grid.get_reward()
            # Get max Q(S',a)
            new_q = model.predict(new_state.reshape(1, input_size), batch_size=1)
            max_q = np.max(new_q)
            y = np.zeros((1, 5))
            y[:] = q_value[:]
            update = reward + (gamma * max_q)
            y[0][action] = update  # target output
            print("Game #: %s" % (i,))
            model.fit(pre_state.reshape(1, input_size), y, batch_size=1, epochs=1, verbose=1)
            clear_output(wait=True)
        indices.append(i)
        fidelities.append(get_fidelity(height, model))
        if epsilon > 0.1:
            epsilon -= (1 / episodes)
    return indices, fidelities


def get_fidelity(height, model):
    input_size = height * 5 * 4
    grid = g.GridWorld(height)
    count = height * 5 - (height - 2)
    fidelity = 0
    for i in range(height):
        for j in range(5):
            # If not wall
            if not grid.state[(i, j)][2]:
                grid.place_player((i, j))
                q_value = model.predict(grid.state.reshape(1, input_size), batch_size=1)
                action = (np.argmax(q_value))
                fidelity += grid.check_optimal_policy((i, j), action)
    print(fidelity / count)
    return fidelity / count


def test_training(model, height=3, num_of_steps=10):
    grid = g.GridWorld(height)
    input_size = height * 5 * 4
    total_reward = 0
    print("Initial State:")
    print(grid.display_grid())
    # while game still in progress
    for i in range(num_of_steps):
        q_value = model.predict(grid.state.reshape(1, input_size), batch_size=1)
        # take action with highest Q-value
        action = (np.argmax(q_value))
        print('Move #: %s; Taking action: %s' % (i, action))
        grid.agent_move(action)
        grid.display_grid()
        reward = grid.get_reward()
        total_reward += reward
    print("Max steps reached, total reward: {}".format(total_reward))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        height = sys.argv[1]
        env = g.GridWorld(height)
    else:
        height = 3
        env = g.GridWorld()

    num_of_steps = 10
    for index in range(5):
        model = model_init(height)
        f = training_easy(env, model, 1, height, num_of_steps)
        test_training(model)
        plt.plot(f[0], f[1])
        plt.show()
