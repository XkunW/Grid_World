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

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output


def model_init(height):
    input_size = height * 5 * 4
    size_1 = height * 5 * 11
    size_2 = height * 5 * 10
    nn = Sequential()
    nn.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(input_size,)))
    nn.add(Activation('relu'))

    nn.add(Dense(150, kernel_initializer='lecun_uniform'))
    nn.add(Activation('relu'))

    nn.add(Dense(4, kernel_initializer='lecun_uniform'))
    nn.add(Activation('linear'))

    nn.compile(loss='mse', optimizer=RMSprop())
    return nn


def training_easy(model):
    episodes = 1000
    gamma = 0.8  # since it may take several moves to goal, making gamma high
    epsilon = 1
    for i in range(episodes):

        state = g.init_grid_dynamic_size(3)
        status = 1
        # while game still in progress
        while status == 1:
            # In state S, run Q function on S to get Q values for all possible actions
            q_value = model.predict(state.reshape(1, 60), batch_size=1)
            if random.random() < epsilon:  # choose random action
                action = np.random.randint(0, 4)
            else:  # choose best action from Q(s,a) values
                action = (np.argmax(q_value))
            # Take action, observe new state S'
            new_state = g.make_move(state, action)
            # Observe reward
            reward = g.get_reward(new_state)
            # Get max Q(S',a)
            new_q = model.predict(new_state.reshape(1, 60), batch_size=1)
            max_q = np.max(new_q)
            y = np.zeros((1, 4))
            y[:] = q_value[:]
            if reward == 0:  # non-terminal state
                update = (reward + (gamma * max_q))
            else:  # terminal state
                update = reward
            y[0][action] = update  # target output
            print("Game #: %s" % (i,))
            model.fit(state.reshape(1, 60), y, batch_size=1, epochs=1, verbose=1)
            state = new_state
            if reward != 0:
                status = 0
            clear_output(wait=True)
        if epsilon > 0.1:
            epsilon -= (1 / episodes)


def training_hard(model, n, height):
    episodes = n * 1000
    gamma = 0.8
    epsilon = 1
    batch_size = 40
    buffer = 80
    replay = []
    # stores tuples of (S, A, R, S')
    h = 0
    fidelity = {}
    all = {}
    start = t.time()
    for i in range(episodes):

        state = g.init_grid_dynamic_size(height)  # using the harder state initialization function
        status = 1
        input_size = height * 5 * 4
        # while game still in progress
        while status == 1:
            # In state S, run Q function on S to get Q values for all possible actions
            q_value = model.predict(state.reshape(1, input_size), batch_size=1)
            if random.random() < epsilon:  # choose random action
                action = np.random.randint(0, 4)
            else:  # choose best action from Q(s,a) values
                action = (np.argmax(q_value))
            # Take action, observe new state S'
            new_state = g.make_move(state, action)
            # Observe reward
            reward = g.get_reward(new_state)
            # Experience replay storage
            if len(replay) < buffer:  # if buffer not filled, add to it
                replay.append((state, action, reward, new_state))
            else:  # if buffer full, overwrite old values
                if h < (buffer - 1):
                    h += 1
                else:
                    h = 0
                replay[h] = (state, action, reward, new_state)
                # randomly sample our experience replay memory
                mini_batch = random.sample(replay, batch_size)
                x_train = []
                y_train = []
                for memory in mini_batch:
                    # Get max_Q(S',a)
                    old_state, action, reward, new_state = memory
                    old_q_value = model.predict(old_state.reshape(1, input_size), batch_size=1)
                    new_q = model.predict(new_state.reshape(1, input_size), batch_size=1)
                    max_q = np.max(new_q)
                    y = np.zeros((1, 4))
                    y[:] = old_q_value[:]
                    if reward == -1:  # non-terminal state
                        update = (reward + (gamma * max_q))
                    else:  # terminal state
                        update = reward
                    y[0][action] = update
                    x_train.append(old_state.reshape(input_size, ))
                    y_train.append(y.reshape(4, ))

                x_train = np.array(x_train)
                y_train = np.array(y_train)
                print("Game #: %s" % (i,))
                model.fit(x_train, y_train, batch_size=batch_size, epochs=1, verbose=1)
                state = new_state
            if reward != -1:  # if reached terminal state, update game status
                status = 0
            clear_output(wait=True)
        if epsilon > 0.1:  # decrement epsilon over time
            epsilon -= (1 / episodes)
        all[i] = get_fidelity(height, model)
        if i % 100 == 0:
            fidelity[i] = get_fidelity(height, model)
    end = t.time()
    interval = {"Time Elapsed":  format(end - start, '.3f')}
    all = [interval, all]
    data = [all, fidelity]
    return data


def get_fidelity(height, model):
    input_size = height * 5 * 4
    state = g.init_grid_dynamic_size(height)
    count = height * 5 - 4 - (height - 2)
    fidelity = 0
    for i in range(height):
        for j in range(5):
            if g.check_availability(state, (i, j)) == 0:
                state = g.place_player(state, (i, j))
                q_value = model.predict(state.reshape(1, input_size), batch_size=1)
                action = (np.argmax(q_value))
                fidelity += g.check_optimal_policy(height, (i, j), action)
    print(fidelity / count)
    return fidelity / count


def test_training(init=0, height=0):
    i = 0
    if init == 0:
        state = g.init_grid()
    elif init == 1:
        state = g.init_grid_player()
    elif init == 2:
        state = g.init_grid_rand()
    elif init == 3:
        state = g.init_grid_dynamic_size(height)

    print("Initial State:")
    print(g.display_grid(state))
    height = len(state)
    input_size = height * 5 * 4
    status = 1
    # while game still in progress
    while status == 1:
        q_value = model.predict(state.reshape(1, input_size), batch_size=1)
        # take action with highest Q-value
        action = (np.argmax(q_value))
        print('Move #: %s; Taking action: %s' % (i, action))
        state = g.make_move(state, action)
        print(g.display_grid(state))
        reward = g.get_reward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1
        # If we're taking more than 10 actions, just stop, we probably can't win this game
        if i > height * 2 + 4:
            print("Game lost; too many moves.")
            break


if __name__ == "__main__":
    height = input("Enter the height of the grid: ")
    height = int(height)
    """
    model = model_init(height)
    count = 0
    while True:
        n = input("Enter the number of episodes to train (in thousands): ")
        n = int(n)
        count += n
        training_hard(model, n, height)
        # training_easy(model)
        print("Model was trained for {} episodes in total".format(count * 1000))
        input("Press Enter to test model...")
        while True:
            test_training(3, height)
            response = input("Press Enter to test model or Q to "
                             "finish testing...")
            if response.lower() == 'q':
                break
        response = input("Press Q to abort or else to continue training: ")
        if response.lower() == 'q':
            break
    """
    fidelity = []
    for index in range(10):
        model = model_init(height)
        f = training_hard(model, 5, height)
        fidelity.append(f[0])
        plt.plot(list(f[1].keys()), list(f[1].values()), label="Agent {}".format(index + 1))
        print("Model was trained for 5000 episodes in total")

    plt.title("Fidelity plot on {}x5 gird".format(height))
    plt.xlabel("Episodes")
    plt.ylabel("Fidelity")
    plt.legend(loc='best')
    plt.show()

    # fidelity = json.dump(fidelity)
    with open('data.json', 'w') as json_file:
        json.dump(fidelity, json_file)

