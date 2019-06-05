"""
This file uses the grid world functions to train a Q-learning agent to play the game. 
The Q function is built as a neural network based on the keras Sequential API running on TensorFlow
"""
import gridworld as g
import numpy as np
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output


def model_init():
    nn = Sequential()
    nn.add(Dense(164, kernel_initializer='lecun_uniform', input_shape=(60,)))
    nn.add(Activation('relu'))

    nn.add(Dense(150, kernel_initializer='lecun_uniform'))
    nn.add(Activation('relu'))

    nn.add(Dense(4, kernel_initializer='lecun_uniform'))
    nn.add(Activation('linear'))

    rms = RMSprop()
    nn.compile(loss='mse', optimizer=rms)
    return nn


def training_easy(model):
    epochs = 1000
    gamma = 0.9  # since it may take several moves to goal, making gamma high
    epsilon = 1
    for i in range(epochs):

        state = g.init_grid()
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
            if reward == -1:  # non-terminal state
                update = (reward + (gamma * max_q))
            else:  # terminal state
                update = reward
            y[0][action] = update  # target output
            print("Game #: %s" % (i,))
            model.fit(state.reshape(1, 60), y, batch_size=1, epochs=1, verbose=1)
            state = new_state
            if reward != -1:
                status = 0
            clear_output(wait=True)
        if epsilon > 0.1:
            epsilon -= (1 / epochs)


def training_hard(model, n):
    epochs = n*1000
    gamma = 0.975
    epsilon = 1
    batch_size = 40
    buffer = 80
    replay = []
    # stores tuples of (S, A, R, S')
    h = 0
    for i in range(epochs):

        state = g.init_grid_player()  # using the harder state initialization function
        status = 1
        # while game still in progress
        while status == 1:
            # We are in state S
            # Let's run our Q function on S to get Q values for all possible actions
            q_value = model.predict(state.reshape(1, 60), batch_size=1)
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
                    old_q_value = model.predict(old_state.reshape(1, 60), batch_size=1)
                    new_q = model.predict(new_state.reshape(1, 60), batch_size=1)
                    max_q = np.max(new_q)
                    y = np.zeros((1, 4))
                    y[:] = old_q_value[:]
                    if reward == -1:  # non-terminal state
                        update = (reward + (gamma * max_q))
                    else:  # terminal state
                        update = reward
                    y[0][action] = update
                    x_train.append(old_state.reshape(60, ))
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
            epsilon -= (1 / epochs)


def test_training(init=0):
    i = 0
    if init == 0:
        state = g.init_grid()
    elif init == 1:
        state = g.init_grid_player()
    elif init == 2:
        state = g.init_grid_rand()

    print("Initial State:")
    print(g.display_grid(state))
    status = 1
    # while game still in progress
    while status == 1:
        q_value = model.predict(state.reshape(1, 60), batch_size=1)
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
        if i > 10:
            print("Game lost; too many moves.")
            break


if __name__ == "__main__":
    model = model_init()
    count = 0
    while True:
        n = input("Enter the number of epochs to train (in thousands): ")
        n = int(n)
        count += n
        training_hard(model, n)
        print("Model was trained for {} epochs in total".format(count*1000))
        input("Press Enter to test model...")
        while True:
            test_training(1)
            response = input("Press Enter to test model or Q to "
                             "finish testing...")
            if response.lower() == 'q':
                break
        response = input("Press Q to abort or else to continue training: ")
        if response.lower() == 'q':
            break
