import gridworld as g
import numpy as np
import random

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from IPython.display import clear_output


def model_init():
    nn = Sequential()
    nn.add(Dense(164, init='lecun_uniform', input_shape=(64,)))
    nn.add(Activation('relu'))

    nn.add(Dense(150, init='lecun_uniform'))
    nn.add(Activation('relu'))

    nn.add(Dense(4, init='lecun_uniform'))
    nn.add(Activation('linear'))

    rms = RMSprop()
    nn.compile(loss='mse', optimizer=rms)
    return nn


def simple_training(model):
    epochs = 1000
    gamma = 0.9  # since it may take several moves to goal, making gamma high
    epsilon = 1
    for i in range(epochs):

        state = g.init_grid()
        status = 1
        # while game still in progress
        while status == 1:
            # In state S, run Q function on S to get Q values for all possible actions
            qval = model.predict(state.reshape(1, 64), batch_size=1)
            if random.random() < epsilon:  # choose random action
                action = np.random.randint(0, 4)
            else:  # choose best action from Q(s,a) values
                action = (np.argmax(qval))
            # Take action, observe new state S'
            new_state = g.make_move(state, action)
            # Observe reward
            reward = g.get_reward(new_state)
            # Get max Q(S',a)
            new_q = model.predict(new_state.reshape(1, 64), batch_size=1)
            max_q = np.max(new_q)
            y = np.zeros((1, 4))
            y[:] = qval[:]
            if reward == -1:  # non-terminal state
                update = (reward + (gamma * max_q))
            else:  # terminal state
                update = reward
            y[0][action] = update  # target output
            print("Game #: %s" % (i,))
            model.fit(state.reshape(1, 64), y, batch_size=1, nb_epoch=1, verbose=1)
            state = new_state
            if reward != -1:
                status = 0
            clear_output(wait=True)
        if epsilon > 0.1:
            epsilon -= (1 / epochs)


def test_training(init=0):
    i = 0
    if init == 0:
        state = g.init_grid()
    elif init == 1:
        state = g.inti_grid_player()
    elif init == 2:
        state = g.init_grid_rand()

    print("Initial State:")
    print(g.display_grid(state))
    status = 1
    # while game still in progress
    while status == 1:
        qval = model.predict(state.reshape(1, 64), batch_size=1)
        # take action with highest Q-value
        action = (np.argmax(qval))
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
    simple_training(model)
    test_training()
