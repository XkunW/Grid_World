"""
This file contains all the functions related to the grid world game
"""
import numpy as np
from scipy.stats import bernoulli


def rand_pair(h, w):
    return np.random.randint(0, h), np.random.randint(0, w)


# Initialize stationary grid, all items are placed deterministically
def init_grid():
    state = np.zeros((3, 5, 4))
    # place player
    state[0, 1] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place reward
    state[2, 3] = np.array([1, 0, 0, 0])

    return state


# Initialize player in random location, but keep wall, reward and pit stationary
def init_grid_player():
    state = np.zeros((3, 5, 4))
    # place player
    state[rand_pair(3, 5)] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place reward
    state[1, 2] = np.array([1, 0, 0, 0])

    a = find_location(state, 3)  # find grid position of player (agent)
    w = find_location(state, 2)  # find wall
    g = find_location(state, 0)  # find reward
    p = find_location(state, 1)  # find pit
    if not a or not w or not g or not p:
        # print('Invalid grid. Rebuilding..')
        return init_grid_player()

    return state


# Initialize grid so that reward, pit, wall, player are all randomly placed
def init_grid_rand():
    state = np.zeros((3, 5, 4))
    # place player
    state[rand_pair(3, 5)] = np.array([0, 0, 0, 1])
    # place wall
    state[rand_pair(3, 5)] = np.array([0, 0, 1, 0])
    # place pit
    state[rand_pair(3, 5)] = np.array([0, 1, 0, 0])
    # place reward
    state[rand_pair(3, 5)] = np.array([1, 0, 0, 0])

    a = find_location(state, 3)  # find grid position of player (agent)
    w = find_location(state, 2)  # find wall
    g = find_location(state, 0)  # find reward
    p = find_location(state, 1)  # find pit
    # If any of the "objects" are superimposed, just call the function again to re-place
    if not a or not w or not g or not p:
        # print('Invalid grid. Rebuilding..')
        return init_grid_rand()

    return state


# Initialize grid so that the grid is of size n * 5, where n is a arbitrary positive integer. The walls
def init_grid_dynamic_size(height):
    state = np.zeros((height, 5, 4))

    # place player
    state[height - 1, 4] = np.array([0, 0, 0, 1])
    # place walls
    for i in range(1, height - 1):
        state[i, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[height - 1, 2] = np.array([0, 1, 0, 0])
    # place fixed reward
    state[0, 0] = np.array([1, 0, 0, 0])
    # place stochastic rewards
    state[0, 4] = np.array([2, 0, 0, 0])
    state[2, 0] = np.array([2, 0, 0, 0])

    return state


def place_rand_reward(state):
    height = len(state)
    a = find_location(state, 3)  # find grid position of player (agent)
    w = find_location(state, 2)  # find wall
    g = find_location(state, 0)  # find reward
    p = find_location(state, 1)  # find pit
    pair = rand_pair(height, 5)
    while pair == a or pair == p or pair in w or pair in g:
        pair = rand_pair(height, 5)
    state[pair[0], pair[1]] = np.array([1, 0, 0, 0])


def make_move(state, action):
    # need to locate player in grid
    # need to determine what object (if any) is in the new grid spot the player is moving to
    height = len(state)

    player_loc = find_location(state, 3)
    wall = find_location(state, 2)
    reward = find_location(state, 0)
    pit = find_location(state, 1)
    state = np.zeros((height, 5, 4))
    actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    # e.g. up => (player row - 1, player column + 0)
    new_loc = (player_loc[0] + actions[action][0], player_loc[1] + actions[action][1])
    if new_loc not in wall:
        if (np.array(new_loc) <= (height - 1, 4)).all() and (np.array(new_loc) >= (0, 0)).all():
            state[new_loc][3] = 1

    new_player_loc = find_location(state, 3)
    if not new_player_loc:
        state[player_loc] = np.array([0, 0, 0, 1])
    # re-place pit
    state[pit][1] = 1
    # re-place wall
    for w in wall:
        state[w][2] = 1
    # re-place reward
    for r in reward:
        state[r][0] = 1

    return state


def find_location(state, level):
    height = len(state)
    if level == 0 or level == 2:
        object_list = []
        for i in range(0, height):
            for j in range(0, 5):
                if state[i, j][level] >= 1:
                    object_list.append((i, j))
        return object_list
    else:
        for i in range(0, height):
            for j in range(0, 5):
                if state[i, j][level] == 1:
                    return i, j


def get_reward(state):
    player_loc = find_location(state, 3)
    pit = find_location(state, 1)
    reward = find_location(state, 0)
    if player_loc == pit:
        return 0
    elif player_loc in reward:
        if player_loc[0] == 0 and player_loc[1] == 0:
            return 200
        else:
            bernoulli_list = bernoulli.rvs(0.5, size=100)*200
            index = np.random.randint(0,100)
            return bernoulli_list[index]
    else:
        return 100


def display_grid(state):
    height = len(state)
    grid = np.zeros((height, 5), dtype=str)
    player_loc = find_location(state, 3)
    wall = find_location(state, 2)
    reward = find_location(state, 0)
    pit = find_location(state, 1)
    for i in range(0, height):
        for j in range(0, 5):
            grid[i, j] = ' '

    if player_loc:
        grid[player_loc] = 'P'  # player
    for w in wall:
        if w:
            grid[w] = 'W'  # wall
    for r in reward:
        if r:
            if player_loc != r:
                grid[r] = '+'  # reward
            else:
                grid[r] = 'V'  # Win!
    if pit:
        if player_loc != pit:
            grid[pit] = '-'  # pit
        else:
            grid[pit] = 'L'  # Lose!

    return grid
