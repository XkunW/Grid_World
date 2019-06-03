import numpy as np


def rand_pair(s, e):
    return np.random.randint(s, e), np.random.randint(s, e)


# finds an array in the "depth" dimension of the grid
"""
def find_location(state, obj):
    for i in range(0, 4):
        for j in range(0, 4):
            if (state[i, j] == obj).all():
                return i, j
"""


# Initialize stationary grid, all items are placed deterministically
def init_grid():
    state = np.zeros((4, 4, 4))
    # place player
    state[0, 1] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place goal
    state[3, 3] = np.array([1, 0, 0, 0])

    return state


# Initialize player in random location, but keep wall, goal and pit stationary
def inti_grid_player():
    state = np.zeros((4, 4, 4))
    # place player
    state[rand_pair(0, 4)] = np.array([0, 0, 0, 1])
    # place wall
    state[2, 2] = np.array([0, 0, 1, 0])
    # place pit
    state[1, 1] = np.array([0, 1, 0, 0])
    # place goal
    state[1, 2] = np.array([1, 0, 0, 0])

    a = find_location(state, 3)  # find grid position of player (agent)
    w = find_location(state, 2)  # find wall
    g = find_location(state, 0)  # find goal
    p = find_location(state, 1)  # find pit
    if not a or not w or not g or not p:
        # print('Invalid grid. Rebuilding..')
        return inti_grid_player()

    return state


# Initialize grid so that goal, pit, wall, player are all randomly placed
def init_grid_rand():
    state = np.zeros((4, 4, 4))
    # place player
    state[rand_pair(0, 4)] = np.array([0, 0, 0, 1])
    # place wall
    state[rand_pair(0, 4)] = np.array([0, 0, 1, 0])
    # place pit
    state[rand_pair(0, 4)] = np.array([0, 1, 0, 0])
    # place goal
    state[rand_pair(0, 4)] = np.array([1, 0, 0, 0])

    a = find_location(state, 3)  # find grid position of player (agent)
    w = find_location(state, 2)  # find wall
    g = find_location(state, 0)  # find goal
    p = find_location(state, 1)  # find pit
    # If any of the "objects" are superimposed, just call the function again to re-place
    if not a or not w or not g or not p:
        # print('Invalid grid. Rebuilding..')
        return init_grid_rand()

    return state


def make_move(state, action):
    # need to locate player in grid
    # need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = find_location(state, 3)
    wall = find_location(state, 2)
    goal = find_location(state, 0)
    pit = find_location(state, 1)
    state = np.zeros((4, 4, 4))

    actions = [[-1, 0], [1, 0], [0, -1], [0, 1]]
    # e.g. up => (player row - 1, player column + 0)
    new_loc = (player_loc[0] + actions[action][0], player_loc[1] + actions[action][1])
    if new_loc != wall:
        if (np.array(new_loc) <= (3, 3)).all() and (np.array(new_loc) >= (0, 0)).all():
            state[new_loc][3] = 1

    new_player_loc = find_location(state, 3)
    if not new_player_loc:
        state[player_loc] = np.array([0, 0, 0, 1])
    # re-place pit
    state[pit][1] = 1
    # re-place wall
    state[wall][2] = 1
    # re-place goal
    state[goal][0] = 1

    return state


def find_location(state, level):
    for i in range(0, 4):
        for j in range(0, 4):
            if state[i, j][level] == 1:
                return i, j


def get_reward(state):
    player_loc = find_location(state, 3)
    pit = find_location(state, 1)
    goal = find_location(state, 0)
    if player_loc == pit:
        return -10
    elif player_loc == goal:
        return 10
    else:
        return -1


def display_grid(state):
    grid = np.zeros((4, 4), dtype=str)
    player_loc = find_location(state, 3)
    wall = find_location(state, 2)
    goal = find_location(state, 0)
    pit = find_location(state, 1)
    for i in range(0, 4):
        for j in range(0, 4):
            grid[i, j] = ' '

    if player_loc:
        grid[player_loc] = 'P'  # player
    if wall:
        grid[wall] = 'W'  # wall
    if goal:
        if player_loc != goal:
            grid[goal] = '+'  # goal
        else:
            grid[goal] = 'V'  # Win!
    if pit:
        if player_loc != pit:
            grid[pit] = '-'  # pit
        else:
            grid[pit] = 'L'  # Lose!

    return grid


# if __name__ == '__main__':
#     state = init_grid_rand()
#     display_grid(state)
