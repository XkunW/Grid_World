import numpy as np
from scipy.stats import bernoulli


class GridWorld(object):
    actions = {
        0: (-1, 0),  # up
        1: (1, 0),  # down
        2: (0, -1),  # left
        3: (0, 1),  # right
        4: (0, 0)  # stay
    }

    def __init__(self, height=3, random_reward=False, reward=None):
        if reward is None:
            reward = [-1, -10, 10]
        self.height = height
        self.random_reward = random_reward
        self.state = np.zeros((height, 5, 4))
        self.step = reward[0]
        self.penalty = reward[1]
        self.reward = reward[2]
        self.optimal_policy = {}
        self.init_grid()
        self.init_optimal_policy()

    def init_grid(self):
        # place player
        self.state[self.height - 1, 4] = np.array([0, 0, 0, 1])
        # place walls
        for i in range(1, self.height - 1):
            self.state[i, 2] = np.array([0, 0, 1, 0])
        # place pit
        self.state[self.height - 1, 2] = np.array([0, 1, 0, 0])
        # place fixed reward
        self.state[0, 0] = np.array([1, 0, 0, 0])
        # place stochastic rewards if applicable
        if self.random_reward:
            self.state[0, 4] = np.array([1, 0, 0, 0])
            self.state[self.height - 1, 0] = np.array([1, 0, 0, 0])

    def init_optimal_policy(self):
        # 0 = up, 1 = down, 2 = left, 3 = right, 4 = stay.
        for i in range(self.height):
            for j in range(5):
                self.optimal_policy[(i, j)] = [4]

        # Moving up or left at reward would be the same as stay
        self.optimal_policy[(0, 0)] = [4, 2, 0]

        # First row
        for j in range(1, 5):
            self.optimal_policy[(0, j)] = [2]

        # Last row
        for j in range(5):
            if j == 2:
                self.optimal_policy[(self.height - 1, j)] = [2]
            elif j == 4:
                self.optimal_policy[(self.height - 1, j)] = [2, 0]
            else:
                self.optimal_policy[(self.height - 1, j)] = [0]

        # Rows in between
        for i in range(1, self.height - 1):
            self.optimal_policy[(i, 1)] = [2, 0]
            self.optimal_policy[(i, 4)] = [2, 0]
            self.optimal_policy[(i, 0)] = [0]
            self.optimal_policy[(i, 3)] = [0]

    def find_location(self, level):
        if level == 0 or level == 2:
            object_list = []
            for i in range(self.height):
                for j in range(5):
                    if self.state[i, j][level] == 1:
                        object_list.append((i, j))
            return object_list
        else:
            for i in range(self.height):
                for j in range(5):
                    if self.state[i, j][level] == 1:
                        return i, j

    def display_grid(self):
        grid = np.zeros((self.height, 5), dtype=str)
        player_loc = self.find_location(3)
        wall = self.find_location(2)
        reward = self.find_location(0)
        pit = self.find_location(1)
        for i in range(self.height):
            for j in range(5):
                grid[i, j] = ' '

        if player_loc:
            grid[player_loc] = 'P'  # player
        for w in wall:
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
        print(grid)

    def check_optimal_policy(self, location, action):
        return action in self.optimal_policy[location]

    def get_reward(self):
        player_loc = self.find_location(3)
        pit = self.find_location(1)
        reward = self.find_location(0)
        if player_loc == pit:
            return self.penalty
        elif player_loc in reward:
            """
            # The following snippet is for stochastic reward, currently not used
            if player_loc[0] == 0 and player_loc[1] == 0:
                return 2 
            else:
                bernoulli_list = bernoulli.rvs(0.5, size=10) * 20
                index = np.random.randint(0, 10)
                return bernoulli_list[index] - 10
            """
            return self.reward
        else:
            return self.step

    def agent_move(self, action):
        # need to locate player in grid
        # need to determine what object (if any) is in the new grid spot the player is moving to
        player_loc = find_location(self.state, 3)
        wall = find_location(self.state, 2)
        # e.g. up => (player row - 1, player column + 0)
        new_loc = (player_loc[0] + self.actions[action][0], player_loc[1] + self.actions[action][1])
        if new_loc not in wall:
            if (np.array(new_loc) <= (self.height - 1, 4)).all() and (np.array(new_loc) >= (0, 0)).all():
                self.state[player_loc][3] = 0
                self.state[new_loc][3] = 1

    def place_player(self, new_loc):
        curr_loc = self.find_location(3)
        self.state[curr_loc][3] = 0
        self.state[new_loc][3] = 1


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
    # state[0, 4] = np.array([1, 0, 0, 0])
    # state[height - 1, 0] = np.array([1, 0, 0, 0])

    return state


def check_availability(state, loc):
    return state[loc][0] or state[loc][1] or state[loc][2]


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


def find_location(state, level):
    height = len(state)
    if level == 0 or level == 2:
        object_list = []
        for i in range(height):
            for j in range(5):
                if state[i, j][level] == 1:
                    object_list.append((i, j))
        return object_list
    else:
        for i in range(height):
            for j in range(5):
                if state[i, j][level] == 1:
                    return i, j
