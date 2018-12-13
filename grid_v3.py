import numpy as np
import matplotlib.pyplot as plt
from random import choice


class GridMDP:
    ACTION_LIST = ('R', 'U', 'L', 'D')

    def __init__(self, width, height, start, actions, walls, traps, goal):
        self.width = width # grid width
        self.height = height # grid height
        self.x = start[0]  # x = row
        self.y = start[1]  # y = col
        self.actions = actions # Action = ['R', 'U', 'L', 'D']
        self.walls = walls # grid walls
        self.traps = traps # grid traps
        self.goal = goal # grid goal
        self.start = start # grid start
        self.gamma = 1 # discount factor [** but changed in policy iterations and value iterations
        self.transition_reward = -0.02 # For Q learning and Sarsa

    def set_state(self, s):
        self.x = s[0]
        self.y = s[1]

    def current_state(self):
        return self.x, self.y

    def is_terminal(self, s):
        is_terminal = False
        if s in self.traps:
            is_terminal = True
        if s == self.goal:
            is_terminal = True
        return is_terminal

    def out_of_grid(self):
        # If X goes out of Grid
        if self.x < 0:
            self.x = 0
        if self.x == self.width:
            self.x = self.width - 1
        # If Y goes out of Grid
        if self.y < 0:
            self.y = 0
        if self.y == self.height:
            self.y = self.height - 1

    # Move to next state based on action
    # Returns current state, action, next state, reward for move,
    def move(self, state, action):
        # Basic Actions
        self.x = state[0]
        self.y = state[1]

        if action == 'U':
            self.y -= 1
        elif action == 'D':
            self.y += 1
        elif action == 'R':
            self.x += 1
        elif action == 'L':
            self.x -= 1

        # If agent out of grid - stay on current state
        self.out_of_grid()

        # Undo Move if Walls
        if (self.x, self.y) in self.walls:
            self.undo_move(action)

        next_state = (self.x, self.y)
        reward = self.get_reward(self.x, self.y)

        return state, action, next_state, reward

    def get_reward(self, x, y):
        reward = -0.02
        if (x, y) in self.traps:
            reward = -1
        elif (x, y) == self.goal:
            reward = +1
        return reward

    def undo_move(self, action):
        # These are the opposite of what U/D/L/R should normally do
        if action == 'U':
            self.y += 1
        elif action == 'D':
            self.y -= 1
        elif action == 'R':
            self.x -= 1
        elif action == 'L':
            self.x += 1
        # If agent out of grid - stay on current state
        self.out_of_grid()

    @staticmethod
    def create_grid():
        # X : x-axis, Y : y-axis
        # State positions are indicated as (X, Y)
        # S - Start
        # G - Goal (Reward : +1)
        # T - Trap (Reward : -1)
        # W- Wall

        #     0  1  2  3  4  5  6  7
        #   ------------------------>
        # 0 | T  W  .  .  T  W  W  G
        # 1 | .  .  .  .  .  .  .  .
        # 2 | .  .  W  W  W  .  .  .
        # 3 | .  .  T  .  W  .  W  .
        # 4 | .  W  W  .  .  .  W  .
        # 5 | .  .  W  .  .  W  T  .
        # 6 | .  .  .  .  .  W  W  .
        # 7 | S  .  .  W  .  .  .  .

        width = 8
        height = 8
        start = (0, 7)
        goal = (7, 0)
        walls = [(1, 0), (5, 0), (6, 0), (2, 2), (3, 2), (4, 2), (4, 3), (6, 3), (1, 4), (2, 4), (6, 4), (2, 5), (5, 5),
                 (5, 6), (6, 6), (3, 7)]
        traps = [(0, 0), (4, 0), (2, 3), (6, 5)]
        my_grid = GridMDP(width, height, start, GridMDP.ACTION_LIST, walls, traps, goal)

        # print("Start")
        # my_grid.move('UP')
        # print(my_grid.current_state())

        return my_grid

    def gameover(self):
        self.set_state(self.start)

    def get_sideways_actions(self, action):
        sideways = []
        if action == 'U' or action == 'D':
            sideways = ['L', 'R']
        elif action == 'L' or action == 'R':
            sideways = ['U', 'D']
        return sideways

    @staticmethod
    def turn_right(action):
        return GridMDP.ACTION_LIST[(GridMDP.ACTION_LIST.index(action) + 1) % len(GridMDP.ACTION_LIST)]

    @staticmethod
    def turn_left(action):
        return GridMDP.ACTION_LIST[GridMDP.ACTION_LIST.index(action) - 1]

    # building transition matrix
    def transition_matrix(self, state, action, uniform = False):

        if(uniform):
            uniform_matrix = [(0.25, self.move(state, m)) for m in GridMDP.ACTION_LIST if m != action]
            uniform_matrix.append((0.25, self.move(state, action)))
            return uniform_matrix

        t1 = [0.7, self.move(state, action)]
        t2 = [0.15, self.move(state, GridMDP.turn_right(action))]
        t3 = [0.15, self.move(state, GridMDP.turn_left(action))]

        return [t1, t2, t3]


def get_all_states():
    """

    :return: grid = Grid Object, l = list of accessible states
    """
    grid = GridMDP.create_grid()
    initial_grid = np.zeros(shape=(grid.width, grid.height))
    l = [(i, j) for i, y in enumerate(initial_grid) for j, x in enumerate(y) if (i, j) not in grid.walls]

    # Start state initialization
    l[l.index(grid.start)], l[0] = l[0], l[l.index(grid.start)]

    return grid, l


def reward_matrix():
    pass


g, l = get_all_states()
s = choice(l)
a = choice(GridMDP.ACTION_LIST)
l1 = g.transition_matrix(s, a, True)
# print(l1)


