from grid_v3 import GridMDP
import copy
import numpy as np

THRESHOLD_VALUE = 0.0001


# Building transition matrix using backup diagram
def one_step_look_ahead(grid: GridMDP, state_values, state, action, gamma=0.99, uniform=False):

    # To store the calculated values of the transitions
    calculated_values = list()

    # for a in GridMDP.ACTION_LIST:
    trans_matrix = grid.transition_matrix(state, action, uniform)
    sum_of_look_ahead_tree_child = 0

    for prob, matrix in trans_matrix:
        reward = matrix[3]  # immediate reward
        next_state = matrix[2]  # next state from transition
        v_next_state = state_values[next_state]  # value of the next state

        # sum of first action tree defined by the policy
        val = prob * (reward + gamma * v_next_state)
        sum_of_look_ahead_tree_child += val

    # append sums to calculate max state value
    calculated_values.append((action, np.around(sum_of_look_ahead_tree_child, 4)))

    # print(calculated_values)
    return calculated_values

# For printing in grid
def print_values(V, g: GridMDP):

    v_copy = copy.deepcopy(V)
    for i in range(g.width):
        print("-" * (g.height * g.width))
        for j in range(g.height):

            v = v_copy.get((j, i), None)
            if v is None:
                print(" Wall  |", end="")
            else:
                if v >= 0:
                    print("{0:.4f} |".format(v), end="")
                else:
                    print("{0:.4f}|".format(v), end="")
        print("")

# For Printing policy in grid
def print_policy(P, g: GridMDP):

    copy_p = copy.deepcopy(P)
    for i in range(g.width):
        print("-" * (g.height * g.width))
        for j in range(g.height):

            if (j, i) == g.goal:
                copy_p[j, i] = g.get_reward(j, i)

            if (j, i) in list(g.traps):
                copy_p[j, i] = g.get_reward(j, i)

            a = copy_p.get((j, i), ' ')

            if a == ' ':
                print("  Wall |", end="")
            else:
                print("  {a}    |".format(a=a), end="")
        print("")
