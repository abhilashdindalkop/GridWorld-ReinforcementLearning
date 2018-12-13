import numpy as np
from grid_v3 import GridMDP, get_all_states
from RL_Algorithms.helpers import print_values, print_policy, one_step_look_ahead, THRESHOLD_VALUE
import time


# GAMMA = 0.99, for convergence
def value_iteration():
    """
    grid = GridMDP Object to solve the MDP
    list_of_accessible_states = all possible states where agent can move
    :return:
    """
    start = time.perf_counter()
    grid, list_of_accessible_states = get_all_states()

    # Initializing value of all accessible states -> 0
    v_s = {s: 0.0 for s in list_of_accessible_states}
    for key in v_s:
        if key == grid.goal:
            v_s[key] = grid.get_reward(key[0], key[1])
        if key in list(grid.traps):
            v_s[key] = grid.get_reward(key[0], key[1])

    # counting the iteration till convergence
    iter = 0

    while True:
        delta = 0
        # trap_list = list(grid.traps)

        for s in list_of_accessible_states:

            # not updating the value of terminals. By default the values are 0.0 -> possibly buggy. need to update
            # if s == grid.goal or s in trap_list:
            #     continue
            old_v = v_s[s]
            action_values = list()
            for a in GridMDP.ACTION_LIST:
                # one step look ahead for the state and take maximum value from transitions
                action_values += one_step_look_ahead(grid, v_s, s, a)
                # action_values.append(val)

            # best_action_value = max(action_values)
            v_s[s] = max([v[1] for v in action_values])
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(old_v - v_s[s]))

        iter += 1
        # print(f'Iteration: {iter}')

        if delta < THRESHOLD_VALUE:
            print(f'Converging after: {iter} iterations')
            print("Values:")
            print_values(v_s, grid)
            break

    print("\n Policy: \n")
    best_policy = best_policy_finder(grid, v_s, list_of_accessible_states)
    print_policy(best_policy, grid)

    end = time.perf_counter()
    print(f'\nTotal time taken {end - start}s')


# Find the best policy
def best_policy_finder(grid: GridMDP, state_values, states_list):
    policy = {}

    for s in states_list:

        max_v = 0
        max_action = ""

        for a in GridMDP.ACTION_LIST:

            if s == grid.goal or s in list(grid.traps):
                continue

            action_values = one_step_look_ahead(grid, state_values, s, a)

            for v in action_values:
                if v[1] > max_v:
                    max_v = v[1]
                    max_action = v[0]
            policy[s] = max_action

    return policy


value_iteration()
