import numpy as np
from grid_v3 import GridMDP, get_all_states
from RL_Algorithms.helpers import print_values, print_policy, one_step_look_ahead
import time
import random

THRESHOLD_VALUE = 0.0001
i = 0

def Initialization():
    grid, list_of_accessible_states = get_all_states()

    # Naive Approach
    # Initializing value of all accessible states -> 0
    v_s = {s: 0.0 for s in list_of_accessible_states}
    for key in v_s:
        if key == grid.goal:
            v_s[key] = grid.get_reward(key[0], key[1])
        if key in list(grid.traps):
            v_s[key] = grid.get_reward(key[0], key[1])

    # Initialize policies randomly
    pi = {s: random.choice(grid.ACTION_LIST)
          for s in list_of_accessible_states
          if s != grid.goal and s not in list(grid.traps)}

    return grid, v_s, pi, list_of_accessible_states


def policy_evaluation(grid: GridMDP, states, pi, state_values):
    while True:
        delta = 0
        for s in states:

            if s == grid.goal or s in list(grid.traps):
                continue
            old_v = state_values[s]
            pi_s = pi[s]
            action_value = one_step_look_ahead(grid, state_values, s, pi_s)
            # state_values[s] = sum(action_values)
            state_values[s] = action_value[1]
            delta = max(delta, np.abs(old_v - state_values[s]))

        if delta < THRESHOLD_VALUE:
            policy_improvement(grid, state_values, pi, states)
            break


def policy_improvement(grid: GridMDP, state_values, pi, states):

    # global i
    policy_stable = True
    for s in states:

        if s == grid.goal or s in list(grid.traps):
            continue

        old_action = pi[s]
        max_v = state_values[s]
        max_action = old_action

        for a in GridMDP.ACTION_LIST:

            action_values = one_step_look_ahead(grid, V_s, s, a)

            for v in action_values:
                if v[1] > max_v:
                    max_v = v[1]
                    max_action = v[0]
        pi[s] = max_action

        if old_action != pi[s]:
            policy_stable = False

    if policy_stable:
        # print(f"Total Iterations: {iter}")
        print("Values:")
        print_values(state_values, grid)
        print("\nPolicy")
        print_policy(pi, grid)
    else:
        # i += 1
        policy_evaluation(grid, states, pi, state_values)


grid, V_s, pi, states = Initialization()
policy_improvement(grid, V_s, pi, states)
