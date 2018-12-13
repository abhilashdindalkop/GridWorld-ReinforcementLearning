
import numpy as np
from grid_v3 import GridMDP, get_all_states
from RL_Algorithms.helpers import *
import random
import time

THRESHOLD_VALUE = 0.0001
GAMMA = 0.9



start = time.perf_counter()
if __name__ == '__main__':

  grid, list_of_accessible_states = get_all_states()
  iterations = 0


  # randomly choose an action and update as we learn
  V = {s: 0.0 for s in list_of_accessible_states}

  # Initialize policies randomly
  # policy = {s: random.choice(grid.ACTION_LIST) for s in list_of_accessible_states if s != grid.goal or s not in list(grid.traps)}
  policy = {}
  for s in list_of_accessible_states:
      if s == grid.goal:
          continue
      elif s in grid.traps:
          continue
      else:
          policy[s] = random.choice(GridMDP.ACTION_LIST)

  # initial policy
  print("initial policy:")
  print_policy(policy, grid)


  # repeat until convergence - is going to break out when policy does not change
  while True:

    # Policy Evaluation Step
    iterations += 1
    while True:

      iterations += 1

      biggest_change = 0
      for s in list_of_accessible_states:
        old_v = V[s]

        # V(s) only has value if it's not a terminal state
        policy_states = list(policy)
        if s in policy_states:
          a = policy[s]

          action_value = [v for _, v in one_step_look_ahead(grid, V, s, a)]
          V[s] = action_value[0]
          biggest_change = max(biggest_change, np.abs(old_v - V[s]))

      if biggest_change < THRESHOLD_VALUE:
        break

    # policy improvement step
    is_policy_converged = True
    for s in list_of_accessible_states:
      if s in policy:
        old_a = policy[s]
        new_a = None
        best_value = float('-inf')
        # loop through all possible actions to find the best current action
        for a in GridMDP.ACTION_LIST:

            action_values = one_step_look_ahead(grid, V, s, a)

            for v in action_values:
                if v[1] > best_value:
                    best_value = v[1]
                    new_a = v[0]

        policy[s] = new_a
        if new_a != old_a:
          is_policy_converged = False

    if is_policy_converged:
      break

end = time.perf_counter()

print(f"\nTotal Iterations: {iterations}\n")
print(f"Total Time:{end - start}s\n")

print("values:")
print_values(V, grid)
print("\npolicy:")
print_policy(policy, grid)