import numpy as np
from grid_v3 import get_all_states
import operator
from datetime import datetime
import matplotlib.pyplot as plt

alpha = 0.5                     # Learning Rate
epsilon = 0.1
intended_probability = 0.7
sideways_probability = 0.15
threshold = 0.03

# Initialize Q Values for each state and action as 0
def initialize_Q_values(states, actions, traps, goal):
    Q_values = {}
    # Initialize Utility for all states
    for state in states:
        if(state in traps):
            Q_values[state] = 0.0
        else:
            action_value = {}
            for action in actions:
                action_value[action] = 0.0
            Q_values[state] = action_value

    # Initialize Q for Goal
    Q_values[goal] = 0.0

    return Q_values


# Choose action with 0.7 and 0.15 probabilities
def choose_action(intended_action, grid):
    sideways_actions = grid.get_sideways_actions(intended_action)
    action_list = [intended_action] + sideways_actions
    action = np.random.choice(action_list, 1, p=[intended_probability, sideways_probability, sideways_probability])
    return action[0]


#  Equation to calculate Q value
def utility_equation(Qt, Qtnext, reward_next, gamma):
    #  Qtnext will be max(Qtnext) for Q learning
    #  Qtnext will be Qtnext of selected action for Sarsa
    Qt = Qt + alpha * (reward_next + (gamma * Qtnext) - Qt)
    return Qt


# Get Max Q with action of given state for Q-learning
def get_max_Q(Q_values, state):
    # Get action values of state
    state_action_values = Q_values[state]
    # Get max action value
    action_value = max(state_action_values.items(), key=operator.itemgetter(1))
    action = action_value[0]
    Q = action_value[1]
    return Q, action

# Choose action with 1 - epsilon for greedy action and epsilon/3 for other 3 actions
def egreedy_policy(grid, intended_action, epsilon):
    other_actions = list(grid.actions)
    other_actions.remove(intended_action)
    action_list = [intended_action] + other_actions
    intended_action_prob = 1 - epsilon
    other_action_prob = epsilon/3
    action = np.random.choice(action_list, 1, p=[intended_action_prob, other_action_prob, other_action_prob, other_action_prob])
    return action[0]


# ------------- Main Function ---------------------
# is_sarsa = False for Q Learning
# is_sarsa = True for Sarsa Learning
# episode_limit = Number of episodes for Q Learning or Sarsa

def main(is_sarsa, episode_limit):
    # Initialize Grid and get all possible states
    grid, all_states = get_all_states()
    transition_reward = grid.transition_reward   # Transition Reward
    gamma = grid.gamma                           # Discount
    # Initialize Episode Count
    episode = 0                                  # Episode Count

    # Initialize Q values as 0
    Q_values = initialize_Q_values(all_states, grid.actions, grid.traps, grid.goal)

    # Start Total Algorithm Time
    start_time = datetime.now()
    # List of time taken for each Episode
    episode_time = []
    # Start Episode Time
    episode_start_time = datetime.now()

    while episode < episode_limit:
        # Get Current state
        cur_state = grid.current_state()

        # If current state is goal or trap Finish Episode and start new episode
        # ------ Finish Episode ------
        if cur_state == grid.goal or cur_state in grid.traps:
            # Gameover - Set the Agent back to Start State after terminating at Trap or Goal
            grid.gameover()
            cur_state = grid.current_state()

            # End Episode Time and append to list
            episode_time = generate_episode_graph(episode_start_time, episode_time, episode)
            # Restart Episode Time
            episode_start_time = datetime.now()
            # Start next episode
            episode += 1

        # Get Best Qt and action in the current state
        Qt, action = get_max_Q(Q_values, cur_state)

        # Choose action using e-greedy policy
        action_egreedy = egreedy_policy(grid, action, epsilon)
        if action_egreedy != action:
            # Change Q value with new selected action
            Qt = Q_values[cur_state][action_egreedy]
            action = action_egreedy

        # Choose action by applying Probability of 0.7 on intended action and 0.15 on the sideways actions
        action_new = choose_action(action, grid)
        # Get Qt if chosen different action
        if action_new != action:
            Qt = Q_values[cur_state][action_new]
            action = action_new

        # Move Agent to Next State
        cur_state, action, next_state, reward = grid.move(cur_state, action)

        # if next state is goal or trap
        if next_state == grid.goal or next_state in grid.traps:
            Qtnext = Q_values[next_state]
            # Set Q Value for Goal or Traps
            Q_values[next_state] = utility_equation(0, Qtnext, reward, gamma)
        else:
            # Get max Q value : Qtnext = max(Qt+1) for Q learning
            Qtnext, action_next_max = get_max_Q(Q_values, next_state)
            if is_sarsa:
                # Choose action using epsilon greedy : Qtnext = Qt+1 for Sarsa learning
                action_next_egreedy = egreedy_policy(grid, action_next_max, epsilon)
                Qtnext = Q_values[next_state][action_next_egreedy]

        # Compute Qt value
        # Qtnext is max(Qt+1) for Q Learning
        # Qtnext is Qt+1 of selected action for Sarsa
        Qt = utility_equation(Qt, Qtnext, transition_reward, gamma)

        # Update Qt
        Q_values[cur_state][action] = round(Qt, 4)

    # End of Loop

    # Compute Total Time Taken
    end_time = datetime.now()
    time_difference = end_time - start_time
    print("Time Taken (MilliSeconds) : " + str(int(time_difference.total_seconds() * 1000)))

    print("Episodes : " + str(episode))
    print("Alpha : " + str(alpha))
    print("Gamma : " + str(gamma))
    print("Epsilon : " + str(epsilon))

    print("\n Paths")
    print_paths(grid, Q_values)
    print("\n Values")
    print_values(grid, Q_values)
    return episode_time, episode


# Add each episode time taken in a list
def generate_episode_graph(start_time, episode_time, episode):
    end_time = datetime.now()
    time_diff = end_time - start_time
    episode_time.append(time_diff.total_seconds() * 1000)
    return episode_time

# Print optimal directions for each state
def print_paths(grid, Q_values):
    for y in range(0, grid.height):
        print("-----------------------------------------------------------------")
        for x in range(0, grid.width):
            # print((x, y), "|", end="")
            if (x, y) in grid.walls:
                print("W\t| ", end="")
            else:
                if (x, y) == grid.goal or (x, y) in grid.traps:
                    Q = Q_values[(x, y)]
                    print("%.2f\t| " % Q, end="")
                else:
                    # action_values = Q_values[(x, y)]
                    Qmax, action = get_max_Q(Q_values, (x, y))
                    print(action + "\t| ", end="")
        print("")

# Print Q values for each state and action
def print_values(grid, Q_values):
    for y in range(0, grid.height):
        print("-----------------------------------------------------------------")
        for x in range(0, grid.width):
            # print((x, y), "|", end="")
            if (x, y) in grid.walls:
                print("Wall| ", end="")
            else:
                if (x, y) == grid.goal or (x, y) in grid.traps:
                    Q = Q_values[(x, y)]
                    print("%.2f| " % Q, end="")
                else:
                    action_values = Q_values[(x, y)]
                    for action in grid.actions:
                        print("%.2f" % action_values[action], end=""+action)
                    print("|", end="")
        print("")

# Q Learning
print("\n Q Learning")
episode_time_Q, episode_Q = main(False, 300)


# Sarsa
print("\n Sarsa")
episode_time_sarsa, episode_sarsa = main(True, 300)

# Print Final State Value
plt.plot(list(range(0, episode_Q)), episode_time_Q, list(range(0, episode_sarsa)), episode_time_sarsa)
plt.axis([0, episode_sarsa, 0, 20])
plt.show()