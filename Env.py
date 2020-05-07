# Import routines

import numpy as np
import math
import random

# random.seed(0)
# np.random.seed(0)

# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger

"""
Location λ(of Poisson Distribution)
    0       2
    1       12
    2       4
    3       7
    4       8
"""
poisson_mean = { '0' : 2, '1' : 12, '2' : 4, '3' : 7, '4' : 8 }

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = tuple([(0, 0)]) + tuple(((x, y) for x in range(m) for y in range(m) if x != y))
        self.state_space = tuple(((x, y, z) for x in range(m) for y in range(t) for z in range(d)))
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """
        convert the state into a vector so that it can be fed to the NN. 
        This method converts a given state into a vector format. 
        Hint: The vector is of size m + t + d.
        """
        curr_loc, curr_hour, curr_day = state
        loc_vector = tuple((0 if x != curr_loc else 1 for x in range(m)))
        hour_vector = tuple((0 if x != curr_hour else 1 for x in range(t)))
        week_day_vector = tuple((0 if x != curr_day else 1 for x in range(d)))

        state_encod = loc_vector + hour_vector + week_day_vector

        return state_encod



    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]

        requests = np.random.poisson(poisson_mean.get(str(location)))

        if requests > 15:
            requests = 15

        actions = random.sample(self.action_space[1:], requests) #(0,0) is not considered as customer request
        actions.append((0, 0))

        return tuple(actions)


    def get_updated_hour_and_day(self, present_hour, present_day, additional_hours):
        updated_hour = present_hour + additional_hours
        updated_day = present_day

        if updated_hour >= t :
            updated_hour = updated_hour % t
            updated_day = updated_day + 1
            if updated_day == d:
                updated_day = 0
        
        return (int(updated_hour), updated_day)


    def reward_func(self, state, action, Time_matrix):
        """Takes in state, action and Time-matrix and returns the reward"""
        curr_loc, curr_hour, curr_day = state
        pickup_loc, drop_loc = action

        if action in [(0, 0)] :
            reward = -C
        else:
            curr_to_pickup_loc_time = Time_matrix[curr_loc][pickup_loc][curr_hour][curr_day]

            ride_start_hour, ride_day = self.get_updated_hour_and_day(curr_hour, curr_day, curr_to_pickup_loc_time)

            pickup_to_drop_loc_time = Time_matrix[pickup_loc][drop_loc][ride_start_hour][ride_day]

            reward = (R * pickup_to_drop_loc_time) - (C * (curr_to_pickup_loc_time + pickup_to_drop_loc_time))

        return reward


    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        curr_loc, curr_hour, curr_day = state
        pickup_loc, drop_loc = action

        if action is [(0, 0)]:
            updated_hour, updated_day = self.get_updated_hour_and_day(curr_hour, curr_day, 1)
        else:
            curr_to_pickup_loc_time = Time_matrix[curr_loc][pickup_loc][curr_hour][curr_day]

            ride_start_hour, ride_day = self.get_updated_hour_and_day(curr_hour, curr_day, curr_to_pickup_loc_time)

            pickup_to_drop_loc_time = Time_matrix[pickup_loc][drop_loc][ride_start_hour][ride_day]

            updated_hour, updated_day = self.get_updated_hour_and_day(ride_start_hour, ride_day, pickup_to_drop_loc_time)      
            
        next_state = (curr_loc, updated_hour, updated_day)
        return next_state



    def reset(self):
        return self.action_space, self.state_space, self.state_init



####################################
# Unit Testing 
####################################
if __name__ == "__main__":
    env = CabDriver()
    Time_matrix = np.load("TM.npy")

    action_space, state_space, initial_state = env.reset()
    # print('Cab Driver Action Space:[{}]'.format(action_space))
    # print('Cab Driver State Space:[{}]'.format(state_space))
    print('Cab Driver random initial Space:[{}] \n'.format(initial_state))


    print('Cab Driver Request API:')
    for x in range(10):
        print(env.requests(initial_state))
    print()

    print('Cab Driver State Vector: [{}]\n'.format(env.state_encod_arch1(initial_state)))

    print('Cab Driver rewards on action:')
    for x in range(30):
        action = random.choice(env.requests(initial_state))
        print('Action:[{}] Reward:[{}]'.format(action, env.reward_func(initial_state, action, Time_matrix)))
    print()

    print('Cab Driver action and new state:')
    state = initial_state
    action = random.choice(env.requests(initial_state))
    for x in range(30):
        state = env.next_state_func(state, action, Time_matrix)
        print('Action:[{}] Next State:[{}]'.format(action, state))
        action = random.choice(env.requests(state))
    print()