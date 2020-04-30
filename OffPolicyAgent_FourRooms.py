import numpy as np
import gym
import tensorflow as tf
from tensorflow.keras.optimizers import Adam,SGD
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, TerminateOnNaN
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from random_walk_env import RandomWalkEnv
from four_rooms_env import FourRoomsEnv
import time
from OffPolicyAgent import OffPolicyAgent
tf.compat.v1.disable_eager_execution()


# get(dict,key) returns value for key


# Agent class with defaults for importance sampling
# can write new classes for WIS, IR, etc. by overwriting key functions
# Compatible with problem_name='RandomWalk' or 'FourRooms'
class OffPolicyAgent_FourRooms(OffPolicyAgent):
    # construct agent's model separately, so it can be sized according to problem
    def __init__(self, n_replay, env, target_policy, behavior_policy, lr, discount, type='IS'):
        super().__init__(n_replay, env, target_policy, behavior_policy, lr, discount, type)

    # reseed numpy, reset weights of network
    # Reset must be performed before every episode
    def reset(self,seed):
        # Reset time
        self.t=0

        # Set seed value
        np.random.seed(seed)

        # Reset replay buffer
        self.replay_buffer = np.zeros(shape=(self.n_replay), dtype=[('s',(np.int32,2)), ('s2',(np.int32,2)), ('r',np.int32), ('ratio', np.float)]) # state, next state, reward, ratio

        # Rebuild model
        self.build_model(self.env.size[0]*self.env.size[1],1)

    # Generate steps of experience
    def generate_experience(self, k=16):

        # Initialize environment
        s = self.env.reset()
        done = False
        steps = 0

        # For each step
        while steps < k:

            # choose action according to behavior policy
            a = np.random.choice(a=self.actions, p=self.behavior_policy[s[0],s[1]])

            # Take a step in environment based on chosen action
            (s2,r,done,_) = self.env.step(a)

            # Compute importance ratios
            ratio = self.target_policy[s[0],s[1],a] / self.behavior_policy[s[0],s[1],a]

            # Add experience to IS replay buffer
            self.replay_buffer[self.t%self.n_replay] = (s,s2,r,ratio)

            # Set for next step
            s=s2
            self.t += 1
            steps += 1

            # If episode ends, reset environment
            if done:
                done = False
                s = self.env.reset()

    # Check if terminal state
    def check_terminal(self, sample):
        if (self.replay_buffer['s'][sample] ==  self.replay_buffer['s2'][sample]).all():
            return True
        else:
            return False

    # need to choose tiling (will depend on child class for each environment
    def construct_features(self,states):
        return np.array([[np.float((i == s[0]) and (j == s[1])) for i in range(11) for j in range(11)] for s in states])

    # Return the entire value function estimate
    def value_function(self):
        test_states = [[i,j] for i in range(11) for j in range(11)]
        test_features = self.construct_features(test_states)
        values = np.reshape(self.model.predict([test_features, np.array([0.]*121)]), (11,11))
        values = values*np.array(np.invert(self.env.rooms), dtype=float) # zero the walls
        return values
