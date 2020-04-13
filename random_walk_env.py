import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class RandomWalkEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  # size = number of states
    # ex: size=10, there are states 0-9, going left at state 0 terminates with r=0, right at state 9 terminates with r=1
  def __init__(self, size):
    self.action_space = spaces.Discrete(2)
    self.size = size # number of valid states
    #print("init")
  def step(self, action):
    #print("step")
    reward = 0
    done = False
    if (action == 0):
       self.state -= 1
    if (action == 1):
        self.state += 1
    if (self.state >= self.size):
        reward = 1
        done = True
    if (self.state < 0):
        done = True
    return np.array(self.state), reward, done, {}
  def reset(self):
    #print("reset")
    # print("#self.size:",self.size)
    self.state =  np.random.randint(0,self.size-1)
    return self.state
    # print("starting: ", self.state)
#   def render(self, mode='human', close=False):
#     if close:
#         return
#     #print("render")
#     print("current state: ",self.state)

#   def close(self):