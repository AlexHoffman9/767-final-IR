from OffPolicyAgent_FourRooms import OffPolicyAgent_FourRooms
from random_walk_env import RandomWalkEnv
import numpy as np
import gym
from four_rooms_env import FourRoomsEnv

# testing
env = FourRoomsEnv()
lr=.01
discount=.5
# 0.25 prob of each direction
uniform_random_behavior=np.full(shape=(11,11,4), fill_value=0.25, dtype=np.float)
# construct target policy: deterministic to the right
target=np.zeros(shape=(11,11,4), dtype=np.float)
target[:,:,2] = 1.0 # deterministically choose down

agent = OffPolicyAgent_FourRooms('FourRooms', 256, env, target, uniform_random_behavior, lr, discount)
# print out initial value function for states in column 2
states=np.zeros((11,2), dtype=np.float)
states[:,0] = range(11)
states[:,1] = 1
state_features = agent.construct_features(states)
# print(state_features)
print(agent.model.predict([state_features, np.array([0.]*10)]))
true_value = [discount**i for i in reversed(range(11))]
mses=[]
for j in range(10):
    for i in range(50):
        agent.generate_episode()
        agent.train_batch(32, 1)
    # print current value function
    prediction = agent.model.predict([state_features, np.array([0.]*10)])
    print(prediction)
    prediction = prediction.flatten()
#     mse = np.mean(np.square(true_value-prediction))
#     mses.append(mse)

# print("final MeanSquareError", mses[-1])