from OffPolicyAgent import OffPolicyAgent
from random_walk_env import RandomWalkEnv
import numpy as np
import gym
from IRAgent import IRAgent

# testing
env = RandomWalkEnv(10)
lr=.01
discount=.5
# 0.5 probability of choosing left or right in randomwalk
uniform_random_behavior=np.full(shape=(10,2), fill_value=0.5, dtype=np.float)
# construct target policy: deterministic to the right
target=np.zeros(shape=(10,2), dtype=np.float)
for i in range(10):
    target[i,1] = 1.0

agent = IRAgent('RandomWalk', 256, env, target, uniform_random_behavior, lr, discount, batch_corr = True)
# print out initial value function
states = agent.construct_features(range(10))
print(agent.model.predict([states, np.array([0.]*10)]))
true_value = [discount**i for i in reversed(range(10))]
mses=[]
for j in range(50):
    # generate 100 episodes, training after each
    for i in range(50):
        agent.generate_episode()
        agent.train_batch(32, 1)
    # print current value function
    states = agent.construct_features(range(10))
    prediction = agent.model.predict([states, np.array([0.]*10)])
    print(prediction)
    prediction = prediction.flatten()
    mse = np.mean(np.square(true_value-prediction))
    mses.append(mse)

print("final MeanSquareError", mses[-1])
