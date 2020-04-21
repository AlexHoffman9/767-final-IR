from OffPolicyAgent import OffPolicyAgent
from random_walk_env import RandomWalkEnv
import numpy as np
import gym
from IRAgent import IRAgent
from OffPolicyAgent_FourRooms import OffPolicyAgent_FourRooms
from four_rooms_env import FourRoomsEnv
import matplotlib.pyplot as plt

def dynamic_programming_FourRooms(env, discount, target_policy, thresh=.001):
    # init values to 0
    v = np.zeros((11,11), dtype=np.float)
    states = [[i,j] for i in range(11) for j in range(11) if not env.rooms[i,j]]
    actions = range(4)
    delta = 100
    itrs = 0
    while thresh < delta:
        delta = 0
        for s in states:
            vnew = 0.0
            for a in actions:
                env.state = s
                s2,r,done,_ = env.step(a)
                target = r + discount*v[s2[0],s2[1]]
                if r!=0:
                    target = r
                vnew += target_policy[s[0],s[1],a]*target
            delta = max(delta, abs(v[s[0],s[1]]-vnew))
            v[s[0],s[1]] = vnew
        itrs += 1
    return v


# test agent
lr=.01
discount=.5

# Random walk policies and true value function
env_walk = RandomWalkEnv(10)
uniform_random_behavior=np.full(shape=(10,2), fill_value=0.5, dtype=np.float)
target_policy=np.zeros(shape=(10,2), dtype=np.float)
for i in range(10):
    target_policy[i,1] = 1.0 # deterministic to right
true_value_walk = [discount**i for i in reversed(range(10))]
ir_agent_walk = IRAgent('RandomWalk', 256, env_walk, target_policy, uniform_random_behavior, lr, discount)
is_agent_walk = OffPolicyAgent('RandomWalk', 256, env_walk, target_policy, uniform_random_behavior, lr, discount)
wis_minibatch_agent_walk = OffPolicyAgent('RandomWalk', 256, env_walk, target_policy, uniform_random_behavior, lr, discount, 'WIS_minibatch')

# Four Rooms params and agents
env_rooms = FourRoomsEnv()
uniform_random_behavior=np.full(shape=(11,11,4), fill_value=0.25, dtype=np.float)
target_policy=np.zeros(shape=(11,11,4), dtype=np.float)
target_policy[:,:,2] = 1.0 # deterministically choose down
true_value_rooms = dynamic_programming_FourRooms(env_rooms, discount, target_policy, .000001).flatten()
is_agent_rooms = OffPolicyAgent_FourRooms('FourRooms', 2500, env_rooms, target_policy, uniform_random_behavior, lr, discount)
wis_minibatch_agent_rooms = OffPolicyAgent_FourRooms('FourRooms', 2500, env_rooms, target_policy, uniform_random_behavior, lr, discount, 'WIS_minibatch')

# choose agent
true_value = true_value_walk
agents = [is_agent_walk, wis_minibatch_agent_walk, ir_agent_walk]

n_updates = 500
n_runs = 1 # cannot increase until we have reset function implemented for agents
steps_per_update = 16
colors = ['r','b','k','o','y']
for i in range(len(agents)):
    agent = agents[i]
    mse_array = np.zeros((n_runs,n_updates+1), dtype=np.float)
    for run in range(n_runs): # need a reset function for multiple runs, so the model is reinitialized and agent gets new random seed
        prediction = agent.value_function().flatten()
        mse_array[run,0] = np.mean(np.square(true_value-prediction))
        for j in range(n_updates):
            agent.generate_episode(steps_per_update)
            agent.train_batch(16, 16)
            prediction = agent.value_function().flatten()
            mse_array[run,j+1] = np.mean(np.square(true_value-prediction))
    mean = np.mean(mse_array,axis=0)
    std = np.std(mse_array,axis=0)
    plt.figure(1)
    steps = [i*steps_per_update for i in range(n_updates+1)]
    plt.plot(steps, mean, color=colors[i], label=agent.name)
    # plt.fill_between(steps,mean+std, mean-std, color = colors[i]) # need to implement multiple runs first
    plt.legend()
    plt.axis([0,steps[-1],0,0.5])
    plt.xlabel('Training steps')
    plt.ylabel('MSE')
plt.title('lr={},discount={}'.format(lr,discount))
plt.show()


# technically the paper used a random walk chain of 8 non terminating states, 2 terminating states, but my code
# has 10 non-terminating states. can fix later because don't want to break things now