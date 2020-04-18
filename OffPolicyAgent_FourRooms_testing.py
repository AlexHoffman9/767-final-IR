from OffPolicyAgent_FourRooms import OffPolicyAgent_FourRooms
from random_walk_env import RandomWalkEnv
import numpy as np
import gym
from four_rooms_env import FourRoomsEnv

# function DynamicProgramming(env::FourRooms, gvf::GVF, thresh=1e-20)
#     v = zeros(size(env))
#     states = get_states(env)
#     delta = 100000
#     while thresh < delta
#         delta = 0
#         for state in states
#             _v = v[state]
#             v_new = 0.0
#             for action in JuliaRL.get_actions(env)
#                 new_state, _, _ = _step(env, state, action)
#                 value_act = v[new_state]
#                 c, γ, π = get(gvf, state, action, new_state)
#                 v_new += π*(c + γ*value_act)
#             end
#             v[state] = v_new
#             delta = max(delta, abs(_v - v[state]))
#         end
#     end
#     return v
# end

np.set_printoptions(precision=2, suppress=True)
def dynamic_programming(env, discount, target_policy, thresh=.001):
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
    
        


# Test four rooms environment
# trying to match figure 1 of paper
# n=2500 buffer size, k=16 steps/update, batchsize=16, 25 runs
# 250k iterations, 1k warmup (experience without training)
env = FourRoomsEnv()
lr=.1
discount=.5
# behavior policy: 0.25 prob of each direction
uniform_random_behavior=np.full(shape=(11,11,4), fill_value=0.25, dtype=np.float)
# target policy: deterministic to the right
target=np.zeros(shape=(11,11,4), dtype=np.float)
target[:,:,2] = 1.0 # deterministically choose down

# true values from dynamic programming
v_true = dynamic_programming(env, 0.5, target, .000001)
# print(v_true)
test_states = [[i,j] for i in range(11) for j in range(11)]

agent = OffPolicyAgent_FourRooms('FourRooms', 2500, env, target, uniform_random_behavior, lr, discount)
# convert coordinates to one hot encoding
test_features = agent.construct_features(test_states)
prediction = np.reshape(agent.model.predict([test_features, np.array([0.]*121)]), (11,11))
prediction = prediction*np.array(np.invert(env.rooms), dtype=float)
mse = np.mean(v_true-prediction)
mses=[]
mses.append(mse)
for j in range(5):
    for i in range(256):
        agent.generate_episode()
        agent.train_batch(16, 4)
    # print current value function
    prediction = agent.model.predict([test_features, np.array([0.]*121)])
    prediction = np.reshape(prediction, (11,11))
    prediction = prediction*np.array(np.invert(env.rooms), dtype=float) # zero the values along the walls
    print(prediction)
    mse = np.mean(abs(v_true-prediction))
    mses.append(mse)
print(prediction)
print("error sequence:", mses)
print("final MeanVError", mses[-1])
print("t:", agent.t)