from OffPolicyAgent import OffPolicyAgent
from random_walk_env import RandomWalkEnv
import numpy as np
import gym
from IRAgent import IRAgent
from OffPolicyAgent_FourRooms import OffPolicyAgent_FourRooms
from four_rooms_env import FourRoomsEnv
import matplotlib.pyplot as plt
from IRAgent_FourRooms import IRAgent_FourRooms
from PERAgent import PERAgent
import pandas as pd
from datetime import datetime
from DP_walk import dynamic_programming_random_walk


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


def learning_curve(agents, true_value, discount):
    n_updates = 1000
    runs = np.random.randint(0,999,10)
    steps_per_update = 16
    colors = ['r','b','k','g','y']
    df = pd.DataFrame()
    for i in range(len(agents)):
        agent = agents[i]
        mave_array = np.zeros((len(runs),n_updates+1), dtype=np.float)
        for run_idx, run in enumerate(runs):
            agent.reset(run) # run == random seed
            all_actions = agent.generate_all_actions(True)
            prediction = np.choose(all_actions,agent.value_function().T)
            mave_array[run_idx,0] = np.mean(np.abs(true_value-prediction))
            for j in range(n_updates):
                agent.generate_experience(steps_per_update)
                agent.train_batch(16, 16)
                prediction = np.choose(agent.generate_all_actions(True),agent.value_function().T)
                mave_array[run_idx,j+1] = np.mean(np.abs(true_value-prediction))
        mean = np.mean(mave_array,axis=0)
        std = np.std(mave_array,axis=0)
        plt.figure(1)
        steps = [update*steps_per_update for update in range(n_updates+1)]
        plt.plot(steps, mean, color=colors[i], label=agent.name)
        plt.fill_between(steps,mean-std, mean+std, color = colors[i], alpha=.3)
        df = pd.concat([df, pd.DataFrame(data={'steps':steps,'mean':mean,'std':std,'agent':type(agents[i]).__name__})])
        plt.legend()
    plt.axis([0,steps[-1],0,1])
    df.to_csv("results_"+str(lr)+"_"+str(discount)+"_"+datetime.strftime(datetime.now(), "%y%m%d%H%M")+".csv",index=False)
    plt.xlabel('Training steps')
    plt.ylabel('MAVE')
    plt.title('Learning Curve\nlr={},discount={}'.format(lr,discount))
    plt.show()


def learning_rate_sensitivity(agents, true_value,discount):
    n_updates = 1000
    filename = "results_"+"lrsens"+"_"+str(discount)+"_"+datetime.strftime(datetime.now(), "%y%m%d%H%M%S")+".csv"
    # [[0.0, 0.001, 0.01]; collect(0.025:0.025:0.2); collect(0.25:0.05:1.0); collect(1.25:0.25:2.0)]
    #lrs = [0.01]+np.arange(0, 6.1 ,.1)
    df = pd.DataFrame(data={'lr':[], 'mean':[], 'agent':[]})
    df.to_csv(filename, header=True,index=False)

    lrs = (0.01*((10/5.5)**np.arange(1,13,1)))*1000//10/100
    print(lrs)
    runs = np.random.randint(0,999,25)
    steps_per_update = 16
    colors = ['r','b','k','g','y']

    for i in range(len(agents)):
        agent = agents[i]
        mave_array = np.zeros((len(runs),len(lrs)), dtype=np.float)

        for lr_idx in range(len(lrs)):
            print('agent:',agent.name,'lr=',lrs[lr_idx])
            agent.lr = lrs[lr_idx]
            diverged = 0
            for run_idx, run in enumerate(runs):
                agent.reset(run) # run == random seed
                for k in range(n_updates):
                    diverged = agent.generate_experience(steps_per_update)
                    if diverged:
                        print("diverged")
                        break
                    agent.train_batch(16, 16)
                if diverged:
                    break
                prediction = np.choose(agent.generate_all_actions(True),agent.value_function().T)
                mave_array[run_idx,lr_idx] = np.mean(np.abs(true_value-prediction))
                if mave_array[run_idx,lr_idx] != mave_array[run_idx,lr_idx]: # nan, diverged
                    print("diverged")
                    mave_array[run_idx:,lr_idx] = np.nan # fill rest of array
                    break
        mean = np.mean(mave_array,axis=0)
        std = np.std(mave_array,axis=0)
        plt.figure(1)
        plt.plot(lrs, mean, color=colors[i], label=agent.name)
        df = pd.DataFrame(data={'lr':lrs, 'mean':mean, 'agent':type(agents[i]).__name__})
        df.to_csv(filename, header=False,index=False,mode='a')

        # plt.fill_between(lrs,mean-std, mean+std, color = colors[i], alpha=.3)
        plt.legend()
    plt.axis([0,lrs[-1],0,1])
    plt.xlabel('Learning Rate')
    plt.ylabel('MAVE')
    plt.title('Learning Rate Sensitivity\nlr={},discount={}'.format(lr,discount))
    plt.show()

def steps_per_update(agent,true_value):
    n_steps = 1024*16
    step_sizes = [4,8,16,32,64,128,256,512]
    n_update_arr = np.floor_divide(n_steps,step_sizes)
    print(n_update_arr)
    runs = np.random.randint(0,999,5)
    colors = ['r','b','k','g','y']
    for i in range(len(agents)):
        print("agent:",i)
        agent = agents[i]
        mave_array = np.zeros((len(runs),len(n_update_arr)), dtype=np.float)
        for update_idx in range(len(n_update_arr)):
            steps_per_update = step_sizes[update_idx]
            n_updates = n_update_arr[update_idx]
            print('steps=',steps_per_update)
            for run_idx, run in enumerate(runs):
                agent.reset(run) # run == random seed
                for k in range(n_updates):
                    agent.generate_experience(steps_per_update)
                    agent.train_batch(16, 16)
                prediction = agent.value_function()
                mave_array[run_idx,update_idx] = np.mean(np.abs(true_value-prediction))
        mean = np.mean(mave_array,axis=0)
        std = np.std(mave_array,axis=0)
        plt.figure(1)
        plt.plot(np.log2(n_update_arr), mean, color=colors[i], label=agent.name)
        plt.fill_between(lrs,mean-std, mean+std, color = colors[i], alpha=.3)
        plt.legend()
    plt.axis([0,np.log2(n_update_arr[0]),0,1])
    plt.xlabel('Number of Updates (log2)')
    plt.ylabel('MAVE')
    plt.title('Steps per Sensitivity\nlr={},discount={}'.format(lr,discount))
    plt.show()


# test agent
lr=0.7
discount=.9

# Random walk policies and true value function
env_walk = RandomWalkEnv(10)
uniform_random_behavior=np.full(shape=(10,2), fill_value=0.5, dtype=np.float)
left_behavior = np.zeros(shape=(10,2), dtype=np.float)
left_behavior[:,0] = 0.5
left_behavior[:,1] = 0.5
target_policy=np.zeros(shape=(10,2), dtype=np.float)
target_policy[:,0] = 0.1
target_policy[:,1] = 0.9
true_value_walk = dynamic_programming_random_walk(env_walk, discount, target_policy, .000000000001)
per_agent_walk = PERAgent(15000, env_walk, target_policy, left_behavior, lr, discount, 'IR') # best lr ~= 2 diverges after 3
bc_agent_walk = IRAgent(15000, env_walk, target_policy, left_behavior, lr, discount, 'BC') # best lr = 2 to 4 but diverges after 4
is_agent_walk = OffPolicyAgent(15000, env_walk, target_policy, left_behavior, lr, discount) # best lr = 0.4, diverges at 1
wis_minibatch_agent_walk = OffPolicyAgent(15000, env_walk, target_policy, left_behavior, lr, discount, 'WIS_minibatch') # best lr = <1
# Four Rooms params and agents
env_rooms = FourRoomsEnv()
grid_size = 11
uniform_random_behavior=np.full(shape=(grid_size,grid_size,4), fill_value=0.25, dtype=np.float)
random_sel_states = np.random.choice(grid_size**2, 25)
for random_sel_state in random_sel_states:
    uniform_random_behavior[random_sel_state//grid_size, random_sel_state%grid_size,2] = 0.05
    uniform_random_behavior[random_sel_state//grid_size, random_sel_state%grid_size, [0,1,3]] = 0.95/3
target_policy=np.zeros(shape=(11,11,4), dtype=np.float)
target_policy[:,:,2] = 1.0 # deterministically choose down
true_value_rooms = dynamic_programming_FourRooms(env_rooms, discount, target_policy, .000000000001)
is_agent_rooms = OffPolicyAgent_FourRooms(2500, env_rooms, target_policy, uniform_random_behavior, lr, discount)
wis_minibatch_agent_rooms = OffPolicyAgent_FourRooms(2500, env_rooms, target_policy, uniform_random_behavior, lr, discount, 'WIS_minibatch')
ir_agent_rooms = IRAgent_FourRooms(2500, env_rooms, target_policy, uniform_random_behavior, lr, discount, "IR")
bc_agent_rooms = IRAgent_FourRooms(2500, env_rooms, target_policy, uniform_random_behavior, lr, discount, "BC")


# choose agent
# true_value = true_value_rooms
true_value = true_value_walk
agents = [per_agent_walk]
# agents = [is_agent_rooms, wis_minibatch_agent_rooms, ir_agent_rooms, bc_agent_rooms]

# figure 1 final plot:

# find best learning rate for each method
learning_curve(agents, true_value,discount)

# learning_curve(agents_temp,true_value)
# steps_per_update(agents,true_value)
