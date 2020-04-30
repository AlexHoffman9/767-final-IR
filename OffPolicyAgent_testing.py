from OffPolicyAgent import OffPolicyAgent
from random_walk_env import RandomWalkEnv
import numpy as np
import gym
from IRAgent import IRAgent
from OffPolicyAgent_FourRooms import OffPolicyAgent_FourRooms
from four_rooms_env import FourRoomsEnv
import matplotlib.pyplot as plt
from IRAgent_FourRooms import IRAgent_FourRooms


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

def dynamic_programming_random_walk(env, discount, target_policy, thresh=.001):
    # init values to 0
    v = np.zeros((10,1), dtype=np.float)
    states = range(10)
    actions = range(2)
    delta = 100
    itrs = 0
    while thresh < delta:
        delta = 0
        for s in states:
            vnew = 0.0
            for a in actions:
                env.state = s
                s2,r,done,_ = env.step(a)
                if r!=0: # terminated, ignore s2
                    target = r
                else:
                    target = r + discount*v[s2]
                
                vnew += target_policy[s,a]*target
            delta = max(delta, abs(v[s]-vnew))
            v[s] = vnew
        itrs += 1
    return v


def learning_curve(agents, true_value):
    n_updates = 4000 # 4000
    runs = np.random.randint(0,999,25) # set to 25
    mean_save_file = np.zeros((4,n_updates+1))
    std_save_file = np.zeros((4,n_updates+1))
    steps_per_update = 16
    colors = ['r','b','k','g','y']
    for i in range(len(agents)):
        agent = agents[i]
        mave_array = np.zeros((len(runs),n_updates+1), dtype=np.float)
        for run_idx, run in enumerate(runs):
            print(agent.name,run_idx)
            agent.reset(run) # run == random seed
            prediction = agent.value_function()
            mave_array[run_idx,0] = np.mean(np.abs(true_value-prediction))
            for j in range(n_updates):
                agent.generate_episode(steps_per_update)
                agent.train_batch(16, 16)
                prediction = agent.value_function()
                mave_array[run_idx,j+1] = np.mean(np.abs(true_value-prediction))
        mean = np.mean(mave_array,axis=0)
        std = np.std(mave_array,axis=0)
        mean_save_file[i,:] = mean
        std_save_file[i,:] = std
        plt.figure(1)
        steps = [update*steps_per_update for update in range(n_updates+1)]
        plt.plot(steps, mean, color=colors[i], label=agent.name)
        plt.fill_between(steps,mean-std, mean+std, color = colors[i], alpha=.3)
        plt.legend()
    plt.axis([0,steps[-1],0,1])
    plt.xlabel('Training steps')
    plt.ylabel('MAVE')
    plt.show()
    np.savetxt('.3mean_lrncurve_4rooms.csv', mean_save_file,delimiter=',')
    np.savetxt('.3std_lrncurve_4rooms.csv', std_save_file,delimiter=',')




def learning_rate_sensitivity(agents, true_value):
    n_updates = 1000
    # [[0.0, 0.001, 0.01]; collect(0.025:0.025:0.2); collect(0.25:0.05:1.0); collect(1.25:0.25:2.0)]
    # lrs = [0.01]+np.arange(0, 6.1 ,.1)
    lrs = [.01,.1,.3,.5,.8,1.1,1.5,2,2.5,3,3.5,4,5,6,7,8,9]
    runs = np.random.randint(0,999,5)
    save_file = np.zeros((4,len(lrs)),dtype=float)
    # runs = np.random.randint(0,999,5) 
    steps_per_update = 16
    colors = ['r','b','k','g','y']
    for i in range(len(agents)):
        agent = agents[i]
        mave_array = np.zeros((len(runs),len(lrs)), dtype=np.float)
        for lr_idx in range(len(lrs)):
            print('agent:',agent.name,'lr=',lrs[lr_idx])
            agent.lr = lrs[lr_idx]
            for run_idx, run in enumerate(runs):
                agent.reset(run) # run == random seed
                for k in range(n_updates):
                    agent.generate_episode(steps_per_update)
                    agent.train_batch(16, 16)
                prediction = agent.value_function()
                mave_array[run_idx,lr_idx] = np.mean(np.abs(true_value-prediction))
                if mave_array[run_idx,lr_idx] != mave_array[run_idx,lr_idx]: # nan, diverged
                    print("diverged")
                    mave_array[run_idx:,lr_idx] = 100 # fill rest of array
                    break
        mean = np.mean(mave_array,axis=0)
        save_file[i,:] = mean
        plt.figure(1)
        plt.plot(lrs, mean, color=colors[i], label=agent.name)
        plt.legend()
    plt.axis([0,lrs[-1],0,.1])
    plt.xlabel('Learning Rate')
    plt.ylabel('MAVE')
    plt.show()
    np.savetxt('lr_4rooms.csv', save_file,delimiter=',')

def steps_per_update(agent,true_value):
    n_steps = 1024*32
    step_sizes = [8,16,32,64,128,256,512]
    n_update_arr = np.floor_divide(n_steps,step_sizes)
    mean_save_file = np.zeros((4,len(step_sizes)))
    std_save_file = np.zeros((4,len(step_sizes)))
    runs = np.random.randint(0,999,10) 
    colors = ['r','b','k','g','y']
    for i in range(len(agents)):
        agent = agents[i]
        mave_array = np.zeros((len(runs),len(n_update_arr)), dtype=np.float)
        for update_idx in range(len(n_update_arr)):
            steps_per_update = step_sizes[update_idx]
            n_updates = n_update_arr[update_idx]
            print('steps=',steps_per_update)
            for run_idx, run in enumerate(runs):
                print(agent.name,run_idx)
                agent.reset(run) # run == random seed
                for k in range(n_updates):
                    agent.generate_episode(steps_per_update)
                    agent.train_batch(16, 16)
                prediction = agent.value_function()
                mave_array[run_idx,update_idx] = np.mean(np.abs(true_value-prediction))
        mean = np.mean(mave_array,axis=0)
        std = np.std(mave_array,axis=0)
        mean_save_file[i,:] = mean
        std_save_file[i,:] = std
        plt.figure(1)
        plt.plot(np.log2(n_update_arr), mean, color=colors[i], label=agent.name)
        plt.fill_between(np.log2(n_update_arr),mean-std, mean+std, color = colors[i], alpha=.3)
        plt.legend()
    plt.axis([np.log2(n_update_arr[-1]),np.log2(n_update_arr[0]),0,0.3])
    plt.xlabel('Number of Updates (log2)')
    plt.ylabel('MAVE')
    plt.show()
    np.savetxt('4roomsupdatesens_mean.csv', mean_save_file,delimiter=',')
    np.savetxt('4roomsupdatesens_std.csv', std_save_file, delimiter=',')


# test agent
lr=0.1
discount=.9

# Random walk policies and true value function
env_walk = RandomWalkEnv(10)
uniform_random_behavior=np.full(shape=(10,2), fill_value=0.5, dtype=np.float)
left_behavior = np.zeros(shape=(10,2), dtype=np.float)
left_behavior[:,0] = 0.9
left_behavior[:,1] = 0.1
target_policy=np.zeros(shape=(10,2), dtype=np.float)
target_policy[:,0] = 0.1
target_policy[:,1] = 0.9
true_value_walk = dynamic_programming_random_walk(env_walk, discount, target_policy, thresh=.000001)
print(true_value_walk)
ir_agent_walk = IRAgent(15000, env_walk, target_policy, left_behavior, lr, discount, 'IR') # best lr ~= 2 diverges after 3
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
is_agent_rooms = OffPolicyAgent_FourRooms(2500, env_rooms, target_policy, uniform_random_behavior, 0.3, discount) # best 0.3, diverge .5
wis_minibatch_agent_rooms = OffPolicyAgent_FourRooms(2500, env_rooms, target_policy, uniform_random_behavior, 0.3, discount, 'WIS_minibatch') # best 0.5, diverge 1
ir_agent_rooms = IRAgent_FourRooms(2500, env_rooms, target_policy, uniform_random_behavior, .3, discount, "IR") # best about 1, diverge 3.5
bc_agent_rooms = IRAgent_FourRooms(2500, env_rooms, target_policy, uniform_random_behavior, .3, discount, "BC") # best about 1, diverge 6
 

# choose agent
true_value = true_value_rooms
# true_value = true_value_walk
# agents = [is_agent_walk, wis_minibatch_agent_walk, ir_agent_walk]
# agents = [is_agent_rooms, wis_minibatch_agent_rooms, ir_agent_rooms, bc_agent_rooms]

# test lr for rooms to find optimal
# learning_rate_sensitivity(agents, true_value)

# figure 1 learning curve final plot: .3,.5,1,1
# true_value = true_value_rooms
# agents = [is_agent_rooms, wis_minibatch_agent_rooms, ir_agent_rooms, bc_agent_rooms]
# learning_curve(agents,true_value)

# figure 1 update sens .3,.3,.3,.3
# true_value = true_value_rooms
# agents = [is_agent_rooms, wis_minibatch_agent_rooms, ir_agent_rooms, bc_agent_rooms]
# steps_per_update(agents,true_value)

#figure 1 lr sens
# true_value = true_value_rooms
# agents = [is_agent_rooms, wis_minibatch_agent_rooms, ir_agent_rooms, bc_agent_rooms]
# learning_rate_sensitivity(agents,true_value)

# figure 4 final plot
true_value = true_value_walk
agents = [is_agent_walk, wis_minibatch_agent_walk, ir_agent_walk, bc_agent_walk]
learning_rate_sensitivity(agents, true_value)


# steps_per_update(agents,true_value)