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
