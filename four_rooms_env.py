# four rooms details
    # not doing continuous 4 rooms, because it will take longer to code and there are many details in their code that are hard
    # to reproduce (noise params, etc)
    # 11x11 grid
    # 64 tilings, 8 tiles
    # world looks like this:
    # Walls    [0 0 0 0 0 1 0 0 0 0 0;
    #           0 0 0 0 0 1 0 0 0 0 0;
    #           0 0 0 0 0 0 0 0 0 0 0;
    #           0 0 0 0 0 1 0 0 0 0 0;
    #           0 0 0 0 0 1 0 0 0 0 0;
    #           1 0 1 1 1 1 0 0 0 0 0;
    #           0 0 0 0 0 1 1 1 0 1 1;
    #           0 0 0 0 0 1 0 0 0 0 0;
    #           0 0 0 0 0 1 0 0 0 0 0;
    #           0 0 0 0 0 0 0 0 0 0 0;
    #           0 0 0 0 0 1 0 0 0 0 0;]
    # state= row,col -> 0,0 is top left corner

import numpy as np
class FourRoomsEnv():
    def __init__(self):
        self.rooms = self.build_rooms()
        self.reset()
        self.size = (11,11)
        self.nA = 4

    # action = 0,1,2,3 -> up,right,down,left
    def step(self, action):
        s2 = [self.state[0], self.state[1]]
        if action==0:
            s2[0] -= 1
        elif action==1:
            s2[1] += 1
        elif action==2:
            s2[0] += 1
        else:
            s2[1] -= 1
        s2 = np.clip(s2,0,10)       # agent hit outer wall, Ouch! cancel movement
        if self.rooms[s2[0],s2[1]]: # agent hit inner wall, Ouch!
            s2[:] = self.state[:]         # cancel movement
        r = 0
        done = False
        if ((s2[0] == self.state[0]) and (s2[1] == self.state[1])): # if agent hit wall it was to be reset to previous position (via np.clip or previous if condition)
            r += 1
            done = True
        self.state = s2[:]
        return (s2,r,done,None)

    def reset(self):
        state = np.random.randint(low=0,high=10, size=2)
        # print(self.rooms[state[0],state[1]])
        while self.rooms[state[0],state[1]]: # if state is on a wall
            state = np.random.randint(low=0,high=10, size=2)
        self.state = state
        return state

    def build_rooms(self):
        rooms = np.zeros((11,11), dtype=bool)
        # walls
        for i in range(11):
            rooms[i,5] = True
        for i in range(5):
            rooms[5,i] = True
            rooms[6,10-i] = True
        # doors
        rooms[2,5] = False
        rooms[9,5] = False
        rooms[5,1] = False
        rooms[6,8] = False
        return rooms
