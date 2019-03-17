#!/usr/bin/env python
# coding: utf-8

# In[39]:


import gym
import numpy
from numpy.random import randint as rand
import matplotlib.pyplot as pyplot

def maze(width=81, height=51, complexity=.75, density=.75):
    # Only odd shapes
    shape = ((height // 2) * 2 + 1, (width // 2) * 2 + 1)
    # Adjust complexity and density relative to maze size
    complexity = int(complexity * (5 * (shape[0] + shape[1]))) # number of components
    density    = int(density * ((shape[0] // 2) * (shape[1] // 2))) # size of components
    # Build actual maze
    Z = numpy.zeros(shape, dtype=bool)
    # Fill borders
    Z[0, :] = Z[-1, :] = 1
    Z[:, 0] = Z[:, -1] = 1
    # Make aisles
    for i in range(density):
        x, y = rand(0, shape[1] // 2) * 2, rand(0, shape[0] // 2) * 2 # pick a random position
        Z[y, x] = 1
        for j in range(complexity):
            neighbours = []
            if x > 1:             neighbours.append((y, x - 2))
            if x < shape[1] - 2:  neighbours.append((y, x + 2))
            if y > 1:             neighbours.append((y - 2, x))
            if y < shape[0] - 2:  neighbours.append((y + 2, x))
            if len(neighbours):
                y_,x_ = neighbours[rand(0, len(neighbours) - 1)]
                if Z[y_, x_] == 0:
                    Z[y_, x_] = 1
                    Z[y_ + (y - y_) // 2, x_ + (x - x_) // 2] = 1
                    x, y = x_, y_
    return Z

pyplot.figure(figsize=(10, 5))


pyplot.imshow(Z, cmap=pyplot.cm.binary, interpolation='nearest')
pyplot.xticks([]), pyplot.yticks([])
pyplot.show()



# In[40]:


class Maze(gym.Env):
    def __init__(self, size = 5):
        self.mapAction = {0:(-1,0),
                         1: (0,1),
                         2: (1,0),
                         3: (0,-1)}
        self.maze = maze(size,size).astype(float)
        self.pos = (1,1)
        print(self.pos)
        self.goal = (size-2, size-2)
        self.maze[self.pos] = 2
        print(self.maze[self.pos])
        self.maze[self.goal] = 3
        self.actionHistory = []
    def render(self):
        Z = self.maze * 255.0 / 5.0
        #print(Z)
        pyplot.imshow(Z, cmap=pyplot.cm.tab20c, interpolation='nearest')
        #pyplot.imshow()
        pyplot.xticks([]), pyplot.yticks([])
        pyplot.pause(0.1)
        pyplot.show()
        
    def step(self, action):
        newPos = self.mapPos(self.pos,action)
        if(self.maze[newPos] == 1): #is wall
           return self.pos,-2,False,""
        elif(newPos == self.goal): # is finished
            return self.pos, 10, True, ""
        else:
            self.maze[self.pos] = 4
            self.maze[newPos] = 2
            self.pos = newPos
            return self.pos, -1, False, ""
    def sample(self):
        return rand(4)
            
            
    def mapPos(self, a, d):
        b = self.mapAction[d]
        c = (a[0] + b[0], a[1] + b[1])
        #print(c)
        return c
        
        


# In[41]:


env = Maze()
env.render()
isDone = False
while not isDone:
    s_next,r,isDone,_ = env.step(env.sample())
    
    env.render()


# In[ ]:




