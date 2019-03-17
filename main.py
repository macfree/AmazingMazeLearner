#!/usr/bin/python

from AmazingMaze import Maze
from q_learning import QLearningAgent
import matplotlib.pyplot as plt


env  = Maze(20)

agent = QLearningAgent(env,0.3,eps=0.8)

reward = agent.train(100)

plt.figure(1)
plt.plot(reward)
plt.show()

agent.train(1,render=True)