#!/usr/bin/python

from AmazingMaze import Maze
from q_learning import QLearningAgent
from dynaq_learning import DynaQLearningAgent
from SARSA import ExpectedSARSAagent, SARSAAgent

import matplotlib.pyplot as plt

env  = Maze(10)

#agent = QLearningAgent(env,0.5,eps=0.8)
agent = DynaQLearningAgent(env,alpha=0.35,eps=0.99,nr_updates=10)

#agent = ExpectedSARSAagent(env,alpha=0.34,eps=0.99)
#agent = SARSAAgent(env,alpha=0.35,eps=0.99)

reward = agent.train(100,render=False)

_ = agent.train(1,render=True)

# agent.run_interactive(True)

plt.figure(2)
plt.plot(reward)
plt.show()
