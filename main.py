#!/usr/bin/python

from AmazingMaze import Maze, InterActiveAgent
from q_learning import QLearningAgent
from dynaq_learning import DynaQLearningAgent
from SARSA import ExpectedSARSAagent, SARSAAgent, NStepSARSAAgent
from MCC import FirstVistMCC
import matplotlib.pyplot as plt

env  = Maze(15)

agents = [NStepSARSAAgent(env, alpha=0.3, N=5, gamma=0.99, eps=0.9, eps_decay=0.99),
          SARSAAgent(env,alpha=0.35,eps=0.99),
          QLearningAgent(env,0.5,eps=0.8),
          ExpectedSARSAagent(env,alpha=0.34,eps=0.99),
          FirstVistMCC(env,gamma=0.99,eps=0.9,eps_decay=0.99),
          DynaQLearningAgent(env,alpha=0.9, eps=0.99,nr_updates=10)
        ]

rewards = {str(agent):agent.train(100,render=False) for agent in agents}

# apa = InterActiveAgent(agent,env)    
# apa.run_interactive()
#reward = agent.train(1)

plt.figure(2)

for agent_name in rewards:
    plt.plot(rewards[agent_name],label=agent_name)

plt.legend(loc="best")
plt.show()


