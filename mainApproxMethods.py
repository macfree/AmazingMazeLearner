#!/usr/bin/python

from AmazingMaze import Maze, InterActiveAgent
from q_learning import QLearningAgent
from dynaq_learning import DynaQLearningAgent

from SARSA import ExpectedSARSAagent, SARSAAgent, NStepSARSAAgent
from MCC import FirstVistMCC
import matplotlib.pyplot as plt

env  = gym.make("CartPole-v0")

agents = [
        
        ]

rewards = {str(agent):agent.train(1,render=False) for agent in agents}

# apa = InterActiveAgent(agent,env)    
# apa.run_interactive()
#reward = agent.train(1)

plt.figure(2)

for agent_name in rewards:
    plt.plot(rewards[agent_name],label=agent_name)

plt.legend(loc="best")
plt.show()


