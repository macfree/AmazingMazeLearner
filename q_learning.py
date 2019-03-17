#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 20:48:24 2019

@author: x
"""
import gym
from collections import defaultdict
import random
import matplotlib.pyplot as plt 
import numpy as np
from mpl_toolkits.mplot3d import axes3d

class QLearningAgent:
    
    
    def __init__(self,env,alpha,gamma=0.99,base_value=-1,eps=0.9):
        
        base_value = -1.0
        self.n_actions = env.action_space.n
        self.q_table = defaultdict(lambda: base_value)
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        
        
    def Q_value(self,state,action):
        return self.q_table[(state,action)]
    
    def set_Q_value(self,state,action,Q_new):
        self.q_table[(state,action)] = Q_new 
        
    def policy(self,state,greedy=False):
        
        if random.random() < self.eps or greedy:
            
            action_max = 0
            q_max = self.Q_value(state,action_max) 
            
            for action in range(self.n_actions):
                
                q = self.Q_value(state,action) 
                
                if q > q_max:
                    action_max = action
            
            action = action_max
            
        else:
            
            action = random.randint(0,self.n_actions-1)
            
        return action
    
    def train(self,n_eps=100,render=False):
        
        env = self.env
        
        reward_history = []
        
        for i in range(n_eps):
        
            is_done = False
            
            state = env.reset()
            
            acc_reward  = 0.0
            i = 0
            
            if render:
                env.render(title="Step %i, accumulated reward: %0.2f"%(i,acc_reward))
            
            
        
            while not is_done:
                
                action = self.policy(state)
                
                nxt_state, reward, is_done, _ =  env.step(action)
                
                self.update(state,action,reward,is_done,nxt_state)
                
                state = nxt_state
                
                acc_reward += reward
                
                if render:
                    env.render(title="Step %i, accumulated reward: %f"%(i,acc_reward))
                
                
            reward_history.append(acc_reward)
            
        return reward_history
                
        
    def update(self,state,action,reward,is_done,nxt_state):
        
        Q_old = self.Q_value(state,action)
        
        Q_nxt_state = self.Q_value(nxt_state,self.policy(nxt_state,greedy=True))
        
        Q_target = reward + self.gamma*Q_nxt_state * (1.0-is_done)
        
        Q_update = Q_old + self.alpha * (Q_target-Q_old)
        
        self.set_Q_value(state,action,Q_update)

class DoubleQLearningAgent:
    
    def __init__(self,env,alpha,gamma=0.99,base_value=-1,eps=0.9):
        pass
    
    
def evaluate_agent(agent,n_epochs,n_iterations):    
    
    reward_history = []
    
    for epoch in range(n_epochs):
        
        rewards = agent.train(n_iterations)
        
        reward_history.append(sum(rewards)/n_iterations)
        
    return sum(reward_history)/n_epochs
    
if __name__=="__main__":
    pass
    
    
        
        
