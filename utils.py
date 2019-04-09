#!/usr/bin/env python2
# -*- coding: utf-8 -*-
        
from collections import defaultdict
import numpy as np
import random

def evaluate_agent(agent,n_epochs,n_iterations):    
    
    reward_history = []
    
    for epoch in range(n_epochs):
        
        rewards = agent.train(n_iterations)
        
        reward_history.append(sum(rewards)/n_iterations)
        
    return sum(reward_history)/n_epochs

class BaseValueAgent:
    
    def __init__(self,env,eps,alpha,gamma,eps_decay):
        
        base_value = 0.0
        self.n_actions = env.action_space.n
        self.q_table = defaultdict(lambda: base_value)
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.eps_decay = eps_decay
        self.nr_env_interactions = 0
        self.env.set_agent(self)
        
    def Q_value(self,state,action):
        return self.q_table[(state,action)]

    def train(self):
        raise NotImplementedError()
        
    def policy():
        raise NotImplementedError()
    
    def set_Q_value():
        raise NotImplementedError()
    
    def greedy_policy(self,state):

        action_max = 0
        q_max = self.Q_value(state,action_max) 
        
        for action in range(self.n_actions):
            
            q = self.Q_value(state,action) 
            
            if q > q_max:
                
                action_max = action
                q_max = q
                
            elif q == q_max and random.random()>=0.5:
                
                action_max = action
        
        return action_max
    
    def eps_greedy_policy(self,state,greedy=False):
        
        if random.random() > self.eps or greedy:
            action = self.greedy_policy(state)
        else:
            action = random.randint(0,self.n_actions-1)
            
        return action
    
    def state_action_info(self,state,action):
        return ""
    
    def q_values_for_state(self, state):
        q_values = np.zeros((4))
        for action in range(self.n_actions):            
            q_values[action] = self.Q_value(state, action)
        return q_values
            
    