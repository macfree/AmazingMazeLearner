#!/usr/bin/env python2
# -*- coding: utf-8 -*-
        
def evaluate_agent(agent,n_epochs,n_iterations):    
    
    reward_history = []
    
    for epoch in range(n_epochs):
        
        rewards = agent.train(n_iterations)
        
        reward_history.append(sum(rewards)/n_iterations)
        
    return sum(reward_history)/n_epochs
    