#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:04:38 2019

@author: x
"""

from q_learning import QLearningAgent

class ExpectedSARSAagent(QLearningAgent):
    
    def __init__(self,env,alpha=0.3,gamma=0.99,base_value=0.0,eps=0.9):
        QLearningAgent.__init__(self,env,alpha,gamma=0.99,base_value=0.0,eps=0.9)
    
    def _expected_Q_value(self,state):
        
        N = self.n_actions
        
        Q_weighted = 0
        
        action_max = self.greedy_policy(state)
        
        for action in range(N):
            
            q = self.Q_value(state,action) 
            
            if action == action_max:
                Q_weighted += q * (self.eps + (1-self.eps)*1/N)
            else:
                Q_weighted += q *(1-self.eps)*1/N
        
        return Q_weighted
    
    def update(self,state,action,reward,is_done,nxt_state):
        
        Q_nxt_state = self._expected_Q_value(nxt_state)
        
        Q_target = reward + self.gamma*Q_nxt_state * (1.0-is_done)
        
        self.set_Q_value(state,action,Q_target)
        
class SARSAAgent(QLearningAgent):
    
    def __init__(self,env,alpha=0.3,gamma=0.99,base_value=0.0,eps=0.9):
        QLearningAgent.__init__(self,env,alpha,gamma=0.99,base_value=0.0,eps=0.9)
    
    def _step(self,state,action):
        
        
        greedy_action = self.greedy_policy(state)
        
        self.env.save_greedy_policy(state,greedy_action)
        
        nxt_state, reward, is_done, _ =  self.env.step(action)
        
        nxt_action = self.policy(nxt_state)
        
        self.update(state,action,reward,is_done,nxt_state,nxt_action)
        
        self.nr_env_interactions +=1
        
        return nxt_state, nxt_action, reward, is_done
    
    def train(self,n_eps=100,render=False):
        
        env = self.env
        
        reward_history = []
        
        for i in range(n_eps):
        
            is_done = False
            
            state = env.reset()
            
            action = self.policy(state)
            
            self.nr_env_interactions +=1
            
            acc_reward  = 0.0
            i = 0
            
            self._if_render(render,i,acc_reward,state,0)
            
            while not is_done:
                
                nxt_state, nxt_action, reward, is_done = self._step(state,action)
                
                self._if_render(render,i,acc_reward,state,action)
                
                state = nxt_state
                action = nxt_action
                
                i+=1
                
                acc_reward += reward
                
            self.eps *= self.eps_decay
            
            reward_history.append(acc_reward)
            
        return reward_history
    
    def update(self,state,action,reward,is_done,nxt_state,nxt_action):
        
        Q_nxt_state = self.Q_value(nxt_state,nxt_action)
        
        Q_target = reward + self.gamma*Q_nxt_state * (1.0-is_done)
        
        self.set_Q_value(state,action,Q_target)
    