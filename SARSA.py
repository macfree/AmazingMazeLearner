#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 21:04:38 2019

@author: x
"""

from q_learning import QLearningAgent
from collections import deque

class ExpectedSARSAagent(QLearningAgent):
    
    def __init__(self,env,alpha,gamma=0.99,base_value=-1,eps=0.9,eps_decay = 0.9):
        QLearningAgent.__init__(self,env,alpha,gamma,base_value,eps,eps_decay)

    
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
        
    def __repr__(self):
        return "Expected_SARSA(0)"
    
        
class SARSAAgent(QLearningAgent):
    
    def __init__(self,env,alpha,gamma=0.99,base_value=-1,eps=0.9,eps_decay = 0.9):
        QLearningAgent.__init__(self,env,alpha,gamma,base_value,eps,eps_decay)
    
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
        
    def __repr__(self):
        return "SARSA(0)"
        
        
class NStepSARSAAgent(QLearningAgent):
    
    def __init__(self,env,alpha=.3, N=4,gamma=0.99,base_value=-1,eps=0.9,eps_decay = 0.9):
        
        QLearningAgent.__init__(self,env,alpha,gamma,base_value,eps,eps_decay)
        self.N = N
        self.gamma_weights = [ gamma**i for i in range(1,self.N+1) ]
    
    def run_episode(self,render):
        
        
        env = self.env
        
        state = env.reset()
        action = self.policy(state)
        
        is_done = False
        T = 1000000000
        acc_reward  = 0.0
        i = 0
        tau = i - self.N - 1 
        
        self._if_render(render,i,acc_reward,state,0)
        
        trajectory = []
        rewards = []
        
        while not is_done and tau  < T: # hmmm 
            
            if not is_done:
                
                self.nr_env_interactions += 1
                greedy_action = self.greedy_policy(state)
                self.env.save_greedy_policy(state,greedy_action)
                
                nxt_state, reward, is_done, _ =  self.env.step(action)
                nxt_action = self.policy(nxt_state)
                trajectory.append((state,action))
                rewards.append(reward)
            
                if is_done:
                    T = len(trajectory)
            
            if tau>=0:
                
                N_min = min(self.N - 1, len(trajectory) - tau) # should be at least 1 
                
                Gt = sum(map(lambda x,y: x*y,self.gamma_weights[0:N_min],rewards[tau:tau+N_min]))
                
                if tau + self.N < T: # same as  i+1 < len(trajectory) or is_done
                    
                    Gt += self.gamma_weights[-1] * self.Q_value(state, action)
                    
                    
                
                state_tau, action_tau = trajectory[tau]
                
                self.set_Q_value(state_tau, action_tau, Gt)
            
            self._if_render(render,i,acc_reward,state,action)
            
            i += 1
            tau += 1
            
            if not is_done:    
                state = nxt_state
                action = nxt_action                
                acc_reward += reward
                
        return acc_reward
    
    def __repr__(self):
        return "SARSA(%i)"%(self.N)
    
    def train(self,n_eps=100,render=False):
        
        reward_history = []
        
        for i in range(n_eps):
        
            acc_reward = self.run_episode(render)
            
#            print "Nr_interactions: %i eps: %f"%(self.nr_env_interactions,self.eps)
            
            self.eps *= self.eps_decay            
            reward_history.append(acc_reward)
            
            
        return reward_history
    