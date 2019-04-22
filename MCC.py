#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 15:26:50 2019

@author: x
"""

from utils import BaseValueAgent
import numpy as np
from collections import defaultdict

class FirstVistMCC(BaseValueAgent):
    
    def __init__(self,env,gamma,eps,eps_decay=1.0):
        
        BaseValueAgent.__init__(self,env,eps,0.0,gamma,eps_decay)
        self._counter = defaultdict(lambda : 0)
        
        prob = 1.0/self.n_actions
        self.pi =  defaultdict(lambda : [prob for a in range(self.n_actions)] )
        
    def policy(self,state):
        return np.random.choice(self.n_actions, p=self.pi[state])
    
    def _if_render(self,render,step_nr,acc_reward,state,action):
        
        if render:
            Q_value = self.Q_value(state,action)
            self.env.render(title="Step %i,eps: %f, accumulated reward: %0.2f, action: %i Q(s,a): %f"%(step_nr,self.eps,acc_reward,action,Q_value),plot_greedy_action=True)
    
    def generate_episode(self,render=False):
        
        s = self.env.reset()
        is_done = False
        
        rewards = []
        state_actions = []
        
        acc_reward = 0.0
        i = 0
        
        while not is_done:
            
            a = self.policy(s)
            nxt_s, r, is_done, _= self.env.step(a)
            
            rewards.append(r)
            state_actions.append((s,a))
            
            self._if_render(render,i,acc_reward,s,a)
            
            acc_reward += r
            i+=1
            s = nxt_s
            
        return state_actions, rewards
    
    def set_Q_value(self,s,a,Gt):
        
        N = self._counter[(s,a)]
                    
        self.q_table[(s,a)] = float((self.q_table[(s,a)]*N + Gt))/(N+1)
        
        self._counter[(s,a)] += 1
        
    
    def train(self,n_eps=10,render=False):
        
        reward_history = []
        
        for i_eps in range(n_eps):
            
            state_actions, rewards = self.generate_episode(render)
            
            self.eps *= self.eps_decay
            
            Gt = 0.0
            rev_state_actions = reversed(state_actions)
            rev_rewards = reversed(rewards)
            
            for i, ((s,a), r) in enumerate(zip(rev_state_actions,rev_rewards)):

                Gt = r + Gt*self.gamma
                
                visited_states = []
                
                if (s,a) not in state_actions[0:-(i+1)]:
                    self.set_Q_value(s,a,Gt)
                    visited_states.append(s)
                    
                for s in set(visited_states):
                    
                    a_max = self.greedy_policy(s)
                    self.env.save_greedy_policy(s,a_max)
                    
                    self.pi[s] = [ self.eps/self.n_actions ]*self.n_actions
                    self.pi[s][a_max] +=  (1-self.eps)
                    
            reward_history.append(sum(rewards))
        
        return reward_history
    
    def state_action_info(self,state,action):
        return "counter: %i pi(a|a): %f "%(self._counter[(state,action)],self.pi[state][action])
    
    def __repr__(self):
        return "FirstVistitMCC"

    
    