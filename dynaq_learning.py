#!/usr/bin/env python2
# -*- coding: utf-8 -*-


import random
from q_learning import QLearningAgent

class DynaQLearningAgent(QLearningAgent):
    
    def __init__(self,env,alpha,gamma=0.99,base_value=-1,eps=0.9,nr_updates=50):
        
        QLearningAgent.__init__(self,env,alpha,gamma=0.99,base_value=-1,eps=0.9)
        self.nr_updates = nr_updates
        self.model = dict()
        
    def train(self,n_eps=100,render=False):
        
        env = self.env
        
        reward_history = []
        
        for i in range(n_eps):
        
            is_done = False
            
            state = env.reset()
            
            acc_reward  = 0.0
            i = 0
            
            self._if_render(render,i,acc_reward,state,0)
           
            while not is_done:
                
                action, nxt_state, reward, is_done = self._step(state)
                
                self.update_model(state,action,nxt_state,reward)
                
                self.run_model()
                
                state = nxt_state
                
                acc_reward += reward
                
                self._if_render(render,i,acc_reward,state,action)
              
                i += 1
                
            self.eps *= self.eps_decay
                
            reward_history.append(acc_reward)
            
        return reward_history
        
    def update_model(self,state,action,next_state,reward):
        self.model[(state,action)] = (next_state,reward) 
        
    def run_model(self):
        
        nr_updates = min(self.nr_updates,len(self.model))
        
        vals = random.sample(self.model,nr_updates)
        
        for state, action in vals:
            
            nxt_state, reward = self.model[(state,action)]
            
            greedy_action = self.greedy_policy(nxt_state)
            
            Q_nxt_state = self.Q_value(nxt_state,greedy_action)
            
            Q_target = reward + self.gamma*Q_nxt_state
            
            self.set_Q_value(state,action,Q_target)
            
            self.env.save_greedy_policy(state,greedy_action)
    def __repr__(self):
        return "DynaQ"
            
    
if __name__=="__main__":
    pass
    
    
        
        
