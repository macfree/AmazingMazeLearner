#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from utils import BaseValueAgent

class QLearningAgent(BaseValueAgent):
    
    def __init__(self,env,alpha,gamma=0.99,base_value=-1,eps=0.9,eps_decay = 0.9):
        
        self.base_value = base_value
        BaseValueAgent.__init__(self,env,eps,alpha,gamma,eps_decay)
        
    def set_Q_value(self,state,action,Q_target):
        
        Q_old = self.Q_value(state,action)
        
        Q_update = Q_old + self.alpha * (Q_target-Q_old)
        
        self.q_table[(state,action)] = Q_update 
    
    def policy(self,state,greedy=False):
        return self.eps_greedy_policy(state,greedy)
    
    def _step(self,state):
        
        action = self.policy(state)
        
        greedy_action = self.greedy_policy(state)
        
        self.env.save_greedy_policy(state,greedy_action)
        
        nxt_state, reward, is_done, _ =  self.env.step(action)
        
        self.update(state,action,reward,is_done,nxt_state)
        
        self.nr_env_interactions +=1
        
        return action, nxt_state, reward, is_done

    def _if_render(self,render,step_nr,acc_reward,state,action):
        
        if render:
            Q_value = self.Q_value(state,action)
            self.env.render(title="Step %i,eps: %f, accumulated reward: %0.2f, action: %i Q(s,a): %f"%(step_nr,self.eps,acc_reward,action,Q_value),plot_greedy_action=True)

    def train(self,n_eps=100,render=False):
        
        env = self.env
        
        reward_history = []
        
        for i in range(n_eps):
        
            is_done = False
            
            state = env.reset()
            self.nr_env_interactions +=1
            
            acc_reward  = 0.0
            i = 0
            
            self._if_render(render,i,acc_reward,state,0)
            
            while not is_done:
                
                action, nxt_state, reward, is_done = self._step(state)
                
                self._if_render(render,i,acc_reward,state,action)
                
                state = nxt_state
                
                i+=1
                
                acc_reward += reward
                
            self.eps *= self.eps_decay
            
            reward_history.append(acc_reward)
            
        return reward_history
        
    def update(self,state,action,reward,is_done,nxt_state):
        
        Q_nxt_state = self.Q_value(nxt_state,self.policy(nxt_state,greedy=True))
        
        Q_target = reward + self.gamma*Q_nxt_state * (1.0-is_done)
        
        self.set_Q_value(state,action,Q_target)
        
    def __repr__(self):
        return "QLearning"

if __name__=="__main__":
    pass
    
    
        
        
