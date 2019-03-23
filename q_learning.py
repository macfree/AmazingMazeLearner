#!/usr/bin/env python2
# -*- coding: utf-8 -*-

from collections import defaultdict
import random

class QLearningAgent:
    
    def __init__(self,env,alpha,gamma=0.99,base_value=-1,eps=0.9,eps_decay = 0.9):
        
        base_value = 0.0
        self.n_actions = env.action_space.n
        self.q_table = defaultdict(lambda: base_value)
        self.eps = eps
        self.alpha = alpha
        self.gamma = gamma
        self.env = env
        self.eps_decay = eps_decay
        self.nr_env_interactions = 0
        
    def Q_value(self,state,action):
        return self.q_table[(state,action)]
    
    def set_Q_value(self,state,action,Q_target):
        
        Q_old = self.Q_value(state,action)
        
        Q_update = Q_old + self.alpha * (Q_target-Q_old)
        
        self.q_table[(state,action)] = Q_update 
        
    def greedy_policy(self,state):
        
        action_max = 0
        q_max = self.Q_value(state,action_max) 
        
        for action in range(self.n_actions):
            
            q = self.Q_value(state,action) 
            
            if q > q_max:
                
                action_max = action
                
            elif q == q_max and random.random()>=0.5:
                
                action_max = action
        
        return action_max
    
    def policy(self,state,greedy=False):
        
        if random.random() > self.eps or greedy:
            action = self.greedy_policy(state)
        else:
            action = random.randint(0,self.n_actions-1)
            
        return action
    
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

    def _interactive_step(self,state):
        
        greedy_action = self.greedy_policy(state)
        
        self.env.save_greedy_policy(state,greedy_action)
        
        str_list = [ ]
        
        for action in range(self.n_actions):
            if action == greedy_action:
                str_list.append("<%s>: %f"%(self.env.map_action_2_name[action],self.Q_value(state,action)))
            else:
                str_list.append("%s: %f"%(self.env.map_action_2_name[action],self.Q_value(state,action)))
        
        action = input("%s Action> "%(" ".join(str_list)))
        
        nxt_state, reward, is_done, _ =  self.env.step(action)
        
        self.update(state,action,reward,is_done,nxt_state)
        
        self.nr_env_interactions +=1
        
        return action, nxt_state, reward, is_done
        
    def run_interactive(self,render):
        
        env = self.env
        
        is_done = False
            
        state = env.reset()
        self.nr_env_interactions +=1
        
        acc_reward  = 0.0
        i = 0
        
        self._if_render(render,i,acc_reward,state,0)
        
        while not is_done:
            
            action, nxt_state, reward, is_done = self._interactive_step(state)
            
            self._if_render(render,i,acc_reward,state,action)
            
            state = nxt_state
            
            i+=1
            
        self.eps *= self.eps_decay
        
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

if __name__=="__main__":
    pass
    
    
        
        
