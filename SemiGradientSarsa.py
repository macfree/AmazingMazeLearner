# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:17:29 2019

@author: Markus
"""

import tensorflow as tf
import gym
import numpy as np
import random
env = gym.make("CartPole-v0")

tf.reset_default_graph()
session=tf.InteractiveSession()

def policy(q_values, eps):
    if random.random() < eps:
        action = np.argmax(q_values[0])
    else:
        action = random.randint(0,n_actions-1)
    return action


session = tf.InteractiveSession()
alpha = 10**-4
eps =0.5
gamma = 0.99
episodes = 1000

try:
    n_actions = env.action_space.n
    obs_dims = env.observation_space.shape
    action_ph = tf.placeholder(tf.int32, (None,), name="action_ph" )
    state_ph = tf.placeholder(tf.float32, (None,obs_dims[0]),name="state_ph" )
    q_targets_ph = tf.placeholder(tf.float32, (None,), name="q_targets_ph")
    
    out = tf.layers.dense(state_ph, 10, activation=tf.nn.relu)
    q_values_nn = tf.layers.dense(out, n_actions, activation=None)
    
    action_ph_one_hot = tf.one_hot(action_ph,n_actions)
    q_values_one_hot = tf.math.multiply(action_ph_one_hot,q_values_nn)
    q_values_reduced_sum = tf.reduce_sum(q_values_one_hot, axis=-1)
    
    TD_error = tf.math.squared_difference(q_targets_ph,q_values_reduced_sum)
    TD_cost = tf.reduce_mean(TD_error)
    
    train_opt = tf.train.AdamOptimizer(learning_rate=alpha).minimize(TD_cost)
    
    session.run(tf.global_variables_initializer())

    
    for ep in range(episodes):
        print(ep)       
        done = False
        #pi = 
        state =  env.reset()
        while not done:
            
            q_values = session.run(q_values_nn, {state_ph:[state]})
            action =  policy(q_values, eps)
            state_next, reward, done, info = env.step(action)
            q_values_next_state = session.run(q_values_nn, {state_ph:[state_next]})
            q_values_next_state = np.ravel(q_values_next_state)
            action_next = policy(q_values_next_state, eps)
            q_target = reward+gamma*q_values_next_state[action_next]*(1-done)
            loss, _ = session.run([TD_cost, train_opt], {state_ph:[state], action_ph:[action], q_targets_ph:[q_target]})
            print(loss)
except Exception as e:
    print(e)
    
session.close()