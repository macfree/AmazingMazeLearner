# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 13:17:29 2019

@author: Markus
"""

import tensorflow as tf
import gym
env = gym.make("CartPole-v0")

tf.reset_default_graph()
session=tf.InteractiveSession()

alpha = 10**-4

n_actions = env.action_space.n
obs_dims = env.observation_space.shape
action_ph = tf.placeholder(tf.int32, (None,) )
state_ph = tf.placeholder(tf.float32, (None,obs_dims[0]) )
q_targets_ph = tf.placeholder(tf.float32, (None,) )

out = tf.layers.dense(state_ph, 10, activation=tf.nn.relu)
q_values = tf.layers.dense(out, n_actions, activation=None)

action_ph_one_hot = tf.one_hot(action_ph,n_actions)
q_values_one_hot = tf.math.multiply(action_ph_one_hot,q_values)
q_values_reduced_sum = tf.reduce_sum(q_values_one_hot, axis=-1)

TD_error = tf.math.squared_difference(q_targets_ph,q_values_reduced_sum)
TD_cost = tf.reduce_mean(TD_error)

train_opt = tf.train.AdamOptimizer(learning_rate=alpha).minimize(TD_cost)

session.run(tf.global_variables_initializer())
