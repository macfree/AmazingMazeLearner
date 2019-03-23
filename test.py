#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 13:44:20 2019

@author: x





"""

import matplotlib.pyplot as plt
import numpy as np




A = np.random.random((10,10))

x = np.arange(0,10)
y = np.arange(0,10)

x_dir = np.ones((10,))
y_dir = np.zeros((10,))

X,Y = np.meshgrid(x,y)

X_dir,Y_dir = np.meshgrid(x_dir,y_dir)
X_dir[0,0] = 0

plt.imshow(A)
plt.quiver(X,Y,X_dir,Y_dir)
