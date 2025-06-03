# -*- coding: utf-8 -*-
"""
Created on Fri May 30 11:47:09 2025

@author: bruschini.1
"""

# Create two example potentials and store them in the correct file format
# to work with the SPARSE module

import numpy as np
import pandas as pd


def potential_well(x):
    return (0.05 / x - 100) * np.exp(- (100 * x) ** 2)


def linear_potential(x):
    return 5e03 * x


def coupling(x):
    return 50 * (100 * x) ** 2 * np.exp(- (100 * x) ** 2)


names = ['Lower (P-wave)', 'Higher (S-wave)']
l = [1, 0]
thresholds = [0, 100]
mu = [1000, 1000]
channels = pd.DataFrame(
    {'channel': names,
     'l': l,
     'threshold': thresholds,
     'mu': mu}
    )

max_radius = 1.
mesh_points = int(1e6) + 2
r = np.linspace(0, max_radius, mesh_points)[1:-1]
pot = np.empty((len(r), len(channels), len(channels)))
pot[:, 0, 0] = thresholds[0] + potential_well(r)
pot[:, 1, 1] = thresholds[1] + potential_well(r)
pot[:, 0, 1] = coupling(r)
pot[:, 1, 0] = coupling(r)
potential = pd.DataFrame(pot.reshape(len(r), -1), index=r)

if __name__ == "__main__":
    print('writing channels... ')
    channels.to_csv('channels.csv', index=False)
    print('writing potential... ')
    potential.to_csv('potential.csv', header=False)
