"""
Basic Rejection ABC.
This is the first Basic Rejection ABC algorithm.
It aims to start with a basic summary statistics to try and
estimate the posterior distribution of beta, gamma, rho,
using a simple to understand summary statistics.

This would later then be modified further, by adjusting
the summary statistics to try and obtain a more accurate
posterior approximation. In short, this acts as a first
attempt at constructing the ABC algorithm.
"""

import numpy as np
from simulator import simulate
import pandas as pd

# Extract the summary statistics required from observations.
infected = pd.read_csv("data/infected_timeseries.csv")
rewiring = pd.read_csv("data/rewiring_timeseries.csv")
degree = pd.read_csv("data/final_degree_histograms.csv")

split_infected = {val: infected[infected['replicate_id'] == val]
                  for val in infected['replicate_id'].unique()}
split_rewiring = {val: rewiring[rewiring['replicate_id'] == val]
                  for val in rewiring['replicate_id'].unique()}
split_degree = {val: degree[degree['replicate_id'] == val]
                for val in degree['replicate_id'].unique()}

infected_max = np.array(
    [np.max(split_infected[i]['infected_fraction']) for i in range(40)])
infected_argmax = np.array([np.max(split_infected[i]['infected_fraction'])/(split_infected[i]
                           ['infected_fraction'].iloc[np.argmax(split_infected[i]['infected_fraction'])+1]) for i in range(40)])
rewire_max = np.array(
    [np.max(split_rewiring[i]['rewire_count']) for i in range(40)])


# Set seed for reproducability.
np.random.seed(3247)

# Fix the number of simulations.
Nsim = 10000

# Generate beta, gamma, rho using the prior distributions.
beta = np.random.uniform(0.05, 0.50, Nsim)
gamma = np.random.uniform(0.02, 0.20, Nsim)
rho = np.random.uniform(0.0, 0.8, Nsim)
