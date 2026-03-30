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
import matplotlib.pyplot as plt

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


# Fix the number of simulations.
Nsim = 50000

# Generate beta, gamma, rho using the prior distributions.
beta = np.random.uniform(0.05, 0.50, Nsim)
gamma = np.random.uniform(0.02, 0.20, Nsim)
rho = np.random.uniform(0.0, 0.8, Nsim)

# Simulate the data using generated beta, gamma, rho.
infected_sim = []
rewire_sim = []
degree_sim = []
for i in range(Nsim):
    infected_fraction, rewire_counts, degree_histogram = simulate(
        beta[i], gamma[i], rho[i])
    infected_sim.append(infected_fraction)
    rewire_sim.append(rewire_counts)
    degree_sim.append(degree_histogram)

infected_sim = np.array(infected_sim)
rewire_sim = np.array(rewire_sim)
degree_sim = np.array(degree_sim)


# Generate summary statistics based on initial choice of summary statistics.
p_sim = np.max(infected_sim, axis=1)
pr_sim = np.max(infected_sim, axis=1) / \
    infected_sim[np.arange(infected_sim.shape[0]),
                 np.argmax(infected_sim, axis=1)+1]
r_sim = np.max(rewire_sim, axis=1)
sd_p = np.std(p_sim)
sd_pr = np.std(pr_sim)
sd_r = np.std(r_sim)
d = np.sqrt(
    ((p_sim - np.mean(infected_max))/sd_p)**2 +
    ((pr_sim - np.mean(infected_argmax))/sd_pr)**2 +
    ((r_sim - np.mean(rewire_max))/sd_r)**2
)

# Accept only the 1% data
epsilon = np.quantile(d, 0.01)
betafilter = beta[d <= epsilon]
gammafilter = gamma[d <= epsilon]
rhofilter = rho[d <= epsilon]

plt.title("Beta Approximate Posterior")
plt.hist(betafilter, bins=50, density=True)
plt.axvline(x=np.mean(betafilter), color='r')
plt.show()

plt.title("Gamma Approximate Posterior")
plt.hist(gammafilter, bins=50, density=True)
plt.axvline(x=np.mean(gammafilter), color='r')
plt.show()

plt.title("Rho Approximate Posterior")
plt.hist(rhofilter, bins=50, density=True)
plt.axvline(x=np.mean(rhofilter), color='r')
plt.show()
