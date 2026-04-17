"""
This code runs the simulation 40 times, then compares
it to the observed data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from simulator import simulate

# Fix the number of simulations.
Nsim = 40

# Simulate the data using given beta, gamma, rho.
infected_sim = []
rewire_sim = []
degree_sim = []
for i in range(Nsim):
    infected_fraction, rewire_counts, degree_histogram = simulate(
        0.160, 0.08, 0.30)
    infected_sim.append(infected_fraction)
    rewire_sim.append(rewire_counts)
    degree_sim.append(degree_histogram)

infected_sim = np.array(infected_sim)
rewire_sim = np.array(rewire_sim)
degree_sim = np.array(degree_sim)

infs_mean = np.mean(infected_sim, axis=0)
rews_mean = np.mean(rewire_sim, axis=0)
degs_mean = np.mean(degree_sim, axis=0)

inf_df = pd.read_csv('data/infected_timeseries.csv')
rew_df = pd.read_csv('data/rewiring_timeseries.csv')
deg_df = pd.read_csv('data/final_degree_histograms.csv')

inf_mean = inf_df.groupby('time')['infected_fraction'].mean()
rew_mean = rew_df.groupby('time')['rewire_count'].mean()
deg_mean = deg_df.groupby('degree')['count'].mean()

plt.title("Infected Mean Plot over 40 Simulations")
plt.plot(np.arange(0, 201), inf_mean, label='Observed Data')
plt.plot(np.arange(0, 201), infs_mean, label='Simulated Data')
plt.legend()
plt.show()

plt.title("Rewire Mean Plot over 40 Simulations")
plt.plot(np.arange(0, 201), rew_mean, label='Observed Data')
plt.plot(np.arange(0, 201), rews_mean, label='Simulated Data')
plt.legend()
plt.show()

plt.title("Degree Mean Plot over 40 Simulations")
plt.plot(np.arange(0, 31), deg_mean, label='Observed Data')
plt.plot(np.arange(0, 31), degs_mean, label='Simulated Data')
plt.legend()
plt.show()
