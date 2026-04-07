"""
This code creates the simulated data for the project.
Note that we only run this once; the data is then
resued for all subsequent code, to reduce time on simulation.
"""

import numpy as np
from simulator import simulate

# Fix the number of simulations.
Nsim = 50000

# Generate beta, gamma, rho using the prior distributions.
beta = np.random.uniform(0.05, 0.50, Nsim)
gamma = np.random.uniform(0.02, 0.20, Nsim)
rho = np.random.uniform(0.0, 0.8, Nsim)
np.save("beta.npy", beta)
np.save("gamma.npy", gamma)
np.save("rho.npy", rho)

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
np.save("infected_sim.npy", infected_sim)
np.save("rewire_sim.npy", rewire_sim)
np.save("degree_sim.npy", degree_sim)
