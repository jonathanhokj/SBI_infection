"""
Rejection ABC Updated.
This is the second Basic Rejection ABC algorithm.
The code is mostly the same as the first, except
that we add a new summary statistic for the ratio
for infection proportion.
"""

import numpy as np
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
infected_timemax = np.array(
    [np.argmax(split_infected[i]['infected_fraction']) for i in range(40)])

# Load generated data from another file
beta = np.load("simulated_data/beta.npy")
gamma = np.load("simulated_data/gamma.npy")
rho = np.load("simulated_data/rho.npy")
infected_sim = np.load("simulated_data/infected_sim.npy")
rewire_sim = np.load("simulated_data/rewire_sim.npy")
degree_sim = np.load("simulated_data/degree_sim.npy")

# Generate summary statistics based on initial choice of summary statistics.
p_sim = np.max(infected_sim, axis=1)
pr_sim = np.max(infected_sim, axis=1) / \
    infected_sim[np.arange(infected_sim.shape[0]),
                 np.argmax(infected_sim, axis=1)+1]
r_sim = np.max(rewire_sim, axis=1)
infr_sim = np.max(infected_sim, axis=1) / \
    infected_sim[np.arange(infected_sim.shape[0]),
                 np.argmax(infected_sim, axis=1)-1]
tmax_sim = np.argmax(infected_sim, axis=1)
sd_p = np.std(p_sim)
sd_pr = np.std(pr_sim)
sd_r = np.std(r_sim)
sd_tmax = np.std(tmax_sim)
d = np.sqrt(
    ((p_sim - np.mean(infected_max))/sd_p)**2 +
    ((pr_sim - np.mean(infected_argmax))/sd_pr)**2 +
    ((r_sim - np.mean(rewire_max))/sd_r)**2 +
    ((tmax_sim - np.mean(infected_timemax))/sd_tmax)**2
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

plt.title("Beta against Gamma")
plt.scatter(betafilter, gammafilter)
plt.xlabel("Beta")
plt.ylabel("Gamma")
plt.show()

plt.title("Beta against Rho")
plt.scatter(betafilter, rhofilter)
plt.xlabel("Beta")
plt.ylabel("Rho")
plt.show()

plt.title("Gamma against Rho")
plt.scatter(gammafilter, rhofilter)
plt.xlabel("Gamma")
plt.ylabel("Rho")
plt.show()
