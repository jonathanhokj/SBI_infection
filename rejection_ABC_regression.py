"""
Regression ABC.
This is the Fourth Rejection ABC algorithm.
The code is mostly the same as the third, except
that now we apply regression adjustment at the
end to try and see if our results improve.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

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
degree_max = np.array([
    v.loc[v['count'] > 0, 'degree'].max()
    for k, v in split_degree.items()
])
degree_var = [
    np.average(
        (split_degree[k]['degree']
         - np.average(split_degree[k]['degree'],
                      weights=split_degree[k]['count']))**2,
        weights=split_degree[k]['count']
    )
    for k in sorted(split_degree)
]

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
degmax_sim = np.array([np.where(hist > 0)[0].max() for hist in degree_sim])
vardeg_sim = [
    np.average((np.arange(len(hist)) -
               np.average(np.arange(len(hist)), weights=hist))**2, weights=hist)
    for hist in degree_sim
]

sd_p = np.std(p_sim)
sd_pr = np.std(pr_sim)
sd_r = np.std(r_sim)
sd_tmax = np.std(tmax_sim)
sd_dmax = np.std(degmax_sim)
sd_vdeg = np.std(vardeg_sim)
d = np.sqrt(
    ((p_sim - np.mean(infected_max))/sd_p)**2 +
    ((pr_sim - np.mean(infected_argmax))/sd_pr)**2 +
    ((r_sim - np.mean(rewire_max))/sd_r)**2 +
    ((tmax_sim - np.mean(infected_timemax))/sd_tmax)**2 +
    ((degmax_sim - np.mean(degree_max))/sd_dmax)**2 +
    ((vardeg_sim - np.mean(degree_var))/sd_vdeg)**2
)

# Accept only the 1% data
epsilon = np.quantile(d, 0.01)
betafilter = beta[d <= epsilon]
gammafilter = gamma[d <= epsilon]
rhofilter = rho[d <= epsilon]

# Accepted summary statistics
s_sim_all = np.column_stack([
    p_sim, pr_sim, r_sim, tmax_sim, degmax_sim, vardeg_sim
])

# Observed summary statistics in matrix form
s_accepted = s_sim_all[d <= epsilon]
s_obs = np.array([
    np.mean(infected_max),
    np.mean(infected_argmax),
    np.mean(rewire_max),
    np.mean(infected_timemax),
    np.mean(degree_max),
    np.mean(degree_var)
]).reshape(1, -1)

# Adjusts filtered parameters with linear regression


def apply_regression_adjustment(params_filter, s_acc, s_target):
    model = LinearRegression()
    model.fit(s_acc, params_filter)

    pred_acc = model.predict(s_acc)
    pred_obs = model.predict(s_target)

    return params_filter - (pred_acc - pred_obs)


beta_adj = apply_regression_adjustment(betafilter, s_accepted, s_obs)
gamma_adj = apply_regression_adjustment(gammafilter, s_accepted, s_obs)
rho_adj = apply_regression_adjustment(rhofilter, s_accepted, s_obs)


plt.title("Beta Adjusted Approximate Posterior")
plt.hist(beta_adj, bins=50, density=True)
plt.axvline(x=np.mean(beta_adj), color='r')
plt.show()

plt.title("Gamma Adjusted Approximate Posterior")
plt.hist(gamma_adj, bins=50, density=True)
plt.axvline(x=np.mean(gamma_adj), color='r')
plt.show()

plt.title("Rho Adjusted Approximate Posterior")
plt.hist(rho_adj, bins=50, density=True)
plt.axvline(x=np.mean(rho_adj), color='r')
plt.show()

plt.title("Beta against Gamma")
plt.scatter(beta_adj, gamma_adj)
plt.xlabel("Beta")
plt.ylabel("Gamma")
plt.show()

plt.title("Beta against Rho")
plt.scatter(beta_adj, rho_adj)
plt.xlabel("Beta")
plt.ylabel("Rho")
plt.show()

plt.title("Gamma against Rho")
plt.scatter(gamma_adj, rho_adj)
plt.xlabel("Gamma")
plt.ylabel("Rho")
plt.show()
