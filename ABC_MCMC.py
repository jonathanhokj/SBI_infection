"""
ABC MCMC.
"""

import numpy as np
import pandas as pd
from simulator import simulate
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
epsilon_target = np.quantile(d, 0.1)

s_obs = np.array([
    np.mean(infected_max),
    np.mean(infected_argmax),
    np.mean(rewire_max),
    np.mean(infected_timemax),
    np.mean(degree_max),
    np.mean(degree_var)
])
s_std = np.array([
    sd_p,
    sd_pr,
    sd_r,
    sd_tmax,
    sd_dmax,
    sd_vdeg
])   # SDs for normalization


# Define function to get summary statistics based on given parameters


def get_summary_statistics(beta, gamma, rho):
    inf_frac, rew_counts, deg_hist = simulate(beta, gamma, rho)

    p_sim = np.max(inf_frac)
    idx_max = np.argmax(inf_frac)
    pr_sim = p_sim / inf_frac[idx_max +
                              1] if idx_max < len(inf_frac)-1 else 1.0
    r_sim = np.max(rew_counts)
    tmax_sim = float(idx_max)
    degmax_sim = float(np.where(deg_hist > 0)[0].max())

    degrees = np.arange(len(deg_hist))
    avg_deg = np.average(degrees, weights=deg_hist)
    vardeg_sim = np.average((degrees - avg_deg)**2, weights=deg_hist)

    return np.array([p_sim, pr_sim, r_sim, tmax_sim, degmax_sim, vardeg_sim])


def calculate_distance(s_sim, s_target, s_scale):
    return np.sqrt(np.sum(((s_sim - s_target) / s_scale)**2))

# ABC-MCMC Algorithm


def abc_mcmc(n_iterations, epsilon, initial_theta, proposal_width):
    # Initialize
    chain = np.zeros((n_iterations, 3))
    chain[0] = initial_theta

    # Get initial distance
    s_curr = get_summary_statistics(*chain[0])
    dist_curr = calculate_distance(s_curr, s_obs, s_std)

    # If the starting point is outside epsilon, we keep trying until we find a valid start
    while dist_curr > epsilon:
        chain[0] = np.random.uniform(0, 1, 3)
        s_curr = get_summary_statistics(*chain[0])
        dist_curr = calculate_distance(s_curr, s_obs, s_std)

    accepted_count = 0

    for i in range(1, n_iterations):
        # 1. Propose new parameters (Random Walk)
        theta_current = chain[i-1]
        theta_proposal = theta_current + np.random.normal(0, proposal_width)

        # 2. Check Priors (Uniform [0, 1] for all params)
        if np.any(theta_proposal < 0) or np.any(theta_proposal > 1):
            # Out of bounds: reject and stay at current
            chain[i] = theta_current
            continue

        # 3. Simulate and Calculate Distance
        s_prop = get_summary_statistics(*theta_proposal)
        dist_prop = calculate_distance(s_prop, s_obs, s_std)

        # 4. Acceptance Step
        # In ABC-MCMC, the Likelihood is 1 if dist <= epsilon, else 0.
        if dist_prop <= epsilon:
            # Since our priors and proposals (Normal) are symmetric,
            # the MH ratio simplifies to 1. We just accept.
            chain[i] = theta_proposal
            accepted_count += 1
        else:
            # Distance too large: reject and stay at current
            chain[i] = theta_current
    return chain


# Propose a random starting parameter
init = [0.05, 0.02, 0.2]
# Proposal widths (start around 5-10% of param range)
widths = [0.01, 0.01, 0.05]

# Epsilon needs to be small enough to be accurate, but large enough to move.
# Check your rejection file's 'epsilon' (1% quantile) for a good starting value.
results = abc_mcmc(n_iterations=20000, epsilon=epsilon_target,
                   initial_theta=init, proposal_width=widths)

burn_in = int(0.2 * len(results))  # Discard first 20%
# Keep every 5th sample to reduce correlation
thinned_results = results[burn_in::5]

beta_post = thinned_results[:, 0]
gamma_post = thinned_results[:, 1]
rho_post = thinned_results[:, 2]


plt.title("Posterior of Beta")
plt.hist(beta_post, bins=50, density=True)
plt.axvline(np.mean(beta_post), color='r')
plt.show()

plt.title("Posterior of Gamma")
plt.hist(gamma_post, bins=50, density=True)
plt.axvline(np.mean(gamma_post), color='r')
plt.show()

plt.title("Posterior of Rho")
plt.hist(rho_post, bins=50, density=True)
plt.axvline(np.mean(rho_post), color='r')
plt.show()

plt.title("Beta Against Gamma")
plt.scatter(beta_post, gamma_post)
plt.xlabel("Beta")
plt.ylabel("Gamma")
plt.show()

plt.title("Beta Against Rho")
plt.scatter(beta_post, rho_post)
plt.xlabel("Beta")
plt.ylabel("Rho")
plt.show()

plt.title("Gamma Against Rho")
plt.scatter(gamma_post, rho_post)
plt.xlabel("Gamma")
plt.ylabel("Rho")
plt.show()

# Checking convergence of MCMC
fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

axes[0].plot(results[:, 0], color='blue', lw=0.5)
axes[0].set_ylabel(r"$\beta$")
axes[1].plot(results[:, 1], color='green', lw=0.5)
axes[1].set_ylabel(r"$\gamma$")
axes[2].plot(results[:, 2], color='red', lw=0.5)
axes[2].set_ylabel(r"$\rho$")
axes[2].set_xlabel("Iteration")

plt.suptitle("MCMC Trace Plots (Convergence Check)")
plt.show()
