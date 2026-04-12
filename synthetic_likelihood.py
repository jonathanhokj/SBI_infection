"""
Synthetic Likelihood.

This file performs synthetic likelihood by assuming that our summary
statistics follows a multivariate normal distribution rather than a
fixed value. This allows us to improve our ABC-MCMC algorithm that
was done in a previous file.

We use the same initial guess in this file.
"""

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from simulator import simulate
import matplotlib.pyplot as plt


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


def get_multiple_sims(theta, n_sims=50):
    """Runs n simulations for a single theta and returns the summary stats."""
    all_stats = []
    for _ in range(n_sims):
        # Using the summary logic from your previous ABC_MCMC.py
        stats = get_summary_statistics(*theta)
        all_stats.append(stats)
    return np.array(all_stats)


def log_synthetic_likelihood(s_obs, sim_stats):
    """Calculates the log-likelihood of s_obs under the Gaussian of sim_stats."""
    mu = np.mean(sim_stats, axis=0)
    # Adding a small value to diagonal (shrinkage) for numerical stability
    sigma = np.cov(sim_stats, rowvar=False) + np.eye(len(mu)) * 1e-6

    try:
        return multivariate_normal.logpdf(s_obs, mean=mu, cov=sigma)
    except:
        return -np.inf  # Return -inf if the matrix is singular


def bsl_mcmc(n_iterations, n_sims, s_obs, initial_theta, proposal_width):
    n_params = len(initial_theta)
    chain = np.zeros((n_iterations, n_params))
    chain[0] = initial_theta

    # Initial likelihood
    current_sims = get_multiple_sims(chain[0], n_sims)
    current_log_lik = log_synthetic_likelihood(s_obs, current_sims)

    accepted = 0
    for i in range(1, n_iterations):
        # 1. Propose new theta
        proposal = chain[i-1] + np.random.normal(0, proposal_width, n_params)

        # 2. Check Priors
        if np.any(proposal < 0) or np.any(proposal > 1):
            chain[i] = chain[i-1]
            continue

        # 3. Estimate Likelihood for proposal
        prop_sims = get_multiple_sims(proposal, n_sims)
        prop_log_lik = log_synthetic_likelihood(s_obs, prop_sims)

        # 4. Metropolis-Hastings Acceptance
        # Since we use log-likelihood, we use subtraction instead of division
        log_ratio = prop_log_lik - current_log_lik

        if np.log(np.random.rand()) < log_ratio:
            chain[i] = proposal
            current_log_lik = prop_log_lik
            accepted += 1
        else:
            chain[i] = chain[i-1]

        if i % 10 == 0:
            print(f"Iteration {i}: Acceptance Rate = {accepted/i:.2%}")

    return chain


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

s_obs = np.array([
    np.mean(infected_max),
    np.mean(infected_argmax),
    np.mean(rewire_max),
    np.mean(infected_timemax),
    np.mean(degree_max),
    np.mean(degree_var)
])

# Propose a random starting parameter
init = [0.05, 0.02, 0.2]
# Proposal widths (start around 5-10% of param range)
widths = [0.01, 0.01, 0.05]


chain = bsl_mcmc(5000, 50, s_obs, init, widths)

plt.title("Posterior of Beta")
plt.hist(chain[:, 0], bins=30, density=True)
plt.axvline(np.mean(chain[:, 0]), color='r')
plt.show()

plt.title("Posterior of Gamma")
plt.hist(chain[:, 1], bins=30, density=True)
plt.axvline(np.mean(chain[:, 1]), color='r')
plt.show()

plt.title("Posterior of Rho")
plt.hist(chain[:, 2], bins=30, density=True)
plt.axvline(np.mean(chain[:, 2]), color='r')
plt.show()

plt.title("Beta against Gamma")
plt.scatter(chain[:, 0], chain[:, 1])
plt.xlabel("Beta")
plt.ylabel("Gamma")
plt.show()

plt.title("Beta against Rho")
plt.scatter(chain[:, 0], chain[:, 2])
plt.xlabel("Beta")
plt.ylabel("Rho")
plt.show()

plt.title("Gamma against Rho")
plt.scatter(chain[:, 1], chain[:, 2])
plt.xlabel("Gamma")
plt.ylabel("Rho")
plt.show()
