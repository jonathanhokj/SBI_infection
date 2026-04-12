"""
SMC ABC.
This file performs SMC ABC. Instead of considering one fixed epsilon,
define a decreasing sequence of epsilon. For initial epsilon, simulate
dataset and assign equal weights. For every subsequent decreasing
epsilon, pick parameter from previous population based on their weight.
Perturb it, simulate and check if it is less than this epsilon value.
Finally, calculate a new weight to correct it as we sampled from a
proposal distribution rather than a prior.

Summary statistics used are the same as previous ABC algorithms.
"""

import numpy as np
import pandas as pd
from simulator import simulate
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


def calculate_weight(theta, prev_particles, prev_weights, kernel_cov):
    """Calculates the importance weight for a particle in round t."""
    # Prior is Uniform[0, 1] for all, so it's a constant 1.0 (if in bounds)
    prior = 1.0

    # Denominator: Sum of (previous weights * transition kernel)
    # This accounts for the probability of reaching 'theta' from the previous population
    densities = multivariate_normal.pdf(
        prev_particles, mean=theta, cov=kernel_cov)
    denominator = np.sum(prev_weights * densities)

    return prior / denominator


def smc_abc(n_particles, epsilon_schedule, s_obs, s_std):
    # n_particles: Number of samples in each population
    # epsilon_schedule: List of decreasing thresholds [eps1, eps2, eps3...]
    n_rounds = len(epsilon_schedule)
    # Store particles and weights for each round
    all_populations = []
    all_weights = []

    for t in range(n_rounds):
        eps_t = epsilon_schedule[t]
        new_particles = []
        new_weights = []
        accepted_count = 0

        print(f"Starting Round {t} (epsilon = {eps_t})...")

        # Determine Kernel Covariance for perturbation (Standard heuristic: 2 * Var of previous)
        if t > 0:
            kernel_cov = 2 * \
                np.cov(all_populations[t-1], rowvar=False,
                       aweights=all_weights[t-1])

        while len(new_particles) < n_particles:
            # 1. Propose theta
            if t == 0:
                # Round 0 is just Rejection ABC from the Prior
                theta_prop = np.random.uniform(0, 1, 3)
            else:
                # Sample from previous population based on weights
                idx = np.random.choice(n_particles, p=all_weights[t-1])
                parent = all_populations[t-1][idx]
                # Perturb
                theta_prop = np.random.multivariate_normal(parent, kernel_cov)

            # 2. Check Prior Bounds
            if np.any(theta_prop < 0) or np.any(theta_prop > 1):
                continue

            # 3. Simulate and Distance check
            s_sim = get_summary_statistics(*theta_prop)
            dist = np.sqrt(np.sum(((s_sim - s_obs) / s_std)**2))

            if dist <= eps_t:
                # 4. Acceptance and Weighting
                new_particles.append(theta_prop)
                if t == 0:
                    new_weights.append(1.0)
                else:
                    w = calculate_weight(theta_prop, all_populations[t-1],
                                         all_weights[t-1], kernel_cov)
                    new_weights.append(w)

                accepted_count += 1
                if len(new_particles) % 10 == 0:
                    print(
                        f"Round {t}: {len(new_particles)}/{n_particles} particles found.")

        # Normalize weights
        new_weights = np.array(new_weights)
        new_weights /= np.sum(new_weights)

        all_populations.append(np.array(new_particles))
        all_weights.append(new_weights)

    return all_populations, all_weights


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
epsilon_target = np.quantile(d, 0.05)

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


epsilons = [5.0, 3.0, 2.0, 1.5]  # Gradually tightening thresholds
populations, weights = smc_abc(n_particles=500, epsilon_schedule=epsilons,
                               s_obs=s_obs, s_std=s_std)

# Final posterior is the last population
final_particles = populations[-1]

plt.title("Posterior of Beta")
plt.hist(final_particles[:, 0], bins=30, weights=weights[-1], density=True)
plt.axvline(np.mean(final_particles[:, 0]), color='r')
plt.show()

plt.title("Posterior of Gamma")
plt.hist(final_particles[:, 1], bins=30, weights=weights[-1], density=True)
plt.axvline(np.mean(final_particles[:, 1]), color='r')
plt.show()

plt.title("Posterior of Rho")
plt.hist(final_particles[:, 2], bins=30, weights=weights[-1], density=True)
plt.axvline(np.mean(final_particles[:, 2]), color='r')
plt.show()

plt.title("Beta against Gamma")
plt.scatter(final_particles[:, 0], final_particles[:, 1])
plt.xlabel("Beta")
plt.ylabel("Gamma")
plt.show()

plt.title("Beta against Rho")
plt.scatter(final_particles[:, 0], final_particles[:, 2])
plt.xlabel("Beta")
plt.ylabel("Rho")
plt.show()

plt.title("Gamma against Rho")
plt.scatter(final_particles[:, 1], final_particles[:, 2])
plt.xlabel("Gamma")
plt.ylabel("Rho")
plt.show()
