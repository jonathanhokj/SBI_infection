# SBI for Epidemic Parameter Inference on Adaptive Networks

Companion code and data for the simulation-based inference (SBI) class assignment.
Given a stochastic SIR epidemic model on an adaptive network, infer the unknown parameters $(\beta, \gamma, \rho)$ from observed data using SBI methods.

## The model

A population of 200 agents interact on an undirected contact network.
Each agent is **Susceptible (S)**, **Infected (I)**, or **Recovered (R)**.
The initial network is an Erdos-Renyi graph $G(N, p)$ with $p = 0.05$, giving an expected degree of about 10.
At time 0, five agents chosen uniformly at random are infected.

Three parameters govern the dynamics:

| Parameter | Meaning | Prior |
|-----------|---------|-------|
| $\beta$ | Infection probability per S--I edge per step | Uniform(0.05, 0.50) |
| $\gamma$ | Recovery probability per infected agent per step | Uniform(0.02, 0.20) |
| $\rho$ | Rewiring probability per S--I edge per step | Uniform(0.0, 0.8) |

At each of the 200 time steps, three phases are applied synchronously:

1. **Infection.** Each susceptible neighbor of an infected agent becomes infected with probability $\beta$.
2. **Recovery.** Each infected agent recovers with probability $\gamma$.
3. **Rewiring.** For each S-I edge, with probability $\rho$ the susceptible agent breaks the link and connects to a random non-neighbor. This models behavioral avoidance of infected contacts.

## Repository contents

```
simulator.py                       # Python implementation of the model
data/
  infected_timeseries.csv          # Fraction infected over time (40 replicates)
  rewiring_timeseries.csv          # Rewiring counts over time (40 replicates)
  final_degree_histograms.csv      # Degree distribution at t=200 (40 replicates)
create_simulations.py              # Simulate data using simulator (50000 iterations)
basic_rejection_ABC.py             # Rejection ABC to estimate parameters
rejection_ABC_updated.py           # Rejection ABC method with improved summary statistics
rejection_ABC_finalized_summary.py # ABC method with a finailized summary statistics
rejection_ABC_regression.py        # ABC method with regression adjustment
ABC_MCMC.py                        # ABC method with MCMC adaptation
SMC_MCMC.py                        # SMC MCMC algorithm
synthetic_likelihood.py            # ABC MCMC with synthetic likelihood
check_final_params.py              # Runs synthetic-truth recovery experiment
simulated_data/
  beta.npy                         # Array of simulated beta
  gamma.npy                        # Array of simulated gamma
  rho.npy                          # Array of simulated rho
  infected_sim.npy                 # Simulated infected proportion
  rewire_sim.npy                   # Simulated rewire counts
  degree_sim.npy                   # Simulated degree counts
```

## Simulator usage

```python
import numpy as np
from simulator import simulate

# Run one replicate with specific parameters
rng = np.random.default_rng(42)
infected, rewires, degrees = simulate(beta=0.3, gamma=0.15, rho=0.7, rng=rng)
```

See `simulator.py` for full parameter documentation.

## Observed data

The data files contain 40 independent realizations, all generated with the **same** unknown $(\beta, \gamma, \rho)$.
The contact network is never observed.

| File | Columns |
|------|---------|
| `infected_timeseries.csv` | `replicate_id`, `time`, `infected_fraction` |
| `rewiring_timeseries.csv` | `replicate_id`, `time`, `rewire_count` |
| `final_degree_histograms.csv` | `replicate_id`, `degree` (0-30, clipped), `count` |

## Simulation algorithms

The simulator files contains different algorithms used for trying to infer the $\beta$, $\gamma$, $\rho$ parameters. These include a few Basic Rejection ABC, Regression Adjustment ABC, ABC-MCMC, SMC-MCMC, Synthetic Likelihood MCMC.

## Requirements

- Python 3.8+
- NumPy
