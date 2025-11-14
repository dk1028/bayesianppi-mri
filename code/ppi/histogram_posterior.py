import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- paths relative to repo root ---
REPO_ROOT    = Path(__file__).resolve().parents[2]
RESULTS_ROOT = REPO_ROOT / "results"
SIM_ROOT     = RESULTS_ROOT / "sims" / "histogram_posterior"
FIGS_ROOT    = REPO_ROOT / "figs"

SIM_ROOT.mkdir(parents=True, exist_ok=True)
FIGS_ROOT.mkdir(parents=True, exist_ok=True)


def main():
    # True parameter values and target quantity
    true_theta_A = 0.6
    true_theta_H_A1 = 0.8
    true_theta_H_A0 = 0.3
    true_g = true_theta_A * true_theta_H_A1 + (1 - true_theta_A) * true_theta_H_A0

    # Simulation settings
    M = 50  # number of simulated data sets
    N_A = 1000
    N_H = 100

    # Storage for summary statistics
    means = np.zeros(M)
    medians = np.zeros(M)
    ci_lower = np.zeros(M)
    ci_upper = np.zeros(M)

    # Loop over M simulated data sets
    for i in range(M):
        # 1) Generate a new synthetic data set
        A_data = np.random.binomial(1, true_theta_A, size=N_A)
        A_H = A_data[:N_H]
        H_data = np.array([
            np.random.binomial(1, true_theta_H_A1 if a == 1 else true_theta_H_A0)
            for a in A_H
        ])

        # 2) Fit the Bayesian chain-rule model
        with pm.Model() as model:
            theta_A = pm.Beta("theta_A", 1, 1)
            theta_H_A1 = pm.Beta("theta_H_A1", 1, 1)
            theta_H_A0 = pm.Beta("theta_H_A0", 1, 1)

            pm.Bernoulli("A_obs", p=theta_A, observed=A_data)
            pm.Bernoulli(
                "H_obs",
                p=theta_H_A1 * A_H + theta_H_A0 * (1 - A_H),
                observed=H_data,
            )

            g = pm.Deterministic(
                "g",
                theta_H_A1 * theta_A + theta_H_A0 * (1 - theta_A),
            )

            trace = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                cores=1,
                progressbar=False,
                return_inferencedata=True,
            )

        # 3) Extract posterior draws of g and compute summary stats
        g_samples = trace.posterior["g"].values.flatten()
        means[i] = g_samples.mean()
        medians[i] = np.median(g_samples)
        ci_lower[i], ci_upper[i] = np.percentile(g_samples, [2.5, 97.5])

    # 4) Compute empirical coverage of the 95% credible interval
    coverage = np.mean((ci_lower <= true_g) & (true_g <= ci_upper))
    print(f"Empirical coverage of 95% CI: {coverage*100:.1f}%")

    # 4-1) Save summary stats to disk (for reproducibility)
    np.savez(
        SIM_ROOT / "histogram_posterior_summary.npz",
        means=means,
        medians=medians,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        true_g=true_g,
        coverage=coverage,
        M=M,
        N_A=N_A,
        N_H=N_H,
    )

    # 5) Plot histograms of posterior means and medians across simulations
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    ax = axes[0]
    sns.histplot(means, bins=15, kde=True, color="steelblue", ax=ax)
    ax.axvline(true_g, color="red", linestyle="--", label=f"True g = {true_g:.2f}")
    ax.set_title("Histogram of Posterior Means of g")
    ax.set_xlabel("Posterior mean")
    ax.set_ylabel("Frequency")
    ax.legend()

    ax = axes[1]
    sns.histplot(medians, bins=15, kde=True, color="seagreen", ax=ax)
    ax.axvline(true_g, color="red", linestyle="--", label=f"True g = {true_g:.2f}")
    ax.set_title("Histogram of Posterior Medians of g")
    ax.set_xlabel("Posterior median")
    ax.set_ylabel("Frequency")
    ax.legend()

    fig.tight_layout()
    fig_path = FIGS_ROOT / "histogram_posterior_g.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)
    print(f"Saved histogram figure to: {fig_path}")


if __name__ == "__main__":
    main()
