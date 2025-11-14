from pathlib import Path
import numpy as np
import pandas as pd
import pymc as pm
from tqdm.auto import tqdm  # 진행상황 표시용

# --- repo-relative paths (repo_root/code/ppi/this_file.py 기준) ---
REPO_ROOT    = Path(__file__).resolve().parents[2]
DATA_ROOT    = REPO_ROOT / "data"
CSV_ROOT     = DATA_ROOT / "csv"
RESULTS_ROOT = REPO_ROOT / "results"

AUTORATER_ROOT = RESULTS_ROOT / "autorater"
COVERAGE_ROOT  = RESULTS_ROOT / "coverage"
COVERAGE_ROOT.mkdir(parents=True, exist_ok=True)

CSV_PATH = AUTORATER_ROOT / "autorater_predictions_all.csv"


def main():
    # 1) Load full dataset 
    df = pd.read_csv(CSV_PATH)

    # label / autorater 컬럼 이름은 CSV에 맞게 조정해야 함
    df['H'] = (df['label'] == 'AD').astype(int)
    df['A_class'] = (df['autorater_prediction'] >= 0.5).astype(int)

    # Full pool size and true g
    N      = len(df)
    g_true = df['H'].mean()
    NA1    = int(df['A_class'].sum())

    # 2) Chain-rule estimator (takes prior α, β as arguments)
    def chain_rule_estimator(labeled_idx, alpha, beta):
        sub = df.iloc[labeled_idx]
        n1 = int((sub['A_class'] == 1).sum())
        H1 = int(sub.loc[sub['A_class'] == 1, 'H'].sum())
        n0 = int((sub['A_class'] == 0).sum())
        H0 = int(sub.loc[sub['A_class'] == 0, 'H'].sum())

        with pm.Model() as model:
            θA  = pm.Beta('θA', alpha, beta)
            θH1 = pm.Beta('θH1', alpha, beta)
            θH0 = pm.Beta('θH0', alpha, beta)

            pm.Binomial('obs_A',  N,  θA,  observed=NA1)
            pm.Binomial('obs_H1', n1, θH1, observed=H1)
            pm.Binomial('obs_H0', n0, θH0, observed=H0)

            g = pm.Deterministic('g', θA * θH1 + (1 - θA) * θH0)

            idata = pm.sample(
                draws=500,
                tune=500,
                chains=2,
                cores=1,
                progressbar=False,
                target_accept=0.9,
            )

        g_samples = idata.posterior['g'].values.flatten()
        return g_samples.mean(), np.quantile(g_samples, [0.025, 0.975])

    # 3) Naïve estimator (takes prior α, β as arguments)
    def naive_estimator(labeled_idx, alpha, beta):
        sub   = df.iloc[labeled_idx]
        n     = len(sub)
        H_sum = int(sub['H'].sum())
        a, b  = alpha + H_sum, beta + n - H_sum
        samples = np.random.beta(a, b, size=2000)
        return samples.mean(), np.quantile(samples, [0.025, 0.975])

    # 4) Difference estimator (no prior needed)
    def difference_estimator(labeled_idx):
        A_bar = df['A_class'].mean()
        resid = df.iloc[labeled_idx]['H'] - df.iloc[labeled_idx]['A_class']
        g_hat = A_bar + resid.mean()

        # Bootstrap CI
        boots = [
            A_bar + resid.sample(frac=1, replace=True).mean()
            for _ in range(1000)
        ]
        ci = np.quantile(boots, [0.025, 0.975])
        return g_hat, ci

    # 4.5) PPI (analytic) — closed-form CI: Var(A)/N + Var(H−A)/n
    def ppi_analytic_estimator(labeled_idx):
        Z975 = 1.959963984540054  # z-value for 95% two-sided CI

        # Mean and variance of A (binary) over the full pool
        A_all = df['A_class'].astype(float).values
        N_all = len(A_all)
        A_bar = A_all.mean()
        varA  = A_all.var(ddof=1) if N_all > 1 else 0.0

        # Mean and variance of (H - A) in the labeled subset
        sub   = df.iloc[labeled_idx]
        R     = (sub['H'].astype(float) - sub['A_class'].astype(float)).values
        n_lab = len(R)
        r_bar = R.mean() if n_lab > 0 else 0.0
        varR  = R.var(ddof=1) if n_lab > 1 else 0.0

        # Point estimate, standard error, and CI
        g_hat = A_bar + r_bar
        se    = np.sqrt(varA / N_all + varR / n_lab) if (N_all > 0 and n_lab > 0) else 0.0
        ci_lo = g_hat - Z975 * se
        ci_hi = g_hat + Z975 * se
        return g_hat, (ci_lo, ci_hi)

    # 5) Simulation settings
    nsim        = 50
    label_sizes = [10, 20, 40, 80]
    priors = [
        {'name': 'uniform',  'alpha': 1.0, 'beta': 1.0},
        {'name': 'jeffreys', 'alpha': 0.5, 'beta': 0.5},
    ]

    all_results = []

    # 재현 가능성을 위해 seed 고정 (원하면)
    np.random.seed(123)

    for prior in priors:
        pname, α, β = prior['name'], prior['alpha'], prior['beta']
        for nh in label_sizes:
            cov_counts = {'chain': 0, 'naive': 0, 'diff': 0, 'ppi': 0}
            widths     = {'chain': [], 'naive': [], 'diff': [], 'ppi': []}

            for _ in tqdm(range(nsim), desc=f"{pname} | labels={nh}"):
                idx = np.random.choice(N, size=nh, replace=False)

                # Chain-rule
                m_c, ci_c = chain_rule_estimator(idx, α, β)
                if ci_c[0] <= g_true <= ci_c[1]:
                    cov_counts['chain'] += 1
                widths['chain'].append(ci_c[1] - ci_c[0])

                # Naïve
                m_n, ci_n = naive_estimator(idx, α, β)
                if ci_n[0] <= g_true <= ci_n[1]:
                    cov_counts['naive'] += 1
                widths['naive'].append(ci_n[1] - ci_n[0])

                # PPI (analytic)
                m_p, ci_p = ppi_analytic_estimator(idx)
                if ci_p[0] <= g_true <= ci_p[1]:
                    cov_counts['ppi'] += 1
                widths['ppi'].append(ci_p[1] - ci_p[0])

                # Difference
                m_d, ci_d = difference_estimator(idx)
                if ci_d[0] <= g_true <= ci_d[1]:
                    cov_counts['diff'] += 1
                widths['diff'].append(ci_d[1] - ci_d[0])

            all_results.append({
                'prior':     pname,
                'n_labels':  nh,
                'chain_cov': cov_counts['chain'] / nsim,
                'chain_w':   np.mean(widths['chain']),
                'naive_cov': cov_counts['naive'] / nsim,
                'naive_w':   np.mean(widths['naive']),
                'diff_cov':  cov_counts['diff'] / nsim,
                'diff_w':    np.mean(widths['diff']),
                'ppi_cov':   cov_counts['ppi'] / nsim,
                'ppi_w':     np.mean(widths['ppi']),
            })

    # 6) Save and display results
    res_df = pd.DataFrame(all_results)
    print(res_df)

    out_csv = COVERAGE_ROOT / "coverage_all.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"\nSaved coverage results to: {out_csv}")


if __name__ == "__main__":
    main()
