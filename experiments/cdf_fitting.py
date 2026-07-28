import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import genextreme
import pandas as pd
from scipy.optimize import minimize

def fit_and_plot_mixed_cdf(epsilons, n, m, tol=1e-6):
    """
    Fits a mixed distribution (Continuous GEV + Discrete Point Mass) to the empirical CDF.

    Parameters:
    epsilons (np.array): Your sorted array of epsilon values.
    n (int): The width of the grid (e.g., 200, 500, 1000).
    m (int): The height of the grid (e.g., 100, 250, 500).
    """
    total_samples = len(epsilons)

    # 1. Calculate the exact theoretical maximum epsilon based on the Rust script's logic
    # The max x-coordinate is (n - 1), center is (n - 1) / 2.
    # Max deviation is (n - 1) / 2, scaled by m.
    eps_max = (n - 1) / (2.0 * m)

    # 2. Partition the data (using a tiny tolerance for floating-point safety)
    continuous_data = epsilons[epsilons < (eps_max - tol)]
    boundary_data = epsilons[epsilons >= (eps_max - tol)]

    # Calculate mixture probabilities
    p = len(continuous_data) / total_samples
    q = len(boundary_data) / total_samples

    print(f"Grid: {n}x{m} (rho = {n / m})")
    print(f"Theoretical Max Epsilon: {eps_max:.4f}")
    print(f"Continuous Portion (p): {p * 100:.2f}%")
    print(f"Boundary Point Mass (q): {q * 100:.2f}%\n")

    # 3. Fit a Generalized Extreme Value (GEV) distribution to the continuous portion
    # The fit returns (shape, loc, scale)
    c_fit, loc_fit, scale_fit = genextreme.fit(continuous_data)
    print(f"Shape parameter (c): {c_fit:.4f}")
    print(f"Location parameter (loc): {loc_fit:.4f}")
    print(f"Scale parameter (scale): {scale_fit:.4f}")

    # 4. Define the piecewise theoretical CDF
    def theoretical_mixed_cdf(x):
        # Initialize an array of zeros with the same shape as x
        y = np.zeros_like(x)

        # Condition 1: x is less than the max boundary
        mask_continuous = x < eps_max
        y[mask_continuous] = p * genextreme.cdf(x[mask_continuous], c_fit, loc=loc_fit, scale=scale_fit)

        # Condition 2: x is greater than or equal to the max boundary
        mask_boundary = x >= eps_max
        y[mask_boundary] = 1.0

        return y

    # 5. Plotting
    plt.figure(figsize=(9, 6))

    # Empirical CDF
    y_empirical = np.arange(1, total_samples + 1) / total_samples
    plt.step(epsilons, y_empirical, label=f"Empirical CDF ({n}x{m})", where='post', color='royalblue', linewidth=2)

    # Theoretical Fitted Curve
    # Generate a smooth range of x-values extending slightly past the max boundary to visualize the jump
    x_smooth = np.linspace(min(epsilons), eps_max + 0.1, 2000)
    y_smooth = theoretical_mixed_cdf(x_smooth)

    plt.plot(x_smooth, y_smooth, label="Fitted Mixed CDF (GEV + Point Mass)", color='darkorange', linewidth=2.5,
             linestyle='--')

    # Highlight the boundary cutoff with a vertical line
    plt.axvline(x=eps_max, color='red', linestyle=':',
                label=f"Max Boundary ($\epsilon_{{max}} \\approx {n / m / 2:.2f}$)")

    plt.xlabel(r"Maximum Scaled Deviation ($\epsilon$)")
    plt.ylabel("Cumulative Probability")
    plt.title(f"Mixed Distribution Fit for {n}x{m} Grid Bisection")
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Set plot limits to match your reference images
    plt.ylim(0, 1.0)
    plt.xlim(left=0)

    plt.show()


def fit_weighted_mixed_cdf(filename, n, m):
    # 1. Load Data
    # Assuming the file has lines like "0.354, 152"
    data = np.loadtxt(filename, delimiter=',')
    epsilons_raw = data[:, 0]
    boundary_lengths = data[:, 1]

    # Sort data by epsilon to build the CDF properly
    sort_idx = np.argsort(epsilons_raw)
    epsilons = epsilons_raw[sort_idx]
    lengths = boundary_lengths[sort_idx]

    # 2. Calculate and Normalize Weights
    # Weight = 1 / boundary_length
    raw_weights = 1.0 / lengths
    weights = raw_weights / np.sum(raw_weights)  # Normalize so sum(weights) = 1.0

    # 3. Construct the Weighted Empirical CDF
    # Instead of step size 1/N, the step size is the weight of the sample
    y_empirical = np.cumsum(weights)

    # 4. Partition the Data (Continuous vs Boundary)
    eps_max = (n - 1) / (2.0 * m)
    tol = 1e-6

    mask_cont = epsilons < (eps_max - tol)
    mask_bound = epsilons >= (eps_max - tol)

    continuous_eps = epsilons[mask_cont]
    continuous_weights = weights[mask_cont]

    # The mixture probabilities are now based on WEIGHTS, not raw counts
    p = np.sum(continuous_weights)
    q = np.sum(weights[mask_bound])

    # 5. Define Custom Weighted Negative Log-Likelihood for GEV
    def weighted_nll(params):
        c, loc, scale = params

        # Scale must be strictly positive. If the optimizer guesses a negative scale, penalize it heavily.
        if scale <= 0:
            return np.inf

            # Calculate the log-probability density for each continuous epsilon
        logpdf = genextreme.logpdf(continuous_eps, c, loc=loc, scale=scale)

        # Multiply by weights and sum, then negate to turn maximization into minimization
        return -np.sum(continuous_weights * logpdf)

    # 6. Optimize!
    # We use the unweighted fit as a good starting guess for the optimizer
    initial_guess = genextreme.fit(continuous_eps)

    result = minimize(weighted_nll, initial_guess, method='Nelder-Mead')

    if not result.success:
        print("Warning: Optimization did not converge!")

    c_fit, loc_fit, scale_fit = result.x

    print(f"Weighted p (Continuous): {p:.4f}")
    print(f"Weighted q (Point Mass): {q:.4f}")
    print(f"Weighted Fit -> c: {c_fit:.4f}, loc: {loc_fit:.4f}, scale: {scale_fit:.4f}")

    # 7. Plotting
    plt.figure(figsize=(9, 6))

    # Plot Weighted Empirical CDF
    plt.step(epsilons, y_empirical, label=f"Weighted Empirical CDF ({n}x{m})", where='post', color='royalblue')

    # Generate and plot Theoretical Fitted Curve
    def theoretical_mixed_cdf(x):
        y = np.zeros_like(x)
        mask_c = x < eps_max
        y[mask_c] = p * genextreme.cdf(x[mask_c], c_fit, loc=loc_fit, scale=scale_fit)
        y[x >= eps_max] = 1.0
        return y

    x_smooth = np.linspace(min(epsilons), eps_max + 0.1, 2000)
    plt.plot(x_smooth, theoretical_mixed_cdf(x_smooth), label="Weighted Fitted Mixed CDF", color='darkorange',
             linestyle='--')

    plt.axvline(x=eps_max, color='red', linestyle=':', label="Max Boundary")
    plt.xlabel(r"Maximum Scaled Deviation ($\epsilon$)")
    plt.ylabel("Cumulative Weighted Probability")
    plt.legend(loc="lower right")
    plt.ylim(0, 1.0)
    plt.xlim(left=0)
    plt.show()


if __name__ == '__main__':
    n = 4000
    m = 1000
    num_trials = 1000
    ver = 2
    fname = None
    if ver == 1:
        fname = f'results/epsilons{n}x{m}-{num_trials}.txt'
    else:
        fname = f'results/epsilons{n}x{m}-{num_trials}_{ver}.txt'
    # input = np.loadtxt(f"results/epsilons{n}x{m}-1000.txt")

    df = pd.read_csv(fname, header=None)
    # print(len(df.columns))

    epsilons = None
    boundary_lengths = None
    if len(df.columns) == 1:
        df.columns = ['epsilons']
    else:
        df.columns = ['epsilons', 'boundary_lengths']

    df.sort_values(by='epsilons', inplace=True)

    epsilons = df['epsilons'].to_numpy(np.float32)

    if len(df.columns) == 1:
        fit_and_plot_mixed_cdf(epsilons, n, m)
    else:
        boundary_lengths = df['boundary_lengths'].to_numpy(np.int16)
        fit_weighted_mixed_cdf(fname, n, m)


