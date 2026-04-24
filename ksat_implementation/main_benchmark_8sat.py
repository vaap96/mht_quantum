import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

from src.ksat.sat_utils import generate_benchmark_instances, generate_training_set
from src.ksat.qaoa import train_lr_grid_search
from src.ksat.evaluation import (
    evaluate_model,
    evaluate_model_walksat_fast,
    evaluate_model_walksatlm,
    fit_scaling
)

sns.set_style("whitegrid")

mpl.rcParams.update({
    "text.usetex": True,        # Use LaTeX for all text
    "font.family": "serif",     # Serif font (Computer Modern)
    "font.serif": ["Computer Modern Roman"],
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def main():
    # --- 1. Hyperparameters ---
    train_n = 12            # Train on n=12 (like in the paper)
    test_n_range = range(9, 17)  # Evaluate on n=10 to 16
    clause_density = 176.54  # Strict 8-SAT threshold
    train_size = 50         # Size of training set
    test_size = 200          # Size of test set per n
    k_sat = 8
    np.random.seed(27)

    # QAOA specific parameters
    depths = [410, 430, 450, 480, 500]
    exponents_qaoa = []

    # --- 2. Data Generation ---
    print("Generating Training Dataset...")
    train_dataset = generate_training_set(
        train_n=train_n,
        train_size=train_size,
        clause_density=clause_density,
        k_sat=k_sat
    )

    print("\nGenerating Testing Dataset...")
    test_dataset = generate_benchmark_instances(
        n_range=test_n_range,
        test_size=test_size,
        clause_density=clause_density,
        k_sat=k_sat
    )

    # --- 3. Classical Benchmarks ---
    # WalkSAT
    walksat_perf = evaluate_model_walksat_fast(test_dataset, max_flips=100000, p_noise=0.5)
    a_ws, b_ws = fit_scaling(walksat_perf['n'], walksat_perf['median_runtime'])
    exponent_ws = b_ws / np.log(2)
    print(f"WalkSAT Scaling Exponent: {exponent_ws:.4f}")

    # WalkSATlm
    walksatlm_perf = evaluate_model_walksatlm(test_dataset, max_flips=100000, p_noise=0.5, w1=6, w2=5)
    a_wslm, b_wslm = fit_scaling(walksatlm_perf['n'], walksatlm_perf['median_runtime'])
    exponent_wslm = b_wslm / np.log(2)
    print(f"WalkSATlm Scaling Exponent: {exponent_wslm:.4f}")

    # --- 4. QAOA Benchmarks ---
    for p in depths:
        # Train to find optimal LR angles
        lr_angles = train_lr_grid_search(train_dataset, depth=p, train_n=TRAIN_N, skip_grid=True)

        # Evaluate on the test set
        qaoa_perf = evaluate_model(test_dataset, lr_angles, p_depth=p)

        # Calculate scaling
        a_lr, b_lr = fit_scaling(qaoa_perf['n'], qaoa_perf['lr_runtime'])
        exponent_lr = b_lr / np.log(2)
        exponents_qaoa.append(exponent_lr)
        print(f"QAOA (p={p}) Scaling Exponent: {exponent_lr:.4f}")

    # --- 5. Plotting ---
    plt.figure(figsize=(8, 5))
    plt.plot(depths, exponents_qaoa, marker='o', label="LR QAOA scaling exponent")

    plt.hlines(y=b_ws, xmin=50, xmax=65, colors='r', label="WalkSAT ", linestyle='dotted')
    plt.hlines(y=b_wslm, xmin=50, xmax=65, colors='b', label="WalkSATlm", linestyle='dashed')
    plt.hlines(y=0.325, xmin=min(depths), xmax=max(depths), colors='g',
               label="Q. WalkSATlm (Boulebnane & Montanaro)", linestyle='dashdot')

    plt.ylabel("Scaling Exponent")
    plt.xlabel("Number of QAOA layers (p)")
    plt.title("Scaling Exponents: QAOA vs Classical WalkSAT")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()


if __name__ == "__main__":
    main()
