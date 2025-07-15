from config import *
from wvag import WVAGProcess
from parameter_generator import draw_wvag_params
import matplotlib.pyplot as plt
import seaborn as sns

# Plots out sample paths
if __name__ == "__main__":
    sns.set_theme()
    print("Entered!")

    print("Setting dimension: " + str(d_test_simulator))

    print("Generating params")
    a, alpha, mu, theta, sigma_sqrt = draw_wvag_params(d_test_simulator)
    print(a, alpha, mu, theta, sigma_sqrt)

    print("Simulating...")
    vals = WVAGProcess(a, alpha, mu, theta, sigma_sqrt).simulate()

    print("Graphing")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            plt.plot(t_values.cpu().numpy(), vals[i, j].cpu().numpy(), alpha=0.3)

    plt.xlabel("t")
    plt.ylabel("X_t^i")
    plt.title("Simulated WVAG Process Paths (All)")
    plt.show()
