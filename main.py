import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import Simulator as sim
from Simulator import t_values
import helper


if __name__ == "__main__":
    sns.set_theme()
    print("Entered!")
    
    d = 50
    print("Setting dimension: " + str(d))

    print('Generating params')
    mu, a, alpha, theta, sigma_sqrt = helper.draw_wvag_params(d)
    print(mu, a, alpha, theta, sigma_sqrt)

    print("Simulating...")
    vals = sim.WVAGProcess(a, alpha, mu, theta, sigma_sqrt).simulate()

    print("Graphing")
    for i, val in enumerate(vals):
        sns.lineplot(x=t_values, y=vals[i], label=f"WVAG Process {i+1}")

    plt.xlabel("t")
    plt.ylabel("X_t^i")
    plt.title("Simulated WVAG Process")
    plt.legend()
    plt.show()





