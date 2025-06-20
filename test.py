from multiprocessing import Pool
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import Simulator as sim
from Simulator import t_values
sns.set_theme()

a = 0.3
alpha = [0.2]
sample_size = 2 ** 15


def simulate_1D_ag_sample(args):
    a, alpha = args
    ag = sim.AGProcess(a, alpha)
    
    return ag.simulate()[0, -1]

def test_gamma():
    args_list = [(a, alpha) for _ in range(sample_size)]    
    
    with Pool() as pool:
        samples = list(pool.map(simulate_1D_ag_sample, args_list))
        
    return stats.describe(np.array(samples), bias=False)
    
def simulate_1D_brownian_sample(_):
    bm = sim.BrownianMotion(1)
    
    return bm.simulate()[0, -1]
    
def test_brownian():
    
    with Pool() as pool:
        samples = list(pool.map(simulate_1D_brownian_sample, range(sample_size)))
        
    return stats.describe(np.array(samples), bias=False)

if __name__ == "__main__":
    print(test_gamma())
    print(test_brownian())
    