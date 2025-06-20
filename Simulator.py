import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm, describe
from scipy.stats.qmc import Sobol
import seaborn as sns

"""Parameters to vary for all models
"""
t_max = 1
sample_size = 2 ** 15
delta_t = t_max / sample_size

t_values = np.linspace(0, t_max, sample_size + 1)

class AGProcess:
    def __init__(self, a, alpha):
        self.dim = len(alpha)
        self.alpha = alpha
        self.marginal_g0 = lambda dt : gamma(dt * a, scale=1)
        self.marginal_g = [
            (lambda dt, alpha_i=alpha_i: gamma(dt * (1 - a * alpha_i) / alpha_i, scale=alpha_i))
            for alpha_i in alpha
        ]
        
    def simulate_increments(self):
        all_paths_increments = []
        
        g0_increments = self.marginal_g0(delta_t).rvs(sample_size)
        
        for i, alpha_i in enumerate(self.alpha):
            g_increments =  self.marginal_g[i](delta_t).rvs(sample_size)
            total_increments = g_increments + alpha_i * g0_increments
            all_paths_increments.append(total_increments)
            
        return all_paths_increments
        
        
    def simulate(self):
        all_paths_increments = self.simulate_increments()
        
        paths = np.cumsum(np.vstack(all_paths_increments), axis=1)
        paths = np.hstack((np.zeros((len(self.alpha), 1)), paths))
        
        return paths
    
    
class BrownianMotion:
    def __init__(self, dim):
        self.dim = dim
        
    def simulate_increments(self):
        w_increments = norm.rvs(
            loc=0,
            scale=math.sqrt(delta_t),
            size=(self.dim, sample_size)
        )
        
        return w_increments
    
    def simulate(self):
        paths = np.cumsum(np.vstack(self.simulate_increments()), axis=1)
        paths = np.hstack((np.zeros((self.dim, 1)), paths))
        
        return paths
    
    
class WVAGProcess:
    def __init__(self, a, alpha, mu, theta, sigma):
        self.dim = len(alpha)
        self.ag_subordinator = AGProcess(a, alpha)
        self.mu = mu.reshape(self.dim, -1) 
        self.theta = theta.reshape(self.dim, -1)
        self.sigma = sigma
        
        
    def simulate(self):
        ag_increments = np.vstack(self.ag_subordinator.simulate_increments())
        wg_increments = norm.rvs(loc=0, scale=np.sqrt(ag_increments))
        
        ag_paths = np.cumsum(ag_increments, axis=1)
        ag_paths = np.hstack((np.zeros((self.dim, 1)), ag_paths))
        
        wg_paths = np.cumsum(wg_increments, axis=1)
        wg_paths = np.hstack((np.zeros((self.dim, 1)), wg_paths))
        
        xt_paths = self.mu * t_values + self.theta * ag_paths + self.sigma @ wg_paths
        
        return xt_paths
