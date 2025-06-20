import numpy as np
from scipy.stats import wishart

def draw_wvag_params(d, rng=None):
    rng   = np.random.default_rng() if rng is None else rng
    a = rng.uniform(0.25, 0.45)
    alpha = rng.uniform(0.05, 0.25, size=d)
    while np.any(1 - a * alpha <= 0):
        alpha = rng.uniform(0.05, 0.25, size=d)

    vol = rng.uniform(0.12, 0.22, size=d)
    W = wishart.rvs(df=d + 5, scale=np.eye(d), random_state=rng)
    D = np.diag(1 / np.sqrt(np.diag(W)))
    C = D @ W @ D                                 
    L = np.linalg.cholesky(np.outer(vol, vol) * C / (a + 1.0))

    sigma_sqrt = L                                       
    theta = rng.uniform(-0.25, 0.25, size=d)

    Gm = 1.0 + alpha * a                         
    var = (sigma_sqrt ** 2) @ Gm
    drift = theta * Gm
    
    mu = rng.uniform(-0.05, 0.05, size=d) - drift

    return mu, a, alpha, theta, sigma_sqrt

