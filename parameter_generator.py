import torch


def draw_wvag_params(d, device=torch.device("cuda"), dtype=torch.double, seed=None):
    if device is None:
        device = torch.device("cpu")
    if seed is not None:
        torch.manual_seed(seed)
    # Uniforms
    a = torch.empty(1, dtype=dtype, device=device).uniform_(0.25, 0.45).item()
    alpha = torch.empty(d, dtype=dtype, device=device).uniform_(0.05, 0.25)
    while torch.any(1 - a * alpha <= 0):
        alpha = torch.empty(d, dtype=dtype, device=device).uniform_(0.05, 0.25)
    vol = torch.empty(d, dtype=dtype, device=device).uniform_(0.12, 0.22)
    # Wishart: simulate by summing outer products of standard normals
    df = d + 5
    W = torch.zeros((d, d), dtype=dtype, device=device)
    for _ in range(df):
        x = torch.randn(d, dtype=dtype, device=device)
        W += torch.ger(x, x)
    D = torch.diag(1 / torch.sqrt(torch.diag(W)))
    C = D @ W @ D
    L = torch.linalg.cholesky(torch.ger(vol, vol) * C / (a + 1.0))
    sigma_sqrt = L
    theta = torch.empty(d, dtype=dtype, device=device).uniform_(-0.25, 0.25)
    Gm = 1.0 + alpha * a
    var = (sigma_sqrt**2) @ Gm
    drift = theta * Gm
    mu = torch.empty(d, dtype=dtype, device=device).uniform_(-0.05, 0.05) - drift

    return a, alpha, mu, theta, sigma_sqrt
