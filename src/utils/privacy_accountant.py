import numpy as np

def compute_rdp(noise_ratio, orders, iterations):
    """
    Computes the Rényi Differential Privacy (RDP) of a Gaussian mechanism.

    Args:
        noise_ratio (float): The standard deviation of the Gaussian noise divided by sensitivity.
        orders (list): List of Renyi orders (alpha) to evaluate RDP at.
        iterations (int): The number of iterations to compose.

    Returns:
        rdp (list): RDP values at the given orders.
    """
    rdp = []
    for alpha in orders:
        if alpha == 1:
            raise ValueError("Rényi order must be greater than 1.")
        term = alpha * (1 / noise_ratio) ** 2 / 2
        rdp.append(iterations * term)
    return rdp


def rdp_to_dp(rdp, orders, delta):
    """
    Converts Rényi Differential Privacy (RDP) to (ε, δ)-DP.

    Args:
        rdp (list): RDP values at the corresponding orders.
        orders (list): List of Renyi orders (alpha).
        delta (float): Target δ value.

    Returns:
        (float): ε value for (ε, δ)-DP.
    """
    epsilons = [rdp[i] - np.log(delta) / (order - 1) for i, order in enumerate(orders)]
    return min(epsilons)


def gaussian_mechanism_dp(noise_ratio, delta, iterations, orders=None):
    """
    Computes the (ε, δ)-DP guarantee of a Gaussian mechanism over multiple iterations.

    Args:
        noise_ratio (float): The standard deviation of the Gaussian noise divided by sensitivity.
        delta (float): Target δ value.
        iterations (int): The number of iterations to compose.
        orders (list, optional): List of Renyi orders (alpha) to evaluate RDP at.
                                 Defaults to a standard range.

    Returns:
        (float): ε value for (ε, δ)-DP.
    """
    if orders is None:
        orders = np.arange(1.01, 100, 0.1)  # Standard range of alpha values.

    rdp = compute_rdp(noise_ratio, orders, iterations)
    epsilon = rdp_to_dp(rdp, orders, delta)
    return epsilon

def sigma_to_eps(noise_ratio, delta):
    epsilon = np.sqrt(2 * np.log(1.25 / delta)) / noise_ratio
    return epsilon

if __name__ == "__main__":
    # Example usage
    noise_ratio = 100   # noise scale divided by sensitivity
    delta = 1e-4        # Target δ
    iterations = 1000     # Number of iterations
    # 10, 100: 4.79
    # 10, 1000: 18.58
    # 10, 5000: 55.34
    # 10, 10000: 92.96
    # 40, 120: 1.21
    # 40, 60: 0.85
    # 40, 10000: 13.85
    # 60, 10000: 8.54
    # 100, 10000: 4.79
    orders = np.arange(1.01, 100, 0.1)  # Range of Renyi orders

    epsilon = gaussian_mechanism_dp(noise_ratio, delta, iterations, orders)
    print(f"(ε, δ)-DP Guarantee for T iterations: ε = {epsilon}, δ = {delta}")
    epsilon = sigma_to_eps(noise_ratio, delta)
    print(f"(ε, δ)-DP Guarantee for one iteration: ε = {epsilon}, δ = {delta}")