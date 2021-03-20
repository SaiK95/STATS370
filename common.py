import numpy as np

data = np.array([[1.1092, 0.9592],
                 [0.7942, -0.5889],
                 [0.0521, -1.1212],
                 [-1.0795, -1.2398],
                 [2.4925, 0.4005],
                 [-1.8491, 1.2854],
                 [0.1581, 3.0208],
                 [0.0776, 2.6427],
                 [0.8305, 0.4605],
                 [-1.2829, 0.9888],
                 [0.191, 1.2963],
                 [0.612, -0.304],
                 [-0.6328, 2.2947],
                 [0.7406, 1.3705],
                 [1.3908, 1.7367],
                 [1.3897,	2.1947],
                ])


def init_theta(data):
    """
    Initialize theta vector.
    
    The 7 elements are sigma_sq, lambda, tau, mu_1, mu_2, gamma_1 and gamma_2. 
    
    Args:
        data: This is used to initialize sigma_sq.
    
    Returns: 
        theta: 7 dimensional vector of parameters.
    """
    theta = np.array([1, 0.5, 0.5, 0, 0, 0, 0]).reshape(-1, 1)
    # theta = np.array([1.5, 0.5, 0, 0, -0.5, 0.5, 1.75]).reshape(-1, 1)
    return theta


def center_data(_mu, _gamma, _lambda, _tau, data):

    _means_a = np.repeat(_mu, 4, axis=0)  # [4,  2]
    _means_b = np.repeat(_gamma, 4, axis=0)
    _means_c = _lambda * _means_a + (1 - _lambda) * _means_b
    _means_d = _tau * _means_a + (1 - _tau) * _means_b
    _means = np.vstack((_means_a, _means_b, _means_c, _means_d))  # [16, 2]
    centered_data = data - _means  # [16, 2]

    return centered_data


def get_joint_log_posterior(theta, data):
    """
    Evaluates joint log posterior likelihood at the given value of theta.
    This is done to avoid precision issues (overflow/underflow).

    Args: 
        theta: 7 dimensional vector of parameters.
        data: N*2 dimensional matrix of data.

    Returns:
        posterior: Value of posterior evaluated at the given value of theta.
    """
    _n, _k = data.shape

    _sigma_sq = theta[0]
    _lambda = theta[1]
    _tau = theta[2]
    _mu = theta[3:5].reshape(1, -1)  # [1, 2]
    _gamma = theta[5:].reshape(1, -1)

    # Center data
    centered_data = center_data(_mu, _gamma, _lambda, _tau, data)  # [16, 2]

    _sigma_term = -2*(_n + 1) * np.log(np.sqrt(_sigma_sq)) # [1, 1]
    posterior = centered_data @ centered_data.T  # [16, 16]
    posterior = _sigma_term + (-0.5/_sigma_sq) * np.trace(posterior)  # [1, 1] + [1, 1] + [1, 1]

    return posterior


def get_gradient_log_posterior(theta, data):
    """
    Evaluates the gradient of log posterior likelihood at the given value of theta.

    Args: 
        theta: 7 dimensional vector of parameters.
        data: N*2 dimensional matrix of data.

    Returns:
        posterior: Value of posterior evaluated at the given value of theta.
    """
    _n, _k = data.shape

    _sigma_sq = theta[0]
    _sigma = np.sqrt(_sigma_sq)
    _lambda = theta[1]
    _tau = theta[2]
    _mu = theta[3:5].reshape(1, -1)  # [1, 2]
    _gamma = theta[5:].reshape(1, -1)

    # Means
    centered_data = center_data(_mu, _gamma, _lambda, _tau, data)  # [16, 2]
    sum_sq_cd = np.trace(centered_data @ centered_data.T)

    # gradient
    gradient = np.zeros_like(theta)

    # The powers are 0.5 and 1.5 because our parameter is sigma_sq while in the equation it's sigma
    gradient[0] = (-2 * (_n + 1) / _sigma) + sum_sq_cd / (_sigma ** 3)
    gradient[1] = np.sum(centered_data[8:12] @ (_mu - _gamma).T) / _sigma_sq
    gradient[2] = np.sum(centered_data[12:] @ (_mu - _gamma).T)
    gradient[3:5] = (np.sum((centered_data[0:4]), axis=0) +
                     np.sum((centered_data[8:12]) * (_lambda), axis = 0) +
                     np.sum((centered_data[12:]) * (_tau), axis=0)
                     ).reshape(-1, 1) / _sigma_sq
    gradient[5:] = (np.sum(centered_data[4:8], axis = 0) +
                    np.sum(centered_data[8:12] * (1 - _lambda), axis=0) +
                    np.sum(centered_data[12:] * (1 - _tau), axis=0)
                    ).reshape(-1, 1) / _sigma_sq

    return gradient


def get_proposal_normal(prop_mean, prop_std, size):
    """
    Draw a value from a normal proposal kernel, with given mean and std_dev.

    Args: 
        mean: Mean of the proposal distribution.
        std: Standard Deviation of the proposal distribution.
        size: Shape of the required vector.

    Returns:
        A vector of dimensions specified by `size` sampled from the normal distribution.
    
    """
    return np.random.normal(prop_mean, prop_std, size = size)


def log_normal(x, mu=0, sigma=1):

    mu = np.ones_like(x) * mu
    numerator = np.exp(-1*((x-mu)**2)/(2*sigma**2))
    denominator = sigma * np.sqrt(2*np.pi)

    return np.sum(np.log(numerator/denominator))


def sample_lambda_cond_post(data, theta):

    _sigma_sq = theta[0]
    _sigma = np.sqrt(_sigma_sq)
    _lambda = theta[1]
    _tau = theta[2]
    _mu = theta[3:5].reshape(1, -1)  # [1, 2]
    _gamma = theta[5:].reshape(1, -1)

    lambda_mu = np.sum(np.divide(data[8:12] - _gamma, (_mu - _gamma)))
    lambda_std = _sigma_sq / np.sum((_gamma - _mu))
    if lambda_std <= 0:
        return _lambda
    else:
        return np.random.normal(lambda_mu, lambda_std)


def sample_tau_cond_post(data, theta):

    _sigma_sq = theta[0]
    _sigma = np.sqrt(_sigma_sq)
    _lambda = theta[1]
    _tau = theta[2]
    _mu = theta[3:5].reshape(1, -1)  # [1, 2]
    _gamma = theta[5:].reshape(1, -1)

    tau_mu = np.sum(np.divide(data[12:] - _gamma, (_mu - _gamma)))
    tau_std = _sigma_sq / np.sum((_gamma - _mu))
    if tau_std <= 0:
        return _tau
    else:
        return np.random.normal(tau_mu, tau_std)


def sample_gamma_cond_post(data, theta):

    _sigma_sq = theta[0]
    _sigma = np.sqrt(_sigma_sq)
    _lambda = theta[1]
    _tau = theta[2]
    _mu = theta[3:5].reshape(1, -1)  # [1, 2]
    _gamma = theta[5:].reshape(1, -1)

    y_i = data[4:8]
    y_i_prime = (data[8:12] - _lambda * _mu) / (1 - _lambda)
    y_i_prime_prime = (data[12:] - _tau * _mu) / (1 - _tau)
    c_prime = (_lambda - 1)** 2
    d_prime = (_tau - 1)** 2
    # The *4 is a multiple because we have to sum over corresponding cell/tissue type
    denominator = 1*4 + c_prime*4 + d_prime*4

    gamma_mu = (np.sum(y_i) + np.sum(y_i_prime * c_prime) + np.sum(y_i_prime_prime * d_prime))/denominator
    gamma_std = _sigma_sq / denominator

    if gamma_std <= 0:
        return _gamma
    else:
        return np.random.normal(gamma_mu, gamma_std)


def sample_mu_cond_post(data, theta):

    _sigma_sq = theta[0]
    _sigma = np.sqrt(_sigma_sq)
    _lambda = theta[1]
    _tau = theta[2]
    _mu = theta[3:5].reshape(1, -1)  # [1, 2]
    _gamma = theta[5:].reshape(1, -1)

    y_i = data[0:4]
    y_i_prime = (data[8:12] - _gamma + (_lambda * _gamma))/_lambda
    y_i_prime_prime = (data[12:] - _gamma + (_tau * _gamma))/_tau
    # The *4 is a multiple because we have to sum over corresponding cell/tissue type
    denominator = 1*4 + _lambda**2*4 + _tau**2*4

    mu_mu = (np.sum(y_i) + np.sum(y_i_prime) + np.sum(y_i_prime_prime))/denominator
    mu_std = _sigma_sq / denominator

    if mu_std <= 0:
        return _mu
    else:
        return np.random.normal(mu_mu, mu_std)
