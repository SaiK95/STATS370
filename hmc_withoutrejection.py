import os

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st

import common

def update_theta(eps, M, phi, curr_theta):
    proposal = curr_theta + eps * np.linalg.inv(M) @ phi
    # Boundary checks
    if proposal[0] >= 0 and 0 <= proposal[1] <= 1 and 0 <= proposal[2] <= 1:
        theta = proposal
    else:
        theta = curr_theta
    return theta


def update_phi(phi, eps, theta, data, step_ratio=1):
    return phi + step_ratio * eps * common.get_gradient_log_posterior(theta, data)


# Fix the random seed for repeatability.
np.random.seed(42)

# Output location
output_dir = "./hmc_withoutrejection"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize data
data = common.data

# Initialize theta
curr_theta = common.init_theta(data)
num_params = np.shape(curr_theta)[0]

# HMC Sampler parameters
num_steps = 5000
burn_in = 500
num_leapfrogs = 10
step_size_epsilon = 1.0/num_leapfrogs
M_var = 1
M = np.eye(num_params) * M_var
num_accepted = 0
num_oob = 0
num_bins = 30

# Store the samples
sampled_theta = np.repeat(np.zeros_like(curr_theta), num_steps, axis=1)
mean_gradient = np.zeros([1, num_steps])

for _step in range(num_steps):
    if _step % 500 == 0:
        print("Completed %s steps", _step)

    phi = st.multivariate_normal.rvs(mean=0, cov=M_var, size=[num_params, 1]).reshape(-1, 1)
    curr_phi = phi  # phi_{t-1}
    theta = curr_theta  # curr_theta is theta_{t-1}

    for leap_frog_step in range(num_leapfrogs):
        if leap_frog_step == 0 or leap_frog_step == num_leapfrogs - 1:
            # Half update phi
            phi = update_phi(phi, step_size_epsilon, theta, data, step_ratio=0.5)
            # Full update theta
            theta = update_theta(step_size_epsilon, M, phi, theta)
            # Half update phi
            phi = update_phi(phi, step_size_epsilon, theta, data, step_ratio=0.5)
        else:
            # Change ordering here to see impact
            # Full update both theta and phi
            theta = update_theta(step_size_epsilon, M, phi, theta)
            phi = update_phi(phi, step_size_epsilon, theta, data, step_ratio=1)

    curr_theta = theta
    num_accepted += 1

    # Store samples
    if _step >= burn_in:
        sampled_theta[:, _step] = curr_theta.reshape((-1,))

print("...Summary...")
print("Acceptance ratio is " + str(num_accepted / (num_steps)))
print("OOB ratio is " + str(num_oob / (num_steps)))
print("Mean theta is " + str(sampled_theta.mean()))
print("Std theta is " + str(sampled_theta.std()))

# # plot things
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(8, 11))
plt.tight_layout()

ax1.hist(sampled_theta[0, burn_in:], num_bins)
ax1.set_xlabel('$sigma^2$')
ax2.hist(sampled_theta[1, burn_in:], num_bins)
ax2.set_xlabel('$lambda$')
ax3.hist(sampled_theta[2, burn_in:], num_bins)
ax3.set_xlabel('$tau$')
ax4.hist(sampled_theta[3, burn_in:], num_bins)
ax4.set_xlabel('$mu_1$')
ax5.hist(sampled_theta[4, burn_in:], num_bins)
ax5.set_xlabel('$mu_2$')
ax6.hist(sampled_theta[5, burn_in:], num_bins)
ax6.set_xlabel('$gamma_1$')
ax7.hist(sampled_theta[6, burn_in:], num_bins)
ax7.set_xlabel('$gamma_2$')
# ax8.hist(mean_gradient[0, burn_in:], num_bins)
# ax8.set_xlabel('$Gradient mean$')

plt.savefig(output_dir + '/hist.png', bbox_inches='tight')

f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7, 1, figsize=(8, 11))
plt.tight_layout()

ax1.plot(sampled_theta[0, burn_in:], 'b')
ax1.set_xlabel('$sigma_squared$')
ax2.plot(sampled_theta[1, burn_in:], 'b')
ax2.set_xlabel('$lambda$')
ax3.plot(sampled_theta[2, burn_in:], 'b')
ax3.set_xlabel('$tau$')
ax4.plot(sampled_theta[3, burn_in:], 'b')
ax4.set_xlabel('$mu_1$')
ax5.plot(sampled_theta[4, burn_in:], 'b')
ax5.set_xlabel('$mu_2$')
ax6.plot(sampled_theta[5, burn_in:], 'b')
ax6.set_xlabel('$gamma_1$')
ax7.plot(sampled_theta[6, burn_in:], 'b')
ax7.set_xlabel('$gamma_2$')
# ax8.plot(mean_gradient[0, burn_in:], 'b')
# ax8.set_xlabel('$Gradient mean$')

plt.savefig(output_dir + '/time_series.png', bbox_inches='tight')
