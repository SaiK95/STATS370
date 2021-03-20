import os

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import invgamma

import common

# Fix the random seed for repeatability.
np.random.seed(42)

# Output location
output_dir = "./gibbs"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize data
data = common.data
num_data_points = data.shape[0]

# Initialize theta
init_theta = common.init_theta(data)

# MH Sampler parameters
num_steps = 3*1500
burn_in = 500
step_size = np.ones([7, 1]) * 0.05
curr_theta = init_theta
num_accepted = 0
num_oob = 0
num_params = np.shape(curr_theta)[0]
num_bins = 30

# Store the samples
sampled_theta = np.repeat(np.zeros_like(init_theta), num_steps, axis=1)

for step in range(0, num_steps, 3):
    if step % 500 == 0:
        print("Completed %s steps", step)

    # Sample sigma_sq from invgamma(n+2)
    sigma_sq_proposal = invgamma.rvs(num_data_points + 2)
    curr_theta[0] = sigma_sq_proposal
    # Store samples at step
    if step >= burn_in:
        sampled_theta[:, step] = curr_theta.reshape((-1,))

    # Sample lambda and tau from their normal distributions
    lambda_proposal = common.sample_lambda_cond_post(data, curr_theta)
    tau_proposal = common.sample_tau_cond_post(data, curr_theta)
    curr_theta[1] = lambda_proposal
    curr_theta[2] = tau_proposal
    # Store samples at step + 1
    if step >= burn_in:
        sampled_theta[:, step + 1] = curr_theta.reshape((-1,))

    # Sample mu and gamma from their normal distributions
    gamma_proposal = common.sample_gamma_cond_post(data, curr_theta)
    mu_proposal = common.sample_mu_cond_post(data, curr_theta)
    curr_theta[3:5] = gamma_proposal
    curr_theta[5:] = mu_proposal
    # Store samples at step + 2
    if step >= burn_in:
        sampled_theta[:, step + 2] = curr_theta.reshape((-1,))


print("...Summary...")
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

plt.savefig(output_dir + '/time_series.png', bbox_inches='tight')
