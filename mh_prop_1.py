import os

import numpy as np
import matplotlib.pyplot as plt

import common

# Fix the random seed for repeatability.
np.random.seed(42)

# Output location
output_dir = "./mh_prop_1"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize data
data = common.data

# Initialize theta
init_theta = common.init_theta(data)

# MH Sampler parameters
num_steps = 5000
burn_in = 500
step_size = np.ones([7, 1]) * 0.05
curr_theta = init_theta
num_accepted = 0
num_oob = 0
num_params = np.shape(curr_theta)[0]
num_bins = 30

# Store the samples
sampled_theta = np.repeat(np.zeros_like(init_theta), num_steps, axis=1)

for step in range(num_steps):
    if step%500 == 0:
        print("Completed %s steps", step)

    proposal = common.get_proposal_normal(0, 1, [num_params, 1])
    proposal = curr_theta + np.multiply(step_size, proposal)

    # Boundary checks
    if proposal[0] >= 0 and 0 <= proposal[1] <= 1 and 0 <= proposal[2] <= 1:

        # Compute the acceptance probability
        curr_posterior = common.get_joint_log_posterior(curr_theta, data)
        prop_posterior = common.get_joint_log_posterior(proposal, data)
        ratio_posterior = prop_posterior - curr_posterior  # - because operating in log scale
        alpha = min(0, ratio_posterior)  # Since np.log(1) = 0
        alpha = np.exp(alpha)  # To convert from log-scale to normal-scale

        # Whether to accept proposal with probability alpha
        if np.random.uniform(0, 1) < alpha:
            curr_theta = proposal
            num_accepted += 1
        else:
            curr_theta = curr_theta

        # Store samples
        if step >= burn_in:
            sampled_theta[:, step] = curr_theta.reshape((-1,))

    else:
        curr_theta = curr_theta
        sampled_theta[:, step] = curr_theta.reshape((-1,))
        num_oob += 1

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
