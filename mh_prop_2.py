import os

import numpy as np
import matplotlib.pyplot as plt

import common
import time
start = time.time()

# Fix the random seed for repeatability.
np.random.seed(42)

# Output location
output_dir = "./mh_prop_2"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Initialize data
data = common.data

# Initialize theta
init_theta = common.init_theta(data)

# MH Sampler parameters
num_steps = 5000
burn_in = 500
step_size_epsilon = np.ones([7, 1]) * 0.01
step_size_delta = np.ones([7, 1]) * 0.001
# If using epsilon = 0 (Rely solely on gradient)
# step_size_delta = np.array([1e-3, 1e-2, 1e-2, 1e-1, 1e-1, 5e-2, 1e-2]).reshape(-1, 1)
curr_theta = init_theta
num_accepted = 0
num_oob = 0
num_params = np.shape(curr_theta)[0]
num_bins = 30

# Store the samples
sampled_theta = np.repeat(np.zeros_like(init_theta), num_steps, axis=1)
mean_gradient = np.zeros([1, num_steps])

for step in range(num_steps):
    if step % 500 == 0:
        print("Completed %s steps", step)

    if np.sum(step_size_delta) > 0:
        if np.random.uniform(0, 1) < 0.5:
            D = 1
        else:
            D = -1
        gradient = common.get_gradient_log_posterior(curr_theta, data)
    else:
        D = 0
        gradient = 0
    mean_gradient[0, step] = np.round(np.mean(gradient), 2)
    
    Z = common.get_proposal_normal(0, 1, [num_params, 1])
    proposal = curr_theta + np.multiply(step_size_epsilon, Z) + np.multiply(step_size_delta * D, gradient)

    # Boundary checks
    if proposal[0] >= 0 and 0 <= proposal[1] <= 1 and 0 <= proposal[2] <= 1:

        # Compute the acceptance probability
        curr_posterior = common.get_joint_log_posterior(curr_theta, data)
        prop_posterior = common.get_joint_log_posterior(proposal, data)
        # - because operating in log scale
        ratio_posterior = prop_posterior - curr_posterior
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
print("Time taken: " + str(np.round(time.time() - start, 2)))

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
