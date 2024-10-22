
# Load required libraries
set.seed(123)

### make simwithlogit a function to simulate

library("aladynoulli")

library(rsvd)  # For fast randomized SVD
library(mgcv) 

data <- generate_tensor_data(num_covariates = 5,K = 3,T = 20,D = 5,N = 100)
#
Y <- data$Y
G <- data$G
plot_individuals(data$S,num_individuals = 3)
# Here you initialize the MCMC
initial_values <- mcmc_init_two(y = Y, G = G, num_topics = 3, length_scales_lambda = rep(10, 3),
                                var_scales_lambda = rep(1, 3),
                                length_scales_phi = rep(10, 3),
                                var_scales_phi = rep(1, 3))

a=aladynoulli(Y, G, n_topics = 3,n_iters=5000,
              initial_values=initial_values,step_size_lambda=0.01, step_size_phi=0.01,
              target_accept_rate = 0.2)
saveRDS(a,"~/Desktop/aladynoulli.rds")
a$acceptance_rates
plot(a$log_posteriors)
plot(a$log_likelihoods)

