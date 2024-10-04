
library(mvtnorm)
library(MASS)
library(pracma)
library(reshape2)
library(dplyr)
set.seed(123)

K = 3
N = 1000
D = 5
T = 50
num_covariates = 5


## things we will treat as fixed:

# Create time differences matrix
time_diff <- outer(seq_len(T), seq_len(T), "-")
# Store scales
length_scales_lambda <- runif(K, T / 3, T / 2)
var_scales_lambda <- runif(K, 0.8, 1.2)  # Reduced range to control the scale


length_scales_phi <- runif(K, T / 3, T / 2)
var_scales_phi <- runif(K, 0.8, 1.2)


length_scales_mu <- runif(D, T / 3, T / 2)
var_scales_mu <- runif(D, 0.8, 1.2)

## simulate mu (we will treat as fixed)


mu_d = array(NA, dim = c(D, T))
# Simulate lambda_ik(t) using a different covariance matrix for each topic

for (d in 1:D) {
  cov_matrix_mu <- exp(-0.5 * var_scales_mu[d] * (time_diff ^ 2) / length_scales_mu[d] ^
                         2)
  mu_d[d, ] <- mvrnorm(1, mu = rep(logit(0.01), T), Sigma = cov_matrix_mu)
}

par(mfrow = c(1, 1))
matplot(
  t(mu_d),
  type = 'l',
  main = "Disease Prevalence (mu_d)",
  xlab = "Time",
  ylab = "mu_d"
)



###


# Create lambda,phi matrix
g_i = array(rnorm(num_covariates * N), dim = c(N, num_covariates))
lambda_ik = array(NA, dim = c(N, K, T))
phi_kd = array(NA, dim = c(K, D, T))
Gamma_k = matrix(rnorm(num_covariates * K), nrow = K, ncol = num_covariates)

## simulate lambda

for (k in 1:K) {
  # Simulate lambda_ik(t) using a different covariance matrix for each topic
  cov_matrix <- exp(-0.5 * var_scales_lambda[k] * (time_diff ^ 2) / length_scales_lambda[k] ^
                      2)
  image(cov_matrix)
  for (i in 1:N) {
    mean_lambda = g_i[i, ] %*% Gamma_k[k, ]
    # lambda_ik[i, k, ] <- log(1 + exp(mvrnorm(
    #   1, mu = rep(mean_lambda, T), Sigma = cov_matrix
    # )))  # Ensure positivity
    lambda_ik[i, k, ] <- mvrnorm(
      1, mu = rep(mean_lambda, T), Sigma = cov_matrix
    )
  }
}

softmax=function(x){
  return(exp(x)/sum(exp(x)))
}

s=apply(lambda_ik,c(1,3),function(x){softmax(x)})

#max_lambda=10
#lambda_ik[lambda_ik>max_lambda]=max_lambda

par(mfrow = c(2, 2))
for (i in sample(1:N, 4)) {
  matplot(
    t(s[,1 , ]),
    type = 'l',
    main = paste("Lambda for individual", i),
    xlab = "Time",
    ylab = "Lambda"
  )
}



## simulate phi

for (k in 1:K) {
  # Simulate lambda_ik(t) using a different covariance matrix for each topic
  cov_matrix <- exp(-0.5 * var_scales_phi[k] * (time_diff ^ 2) / length_scales_phi[k] ^
                      2)
  #image(cov_matrix)
  for (d in 1:D) {
    phi_kd[k, d, ] <- plogis(mvrnorm(1, mu = mu_d[d,], Sigma = cov_matrix))
  }
  image(phi_kd[k,,])
}


par(mfrow = c(2, 2))
for (d in 1:D) {
  matplot(
    t(phi_kd[, d, ]),
    
    main = paste("Phi for disease", d),
    xlab = "Time",
    ylab = "Phi"
  )
}

par(mfrow = c(2, 2))
for (k in 1:K) {
  matplot(
    t(phi_kd[k, , ]),
    
    main = paste("Phi for topic", k),
    xlab = "Time",
    ylab = "Phi"
  )
}
###

y = array(0, dim = c(N, D, T))
pi_values = array(NA, dim = c(N, D, T))

# Simulate data
for (i in 1:N) {
  for (d in 1:D) {
    for (t in 1:T) {
      if (sum(y[i, d, 1:t]) == 0) {
        # Disease hasn't occurred yet
        
        pi_idt <- sum(diag(s[,i , t] %*% t(phi_kd[, d, t])))
        
        pi_values[i, d, t] <- pi_idt  # Store the pi_idt value
        
        # Simulate diagnosis
        y[i, d, t] <- rbinom(1, 1, pi_idt)
      } else {
        break  # Stop once disease is diagnosed
      }
    }
  }
}



par(mfrow = c(2, 2))
for (i in sample(1:N, 4)) {
  matplot(
    t(pi_values[i, , ]),
    type = 'l',
    main = paste("Pi for individual", i),
    xlab = "Time",
    ylab = "Probability"
  )
}


### show nice Topelitz

# Create the Toeplitz matrix
# Load required libraries
library(Matrix)
library(pracma)

# Define the RBF kernel function
rbf_kernel <- function(t1, t2, length_scale, variance) {
  return(variance * exp(-0.5 * (t1 - t2) ^ 2 / length_scale ^ 2))
}

# Generate time points
time_points <- 1:50

# Define kernel hyperparameters
length_scale <- 10
variance <- 1
jitter <- 1e-6  # Small jitter term to improve conditioning

# Compute the first row of the Toeplitz matrix
first_row <- sapply(time_points, function(t)
  rbf_kernel(time_points[1], t, length_scale, variance))

# Create the Toeplitz matrix
K <- toeplitz(first_row)

# Add jitter to the diagonal to improve stability
K <- K + diag(jitter, nrow(K))
image(K)
# Invert the Toeplitz matrix
K_inv <- solve(K)

# Display the inverted matrix
image(K_inv)


par(mfrow = c(1, 2))
image(K, main = "Toeplitz Matrix (K)", col = terrain.colors(100))
image(K_inv, main = "Inverse Toeplitz Matrix (K_inv)", col = terrain.colors(100))

# Assuming Y is your original tensor from the simulation
Y=y
N <- dim(Y)[1]
D <- dim(Y)[2]
T <- dim(Y)[3]

# Convert Y to logit scale, with smoothing to avoid infinite values
Y_smooth <- (Y * (N * D * T - 1) + 0.5) / (N * D * T)
Y_logit <- log(Y_smooth / (1 - Y_smooth))

# Compute mu_d from the logit Y
mu_d <- array(0, dim = c(D, T))
for (d in 1:D) {
  p <- colMeans(Y_logit[, d, ])
  mu_d[d, ] <- p  
}

# Center Y_logit by subtracting mu_d (already log-transformed prevalence)
Y_centered <- array(0, dim = c(N, D, T))
for (d in 1:D) {
  Y_centered[, d, ] <- Y_logit[, d, ] - matrix(mu_d[d, ], nrow = N, ncol = T, byrow = TRUE)
}