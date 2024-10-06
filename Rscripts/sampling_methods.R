elliptical_slice <- function(x, prior_mean, prior_cov, log_likelihood_fn) {
  nu <- mvrnorm(1, prior_mean, prior_cov)
  log_y <- log_likelihood_fn(x, ...) + log(runif(1))
  theta <- runif(1, 0, 2*pi)
  theta_min <- theta - 2*pi
  theta_max <- theta
  
  while (TRUE) {
    proposal <- x * cos(theta) + nu * sin(theta)
    if (log_likelihood_fn(proposal, ...) > log_y) {
      return(proposal)
    }
    if (theta < 0) {
      theta_min <- theta
    } else {
      theta_max <- theta
    }
    theta <- runif(1, theta_min, theta_max)
  }
}
