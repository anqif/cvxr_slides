library(CVXR)
library(MASS)
library(kableExtra)

# Generate problem data
s <- 1
n <- 10
m <- 300
mu <- rep(0, 9)
Sigma <- data.frame(c(1.6484, -0.2096, -0.0771, -0.4088, 0.0678, -0.6337, 0.9720, -1.2158, -1.3219),
                    c(-0.2096, 1.9274, 0.7059, 1.3051, 0.4479, 0.7384, -0.6342, 1.4291, -0.4723),
                    c(-0.0771, 0.7059, 2.5503, 0.9047, 0.9280, 0.0566, -2.5292, 0.4776, -0.4552),
                    c(-0.4088, 1.3051, 0.9047, 2.7638, 0.7607, 1.2465, -1.8116, 2.0076, -0.3377),
                    c(0.0678, 0.4479, 0.9280, 0.7607, 3.8453, -0.2098, -2.0078, -0.1715, -0.3952),
                    c(-0.6337, 0.7384, 0.0566, 1.2465, -0.2098, 2.0432, -1.0666,  1.7536, -0.1845),
                    c(0.9720, -0.6342, -2.5292, -1.8116, -2.0078, -1.0666, 4.0882,  -1.3587, 0.7287),
                    c(-1.2158, 1.4291, 0.4776, 2.0076, -0.1715, 1.7536, -1.3587, 2.8789, 0.4094),
                    c(-1.3219, -0.4723, -0.4552, -0.3377, -0.3952, -0.1845, 0.7287, 0.4094, 4.8406))

X <- mvrnorm(m, mu, Sigma)
X <- cbind(rep(1, m), X)
b <- c(0, 0.8, 0, 1, 0.2, 0, 0.4, 1, 0, 0.7)
y <- X %*% b + rnorm(m, 0, s)

# Construct the OLS problem without constraints
beta <- Variable(n)
obj <- sum_squares(y - X %*% beta)
prob <- Problem(Minimize(obj))

# Solve the OLS problem for beta
result <- solve(prob)
cat("Objective:", result$value)

beta_ols <- result$getValue(beta)
cat("\nOptimal OLS Beta:", beta_ols)

# Add non-negativity constraint on beta
constr <- list(beta >= 0)
prob2 <- Problem(Minimize(obj), constr)

# Solve the NNLS problem for beta
result2 <- solve(prob2)
cat("Objective:", result2$value)

beta_nnls <- result2$getValue(beta)
all(beta_nnls >= 0)   # All resulting beta should be non-negative
cat("\nOptimal NNLS Beta:", beta_nnls)

# Calculate the fitted y values
fit_ols <- X %*% beta_ols
fit_nnls <- X %*% beta_nnls

# Plot coefficients for OLS and NNLS
coeff <- cbind(b, beta_ols, beta_nnls)
colnames(coeff) <- c("Actual", "OLS", "NNLS")
rownames(coeff) <- paste("beta", 1:length(b)-1, sep = "")
barplot(t(coeff), ylab = "Coefficients", beside = TRUE, legend = TRUE)
