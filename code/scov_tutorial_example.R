library(CVXR)
library(ggplot2)
library(grid)
library(Matrix)
library(expm)

# ---- Plotting functions ----
theme_bare <- theme(
  axis.line = element_blank(), 
  axis.text.x = element_blank(), 
  axis.text.y = element_blank(),
  axis.ticks = element_blank(),
  axis.title.y = element_blank(), 
  legend.position = "none", 
  panel.background = element_rect(fill = "white"), 
  panel.border = element_blank(), 
  panel.grid.major = element_blank(), 
  panel.grid.minor = element_blank(), 
)

plotSpMat <- function(S, alpha) {
  n <- nrow(S)
  df <- expand.grid(j = seq_len(n), i = seq_len(n))
  df$z = as.character(as.numeric(S) != 0)
  p <- ggplot(data = df, mapping = aes(x = i, y = j, fill = z)) +
    geom_tile(color = "black") +
    scale_fill_brewer(type = "qual", palette = "Paired") +
    scale_y_reverse()
  if (missing(alpha)) {
    p <- p + xlab("Truth")
  } else {
    p <- p + xlab(parse(text=(paste0("alpha == ", alpha))))
  }
  p + theme_bare
}

# ---- Problem data ----
set.seed(1)
tol <- 1e-4
n <- 10      # Dimension of matrix
m <- 1000    # Number of samples

A <- rsparsematrix(n, n, 0.15, rand.x = stats::rnorm)
S_true <- A %*% t(A) + 0.05 * diag(rep(1, n))    ## Force matrix to be strictly positive definite
R <- base::solve(S_true)
x_sample <- matrix(stats::rnorm(n * m), nrow = m, ncol = n) %*% t(expm::sqrtm(R))
Q <- cov(x_sample)    ## Sample covariance matrix
plotSpMat(S_true)

# alphas <- c(10, 8, 6, 4, 1)
alpha <- 6

# ---- Sparse inverse covariance estimation problem ----
S <- Variable(n, n, PSD = TRUE)
obj <- log_det(S) - matrix_trace(S %*% Q)
constr <- list(sum(abs(S)) <= alpha)
prob <- Problem(Maximize(obj), constr)
result <- solve(prob)
S_res <- result$getValue(S)

S_res[abs(S_res) <= tol] <- 0
plotSpMat(S_res, alpha)
