#' Multiclass Logistic Regression
#'
#' This function performs multiclass logistic regression using gradient descent.
#' It supports regularization and requires a properly formatted design matrix.
#'
#' @param X A numeric matrix where each row represents an observation and each column represents a feature.
#'        The first column of \code{X} should be all ones to account for the intercept term.
#' @param y A numeric vector representing the class labels for each observation. Its length should match the
#'        number of rows in \code{X}.
#' @param numIter An integer specifying the number of iterations for the gradient descent algorithm. Default is 50.
#' @param eta A positive numeric value representing the learning rate. Default is 0.1.
#' @param lambda A non-negative numeric value representing the regularization parameter. Default is 1.
#' @param beta_init A numeric matrix of initial coefficients for each class and feature. If \code{NULL}, the function
#'        initializes the coefficients to zero. The matrix should have dimensions \code{ncol(X)} x \code{length(unique(y))}.
#'
#' @return A numeric matrix representing the final coefficients for each class and feature after training.
#'         The matrix dimensions are \code{ncol(X)} x \code{length(unique(y))}.
#' @export
#'
#' @examples
#' # Example usage:
#' # Assume X is a matrix with observations and features, and y is a vector of class labels
#' X <- matrix(c(1, 1, 1, 1, 1, 2, 3, 4, 5, 6), ncol = 2)  # Add an intercept column (all ones)
#' y <- c(1, 2, 1, 2, 1)  # Class labels
#' beta <- LRMultiClass(X, y, numIter = 100, eta = 0.05, lambda = 0.1)
#' print(beta)
#'
#'
#'

Rcpp::sourceCpp("C:/Users/saraa/Desktop/stat-600-hw6-mohit-sara/src/LRMultiClass.cpp")
LRMultiClass <- function(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL) {
  
  # Compatibility checks
  if (!all(X[, 1] == 1)) stop("First column of X must be all ones for intercept.")
  if (nrow(X) != length(y)) stop("Number of rows in X must match length of y.")
  if (eta <= 0) stop("Learning rate (eta) must be positive.")
  if (lambda < 0) stop("Regularization parameter (lambda) must be non-negative.")
  
  # Initialize beta_init if NULL
  K <- length(unique(y))  # Number of classes
  p <- ncol(X)            # Number of features (including intercept)
  if (is.null(beta_init)) {
    beta_init <- matrix(0, nrow = p, ncol = K)
  } else if (nrow(beta_init) != p || ncol(beta_init) != K) {
    stop("Dimensions of beta_init are incompatible with X and y.")
  }
  
  # Check to ensure numIter is not NA or non-numeric
  if(is.na(numIter) | !is.numeric(numIter)) stop("numIter must be a single positive integer.")
  # Check if numIter is a single positive integer
  if(length(numIter) != 1 | numIter <= 0 | numIter != as.integer(numIter)) stop(" numIter must be a positive integer.")
  
  
  # Call the C++ function with the correct argument order
  out <- LRMultiClass_c(X, y, beta_init, numIter, eta, lambda)
  
  # Return the result
  return(out)
}

# Define a sample dataset
X <- cbind(1, matrix(rnorm(100 * 4), nrow = 100))  # Add intercept column
y <- as.integer(runif(100, 0, 3))  # 3 classes, integers 0, 1, 2
beta_init <- matrix(0, nrow = ncol(X), ncol = length(unique(y)))  # Initialize beta

# Run the function
result <- LRMultiClass(X, y, beta_init = beta_init)
print(result)