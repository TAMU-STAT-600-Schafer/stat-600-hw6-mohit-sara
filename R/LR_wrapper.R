#' Title
#'
#' @param X 
#' @param y 
#' @param numIter 
#' @param eta 
#' @param lambda 
#' @param beta_init 
#'
#' @return
#' @export
#'
#' @examples
#' # Give example
LRMultiClass <- function(X, y, numIter = 50, eta = 0.1, lambda = 1, beta_init = NULL){
  
  # Compatibility checks from HW3 and initialization of beta_init
  
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
  
  # Call C++ LRMultiClass_c function to implement the algorithm
  out = LRMultiClass_c(X, y, numIter, eta, lambda, beta_init)
  
  # Return the class assignments
  return(out)
}