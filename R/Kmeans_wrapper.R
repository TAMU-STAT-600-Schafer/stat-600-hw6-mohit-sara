#' K-means Clustering using Rcpp and Armadillo
#'
#' @param X A numeric matrix with 'n' rows representing data points and 'p' columns representing features.
#' @param K An integer specifying the number of clusters (must be greater than 1).
#' @param M (Optional) A numeric matrix of initial cluster centers with 'K' rows and 'p' columns. 
#'          If 'NULL', 'K' random rows from 'X' are selected as initial centroids.
#' @param numIter An integer specifying the maximum number of iterations. Default is 100.
#'
#' @return A vector of length 'n' with cluster assignments for each data point. 
#'         Each element in the returned vector corresponds to the assigned cluster (0-indexed) for the respective data point.
#' @export
#'
#' @examples
#' # Example usage of MyKmeans
#' set.seed(928)
#' X <- matrix(rnorm(50), nrow = 10, ncol = 5) # A (10 x 5) matrix
#' K <- 3 # Number of clusters
#' 
#' # Executing MyKmeans with random initialization for a maximum of 100 iterations.
#' Y <- MyKmeans(X = X, K = K)
#' print(Y)
#' 
#' M <- matrix(rnorm(15), nrow = 3, ncol = 5) # A (3 x 5) matrix
#' 
#' # Executing MyKmeans with specified initialization for a maximum of 100 iterations.
#' Y <- MyKmeans(X = X, K = K, M = M, numIter = 100)
#' print(Y)
#' 
#' 
#' #' # Edge-case examples
#' # Single-cluster case (K = 2)
#' Y <- MyKmeans(X = X, K = 2)
#' print(Y)
MyKmeans <- function(X, K, M = NULL, numIter = 100){
  
  n = nrow(X) # number of rows in X
  
  # Checks on X
  # Check to ensure X is a matrix or a dataframe
  if(!is.matrix(X)){
    if(!is.data.frame(X)){
      stop("X should be a matrix or a data frame.")
    }else{
      X <- as.matrix(X)
    }
  }
  # Check to ensure X isn't empty
  if(nrow(X) == 0 | ncol(X) == 0) stop("X must be a non-empty matrix.")
  # To check if X is NULL
  if(is.null(X)) stop("X cannot be NULL.")
  # No two rows of X can be identical
  if(!all(dim(X) == dim(unique(X)))) stop("X cannot have duplicate rows.")
  # No values of X can be NA
  if(any(is.na(X))) stop("No element of X can be NA.")
  # All elements of X should strictly be numeric
  if(!all(is.numeric(X))) stop("All elements of X must be numeric.")
  
  # Checks on K
  # Checks the length of K
  if(!(length(K) == 1)) stop("Length of K should be equal to 1.\n")
  # Check if K contains NA or NULL or is not numeric
  if(is.null(K) | is.na(K) | !is.numeric(K)) stop("K should strictly be a positive integer greater than 1.\n")
  # Checks if K is a positive integer greater than 1
  if(!((round(K, 0) == K) & (K > 1))) stop("K should strictly be a positive integer greater than 1.\n")
  # Checks if K is a matrix
  if(is.matrix(K)) K <- as.vector(K)
  
  # Check whether M is NULL or not. If NULL, initialize based on K random points from X. If not NULL, check for compatibility with X dimensions.
  if(is.null(M)){
    M <- X[sample(1:n, K, replace = FALSE) , ]
  } else{
    if(!is.matrix(M) | any(is.na(M)) | !all(is.numeric(M))) stop("M must be a numeric matrix without NA values.")
    if(any(duplicated(M)) | nrow(M) != K | ncol(M) != ncol(X)) stop("M should have K unique rows and the same number of columns as X.")
  }
  
  # Checks on numIter
  # Checks the length of numIter
  if(!(length(numIter) == 1)) stop("Length of number of iterations (numIter) should be equal to 1.\n")
  # Check if numIter contains NA or NULL or is not numeric
  if(is.null(numIter) | is.na(numIter) | !is.numeric(numIter)) stop("Number of iterations (numIter) should strictly be a positive integer greater than 1.\n")
  # Checks if numIter is a positive integer greater than 1
  if(!((round(numIter, 0) == numIter) & (numIter > 1))) stop("Number of iterations (numIter) should strictly be a positive integer greater than 1.\n")
  # Checks if numIter is a matrix
  if(is.matrix(numIter)) numIter <- as.vector(numIter)
  
  # Call C++ MyKmeans_c function to implement the algorithm
  Y = MyKmeans_c(X, K, M, numIter)
  
  # Return the class assignments
  return(Y)
}