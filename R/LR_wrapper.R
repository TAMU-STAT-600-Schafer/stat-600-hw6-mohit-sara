
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
  
  ## Check the supplied parameters as described. You can assume that X, Xt are matrices; y, yt are vectors; and numIter, eta, lambda are scalars. You can assume that beta_init is either NULL (default) or a matrix.
  ###################################
  # Check that the first column of X and Xt are 1s, if not - display appropriate message and stop execution.
  if(!all(X[,1] == 1)){ #for X
    stop("First columns of X is not all ones")
  }
  
  # Check for compatibility of dimensions between X and Y
  if(nrow(X) != length(y)){
    stop("Number of rows in X doesn't match lenght of Y")
  }
  
  
  # Check eta is positive
  if(eta<= 0){
    stop("Eta must be positive")
  }
  
  # Check lambda is non-negative
  if(lambda < 0){
    stop("Lambda must be non-negative.")
  }
  
  
  K = length(unique(y)) #number of classes
  p = ncol(X) #number of features
  
  # Check whether beta_init is NULL. If NULL, initialize beta with p x K matrix of zeroes. If not NULL, check for compatibility of dimensions with what has been already supplied.
  if (is.null(beta_init)){
    #initialize beta with p x K matrix of zeroes
    beta = matrix(0, nrow = p, ncol = K)
  }
  else{
    # not NULL, check for compatibility of dimensions with what has been already supplied
    if(nrow(beta_init) != p || ncol(beta_init) != K){
      stop("beta_init dimensions are incompatible with X and y")
    }
    beta = beta_init
  }
  
  
  # Call C++ LRMultiClass_c function to implement the algorithm
  out = LRMultiClass_c(X, y, numIter, eta, lambda, beta_init)
  
  # Return the class assignments
  return(out)
}