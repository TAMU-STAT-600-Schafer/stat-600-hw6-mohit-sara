// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::uvec MyKmeans_c(const arma::mat& X, int K,
                            const arma::mat& M, int numIter = 100){
    // All input is assumed to be correct
    
    // Initialize some parameters
    int n = X.n_rows;
    int p = X.n_cols;
    arma::uvec Y(n); // to store cluster assignments
    
    // Initialize any additional parameters if needed
    arma::vec X_norm = arma::sum(arma::square(X), 1); // row norms of X
    arma::vec M_norm = arma::sum(arma::square(M), 1); // row norms of M
    
    // For loop with kmeans algorithm
    for (int iter = 0; iter < numIter; iter++) {
      // Calculating squared Euclidean distance between each point and centroid
      arma::mat dist = arma::repmat(X_norm, 1, K) + arma::repmat(M_norm.t(), n, 1) - 2 * X * M.t();
      
      // Assign each data point to the nearest cluster
      // Closest centroid index for each point
      Y = arma::index_min(dist, 1);
      
      // Update centroids
      // Temporary matrix to store new centroids
      arma::mat M_temp = arma::zeros(K, p);
      // Count points in each cluster
      arma::uvec count = arma::zeros<arma::uvec>(K);
    }
    
    // Returns the vector of cluster assignments
    return(Y);
}

