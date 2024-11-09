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
    arma::mat M_copy = M; // A copy of M since it is a constant variable
    arma::vec X_norm = arma::sum(arma::square(X), 1); // row norms of X
    arma::vec M_norm = arma::sum(arma::square(M_copy), 1); // row norms of M
    arma::mat dist(n, K); // Preallocating distance matrix
    arma::mat M_temp(K, p);
    arma::vec count(K);
    
    // For loop with kmeans algorithm
    for (int iter = 0; iter < numIter; iter++) {
      // Calculating squared Euclidean distance between each point and centroid
      // Using norms of rows of X and M to optimize the distance calculation
      dist = arma::repmat(X_norm, 1, K) + arma::repmat(M_norm.t(), n, 1) - 2 * X * M_copy.t();
      
      // Assign each data point to the nearest cluster
      // Closest centroid index for each point
      Y = arma::index_min(dist, 1);
      
      // Update centroids
      // Update the centroids by calculating the mean of all points assigned to each cluster
     
      // Temporary matrix to store new centroids
      //arma::mat M_temp = arma::zeros(K, p);
      M_temp.zeros();
      // Count points in each cluster
      //arma::uvec count = arma::zeros<arma::uvec>(K);
      count.zeros();
      
      for (int i = 0; i < n; i++) {
        // Collect points in cluster
        M_temp.row(Y(i)) += X.row(i);
        // Increment count for this cluster
        count(Y(i))++;
      }
      
      if (arma::any(count == 0)) {
        // No points in cluster
        Rcpp::stop("Error: Change the initialization for M.");
      }
      
      for (int j = 0; j < K; j++) {
        // Calculate cluster means
        M_temp.row(j) /= count(j);
      }
      
      // Check for convergence: if centroid positions haven't changed beyond a threshold, stop the loop
      double centroid_change = arma::accu(arma::square(M_copy - M_temp));
      if (centroid_change < 1e-8) {
        Rcpp::Rcout << "Converged after " << iter + 1 << " iterations.\n";
        break;
      }
      
      // Update centroids and norms
      M_copy = M_temp;
      M_norm = arma::sum(arma::square(M_copy), 1);
      
      // Print that the algorithm could not converge if the for loop has reached the 
      // maximum number of iterations (numIter).
      if (iter == (numIter - 1)) Rcpp::Rcout << "The algorithm did not converge in " << numIter << " iterations.\n";
    }
    
    // Returns the vector of cluster assignments
    return(Y);
}

