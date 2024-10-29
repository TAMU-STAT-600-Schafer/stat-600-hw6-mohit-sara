// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-

// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for
// RcppArmadillo so that the build process will know what to do
//
// [[Rcpp::depends(RcppArmadillo)]]

// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// beta_init - p x K matrix of starting beta values (always supplied in right format)
// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                          int numIter = 50, double eta = 0.1, double lambda = 1){
  // All input is assumed to be correct
  
  // Initialize some parameters
  int K = max(y) + 1; // number of classes
  int p = X.n_cols;
  int n = X.n_rows;
  arma::mat beta = beta_init; // to store betas and be able to change them if needed
  arma::vec objective(numIter + 1); // to store objective values
  
  // Initialize anything else that you may need
  // Initialize beta from the beta_init HW 3 reference
  arma::mat beta = beta_init;  
  arma::vec objective(numIter + 1);  //Storing objective values
  
  
  
  // Helper function to calculate probabilities
  auto calculateProbs = [](const arma::mat& X, const arma::mat& beta) {
    arma::mat linearComb = X * beta;  // n x K matrix
    arma::vec maxVals = arma::max(linearComb, 1);  // Max value per row (n x 1)
    
    // Subtract row-wise maximum for numerical stability
    linearComb.each_col() -= maxVals;
    
    arma::mat expXB = arma::exp(linearComb);  // Exponentiate each element
    return expXB.each_col() / arma::sum(expXB, 1);  // Normalize by row sums
  };
  
  // Helper function to calculate the objective (regularized log-likelihood)
  auto calcObjective = [&](const arma::mat& P, const arma::uvec& y, const arma::mat& beta, double lambda) {
    double logLikelihood = 0;
    for (int i = 0; i < n; ++i) {
      logLikelihood -= std::log(P(i, y(i)) + 1e-10);  // Add epsilon to avoid log(0)
    }
    double regularization = 0.5 * lambda * arma::accu(beta % beta);  // Ridge penalty
    return logLikelihood + regularization;
  };
  
  // Calculate initial probabilities and objective
  arma::mat prob_train = calculateProbs(X, beta);
  objective(0) = calcObjective(prob_train, y, beta, lambda);
  
  
  // Newton's method cycle - implement the update EXACTLY numIter iterations
  
  
  // Create named list with betas and objective values
  return Rcpp::List::create(Rcpp::Named("beta") = beta,
                            Rcpp::Named("objective") = objective);
}