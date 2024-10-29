
// -*- mode: C++; c-indent-level: 4; c-basic-offset: 4; indent-tabs-mode: nil; -*-
// we only include RcppArmadillo.h which pulls Rcpp.h in for us
#include "RcppArmadillo.h"

// via the depends attribute we tell Rcpp to create hooks for RcppArmadillo
//
// [[Rcpp::depends(RcppArmadillo)]]

// For simplicity, no test data, only training data, and no error calculation.
// X - n x p data matrix
// y - n length vector of classes, from 0 to K-1
// numIter - number of iterations, default 50
// eta - damping parameter, default 0.1
// lambda - ridge parameter, default 1
// beta_init - p x K matrix of starting beta values (always supplied in correct format)

// [[Rcpp::export]]
Rcpp::List LRMultiClass_c(const arma::mat& X, const arma::uvec& y, const arma::mat& beta_init,
                          int numIter = 50, double eta = 0.1, double lambda = 1) {
  // Initialize parameters
  int K = arma::max(y) + 1;  // Number of classes
  int p = X.n_cols;           // Number of features
  int n = X.n_rows;           // Number of data points
  
  // Initialize beta from the provided beta_init
  arma::mat beta = beta_init;  
  arma::vec objective(numIter + 1);  // Store objective values
  
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
  
  // Newton's method cycle
  for (int t = 0; t < numIter; ++t) {
    for (int k = 0; k < K; ++k) {
      arma::vec Pk = prob_train.col(k);  // Extract column for class k
      
      // Calculate gradient
      arma::vec gradient = X.t() * (Pk - arma::conv_to<arma::vec>::from(y == k)) + lambda * beta.col(k);
      
      // Calculate diagonal weight matrix for Hessian
      arma::vec WkDiag = Pk % (1 - Pk);  // Element-wise multiplication
      
      // Hessian matrix
      arma::mat hessian = X.t() * (X.each_col() % WkDiag) + lambda * arma::eye(p, p);
      
      // Solve for delta beta using Cholesky decomposition (fast for symmetric matrices)
      arma::vec deltaBeta = arma::solve(hessian, gradient, arma::solve_opts::fast);
      
      // Update beta for class k
      beta.col(k) -= eta * deltaBeta;
    }
    
    // Recalculate probabilities and objective
    prob_train = calculateProbs(X, beta);
    objective(t + 1) = calcObjective(prob_train, y, beta, lambda);
  }
  
  // Return results as a named list
  return Rcpp::List::create(Rcpp::Named("beta") = beta,
                            Rcpp::Named("objective") = objective);
}