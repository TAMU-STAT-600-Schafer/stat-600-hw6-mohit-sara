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
    
    // Helper function to calculate class probabilities using softmax
    auto calculateProbs = [](const arma::mat& X, const arma::mat& beta) {
      arma::mat linearComb = X * beta;
      linearComb.each_row() -= arma::max(linearComb, 1);  // For numerical stability
      arma::mat expXB = arma::exp(linearComb);
      return expXB.each_col() / arma::sum(expXB, 1);  // Softmax normalization
    };
    
    
    // Helper function to calculate the objective value
    auto calcObjective = [&](const arma::mat& P, const arma::uvec& y) {
      arma::vec logLikelihood = arma::log(P(arma::linspace<arma::uvec>(0, n - 1, n), y));
      double regularization = (lambda / 2) * arma::accu(arma::square(beta));
      return -arma::accu(logLikelihood) + regularization;
    };
    
    
    // Initialize probabilities and objective value
    arma::mat prob_train = calculateProbs(X, beta);
    objective(0) = calcObjective(prob_train, y);
    
    
    
    // Newton's method cycle - implement the update EXACTLY numIter iterations
    
    for (int t = 0; t < numIter; ++t) {
      for (int k = 0; k < K; ++k) {
        arma::vec Pk = prob_train.col(k);  // Probabilities for class k
        arma::vec indicator = (y == k);    // Indicator vector for class k
        
        // Calculate gradient
        arma::vec gradient = X.t() * (Pk - indicator) + lambda * beta.col(k);
        
        // Diagonal weight matrix Wk (diagonal stored as a vector)
        arma::vec WkDiag = Pk % (1 - Pk);
        
        // Hessian matrix for class k
        arma::mat hessian = X.t() * (X.each_col() % WkDiag) + lambda * arma::eye(p, p);
        
        // Solving linear system for the update step (currently trying to avoid matrix inversion)
        arma::vec deltaBeta = arma::solve(hessian, gradient);
        
        // Damped Newton update
        beta.col(k) -= eta * deltaBeta;
      }
      
      // Recalculate probabilities after beta update
      prob_train = calculateProbs(X, beta);
      
      // Update objective function
      objective(t + 1) = calcObjective(prob_train, y);
    }
    
    
    // Create named list with betas and objective values
    return Rcpp::List::create(Rcpp::Named("beta") = beta,
                              Rcpp::Named("objective") = objective);
}
