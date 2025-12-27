# ==============================================================================
# BHIFR Simulation Study 2: Heteroscedasticity & Efficiency
# Algorithm: Component-wise Adaptive MALA + Dual Chain Capability
# CODE: DENSE MATRIX VERSION (Simpler & Stable)
# NOTE: Since parameter dim ~ 120x120, Dense Matrix is extremely fast and valid.
# ==============================================================================

rm(list=ls())
set.seed(123) # Global seed for reproducibility

# ==============================================================================
# 1. Environment Loading
# ==============================================================================
pkgs <- c("Rcpp", "RcppArmadillo", "splines2", "MASS", "dplyr", 
          "knitr", "foreach", "doSNOW", "parallel", "tidyr", 
          "kableExtra", "ggplot2", "patchwork", "ggsci", "cowplot")

# Install missing packages
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)

# Load packages
invisible(lapply(pkgs, require, character.only = TRUE))

# Define plotting theme
theme_paper <- function() {
  theme_bw() +
    theme(
      text = element_text(family = "serif"),
      panel.grid.major = element_line(color = "grey90"),
      panel.grid.minor = element_blank(),
      axis.text = element_text(color="black", size=10),
      axis.title = element_text(color="black", size=12, face="bold"),
      legend.position = "bottom",
      strip.background = element_rect(fill="grey95"),
      strip.text = element_text(face="bold", size=10),
      plot.title = element_text(hjust = 0.5, face="bold")
    )
}

# ==============================================================================
# 2. C++ Core Implementation (PURE DENSE VERSION)
# ==============================================================================
cpp_code_string <- '
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>

using namespace arma;
using namespace Rcpp;

// Constants
const double nugget = 1e-6; 

// =============================================================================
// 1. Helper Functions
// =============================================================================

double logit(double p) {
    if(p < 1e-10) p = 1e-10;
    if(p > 1.0 - 1e-10) p = 1.0 - 1e-10;
    return std::log(p / (1.0 - p));
}

double sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

double fisher_z(double rho) { 
    if(rho > 0.995) rho = 0.995; if(rho < -0.995) rho = -0.995;
    return std::atanh(rho); 
}
double inv_fisher_z(double z) { 
    double r = std::tanh(z); if(r > 0.995) r = 0.995; if(r < -0.995) r = -0.995;
    return r;
}

// =============================================================================
// 2. Likelihood and Gradient Calculations
// =============================================================================

// Calculate Marginal Log-Likelihood for Lambda (Dense Version)
double calc_marginal_loglik_lambda_dense(const vec& lambda, const mat& X_L, const mat& X_U,
                                         const mat& BTWB, const mat& BTY_proj, const mat& Prior_Prec) {
    int n = X_L.n_rows; int q = X_L.n_cols;
    
    // Construct subject design matrix P
    mat P = ones(n, q + 1);
    for(int j=0; j<q; ++j) P.col(j+1) = (1.0 - lambda(j)) * X_L.col(j) + lambda(j) * X_U.col(j);
    
    mat PTP = P.t() * P;
    
    // Core: Kronecker Product (Dense)
    // D^T Sigma^-1 D = P^T P (kron) B^T W B
    // Dimension approx 120x120, dense calculation is very fast
    mat Lambda_beta = kron(PTP, BTWB) + Prior_Prec; 
    
    // Calculate mean part mu_beta
    // RHS = D^T Sigma^-1 Y = vec(BTY_proj * P)
    vec RHS = vectorise(BTY_proj * P);
    
    mat L;
    bool success = chol(L, Lambda_beta, "lower");
    if(!success) { // Fallback jitter
         mat I(Lambda_beta.n_rows, Lambda_beta.n_cols, fill::eye);
         if(!chol(L, Lambda_beta + 1e-7 * I, "lower")) return -1e20;
    }
    
    // Log determinant: 2 * sum(log(diag(L)))
    double log_det = 2.0 * sum(log(L.diag()));
    
    // Calculate quadratic form: 0.5 * mu^T Lambda mu
    // solve(L, RHS) solves L * y = RHS
    vec y = solve(trimatl(L), RHS);
    
    // Marginal LogLik = -0.5 * log|Lambda| + 0.5 * y^T y
    return -0.5 * log_det + 0.5 * dot(y, y);
}

double calc_joint_loglik(const mat& E_L, const mat& E_U, const vec& sL, const vec& sU, double rho) {
    int n = E_L.n_rows; int m = E_L.n_cols;
    double omr2 = 1.0 - rho*rho;
    
    double log_det = -n * (accu(log(sL)) + accu(log(sU))) - 0.5 * n * m * std::log(omr2);
    
    double quad = 0.0;
    for(int t=0; t<m; ++t) {
        double vL = sL(t); double vU = sU(t);
        double tL = accu(pow(E_L.col(t), 2)) / (vL*vL);
        double tU = accu(pow(E_U.col(t), 2)) / (vU*vU);
        double tLU = accu(E_L.col(t) % E_U.col(t)) / (vL*vU);
        quad += (tL + tU - 2.0 * rho * tLU);
    }
    return log_det - 0.5 / omr2 * quad;
}

vec calc_gradient_gamma(const mat& E_L, const mat& E_U, const vec& sL, const vec& sU, double rho,
                        const mat& B_sig, const mat& Omega_sig, const vec& gamma_curr, double tau, bool is_lower) {
    int m = sL.n_elem; int n = E_L.n_rows;
    vec v_vec(m);
    double denom = 1.0 / (1.0 - rho*rho);
    for(int t=0; t<m; ++t) {
        double vL = sL(t); double vU = sU(t);
        double SSE_L = accu(pow(E_L.col(t), 2));
        double SSE_U = accu(pow(E_U.col(t), 2));
        double SCP = accu(E_L.col(t) % E_U.col(t));
        
        if(is_lower) v_vec(t) = denom * (SSE_L/(vL*vL) - rho*SCP/(vL*vU)) - n;
        else         v_vec(t) = denom * (SSE_U/(vU*vU) - rho*SCP/(vL*vU)) - n;
    }
    return B_sig.t() * v_vec - tau * Omega_sig * gamma_curr;
}

// =============================================================================
// 3. Model 1: BHIFR-Homo
// =============================================================================
// [[Rcpp::export]]
List BHIFR_Homo_Cpp(mat Y_L, mat Y_U, mat X_L, mat X_U, mat B, mat Omega, int n_iter, int n_burnin) {
    int n = Y_L.n_rows; int m = Y_L.n_cols; int q = X_L.n_cols; int K = B.n_cols;
    int dim_beta = 2 * K * (q + 1);

    // Initialize parameters
    vec lambda = randu(q);
    vec beta = zeros(dim_beta);
    
    mat P_init = ones(n, q + 1);
    for(int j=0; j<q; ++j) P_init.col(j+1) = 0.5 * X_L.col(j) + 0.5 * X_U.col(j);
    mat D_init = kron(P_init, B); 
    int dim_single = K * (q + 1);
    mat DTD = D_init.t() * D_init + 0.1 * eye(dim_single, dim_single);
    vec bL_init = solve(DTD, D_init.t() * vectorise(Y_L.t()));
    vec bU_init = solve(DTD, D_init.t() * vectorise(Y_U.t()));
    for(int j=0; j<=q; ++j) {
        beta.subvec(j*2*K, j*2*K+K-1)      = bL_init.subvec(j*K, j*K+K-1);
        beta.subvec(j*2*K+K, j*2*K+2*K-1) = bU_init.subvec(j*K, j*K+K-1);
    }
    
    double sigma2_L = 0.1, sigma2_U = 0.1, rho = 0.0;
    double nu_sigma_L = 1.0, nu_sigma_U = 1.0; 
    vec tau_beta = ones(2 * (q + 1)) * 10.0; 
    vec nu_beta = ones(2 * (q + 1));
    
    // Adaptive MCMC parameters
    vec step_l_vec = ones(q) * 1.0; 
    
    mat res_b(n_iter-n_burnin, dim_beta); 
    mat res_l(n_iter-n_burnin, q); 
    mat res_s(n_iter-n_burnin, 2); 
    vec res_r(n_iter-n_burnin);
    
    mat BTB = B.t()*B; mat BTY_L = B.t()*Y_L.t(); mat BTY_U = B.t()*Y_U.t();
    
    for(int it=0; it<n_iter; ++it) {
        // Robbins-Monro Adaptation Decay
        double gamma_t = std::min(1.0, 10.0 / (double(it) + 100.0));
        
        // Point 4: Add Nugget to variance
        double sL_val = std::sqrt(sigma2_L + nugget);
        double sU_val = std::sqrt(sigma2_U + nugget);
        
        double sc = 1.0/(1.0 - rho*rho);
        double w11=sc/(sL_val*sL_val); double w22=sc/(sU_val*sU_val); double w12=-sc*rho/(sL_val*sU_val);
        
        // Construct block matrix B^T W B (Dense)
        mat BTWB_blk = zeros(2*K, 2*K);
        BTWB_blk.submat(0,0,K-1,K-1)=w11*BTB; BTWB_blk.submat(K,K,2*K-1,2*K-1)=w22*BTB;
        BTWB_blk.submat(0,K,K-1,2*K-1)=w12*BTB; BTWB_blk.submat(K,0,2*K-1,K-1)=w12*BTB;
        
        mat BTY_proj = zeros(2*K, n);
        BTY_proj.rows(0,K-1) = w11*BTY_L + w12*BTY_U;
        BTY_proj.rows(K,2*K-1) = w12*BTY_L + w22*BTY_U;
        
        // Construct prior precision matrix (Dense)
        mat Prior_Prec = zeros(dim_beta, dim_beta);
        for(int g=0; g<2*(q+1); ++g) {
            Prior_Prec.submat(g*K, g*K, g*K+K-1, g*K+K-1) = tau_beta(g) * Omega;
        }
        
        double current_ll = calc_marginal_loglik_lambda_dense(lambda, X_L, X_U, BTWB_blk, BTY_proj, Prior_Prec);
        
        // --- Step 1: Update Lambda ---
        for(int j=0; j<q; ++j) {
            double eta_curr = logit(lambda(j));
            double eta_prop = eta_curr + step_l_vec(j) * randn();
            double lambda_prop_val = sigmoid(eta_prop);
            
            vec lambda_prop = lambda; 
            lambda_prop(j) = lambda_prop_val;
            
            double prop_ll = calc_marginal_loglik_lambda_dense(lambda_prop, X_L, X_U, BTWB_blk, BTY_proj, Prior_Prec);
            
            double jacobian_diff = std::log(lambda_prop_val * (1.0 - lambda_prop_val)) - 
                                   std::log(lambda(j) * (1.0 - lambda(j)));
            
            double log_ratio = prop_ll - current_ll + jacobian_diff;
            bool accept = std::log(randu()) < log_ratio;
            
            if(accept) {
                lambda(j) = lambda_prop_val;
                current_ll = prop_ll;
            }
            
            // Adaptive Update
            if(it < n_burnin) {
                double alpha = accept ? 1.0 : 0.0;
                double log_step = std::log(step_l_vec(j)) + gamma_t * (alpha - 0.44);
                step_l_vec(j) = std::exp(log_step);
            }
        }
        
        // --- Step 3: Update Beta ---
        mat P(n,q+1); P.col(0).fill(1.0); for(int j=0; j<q; ++j) P.col(j+1)=(1-lambda(j))*X_L.col(j)+lambda(j)*X_U.col(j);
        mat PTP = P.t()*P;
        
        // Dense Kronecker
        mat Prec = kron(PTP, BTWB_blk) + Prior_Prec;
        
        mat L_post;
        bool chol_ok = chol(L_post, Prec, "lower");
        if(!chol_ok) { 
             mat I(dim_beta, dim_beta, fill::eye);
             chol(L_post, Prec + 1e-7 * I, "lower");
        }
        
        vec RHS = vectorise(BTY_proj * P);
        vec z = randn(dim_beta);
        
        // Sampling: beta = L^-T * (L^-1 * RHS + z)
        vec temp1 = solve(trimatl(L_post), RHS);
        vec mean_part = solve(trimatu(L_post.t()), temp1);
        vec temp2 = solve(trimatu(L_post.t()), z);
        
        beta = mean_part + temp2;
        
        // --- Step 4: Update Tau/Nu ---
        for(int g=0; g < 2*(q+1); ++g) {
             vec b_sub = beta.subvec(g*K, g*K+K-1);
             double quad = as_scalar(b_sub.t() * Omega * b_sub);
             tau_beta(g) = R::rgamma(0.5*K + 0.5, 1.0/(0.5*quad + 1.0/nu_beta(g)));
             
             double rate_nu_beta = 1.0/25.0 + tau_beta(g);
             nu_beta(g) = 1.0 / R::rgamma(1.0, 1.0 / rate_nu_beta); 
        }
        
        // --- Step 2: Update Sigma ---
        mat YL_h(n,m), YU_h(n,m); YL_h.zeros(); YU_h.zeros();
        vec aL=beta.head(K); vec aU=beta.subvec(K, 2*K-1); 
        YL_h.each_row() += (B*aL).t(); YU_h.each_row() += (B*aU).t();
        for(int j=0;j<q;++j) { 
            vec bL=beta.subvec((j+1)*2*K, (j+1)*2*K+K-1); vec bU=beta.subvec((j+1)*2*K+K, (j+1)*2*K+2*K-1);
            for(int i=0;i<n;++i) { YL_h.row(i)+=P(i,j+1)*(B*bL).t(); YU_h.row(i)+=P(i,j+1)*(B*bU).t(); } 
        }
        mat EL=Y_L-YL_h; mat EU=Y_U-YU_h;
        
        vec theta_curr(3); 
        theta_curr(0)=log(sqrt(sigma2_L)); theta_curr(1)=log(sqrt(sigma2_U)); theta_curr(2)=fisher_z(rho);
        vec theta_prop = theta_curr + 0.02 * randn(3);
        
        double lsL_p=theta_prop(0); double lsU_p=theta_prop(1); double rho_p=inv_fisher_z(theta_prop(2));
        
        vec sL_c(m); sL_c.fill(sqrt(sigma2_L + nugget)); 
        vec sU_c(m); sU_c.fill(sqrt(sigma2_U + nugget)); 
        
        vec sL_p(m); sL_p.fill(sqrt(exp(2*lsL_p) + nugget)); 
        vec sU_p(m); sU_p.fill(sqrt(exp(2*lsU_p) + nugget)); 
        
        double lp_curr = calc_joint_loglik(EL,EU,sL_c,sU_c,rho) - theta_curr(0) - 1.0/(nu_sigma_L*exp(2*theta_curr(0))) - theta_curr(1) - 1.0/(nu_sigma_U*exp(2*theta_curr(1)));
        double lp_prop = calc_joint_loglik(EL,EU,sL_p,sU_p,rho_p) - theta_prop(0) - 1.0/(nu_sigma_L*exp(2*theta_prop(0))) - theta_prop(1) - 1.0/(nu_sigma_U*exp(2*theta_prop(1)));
        
        if(log(randu()) < (lp_prop - lp_curr + log(1-rho_p*rho_p) - log(1-rho*rho))) {
             sigma2_L = exp(2*lsL_p); sigma2_U = exp(2*lsU_p); rho = rho_p;
        }
        
        double rate_nu_sigma_L = 1.0/225.0 + 1.0/sigma2_L;
        nu_sigma_L = 1.0 / R::rgamma(1.0, 1.0 / rate_nu_sigma_L); 
        
        double rate_nu_sigma_U = 1.0/225.0 + 1.0/sigma2_U;
        nu_sigma_U = 1.0 / R::rgamma(1.0, 1.0 / rate_nu_sigma_U); 

        if(it>=n_burnin) {
             res_b.row(it-n_burnin)=beta.t(); res_l.row(it-n_burnin)=lambda.t();
             res_s(it-n_burnin,0)=sigma2_L; res_s(it-n_burnin,1)=sigma2_U; res_r(it-n_burnin)=rho;
        }
    }
    return List::create(Named("beta")=res_b, Named("lambda")=res_l, Named("sigma")=res_s, Named("rho")=res_r);
}

// =============================================================================
// 4. Model 2: BHIFR-Hetero (Dense Version)
// =============================================================================
// [[Rcpp::export]]
List BHIFR_Hetero_Cpp(mat Y_L, mat Y_U, mat X_L, mat X_U, mat B, mat Omega, mat B_sig, mat Omega_sig, int n_iter, int n_burnin) {
    int n=Y_L.n_rows; int m=Y_L.n_cols; int q=X_L.n_cols; int K=B.n_cols; int K_sig=B_sig.n_cols;
    int dim_beta = 2 * K * (q + 1);
    
    vec lambda=ones(q)*0.5; 
    mat P_init=ones(n,q+1); 
    for(int j=0;j<q;++j) P_init.col(j+1)=0.5*X_L.col(j)+0.5*X_U.col(j);
    
    mat D_init = kron(P_init, B); 
    int dim_single = K * (q + 1); 
    mat DTD = D_init.t() * D_init + 0.1 * eye(dim_single, dim_single); 
    vec bL_init = solve(DTD, D_init.t() * vectorise(Y_L.t()));
    vec bU_init = solve(DTD, D_init.t() * vectorise(Y_U.t()));
    vec beta = zeros(dim_beta); 
    for(int j=0; j<=q; ++j) {
        beta.subvec(j*2*K, j*2*K+K-1)      = bL_init.subvec(j*K, j*K+K-1);
        beta.subvec(j*2*K+K, j*2*K+2*K-1) = bU_init.subvec(j*K, j*K+K-1);
    }
    
    mat YL_h(n,m), YU_h(n,m); YL_h.zeros(); YU_h.zeros();
    vec aL=beta.head(K); vec aU=beta.subvec(K, 2*K-1); 
    YL_h.each_row() += (B*aL).t(); YU_h.each_row() += (B*aU).t();
    for(int j=0;j<q;++j) { 
        vec bL=beta.subvec((j+1)*2*K, (j+1)*2*K+K-1); vec bU=beta.subvec((j+1)*2*K+K, (j+1)*2*K+2*K-1);
        for(int i=0;i<n;++i) { YL_h.row(i)+=P_init(i,j+1)*(B*bL).t(); YU_h.row(i)+=P_init(i,j+1)*(B*bU).t(); } 
    }
    vec mean_log = arma::log(mean(pow(Y_L-YL_h, 2) + pow(Y_U-YU_h, 2), 0).t()/2.0 + 1e-6);
    vec gamma_L = solve(B_sig.t()*B_sig + 0.01*eye(K_sig,K_sig), B_sig.t()*mean_log);
    vec gamma_U = gamma_L; double rho=0.5;
    
    vec tau_beta = ones(2 * (q + 1)) * 10.0; vec nu_beta = ones(2 * (q + 1));
    double tau_gamma_L = 10.0, nu_gamma_L = 1.0;
    double tau_gamma_U = 10.0, nu_gamma_U = 1.0;
    
    // Adaptive parameters
    vec step_l_vec = ones(q) * 1.0; 
    
    double step_MALA = 0.002;
    double step_rho=0.02;
    
    mat res_b(n_iter-n_burnin, dim_beta); mat res_l(n_iter-n_burnin, q); 
    mat res_gL(n_iter-n_burnin, K_sig); mat res_gU(n_iter-n_burnin, K_sig); vec res_r(n_iter-n_burnin);
    mat BTB=B.t()*B; mat BTY_L=B.t()*Y_L.t(); mat BTY_U=B.t()*Y_U.t(); mat BT=B.t();
    
    for(int it=0; it<n_iter; ++it) {
        // Adaptive Decay
        double gamma_t = std::min(1.0, 10.0 / (double(it) + 100.0));
        
        vec sL = exp(B_sig*gamma_L); 
        vec sU = exp(B_sig*gamma_U);
        // Point 4: Nugget
        vec sL_nug = sqrt(square(sL) + nugget);
        vec sU_nug = sqrt(square(sU) + nugget);
        
        double sc=1.0/(1.0-rho*rho);
        
        mat BTWB_acc = zeros(2*K,2*K); 
        mat BTY_p = zeros(2*K,n);
        
        for(int t=0; t<m; ++t) {
             double vL = sL_nug(t); double vU = sU_nug(t);
             double w11=sc/(vL*vL); double w22=sc/(vU*vU); double w12=-sc*rho/(vL*vU);
             
             vec b=BT.col(t); mat bbt=b*b.t();
             BTWB_acc.submat(0,0,K-1,K-1)     += w11*bbt; 
             BTWB_acc.submat(K,K,2*K-1,2*K-1) += w22*bbt;
             BTWB_acc.submat(0,K,K-1,2*K-1)   += w12*bbt; 
             BTWB_acc.submat(K,0,2*K-1,K-1)   += w12*bbt;
             
             for(int k=0; k<K; ++k) { 
                 BTY_p.row(k)    += b(k)*(w11*Y_L.col(t).t()+w12*Y_U.col(t).t()); 
                 BTY_p.row(K+k) += b(k)*(w12*Y_L.col(t).t()+w22*Y_U.col(t).t()); 
             }
        }
        
        // Prior Prec
        mat Prior_Prec = zeros(dim_beta, dim_beta);
        for(int g=0; g<2*(q+1); ++g) {
            Prior_Prec.submat(g*K, g*K, g*K+K-1, g*K+K-1) = tau_beta(g) * Omega;
        }
        
        double current_ll = calc_marginal_loglik_lambda_dense(lambda, X_L, X_U, BTWB_acc, BTY_p, Prior_Prec);
        
        // --- Step 1: Lambda Update ---
        for(int j=0; j<q; ++j) {
            double eta_curr = logit(lambda(j));
            double eta_prop = eta_curr + step_l_vec(j) * randn();
            double lambda_prop_val = sigmoid(eta_prop);
            
            vec lambda_prop = lambda; lambda_prop(j) = lambda_prop_val;
            double prop_ll = calc_marginal_loglik_lambda_dense(lambda_prop, X_L, X_U, BTWB_acc, BTY_p, Prior_Prec);
            
            double jacobian_diff = std::log(lambda_prop_val * (1.0 - lambda_prop_val)) - 
                                   std::log(lambda(j) * (1.0 - lambda(j)));
                                   
            bool accept = std::log(randu()) < (prop_ll - current_ll + jacobian_diff);
            
            if(accept) {
                lambda(j) = lambda_prop_val;
                current_ll = prop_ll;
            }
            
            if(it < n_burnin) {
                 double alpha = accept ? 1.0 : 0.0;
                 double log_step = std::log(step_l_vec(j)) + gamma_t * (alpha - 0.44);
                 step_l_vec(j) = std::exp(log_step);
            }
        }
        
        // --- Step 3: Beta Update ---
        mat P(n,q+1); P.col(0).fill(1.0); for(int j=0;j<q;++j) P.col(j+1)=(1-lambda(j))*X_L.col(j)+lambda(j)*X_U.col(j);
        mat PTP = P.t()*P;
        
        // Dense Kronecker
        mat Prec_Beta = kron(PTP, BTWB_acc) + Prior_Prec;
        mat L_post;
        
        if(chol(L_post, Prec_Beta, "lower")) {
             vec z = randn(dim_beta);
             vec RHS = vectorise(BTY_p * P);
             
             vec temp1 = solve(trimatl(L_post), RHS);
             vec mean_part = solve(trimatu(L_post.t()), temp1);
             vec temp2 = solve(trimatu(L_post.t()), z);
             beta = mean_part + temp2;
        }

        // --- Step 4: Tau/Nu Update ---
        for(int g=0; g < 2*(q+1); ++g) {
             vec b_sub = beta.subvec(g*K, g*K+K-1); double quad = as_scalar(b_sub.t() * Omega * b_sub);
             tau_beta(g) = R::rgamma(0.5*K+0.5, 1.0/(0.5*quad + 1.0/nu_beta(g)));
             double rate_nu_beta = 1/25.0 + tau_beta(g);
             nu_beta(g) = 1.0 / R::rgamma(1.0, 1.0 / rate_nu_beta); 
        }
        
        YL_h.zeros(); YU_h.zeros();
        aL=beta.head(K); aU=beta.subvec(K, 2*K-1); 
        YL_h.each_row() += (B*aL).t(); YU_h.each_row() += (B*aU).t();
        for(int j=0;j<q;++j) { 
            vec bL=beta.subvec((j+1)*2*K, (j+1)*2*K+K-1); vec bU=beta.subvec((j+1)*2*K+K, (j+1)*2*K+2*K-1);
            for(int i=0;i<n;++i) { YL_h.row(i)+=P(i,j+1)*(B*bL).t(); YU_h.row(i)+=P(i,j+1)*(B*bU).t(); } 
        }
        mat EL=Y_L-YL_h; mat EU=Y_U-YU_h;
        
        // --- Step 2: Error Params ---
        bool accepted_gL = false;
        bool accepted_gU = false;

        {
            double quad = as_scalar(gamma_L.t() * Omega_sig * gamma_L);
            tau_gamma_L = R::rgamma(0.5*K_sig+0.5, 1.0/(0.5*quad + 1.0/nu_gamma_L));
            double rate_nu_gamma_L = 1.0/25.0 + tau_gamma_L;
            nu_gamma_L = 1.0 / R::rgamma(1.0, 1.0 / rate_nu_gamma_L); 
            
            vec g_L=calc_gradient_gamma(EL,EU,sL_nug,sU_nug,rho,B_sig,Omega_sig,gamma_L,tau_gamma_L,true);
            vec mu_L=gamma_L+0.5*step_MALA*step_MALA*g_L; vec gL_p=mu_L+step_MALA*randn(K_sig);
            
            vec sL_p = exp(B_sig*gL_p);
            vec sL_p_nug = sqrt(square(sL_p) + nugget); 
            
            vec grad_p=calc_gradient_gamma(EL,EU,sL_p_nug,sU_nug,rho,B_sig,Omega_sig,gL_p,tau_gamma_L,true);
            vec mu_L_rev=gL_p+0.5*step_MALA*step_MALA*grad_p;
            
            double lp_curr=calc_joint_loglik(EL,EU,sL_nug,sU_nug,rho)-0.5*tau_gamma_L*as_scalar(gamma_L.t()*Omega_sig*gamma_L);
            double lp_prop=calc_joint_loglik(EL,EU,sL_p_nug,sU_nug,rho)-0.5*tau_gamma_L*as_scalar(gL_p.t()*Omega_sig*gL_p);
            double lq_fwd= -0.5*dot(gL_p-mu_L,gL_p-mu_L)/(step_MALA*step_MALA);
            double lq_rev= -0.5*dot(gamma_L-mu_L_rev,gamma_L-mu_L_rev)/(step_MALA*step_MALA);
            
            if(std::log(randu()) < lp_prop - lp_curr + lq_rev - lq_fwd) { 
                gamma_L=gL_p; sL=sL_p; sL_nug=sL_p_nug;
                accepted_gL = true;
            }
        }
        
        {
            double quad = as_scalar(gamma_U.t() * Omega_sig * gamma_U);
            tau_gamma_U = R::rgamma(0.5*K_sig+0.5, 1.0/(0.5*quad + 1.0/nu_gamma_U));
            double rate_nu_gamma_U = 1/25.0 + tau_gamma_U;
            nu_gamma_U = 1.0 / R::rgamma(1.0, 1.0 / rate_nu_gamma_U); 
            
            vec g_U=calc_gradient_gamma(EL,EU,sL_nug,sU_nug,rho,B_sig,Omega_sig,gamma_U,tau_gamma_U,false);
            vec mu_U=gamma_U+0.5*step_MALA*step_MALA*g_U; vec gU_p=mu_U+step_MALA*randn(K_sig);
            
            vec sU_p = exp(B_sig*gU_p);
            vec sU_p_nug = sqrt(square(sU_p) + nugget);
            
            vec grad_p=calc_gradient_gamma(EL,EU,sL_nug,sU_p_nug,rho,B_sig,Omega_sig,gU_p,tau_gamma_U,false);
            vec mu_U_rev=gU_p+0.5*step_MALA*step_MALA*grad_p;
            
            double lp_curr=calc_joint_loglik(EL,EU,sL_nug,sU_nug,rho)-0.5*tau_gamma_U*as_scalar(gamma_U.t()*Omega_sig*gamma_U);
            double lp_prop=calc_joint_loglik(EL,EU,sL_nug,sU_p_nug,rho)-0.5*tau_gamma_U*as_scalar(gU_p.t()*Omega_sig*gU_p);
            double lq_fwd= -0.5*dot(gU_p-mu_U,gU_p-mu_U)/(step_MALA*step_MALA);
            double lq_rev= -0.5*dot(gamma_U-mu_U_rev,gamma_U-mu_U_rev)/(step_MALA*step_MALA);
            
            if(std::log(randu()) < lp_prop - lp_curr + lq_rev - lq_fwd) { 
                gamma_U=gU_p; sU=sU_p; sU_nug=sU_p_nug;
                accepted_gU = true;
            }
        }
        
        if(it < n_burnin) {
            double alpha = (double)(accepted_gL + accepted_gU) / 2.0;
            double log_step = std::log(step_MALA) + gamma_t * (alpha - 0.574);
            step_MALA = std::exp(log_step);
        }
        
        double z=fisher_z(rho); double z_p=z+step_rho*randn(); double r_p=inv_fisher_z(z_p);
        if(std::log(randu()) < calc_joint_loglik(EL,EU,sL_nug,sU_nug,r_p) - calc_joint_loglik(EL,EU,sL_nug,sU_nug,rho) + std::log(1-r_p*r_p) - std::log(1-rho*rho)) rho=r_p;
        
        if(it>=n_burnin) { 
            res_l.row(it-n_burnin)=lambda.t(); res_b.row(it-n_burnin)=beta.t(); 
            res_gL.row(it-n_burnin)=gamma_L.t(); res_gU.row(it-n_burnin)=gamma_U.t(); 
            res_r(it-n_burnin)=rho; 
        }
    }
    return List::create(Named("beta")=res_b, Named("lambda")=res_l, Named("alpha_L")=res_gL, Named("alpha_U")=res_gU, Named("rho")=res_r);
}
'

# ==============================================================================
# 3. Helper Functions and Data Generation
# ==============================================================================

create_penalty_matrix <- function(K) {
  D <- diff(diag(K), differences = 2)
  return(t(D) %*% D + diag(1e-5, K))
}

generate_data_hetero <- function(n, log_sigma_mean, log_sigma_sd, m = 51, q = 3, seed = NULL, true_params = NULL) {
  if(!is.null(seed)) set.seed(seed)
  t_grid <- seq(0, 1, length.out = m)
  
  if (is.null(true_params)) {
    lambda_true <- runif(q, 0, 1)
    mu_L_true <- 1 + 2 * t_grid
    mu_U_true <- 2 + 2 * t_grid +3* sin(pi * t_grid)
    beta_1L_true <- 2 * sin(2 * pi * t_grid)
    beta_1U_true <- 2 * sin(2 * pi * t_grid + pi / 4)
    K_beta <- 4
    poly_basis_beta <- poly(t_grid, degree = K_beta, raw = FALSE)
    alpha_2_L_star <- rnorm(K_beta, mean = 0, sd = 1 / (1:K_beta))
    alpha_2_U_star <- rnorm(K_beta, mean = 0, sd = 1 / (1:K_beta))
    beta_2L_true <- as.vector(poly_basis_beta %*% alpha_2_L_star)
    beta_2U_true <- as.vector(poly_basis_beta %*% alpha_2_U_star)
    beta_3L_true <- rep(0, m)
    beta_3U_true <- rep(0, m)
    
    Beta_L_true <- list(beta_1L_true, beta_2L_true, beta_3L_true)
    Beta_U_true <- list(beta_1U_true, beta_2U_true, beta_3U_true)
    
    true_params <- list(mu_L = mu_L_true, mu_U = mu_U_true, Beta_L = Beta_L_true, Beta_U = Beta_U_true, lambda = lambda_true)
  }
  
  c_ij <- matrix(runif(n * q,-10, 10), n, q)
  r_ij <- matrix(runif(n * q, 0, 5), n, q)
  X_L <- c_ij - r_ij
  X_U <- c_ij + r_ij
  
  P_star <- X_L * (1 - matrix(true_params$lambda, n, q, byrow = TRUE)) + X_U * matrix(true_params$lambda, n, q, byrow = TRUE)
  Y_L_mean <- matrix(true_params$mu_L, n, m, byrow = TRUE)
  Y_U_mean <- matrix(true_params$mu_U, n, m, byrow = TRUE)
  for (j in 1:q) {
    Y_L_mean <- Y_L_mean + P_star[, j] %*% t(true_params$Beta_L[[j]])
    Y_U_mean <- Y_U_mean + P_star[, j] %*% t(true_params$Beta_U[[j]])
  }
  
  K_sigma_true <- 4 
  poly_basis_sigma <- poly(t_grid, degree = K_sigma_true, raw = FALSE)
  alpha_sigma_L_star <- rnorm(K_sigma_true, mean = 0, sd = log_sigma_sd / (1:K_sigma_true))
  alpha_sigma_U_star <- rnorm(K_sigma_true, mean = 0, sd = log_sigma_sd / (1:K_sigma_true))
  log_sigma_L_t_true <- log_sigma_mean + as.vector(poly_basis_sigma %*% alpha_sigma_L_star)
  log_sigma_U_t_true <- log_sigma_mean + as.vector(poly_basis_sigma %*% alpha_sigma_U_star)
  sigma_L_t_true <- exp(log_sigma_L_t_true)
  sigma_U_t_true <- exp(log_sigma_U_t_true)
  
  true_params$sigma_L_t <- sigma_L_t_true
  true_params$sigma_U_t <- sigma_U_t_true
  
  errors <- array(NA, dim = c(n, m, 2))
  for (i in 1:n) {
    for (k in 1:m) {
      sigma_L_k <- sigma_L_t_true[k]
      sigma_U_k <- sigma_U_t_true[k]
      error_cov_k <- matrix(c(sigma_L_k ^ 2,
                              0.5 * sigma_L_k * sigma_U_k,
                              0.5 * sigma_L_k * sigma_U_k,
                              sigma_U_k ^ 2), 2, 2)
      errors[i, k, ] <- MASS::mvrnorm(1, mu = c(0, 0), Sigma = error_cov_k)
    }
  }
  Y_L <- Y_L_mean + errors[, , 1]
  Y_U <- Y_U_mean + errors[, , 2]
  
  return(list( Y_L = Y_L, Y_U = Y_U, X_L = X_L, X_U = X_U, t_grid = t_grid, true_params = true_params))
}

get_bs_basis <- function(t, df=10) bSpline(t, df=df, degree=3, intercept=TRUE)

calc_imse <- function(true_curve, est_mean) mean((true_curve - est_mean)^2)
calc_cp <- function(true_curve, lower, upper) mean(true_curve >= lower & true_curve <= upper)

# ==============================================================================
# 4. Execute Simulation 
# ==============================================================================
worker_init <- function(code_str) { 
  library(Rcpp); library(RcppArmadillo); library(splines2); library(MASS); 
  sourceCpp(code = code_str)
  return(TRUE) 
}

Rcpp::sourceCpp(code = cpp_code_string)

n_reps <- 100
sim_grid <- data.frame(rep=1:n_reps, n=100, log_sigma_mean = 1.6, log_sigma_sd = 0.3)

if(exists("cl")) { try(stopCluster(cl), silent=TRUE) }
cl <- makeCluster(detectCores() - 1)
registerDoSNOW(cl)

clusterExport(cl, c("worker_init", "generate_data_hetero", "get_bs_basis", 
                    "create_penalty_matrix", "cpp_code_string", 
                    "calc_imse", "calc_cp"))

clusterEvalQ(cl, worker_init(cpp_code_string))

pb <- txtProgressBar(max=nrow(sim_grid), style=3)
progress <- function(n) setTxtProgressBar(pb, n)

cat("Running Simulation (Pure Dense Matrix Version)...\n")

results_s2 <- foreach(i=1:nrow(sim_grid), 
                      .combine=bind_rows, 
                      .packages=pkgs, .options.snow=list(progress=progress),
                      .noexport = c("BHIFR_Homo_Cpp", "BHIFR_Hetero_Cpp")) %dopar% {
                        p <- sim_grid[i,]
                        
                        dat <- generate_data_hetero(p$n, p$log_sigma_mean, p$log_sigma_sd, seed = p$rep * 12345)
                        test_dat <- generate_data_hetero(100, p$log_sigma_mean, p$log_sigma_sd, seed = p$rep * 99999, true_params = dat$true_params)
                        
                        B <- get_bs_basis(dat$t_grid, df=15)
                        Omega <- create_penalty_matrix(ncol(B))
                        B_sig <- get_bs_basis(dat$t_grid, df=15)
                        Omega_sig <- create_penalty_matrix(ncol(B_sig))
                        K <- ncol(B)
                        
                        get_est <- function(fit, idx_start) {
                          b_samps <- fit$beta[, idx_start:(idx_start+K-1)]
                          est <- colMeans(b_samps) %*% t(B)
                          ci <- apply(b_samps %*% t(B), 2, quantile, probs=c(0.025, 0.975))
                          list(m=as.vector(est), l=ci[1,], u=ci[2,])
                        }
                        
                        predict_y <- function(fit, X_L_new, X_U_new) {
                          l_hat <- colMeans(fit$lambda)
                          muL <- colMeans(fit$beta[, 1:K]) %*% t(B)
                          muU <- colMeans(fit$beta[, (K+1):(2*K)]) %*% t(B)
                          n_t <- nrow(X_L_new); m_t <- ncol(muL)
                          Y_pred_L <- matrix(muL, n_t, m_t, byrow=T)
                          Y_pred_U <- matrix(muU, n_t, m_t, byrow=T)
                          q <- ncol(X_L_new)
                          for(j in 1:q) {
                            bL <- colMeans(fit$beta[, ((j)*2*K+1):((j)*2*K+K)]) %*% t(B)
                            bU <- colMeans(fit$beta[, ((j)*2*K+K+1):((j)*2*K+2*K)]) %*% t(B)
                            P_j <- (1-l_hat[j])*X_L_new[,j] + l_hat[j]*X_U_new[,j]
                            Y_pred_L <- Y_pred_L + P_j %*% bL
                            Y_pred_U <- Y_pred_U + P_j %*% bU
                          }
                          list(L=Y_pred_L, U=Y_pred_U)
                        }
                        
                        # --- 1. BHIFR_Homo ---
                        fit_h <- BHIFR_Homo_Cpp(dat$Y_L, dat$Y_U, dat$X_L, dat$X_U, B, Omega, 15000, 7500)
                        pred_h <- predict_y(fit_h, test_dat$X_L, test_dat$X_U)
                        
                        s2L_h_vec <- rep(mean(fit_h$sigma[,1]), length(dat$t_grid))
                        s2U_h_vec <- rep(mean(fit_h$sigma[,2]), length(dat$t_grid))
                        s2L_h_ci <- quantile(fit_h$sigma[,1], probs=c(0.025, 0.975))
                        s2U_h_ci <- quantile(fit_h$sigma[,2], probs=c(0.025, 0.975))
                        
                        row_h <- data.frame(rep=p$rep, model="BHIFR_Homo", 
                                            IMSPE_YL = mean((test_dat$Y_L - pred_h$L)^2),
                                            IMSPE_YU = mean((test_dat$Y_U - pred_h$U)^2),
                                            IMSE_muL = calc_imse(dat$true_params$mu_L, get_est(fit_h, 1)$m),
                                            IMSE_muU = calc_imse(dat$true_params$mu_U, get_est(fit_h, K+1)$m),
                                            CP_muL = calc_cp(dat$true_params$mu_L, get_est(fit_h, 1)$l, get_est(fit_h, 1)$u),
                                            CP_muU = calc_cp(dat$true_params$mu_U, get_est(fit_h, K+1)$l, get_est(fit_h, K+1)$u),
                                            IMSE_sig2L = calc_imse(dat$true_params$sigma_L_t^2, s2L_h_vec),
                                            IMSE_sig2U = calc_imse(dat$true_params$sigma_U_t^2, s2U_h_vec),
                                            CP_sig2L = calc_cp(dat$true_params$sigma_L_t^2, s2L_h_ci[1], s2L_h_ci[2]),
                                            CP_sig2U = calc_cp(dat$true_params$sigma_U_t^2, s2U_h_ci[1], s2U_h_ci[2]),
                                            Rho_Est = mean(fit_h$rho))
                        
                        # --- 2. BHIFR_Hetero ---
                        fit_ht <- BHIFR_Hetero_Cpp(dat$Y_L, dat$Y_U, dat$X_L, dat$X_U, B, Omega, B_sig, Omega_sig, 15000, 7500)
                        pred_ht <- predict_y(fit_ht, test_dat$X_L, test_dat$X_U)
                        
                        alpL <- fit_ht$alpha_L; alpU <- fit_ht$alpha_U
                        
                        sigma_L_samps <- exp(alpL %*% t(B_sig))
                        s2L_samps <- sigma_L_samps^2
                        
                        sigma_U_samps <- exp(alpU %*% t(B_sig))
                        s2U_samps <- sigma_U_samps^2
                        
                        s2L_mean <- colMeans(s2L_samps)
                        s2U_mean <- colMeans(s2U_samps)
                        s2L_ci_ht <- apply(s2L_samps, 2, quantile, probs=c(0.025, 0.975))
                        s2U_ci_ht <- apply(s2U_samps, 2, quantile, probs=c(0.025, 0.975))
                        
                        row_ht <- data.frame(rep=p$rep, model="BHIFR_Hetero", 
                                             IMSPE_YL = mean((test_dat$Y_L - pred_ht$L)^2),
                                             IMSPE_YU = mean((test_dat$Y_U - pred_ht$U)^2),
                                             IMSE_muL = calc_imse(dat$true_params$mu_L, get_est(fit_ht, 1)$m),
                                             IMSE_muU = calc_imse(dat$true_params$mu_U, get_est(fit_ht, K+1)$m),
                                             CP_muL = calc_cp(dat$true_params$mu_L, get_est(fit_ht, 1)$l, get_est(fit_ht, 1)$u),
                                             CP_muU = calc_cp(dat$true_params$mu_U, get_est(fit_ht, K+1)$l, get_est(fit_ht, K+1)$u),
                                             IMSE_sig2L = calc_imse(dat$true_params$sigma_L_t^2, s2L_mean),
                                             IMSE_sig2U = calc_imse(dat$true_params$sigma_U_t^2, s2U_mean),
                                             CP_sig2L = calc_cp(dat$true_params$sigma_L_t^2, s2L_ci_ht[1,], s2L_ci_ht[2,]),
                                             CP_sig2U = calc_cp(dat$true_params$sigma_U_t^2, s2U_ci_ht[1,], s2U_ci_ht[2,]),
                                             Rho_Est = mean(fit_ht$rho))
                        
                        bind_rows(row_h, row_ht)
                      }
stopCluster(cl); close(pb)

# ==============================================================================
# 5. Table & Plots (Corrected & Complete for Study 2 - Fig 2 Only)
# ==============================================================================

# --- 0. Helper Functions & Config ---
library(ggplot2)
library(patchwork)
library(ggsci)
library(dplyr)
library(tidyr)
library(kableExtra)

mm_to_inch <- 0.0393701
WIDTH_DOUBLE <- 174 * mm_to_inch 

# Minimalist theme (Legend support)
theme_clean_journal <- function() {
  theme_classic(base_family = "sans", base_size = 10) + 
    theme(
      text = element_text(color = "black"),
      axis.text = element_text(color = "black", size = 9),
      axis.title = element_text(face = "bold", size = 10),
      plot.title = element_text(face = "bold", size = 11, hjust = 0.5),
      panel.grid.major.y = element_line(color = "grey90", size = 0.3, linetype = "dashed"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", size = 0.5, fill = NA),
      axis.line = element_blank(),
      strip.background = element_blank(),
      strip.text = element_text(face = "bold", size = 10, margin = margin(b = 5)),
      # Legend settings: Bottom Center
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "horizontal",
      legend.title = element_blank(),
      legend.background = element_blank(),
      legend.key = element_blank(),
      legend.text = element_text(size = 9),
      legend.margin = margin(t = -5, b = 0)
    )
}

fmt <- function(m, s) sprintf("%.4f (%.4f)", m, s)

# --- 1. Generate Table 7 (Simulation Metrics) ---
# Ensure results_s2 exists from simulation run
if(!exists("results_s2")) stop("results_s2 not found. Please run the simulation loop first.")

table_agg <- results_s2 %>%
  group_by(model) %>%
  summarise(
    IMSPE_YL = fmt(mean(IMSPE_YL), sd(IMSPE_YL)),
    IMSPE_YU = fmt(mean(IMSPE_YU), sd(IMSPE_YU)),
    IMSE_muL = fmt(mean(IMSE_muL), sd(IMSE_muL)),
    IMSE_muU = fmt(mean(IMSE_muU), sd(IMSE_muU)),
    CP_muL = fmt(mean(CP_muL), sd(CP_muL)),
    CP_muU = fmt(mean(CP_muU), sd(CP_muU)),
    IMSE_sig2L = fmt(mean(IMSE_sig2L), sd(IMSE_sig2L)),
    IMSE_sig2U = fmt(mean(IMSE_sig2U), sd(IMSE_sig2U)),
    CP_sig2L = fmt(mean(CP_sig2L), sd(CP_sig2L)),
    CP_sig2U = fmt(mean(CP_sig2U), sd(CP_sig2U)),
    Rho_Mean = fmt(mean(Rho_Est), sd(Rho_Est)),
    .groups = 'drop'
  )

pull_val <- function(mod, col) {
  val <- table_agg %>% filter(model == mod) %>% pull(!!sym(col))
  if(length(val) == 0) return("-") else return(val)
}

t7_final <- data.frame(
  Metric = c("IMSPE (Y_L)", "IMSPE (Y_L)", "IMSPE (Y_U)", "IMSPE (Y_U)",
             "IMSE (mu_L)", "IMSE (mu_L)", "IMSE (mu_U)", "IMSE (mu_U)",
             "IMSE (sig2_L)", "IMSE (sig2_L)", "IMSE (sig2_U)", "IMSE (sig2_U)", "Rho"),
  Model = rep(c("BHIFR-Homo", "BHIFR-Hetero"), 7)[1:13],
  IMSE_IMSPE_Mean_SD = c(
    pull_val("BHIFR_Homo", "IMSPE_YL"), pull_val("BHIFR_Hetero", "IMSPE_YL"),
    pull_val("BHIFR_Homo", "IMSPE_YU"), pull_val("BHIFR_Hetero", "IMSPE_YU"),
    pull_val("BHIFR_Homo", "IMSE_muL"), pull_val("BHIFR_Hetero", "IMSE_muL"),
    pull_val("BHIFR_Homo", "IMSE_muU"), pull_val("BHIFR_Hetero", "IMSE_muU"),
    pull_val("BHIFR_Homo", "IMSE_sig2L"), pull_val("BHIFR_Hetero", "IMSE_sig2L"),
    pull_val("BHIFR_Homo", "IMSE_sig2U"), pull_val("BHIFR_Hetero", "IMSE_sig2U"),
    pull_val("BHIFR_Hetero", "Rho_Mean")
  ),
  Coverage_Mean_SD = c(
    "-", "-", "-", "-",
    pull_val("BHIFR_Homo", "CP_muL"), pull_val("BHIFR_Hetero", "CP_muL"),
    pull_val("BHIFR_Homo", "CP_muU"), pull_val("BHIFR_Hetero", "CP_muU"),
    pull_val("BHIFR_Homo", "CP_sig2L"), pull_val("BHIFR_Hetero", "CP_sig2L"),
    pull_val("BHIFR_Homo", "CP_sig2U"), pull_val("BHIFR_Hetero", "CP_sig2U"),
    "-"
  )
)
print(kbl(t7_final, format="simple", caption="Simulation Results (Study 2 Final)"))


# --- 2. Data Preparation for Figures ---
cat("Preparing plotting data (df_mu, df_sig)...\n")

set.seed(999)
d_demo <- generate_data_hetero(100, 1.6, 0.5)
B_demo <- get_bs_basis(d_demo$t_grid, df=15)
O_demo <- create_penalty_matrix(ncol(B_demo))
Bs_demo <- get_bs_basis(d_demo$t_grid, df=15) 
Os_demo <- create_penalty_matrix(ncol(Bs_demo))

# Fit models on this demo dataset
fit_h_demo <- BHIFR_Homo_Cpp(d_demo$Y_L, d_demo$Y_U, d_demo$X_L, d_demo$X_U, B_demo, O_demo, 5000, 2000)
fit_ht_demo <- BHIFR_Hetero_Cpp(d_demo$Y_L, d_demo$Y_U, d_demo$X_L, d_demo$X_U, B_demo, O_demo, Bs_demo, Os_demo, 5000, 2000)

# Extract Mean Estimates
get_mu_ci <- function(f, idx) {
  samps <- f$beta[, idx:(idx+14)] %*% t(B_demo)
  m <- colMeans(samps)
  ci <- apply(samps, 2, quantile, probs=c(0.025, 0.975))
  list(mean=as.vector(m), low=as.vector(ci[1,]), up=as.vector(ci[2,]))
}

res_mu_h <- get_mu_ci(fit_h_demo, 1) # Mu_L for Homo
res_mu_ht <- get_mu_ci(fit_ht_demo, 1) # Mu_L for Hetero

df_mu <- data.frame(
  t = d_demo$t_grid, 
  True = d_demo$true_params$mu_L,
  Homo_Mean = res_mu_h$mean, Homo_L = res_mu_h$low, Homo_U = res_mu_h$up,
  Hetero_Mean = res_mu_ht$mean, Hetero_L = res_mu_ht$low, Hetero_U = res_mu_ht$up
)

# Extract Variance Estimates (Sigma^2)
s2L_h_mean <- mean(fit_h_demo$sigma[,1])
s2L_h_ci <- quantile(fit_h_demo$sigma[,1], probs=c(0.025, 0.975))

sigma_L_ht_samps <- exp(fit_ht_demo$alpha_L %*% t(Bs_demo))
s2L_ht_samps <- sigma_L_ht_samps^2
s2L_ht_mean <- colMeans(s2L_ht_samps)
s2L_ht_ci <- apply(s2L_ht_samps, 2, quantile, probs=c(0.025, 0.975))

df_sig <- data.frame(
  t = d_demo$t_grid,
  True = d_demo$true_params$sigma_L_t^2,
  Homo_Mean = rep(s2L_h_mean, length(d_demo$t_grid)),
  Homo_L = rep(s2L_h_ci[1], length(d_demo$t_grid)),
  Homo_U = rep(s2L_h_ci[2], length(d_demo$t_grid)),
  Hetero_Mean = as.vector(s2L_ht_mean),
  Hetero_L = as.vector(s2L_ht_ci[1,]),
  Hetero_U = as.vector(s2L_ht_ci[2,])
)


# --- 3. Plotting Figure 2: Heteroscedasticity Check ---
cat("Plotting Figure 2...\n")

plot_comparison_curve_with_legend <- function(df, y_col_true, y_col_h, y_col_ht, 
                                              l_col_h, u_col_h, l_col_ht, u_col_ht, 
                                              ylab_expr) {
  ggplot(df, aes(x=t)) +

    geom_ribbon(aes(ymin=.data[[l_col_h]], ymax=.data[[u_col_h]], fill="Homo"), alpha=0.15) +
    geom_ribbon(aes(ymin=.data[[l_col_ht]], ymax=.data[[u_col_ht]], fill="Hetero"), alpha=0.15) +

    geom_line(aes(y=.data[[y_col_true]], linetype="True Value"), color="black", size=0.6) +
    geom_line(aes(y=.data[[y_col_h]], color="Homo", linetype="Estimate"), size=0.6) +
    geom_line(aes(y=.data[[y_col_ht]], color="Hetero", linetype="Estimate"), size=0.6) +
    

    scale_fill_manual(name="Model", values=c("Homo"="#4DBBD5FF", "Hetero"="#E64B35FF")) +
    scale_color_manual(name="Model", values=c("Homo"="#4DBBD5FF", "Hetero"="#E64B35FF")) +
    scale_linetype_manual(name="Type", values=c("True Value"="dashed", "Estimate"="solid")) +
    
    labs(y = ylab_expr, x = expression(italic(t))) +
    theme_clean_journal()
}

# (a) Mean Curve
p2a <- plot_comparison_curve_with_legend(
  df_mu, "True", "Homo_Mean", "Hetero_Mean", 
  "Homo_L", "Homo_U", "Hetero_L", "Hetero_U",
  expression(mu[L]*(t))
) + ggtitle("(a)")

# (b) Variance Curve
p2b <- plot_comparison_curve_with_legend(
  df_sig, "True", "Homo_Mean", "Hetero_Mean", 
  "Homo_L", "Homo_U", "Hetero_L", "Hetero_U",
  expression(sigma[L]^2*(t))
) + ggtitle("(b)")

p2_combined <- (p2a | p2b) + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom", legend.margin = margin(t=0))

ggsave("Fig2.eps", plot=p2_combined, width=WIDTH_DOUBLE, height=4.5, device=cairo_ps)

cat("Done! Study 2 figures saved (Fig2.eps)\n")