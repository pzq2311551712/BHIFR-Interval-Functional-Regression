# ==============================================================================
# BHIFR Empirical Study: Ireland Weather Data (Complete & Final)
# Journal Target: Statistics and Computing (Ultimate Minimalist Style)
# Outputs: 
#   - Table 1 (CV IMSPE)
#   - Table 2 (Posterior Scalars)
#   - Fig3 - Fig5 
# ==============================================================================

rm(list=ls())
set.seed(123)
# ==============================================================================
# PART 0: Environment & Helper Functions
# ==============================================================================
pkgs <- c("Rcpp", "RcppArmadillo", "splines2", "MASS", "dplyr", 
          "knitr", "foreach", "doSNOW", "parallel", "tidyr", 
          "kableExtra", "ggplot2", "patchwork", "ggsci", "lubridate", "scales", "bayesplot")
new_pkgs <- pkgs[!(pkgs %in% installed.packages()[,"Package"])]
if(length(new_pkgs)) install.packages(new_pkgs)
invisible(lapply(pkgs, require, character.only = TRUE))

# --- Dimensions (Statistics and Computing Standard) ---
mm_to_inch <- 0.0393701
WIDTH_SINGLE <- 84 * mm_to_inch   # ~3.3 inch
WIDTH_DOUBLE <- 174 * mm_to_inch  # ~6.85 inch

# --- Theme: Ultimate Minimalist---
theme_elegant_journal <- function() {
  theme_classic(base_family = "sans", base_size = 10) + 
    theme(
      # Text: Black, No Bold
      text = element_text(color = "black"),
      axis.text = element_text(color = "black", size = 9),
      axis.title = element_text(size = 10),
      plot.title = element_text(size = 11, hjust = 0), # Left align
      
      # Grid: Only major Y, thin grey
      panel.grid.major.y = element_line(color = "grey92", size = 0.3, linetype = "solid"),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      
      # Border: Thin black
      panel.border = element_rect(color = "black", size = 0.5, fill = NA),
      axis.line = element_blank(),
      axis.ticks = element_line(size = 0.3),
      
      # Strips (Facets): Minimal
      strip.background = element_blank(),
      strip.text = element_text(size = 10, margin = margin(b = 5)),
      
      # Legend
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(size = 9),
      legend.margin = margin(t = -5),
      legend.key.size = unit(0.4, "cm")
    )
}

# ==============================================================================
# PART 1: C++ Core Definition & Compilation
# ==============================================================================
cpp_code_string <- '
// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
#include <cmath>

using namespace arma;
using namespace Rcpp;

const double nugget = 1e-6; 

// --- Helper Functions ---
bool robust_chol(mat& L, const mat& A_in) {
    if(!A_in.is_finite()) return false;
    if(chol(L, A_in, "lower")) return true;
    mat A = A_in; 
    double diag_mean = mean(A.diag());
    double jitter = 1e-8 * diag_mean; if(jitter < 1e-10) jitter = 1e-10;
    A.diag() += jitter;
    if(chol(L, A, "lower")) return true;
    A.diag() += 100.0 * jitter;
    if(chol(L, A, "lower")) return true;
    return false;
}

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

double calc_marginal_loglik_lambda_dense(const vec& lambda, const mat& X_L, const mat& X_U,
                                         const mat& BTWB, const mat& BTY_proj, const mat& Prior_Prec) {
    int n = X_L.n_rows; int q = X_L.n_cols;
    mat P = ones(n, q + 1);
    for(int j=0; j<q; ++j) P.col(j+1) = (1.0 - lambda(j)) * X_L.col(j) + lambda(j) * X_U.col(j);
    mat PTP = P.t() * P;
    mat Lambda_beta = kron(PTP, BTWB) + Prior_Prec; 
    vec RHS = vectorise(BTY_proj * P);
    mat L;
    if(!robust_chol(L, Lambda_beta)) return -1e20;
    double log_det = 2.0 * sum(log(L.diag()));
    vec y = solve(trimatl(L), RHS);
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

// [[Rcpp::export]]
List BHIFR_Hetero_Cpp(mat Y_L, mat Y_U, mat X_L, mat X_U, mat B, mat Omega, mat B_sig, mat Omega_sig, int n_iter, int n_burnin) {
    int n=Y_L.n_rows; int m=Y_L.n_cols; int q=X_L.n_cols; int K=B.n_cols; int K_sig=B_sig.n_cols;
    int dim_beta = 2 * K * (q + 1);
    
    vec lambda = randu(q) * 0.8 + 0.1; 
    vec beta = zeros(dim_beta);
    // Init via simple ridge
    mat P_init=ones(n,q+1); for(int j=0;j<q;++j) P_init.col(j+1)=0.5*X_L.col(j)+0.5*X_U.col(j);
    mat D_init = kron(P_init, B); 
    int dim_single = K * (q + 1); 
    mat DTD = D_init.t() * D_init + 0.1 * eye(dim_single, dim_single); 
    vec bL_init = solve(DTD, D_init.t() * vectorise(Y_L.t()));
    vec bU_init = solve(DTD, D_init.t() * vectorise(Y_U.t()));
    for(int j=0; j<=q; ++j) {
        beta.subvec(j*2*K, j*2*K+K-1)        = bL_init.subvec(j*K, j*K+K-1);
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
    
    vec step_l_vec = ones(q) * 0.1; double step_MALA = 0.002; double step_rho = 0.02;
    
    mat res_b(n_iter-n_burnin, dim_beta); mat res_l(n_iter-n_burnin, q); 
    mat res_gL(n_iter-n_burnin, K_sig); mat res_gU(n_iter-n_burnin, K_sig); vec res_r(n_iter-n_burnin);
    
    mat BTB=B.t()*B; mat BTY_L=B.t()*Y_L.t(); mat BTY_U=B.t()*Y_U.t(); mat BT=B.t();
    
    for(int it=0; it<n_iter; ++it) {
        double gamma_t = std::min(1.0, 10.0 / (double(it) + 100.0));
        vec sL = exp(B_sig*gamma_L); vec sU = exp(B_sig*gamma_U);
        vec sL_nug = sqrt(square(sL) + nugget); vec sU_nug = sqrt(square(sU) + nugget);
        
        double sc=1.0/(1.0-rho*rho);
        mat BTWB_acc = zeros(2*K,2*K); mat BTY_p = zeros(2*K,n);
        
        for(int t=0; t<m; ++t) {
             double vL = sL_nug(t); double vU = sU_nug(t);
             double w11=sc/(vL*vL); double w22=sc/(vU*vU); double w12=-sc*rho/(vL*vU);
             vec b=BT.col(t); mat bbt=b*b.t();
             BTWB_acc.submat(0,0,K-1,K-1)     += w11*bbt; BTWB_acc.submat(K,K,2*K-1,2*K-1) += w22*bbt;
             BTWB_acc.submat(0,K,K-1,2*K-1)   += w12*bbt; BTWB_acc.submat(K,0,2*K-1,K-1)   += w12*bbt;
             for(int k=0; k<K; ++k) { 
                 BTY_p.row(k)   += b(k)*(w11*Y_L.col(t).t()+w12*Y_U.col(t).t()); 
                 BTY_p.row(K+k) += b(k)*(w12*Y_L.col(t).t()+w22*Y_U.col(t).t()); 
             }
        }
        
        mat Prior_Prec = zeros(dim_beta, dim_beta);
        for(int g=0; g<2*(q+1); ++g) Prior_Prec.submat(g*K, g*K, g*K+K-1, g*K+K-1) = tau_beta(g) * Omega;
        
        double current_ll = calc_marginal_loglik_lambda_dense(lambda, X_L, X_U, BTWB_acc, BTY_p, Prior_Prec);
        
        for(int j=0; j<q; ++j) {
            double eta_curr = logit(lambda(j)); double eta_prop = eta_curr + step_l_vec(j) * randn();
            double lambda_prop_val = sigmoid(eta_prop);
            vec lambda_prop = lambda; lambda_prop(j) = lambda_prop_val;
            double prop_ll = calc_marginal_loglik_lambda_dense(lambda_prop, X_L, X_U, BTWB_acc, BTY_p, Prior_Prec);
            double jacobian_diff = std::log(lambda_prop_val * (1.0 - lambda_prop_val)) - std::log(lambda(j) * (1.0 - lambda(j)));
            bool accept = std::log(randu()) < (prop_ll - current_ll + jacobian_diff);
            if(accept) { lambda(j) = lambda_prop_val; current_ll = prop_ll; }
            if(it < n_burnin) {
                 double alpha = accept ? 1.0 : 0.0;
                 double log_step = std::log(step_l_vec(j)) + gamma_t * (alpha - 0.44);
                 step_l_vec(j) = std::exp(log_step);
            }
        }
        
        mat P(n,q+1); P.col(0).fill(1.0); for(int j=0;j<q;++j) P.col(j+1)=(1-lambda(j))*X_L.col(j)+lambda(j)*X_U.col(j);
        mat PTP = P.t()*P;
        mat Prec_Beta = kron(PTP, BTWB_acc) + Prior_Prec;
        mat L_post;
        if(robust_chol(L_post, Prec_Beta)) {
             vec z = randn(dim_beta);
             vec RHS = vectorise(BTY_p * P);
             vec mean_part = solve(trimatu(L_post.t()), solve(trimatl(L_post), RHS));
             beta = mean_part + solve(trimatu(L_post.t()), z);
        }

        for(int g=0; g < 2*(q+1); ++g) {
             vec b_sub = beta.subvec(g*K, g*K+K-1); double quad = as_scalar(b_sub.t() * Omega * b_sub);
             tau_beta(g) = R::rgamma(0.5*K+0.5, 1.0/(0.5*quad + 1.0/nu_beta(g)));
             nu_beta(g) = 1.0 / R::rgamma(1.0, 1.0 / (1.0/25.0 + tau_beta(g))); 
        }
        
        YL_h.zeros(); YU_h.zeros();
        aL=beta.head(K); aU=beta.subvec(K, 2*K-1); 
        YL_h.each_row() += (B*aL).t(); YU_h.each_row() += (B*aU).t();
        for(int j=0;j<q;++j) { 
            vec bL=beta.subvec((j+1)*2*K, (j+1)*2*K+K-1); vec bU=beta.subvec((j+1)*2*K+K, (j+1)*2*K+2*K-1);
            for(int i=0;i<n;++i) { YL_h.row(i)+=P(i,j+1)*(B*bL).t(); YU_h.row(i)+=P(i,j+1)*(B*bU).t(); } 
        }
        mat EL=Y_L-YL_h; mat EU=Y_U-YU_h;
        
        bool accepted_gL = false; bool accepted_gU = false;
        {
            double quad = as_scalar(gamma_L.t() * Omega_sig * gamma_L);
            tau_gamma_L = R::rgamma(0.5*K_sig+0.5, 1.0/(0.5*quad + 1.0/nu_gamma_L));
            nu_gamma_L = 1.0 / R::rgamma(1.0, 1.0 / (1.0/25.0 + tau_gamma_L)); 
            
            vec g_L=calc_gradient_gamma(EL,EU,sL_nug,sU_nug,rho,B_sig,Omega_sig,gamma_L,tau_gamma_L,true);
            vec mu_L=gamma_L+0.5*step_MALA*step_MALA*g_L; vec gL_p=mu_L+step_MALA*randn(K_sig);
            vec sL_p = exp(B_sig*gL_p); vec sL_p_nug = sqrt(square(sL_p) + nugget); 
            vec grad_p=calc_gradient_gamma(EL,EU,sL_p_nug,sU_nug,rho,B_sig,Omega_sig,gL_p,tau_gamma_L,true);
            vec mu_L_rev=gL_p+0.5*step_MALA*step_MALA*grad_p;
            
            double lp_curr=calc_joint_loglik(EL,EU,sL_nug,sU_nug,rho)-0.5*tau_gamma_L*as_scalar(gamma_L.t()*Omega_sig*gamma_L);
            double lp_prop=calc_joint_loglik(EL,EU,sL_p_nug,sU_nug,rho)-0.5*tau_gamma_L*as_scalar(gL_p.t()*Omega_sig*gL_p);
            double lq_fwd= -0.5*dot(gL_p-mu_L,gL_p-mu_L)/(step_MALA*step_MALA);
            double lq_rev= -0.5*dot(gamma_L-mu_L_rev,gamma_L-mu_L_rev)/(step_MALA*step_MALA);
            if(std::log(randu()) < lp_prop - lp_curr + lq_rev - lq_fwd) { gamma_L=gL_p; sL=sL_p; sL_nug=sL_p_nug; accepted_gL = true; }
        }
        {
            double quad = as_scalar(gamma_U.t() * Omega_sig * gamma_U);
            tau_gamma_U = R::rgamma(0.5*K_sig+0.5, 1.0/(0.5*quad + 1.0/nu_gamma_U));
            nu_gamma_U = 1.0 / R::rgamma(1.0, 1.0 / (1.0/25.0 + tau_gamma_U)); 
            
            vec g_U=calc_gradient_gamma(EL,EU,sL_nug,sU_nug,rho,B_sig,Omega_sig,gamma_U,tau_gamma_U,false);
            vec mu_U=gamma_U+0.5*step_MALA*step_MALA*g_U; vec gU_p=mu_U+step_MALA*randn(K_sig);
            vec sU_p = exp(B_sig*gU_p); vec sU_p_nug = sqrt(square(sU_p) + nugget);
            vec grad_p=calc_gradient_gamma(EL,EU,sL_nug,sU_p_nug,rho,B_sig,Omega_sig,gU_p,tau_gamma_U,false);
            vec mu_U_rev=gU_p+0.5*step_MALA*step_MALA*grad_p;
            
            double lp_curr=calc_joint_loglik(EL,EU,sL_nug,sU_nug,rho)-0.5*tau_gamma_U*as_scalar(gamma_U.t()*Omega_sig*gamma_U);
            double lp_prop=calc_joint_loglik(EL,EU,sL_nug,sU_p_nug,rho)-0.5*tau_gamma_U*as_scalar(gU_p.t()*Omega_sig*gU_p);
            double lq_fwd= -0.5*dot(gU_p-mu_U,gU_p-mu_U)/(step_MALA*step_MALA);
            double lq_rev= -0.5*dot(gamma_U-mu_U_rev,gamma_U-mu_U_rev)/(step_MALA*step_MALA);
            if(std::log(randu()) < lp_prop - lp_curr + lq_rev - lq_fwd) { gamma_U=gU_p; sU=sU_p; sU_nug=sU_p_nug; accepted_gU = true; }
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

# [CRITICAL] Compile C++ Code immediately
cat("Compiling C++ code (BHIFR_Hetero_Cpp)...\n")
sourceCpp(code = cpp_code_string)

# ==============================================================================
# PART 2: Data Processing (Fixing 00:00 Parsing & NA Interpolation)
# ==============================================================================
cat("PART 1: Processing Ireland Data...\n")
cat("Current Working Directory:", getwd(), "\n")
Sys.setlocale("LC_TIME", "C")

zip_path <- file.path("..", "data", "aggregated_hourly_data.zip")
csv_name <- "aggregated_hourly_data.csv"

if(!file.exists(zip_path)) {
  stop(paste0("ERROR: Data file NOT found at: ", normalizePath(zip_path, mustWork = FALSE)))
}

raw_data <- read.csv(unz(zip_path, csv_name), stringsAsFactors = FALSE)
cat("Raw rows:", nrow(raw_data), "\n")

library(lubridate)
if(!"zoo" %in% installed.packages()[,"Package"]) install.packages("zoo")
library(zoo)

data_cleaned <- raw_data %>%
  mutate(
    datetime = parse_date_time(date, orders = c("ymd HMS", "ymd HM", "dmy HMS", "dmy HM", "ymd", "dmy")),
    day = as.Date(datetime), 
    hour = hour(datetime)
  ) %>%
  filter(!is.na(datetime))

cat("Rows after date parsing:", nrow(data_cleaned), "\n")
data_interpolated <- data_cleaned %>%
  arrange(datetime) %>%
  dplyr::select(datetime, day, hour, temp, rhum, msl, wdsp) %>%
  mutate(
    across(where(is.numeric), ~zoo::na.approx(., na.rm = FALSE, rule = 2))
  ) %>%
  filter(!is.na(temp))

response_data <- data_interpolated %>%
  group_by(day, hour) %>%
  summarise(Temp_L = min(temp), Temp_U = max(temp), .groups = 'drop') %>%
  mutate(Temp_L=ifelse(is.infinite(Temp_L),NA,Temp_L), 
         Temp_U=ifelse(is.infinite(Temp_U),NA,Temp_U))

predictor_data <- data_interpolated %>%
  group_by(day) %>%
  summarise(
    rhum_L=min(rhum, na.rm=TRUE), rhum_U=max(rhum, na.rm=TRUE), 
    msl_L=min(msl, na.rm=TRUE), msl_U=max(msl, na.rm=TRUE), 
    wdsp_L=min(wdsp, na.rm=TRUE), wdsp_U=max(wdsp, na.rm=TRUE), 
    .groups='drop'
  ) %>%
  mutate(across(where(is.numeric), ~ifelse(is.infinite(.), NA, .)))

model_data <- inner_join(response_data, predictor_data, by = "day") %>%
  filter(year(day) >= 2018 & year(day) <= 2022) %>%
  na.omit() # Remove rows only if they remain empty after interpolation

cat("Rows after merging & processing:", nrow(model_data), "\n")

complete_days <- model_data %>% group_by(day) %>% filter(n() == 24) %>% ungroup()
unique_days <- complete_days %>% distinct(day) %>% pull(day)

if(length(unique_days) == 0) {
  cat("Debug: Check first few days counts:\n")
  print(head(model_data %>% count(day)))
  stop("ERROR: No days have complete 24-hour records.")
}

Y_L_final <- complete_days %>% filter(day %in% unique_days) %>% dplyr::select(day,hour,Temp_L) %>% pivot_wider(names_from=hour, values_from=Temp_L) %>% dplyr::select(-day) %>% as.matrix()
Y_U_final <- complete_days %>% filter(day %in% unique_days) %>% dplyr::select(day,hour,Temp_U) %>% pivot_wider(names_from=hour, values_from=Temp_U) %>% dplyr::select(-day) %>% as.matrix()

X_pred <- model_data %>% filter(day %in% unique_days) %>% dplyr::select(-hour,-Temp_L,-Temp_U) %>% distinct(day, .keep_all=TRUE) %>% arrange(day) %>% dplyr::select(-day)

# Standardizing
X_scaled <- scale(X_pred)
X_L_final_scaled <- X_scaled[, grep("_L$", colnames(X_scaled))]
X_U_final_scaled <- X_scaled[, grep("_U$", colnames(X_scaled))]

cat("Data Ready: N =", nrow(Y_L_final), "days.\n")

# ==============================================================================
# PART 3: 5-Fold Cross-Validation
# ==============================================================================
cat("PART 2: Running 5-Fold CV...\n")
t_grid <- seq(0, 1, length.out=24)
B <- bSpline(t_grid, df=15, degree=3, intercept=TRUE)
create_penalty_matrix <- function(K) {
  D <- diff(diag(K), differences = 2)
  return(t(D) %*% D + diag(1e-6, K))
}
Omega <- create_penalty_matrix(ncol(B))
B_sig <- bSpline(t_grid, df=15, degree=3, intercept=TRUE)
Omega_sig <- create_penalty_matrix(ncol(B_sig))

K_folds <- 5
set.seed(123)
n_obs <- nrow(Y_L_final)
random_indices <- sample(seq_len(n_obs))
folds <- split(random_indices, rep(1:K_folds, length.out = n_obs))

# Prediction Helper
predict_cpp <- function(res, X_L_new, X_U_new, B) {
  K <- ncol(B); q <- ncol(X_L_new)
  l_hat <- colMeans(res$lambda)
  beta_mean <- colMeans(res$beta)
  muL <- beta_mean[1:K] %*% t(B)
  muU <- beta_mean[(K+1):(2*K)] %*% t(B)
  Y_pred_L <- matrix(muL, nrow(X_L_new), ncol(muL), byrow=TRUE)
  Y_pred_U <- matrix(muU, nrow(X_L_new), ncol(muU), byrow=TRUE)
  for(j in 1:q) {
    bL <- beta_mean[( (j)*2*K+1 ):( (j)*2*K+K )] %*% t(B)
    bU <- beta_mean[( (j)*2*K+K+1 ):( (j)*2*K+2*K )] %*% t(B)
    P_j <- (1-l_hat[j])*X_L_new[,j] + l_hat[j]*X_U_new[,j]
    Y_pred_L <- Y_pred_L + P_j %*% bL
    Y_pred_U <- Y_pred_U + P_j %*% bU
  }
  return(list(L=Y_pred_L, U=Y_pred_U))
}

# --- Benchmark Functions ---
fit_manual_flm <- function(Y_func, X_scalar, B_matrix) {
  Y_coeffs <- tryCatch({ t(MASS::ginv(t(B_matrix) %*% B_matrix) %*% t(B_matrix) %*% t(Y_func)) }, 
                       error = function(e){ t(MASS::ginv(B_matrix) %*% t(Y_func)) })
  fit <- lm(Y_coeffs ~ ., data = as.data.frame(X_scalar))
  coefs <- coef(fit); coefs[is.na(coefs)] <- 0
  
  intercept <- as.vector(B_matrix %*% coefs[1,])
  betas <- list(); names_x <- colnames(X_scalar)
  for(j in 1:length(names_x)) {
    if(names_x[j] %in% rownames(coefs)) betas[[names_x[j]]] <- as.vector(B_matrix %*% coefs[names_x[j],])
    else betas[[names_x[j]]] <- rep(0, nrow(B_matrix))
  }
  list(mu=intercept, beta=betas)
}

predict_manual_flm <- function(fit, X_new) {
  n <- nrow(X_new); m <- length(fit$mu)
  pred <- matrix(fit$mu, n, m, byrow=T)
  for(nm in names(fit$beta)) {
    if(nm %in% colnames(X_new)) pred <- pred + X_new[,nm] %*% t(fit$beta[[nm]])
  }
  pred
}

run_fcm <- function(Y_L, Y_U, X_L, X_U, X_L_test, X_U_test, B) {
  Y_c <- (Y_L+Y_U)/2; X_c <- (X_L+X_U)/2; X_c_test <- (X_L_test+X_U_test)/2
  colnames(X_c) <- paste0("V", 1:ncol(X_c)); colnames(X_c_test) <- paste0("V", 1:ncol(X_c))
  fit <- fit_manual_flm(Y_c, X_c, B)
  pred <- predict_manual_flm(fit, X_c_test)
  list(L=pred, U=pred)
}

run_fcrm <- function(Y_L, Y_U, X_L, X_U, X_L_test, X_U_test, B) {
  Y_c <- (Y_L+Y_U)/2; Y_r <- Y_U-Y_L
  X_c <- (X_L+X_U)/2; X_r <- X_U-X_L
  X_c_test <- (X_L_test+X_U_test)/2; X_r_test <- X_U_test-X_L_test
  colnames(X_c) <- paste0("V", 1:ncol(X_c)); colnames(X_r) <- paste0("V", 1:ncol(X_c))
  colnames(X_c_test) <- paste0("V", 1:ncol(X_c)); colnames(X_r_test) <- paste0("V", 1:ncol(X_c))
  fit_c <- fit_manual_flm(Y_c, X_c, B); pred_c <- predict_manual_flm(fit_c, X_c_test)
  idx_keep <- which(apply(X_r, 2, var) > 1e-10)
  if(length(idx_keep)>0) {
    fit_r <- fit_manual_flm(Y_r, X_r[,idx_keep,drop=F], B)
    pred_r <- predict_manual_flm(fit_r, X_r_test[,idx_keep,drop=F])
  } else { pred_r <- matrix(0, nrow(X_c_test), ncol(Y_c)) }
  pred_r[pred_r<0] <- 0
  list(L=pred_c - 0.5*pred_r, U=pred_c + 0.5*pred_r)
}

run_minmax <- function(Y_L, Y_U, X_L, X_U, X_L_test, X_U_test, B) {
  colnames(X_L) <- paste0("V", 1:ncol(X_L)); colnames(X_L_test) <- paste0("V", 1:ncol(X_L))
  colnames(X_U) <- paste0("V", 1:ncol(X_U)); colnames(X_U_test) <- paste0("V", 1:ncol(X_U))
  fit_l <- fit_manual_flm(Y_L, X_L, B); pred_l <- predict_manual_flm(fit_l, X_L_test)
  fit_u <- fit_manual_flm(Y_U, X_U, B); pred_u <- predict_manual_flm(fit_u, X_U_test)
  list(L=pmin(pred_l, pred_u), U=pmax(pred_l, pred_u))
}

cl <- makeCluster(min(K_folds, detectCores()-1))
registerDoSNOW(cl)
clusterExport(cl, c("cpp_code_string"))
clusterEvalQ(cl, {
  library(Rcpp); library(RcppArmadillo)
  sourceCpp(code = cpp_code_string)
})

cv_res <- foreach(k=1:K_folds, .combine=rbind, .packages=pkgs, .noexport = c("BHIFR_Hetero_Cpp")) %dopar% {
  idx_test <- folds[[k]]
  YL_tr <- Y_L_final[-idx_test, , drop=FALSE]; YL_te <- Y_L_final[idx_test, , drop=FALSE]
  YU_tr <- Y_U_final[-idx_test, , drop=FALSE]; YU_te <- Y_U_final[idx_test, , drop=FALSE]
  XL_tr <- X_L_final_scaled[-idx_test, , drop=FALSE]; XL_te <- X_L_final_scaled[idx_test, , drop=FALSE]
  XU_tr <- X_U_final_scaled[-idx_test, , drop=FALSE]; XU_te <- X_U_final_scaled[idx_test, , drop=FALSE]
  
  fit_b <- BHIFR_Hetero_Cpp(YL_tr, YU_tr, XL_tr, XU_tr, B, Omega, B_sig, Omega_sig, 5000, 2000)
  pred_b <- predict_cpp(fit_b, XL_te, XU_te, B)
  
  res_cm <- run_fcm(YL_tr, YU_tr, XL_tr, XU_tr, XL_te, XU_te, B)
  res_crm <- run_fcrm(YL_tr, YU_tr, XL_tr, XU_tr, XL_te, XU_te, B)
  res_mm <- run_minmax(YL_tr, YU_tr, XL_tr, XU_tr, XL_te, XU_te, B)
  
  calc_imspe <- function(true, pred) mean((true-pred)^2)
  
  rbind(
    data.frame(Fold=k, Method="BHIFR",   Bound="Lower", IMSPE=calc_imspe(YL_te, pred_b$L)),
    data.frame(Fold=k, Method="BHIFR",   Bound="Upper", IMSPE=calc_imspe(YU_te, pred_b$U)),
    data.frame(Fold=k, Method="F-CM",     Bound="Lower", IMSPE=calc_imspe(YL_te, res_cm$L)),
    data.frame(Fold=k, Method="F-CM",     Bound="Upper", IMSPE=calc_imspe(YU_te, res_cm$U)),
    data.frame(Fold=k, Method="F-CRM",    Bound="Lower", IMSPE=calc_imspe(YL_te, res_crm$L)),
    data.frame(Fold=k, Method="F-CRM",    Bound="Upper", IMSPE=calc_imspe(YU_te, res_crm$U)),
    data.frame(Fold=k, Method="F-MinMax", Bound="Lower", IMSPE=calc_imspe(YL_te, res_mm$L)),
    data.frame(Fold=k, Method="F-MinMax", Bound="Upper", IMSPE=calc_imspe(YU_te, res_mm$U))
  )
}
stopCluster(cl)

cv_summary <- cv_res %>% group_by(Method, Bound) %>% 
  summarise(Mean=mean(IMSPE), SD=sd(IMSPE), .groups='drop') %>%
  pivot_wider(names_from=Bound, values_from=c(Mean, SD)) %>%
  dplyr::select(Method, Mean_Lower, SD_Lower, Mean_Upper, SD_Upper)

print(
  kbl(cv_summary, format="latex", booktabs=T, digits=3, caption="5-Fold CV IMSPE") %>% 
    kable_styling(latex_options="striped")
)

# ==============================================================================
# PART 4: Final Model Fit (2 Chains for Inference)
# ==============================================================================
cat("PART 3: Fitting Final Model (2 Chains, 50k Iterations)...\n")

# Run Chain 1
cat("Running Chain 1...\n")
set.seed(101)
fit_chain1 <- BHIFR_Hetero_Cpp(Y_L_final, Y_U_final, X_L_final_scaled, X_U_final_scaled, 
                               B, Omega, B_sig, Omega_sig, 50000, 25000)

# Run Chain 2
cat("Running Chain 2...\n")
set.seed(102)
fit_chain2 <- BHIFR_Hetero_Cpp(Y_L_final, Y_U_final, X_L_final_scaled, X_U_final_scaled, 
                               B, Omega, B_sig, Omega_sig, 50000, 25000)

# Thinning (Keep every 5th sample)
thin_idx <- seq(1, nrow(fit_chain1$beta), by=5)

# Combine for estimation (Using Chain 1 primarily for estimates)
final_fit_thinned <- list(
  beta = fit_chain1$beta[thin_idx,],
  lambda = fit_chain1$lambda[thin_idx,],
  alpha_L = fit_chain1$alpha_L[thin_idx,],
  alpha_U = fit_chain1$alpha_U[thin_idx,],
  rho = fit_chain1$rho[thin_idx]
)

# ==============================================================================
# PART 5: Tables & Visualization (Modified Colors, Style & Titles)
# ==============================================================================
cat("PART 5: Generating Outputs (Table 2 & Figures 3-5)...\n")

output_dir <- file.path(getwd(), "Empirical_Figures_Final")
if(!dir.exists(output_dir)) dir.create(output_dir)

# --- Helper Functions ---
get_curve_summary <- function(samps, idx_start, B_mat) {
  K <- ncol(B_mat)
  cols <- samps[, (idx_start*K + 1) : ((idx_start+1)*K)]
  curves <- cols %*% t(B_mat)
  data.frame(t=0:23, Mean=colMeans(curves), 
             Lower=apply(curves, 2, quantile, 0.025), 
             Upper=apply(curves, 2, quantile, 0.975))
}

extract_params <- function(fit_obj) {
  thin <- seq(1, nrow(fit_obj$lambda), by=5)
  
  # Scalars
  lam <- fit_obj$lambda[thin, ]
  rho <- fit_obj$rho[thin]
  
  # Randomly selected functional coefficients (beta and gamma)
  beta_sel <- fit_obj$beta[thin, c(1, 5)]
  gamma_sel <- fit_obj$alpha_L[thin, c(1, 5)]
  
  res <- cbind(lam, rho, beta_sel, gamma_sel)
  colnames(res) <- c("lambda[1]", "lambda[2]", "lambda[3]", "rho", 
                     "beta[1]", "beta[5]", 
                     "gamma[1]", "gamma[5]")
  return(res)
}

# --- Custom Color Palette ---
custom_cols <- c("Lower" = "#84D8D7", "Upper" = "#B8AEEA")

# Theme with Legend Support (Plain text, no bold)
theme_elegant_journal_legend <- function() {
  theme_classic(base_family = "sans", base_size = 10) +
    theme(
      text = element_text(color = "black"),
      axis.text = element_text(color = "black", size = 9),
      # Changed face="bold" to "plain"
      axis.title = element_text(face = "plain", size = 10),
      plot.title = element_text(face = "plain", size = 11, hjust = 0.5),
      panel.grid.major.y = element_line(color = "grey90", size = 0.3, linetype = "dashed"),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.spacing.x = unit(0.2, 'cm'),
      legend.key.width = unit(0.5, "cm"),
      legend.key.height = unit(0.3, "cm"),
      legend.text = element_text(size = 9)
    )
}

# ==============================================================================
# 1. TABLE 2: Posterior Summary of Scalar Parameters
# ==============================================================================
cat("Generating Table 2 (Posterior Summary)...\n")

if(!exists("final_fit_thinned")) stop("final_fit_thinned not found. Run PART 4 first.")

# Prepare Data Frame
df_lam <- as.data.frame(final_fit_thinned$lambda)
colnames(df_lam) <- c("Humidity", "Pressure", "Wind Speed")
df_rho <- data.frame(rho = final_fit_thinned$rho)

calc_stats <- function(x) {
  c(Mean = mean(x), 
    Lower95 = unname(quantile(x, 0.025)), 
    Upper95 = unname(quantile(x, 0.975)))
}

stats_df <- rbind(
  calc_stats(df_lam$Humidity),
  calc_stats(df_lam$Pressure),
  calc_stats(df_lam$`Wind Speed`),
  calc_stats(df_rho$rho)
) %>% as.data.frame()

stats_df$Parameter <- c("$\\lambda_1$ (Hum)", "$\\lambda_2$ (Pres)", "$\\lambda_3$ (Wind)", "$\\rho$")
stats_df <- stats_df[, c("Parameter", "Mean", "Lower95", "Upper95")]

print(
  kbl(stats_df, format="latex", booktabs=T, digits=3, escape=F, 
      caption="Posterior estimates of scalar parameters (Empirical Study)") %>% 
    kable_styling(latex_options = c("striped", "hold_position"))
)

# ==============================================================================
# 2. Figure 3: Empirical Baseline & Volatility (
# ==============================================================================
cat("Plotting Figure 3 (Baseline & Volatility)...\n")

# Baseline Trend (Intercept)
mu_L <- get_curve_summary(final_fit_thinned$beta, 0, B)
mu_U <- get_curve_summary(final_fit_thinned$beta, 1, B)
df_mu <- rbind(
  data.frame(mu_L, Type="Lower"), 
  data.frame(mu_U, Type="Upper")
)

# Volatility (Sigma)
calc_sigma <- function(alphas, B_s) { 
  log_s <- alphas %*% t(B_s)
  s <- exp(log_s)
  data.frame(t=0:23, Mean=colMeans(s), 
             Lower=apply(s, 2, quantile, 0.025), 
             Upper=apply(s, 2, quantile, 0.975)) 
}
sL <- calc_sigma(final_fit_thinned$alpha_L, B_sig)
sU <- calc_sigma(final_fit_thinned$alpha_U, B_sig)
df_sig <- rbind(data.frame(sL, Type="Lower"), data.frame(sU, Type="Upper"))

plot_emp_curve <- function(df, ylab_expr, title_text) {
  ggplot(df, aes(x=t, y=Mean, color=Type, fill=Type)) +
    geom_line(size=0.8) +
    geom_ribbon(aes(ymin=Lower, ymax=Upper), alpha=0.4, color=NA) + # Increased alpha slightly
    # Custom Colors
    scale_color_manual(values = custom_cols) +
    scale_fill_manual(values = custom_cols) +
    labs(x=expression(italic(t)), y=ylab_expr, title=title_text) + 
    theme_elegant_journal_legend()
}

p3a <- plot_emp_curve(df_mu, expression(mu(t)), "(a) Baseline Temperature Trend")
p3b <- plot_emp_curve(df_sig, expression(sigma(t)), "(b) Daily Volatility Pattern")

p3 <- (p3a | p3b) + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "Fig3.eps"), p3, width=WIDTH_DOUBLE, height=4, device=cairo_ps)

# ==============================================================================
# 3. Figure 4: Empirical Functional Coefficients 
# ==============================================================================
cat("Plotting Figure 4 (Coefficients)...\n")

vars <- c("Humidity", "Pressure", "Wind Speed")
titles <- c("(a) Humidity Effect", "(b) Pressure Effect", "(c) Wind Speed Effect")

df_beta_all <- data.frame()
for(j in 1:3) {
  bL <- get_curve_summary(final_fit_thinned$beta, (j)*2, B)
  bU <- get_curve_summary(final_fit_thinned$beta, (j)*2+1, B)
  tmp <- rbind(data.frame(bL, Bound="Lower"), data.frame(bU, Bound="Upper"))
  tmp$Variable <- vars[j]
  df_beta_all <- rbind(df_beta_all, tmp)
}
df_beta_all$Variable <- factor(df_beta_all$Variable, levels=vars)

p_list <- list()
for(i in 1:3) {
  v <- vars[i]
  sub_df <- df_beta_all %>% filter(Variable == v)
  p <- ggplot(sub_df, aes(x=t, y=Mean, color=Bound, fill=Bound)) +
    geom_hline(yintercept=0, linetype="dashed", color="grey50", size=0.3) +
    geom_line(size=0.7) +
    geom_ribbon(aes(ymin=Lower, ymax=Upper), alpha=0.4, color=NA) + # Increased alpha
    # Custom Colors
    scale_color_manual(values = custom_cols) +
    scale_fill_manual(values = custom_cols) +
    labs(x=expression(italic(t)), y=if(i==1) expression(beta(t)) else NULL, 
         title=titles[i]) + 
    theme_elegant_journal_legend()
  p_list[[i]] <- p
}

p4_combined <- (p_list[[1]] | p_list[[2]] | p_list[[3]]) +
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom")

ggsave(file.path(output_dir, "Fig4.eps"), p4_combined, width=WIDTH_DOUBLE, height=3.5, device=cairo_ps)

# ==============================================================================
# 4. Figure 5: MCMC Convergence Diagnostics 
# ==============================================================================
cat("Plotting Figure 5 (Convergence)...\n")

if(!exists("fit_chain1") || !exists("fit_chain2")) {
  stop("Chain data not found. Please ensure both chains have been fitted.")
}

c1_data <- extract_params(fit_chain1) 
c2_data <- extract_params(fit_chain2)

n_iter <- nrow(c1_data)
n_params <- ncol(c1_data)
chains_array <- array(NA, dim = c(n_iter, 2, n_params))
dimnames(chains_array) <- list(NULL, c("Chain1", "Chain2"), colnames(c1_data))
chains_array[, 1, ] <- c1_data
chains_array[, 2, ] <- c2_data
params_to_plot <- colnames(c1_data)
# Trace plot colors kept as requested (or using distinct colors for chains)
cols_trace <- c("#B8AEEA", "#84D8D7") 
plot_list <- list()

for(p in params_to_plot) {
  p_trace <- mcmc_trace(chains_array, pars = p) + 
    scale_color_manual(values = cols_trace) + 
    theme_classic() + 
    theme(legend.position = "none", axis.title.x = element_blank(), axis.text.x = element_blank(), 
          plot.title = element_text(size=10, face="bold", hjust=0.5)) + ggtitle(p)
  
  val_c1 <- chains_array[, 1, p]; val_c2 <- chains_array[, 2, p]
  df_dens <- data.frame(Value = c(val_c1, val_c2), Chain = rep(c("Chain1", "Chain2"), each = length(val_c1)))
  p_dens <- ggplot(df_dens, aes(x = Value, color = Chain, fill = Chain)) +
    geom_density(alpha = 0.4) + scale_color_manual(values = cols_trace) + scale_fill_manual(values = cols_trace) +
    theme_classic() + theme(legend.position = "none", axis.title.y = element_blank(), axis.text.y = element_blank(),
                            axis.ticks.y = element_blank(), axis.title.x = element_blank(), plot.title = element_text(size=10, face="bold", hjust=0.5)) + ggtitle(p)
  
  plot_list[[paste0(p, "_trace")]] <- p_trace
  plot_list[[paste0(p, "_dens")]] <- p_dens
}

layout_design <- "ABCD\nEFGH\nIJKL\nMNOP"
p5 <- wrap_plots(plot_list[[1]], plot_list[[2]], plot_list[[3]], plot_list[[4]],
                 plot_list[[5]], plot_list[[6]], plot_list[[7]], plot_list[[8]],
                 plot_list[[9]], plot_list[[10]], plot_list[[11]], plot_list[[12]],
                 plot_list[[13]], plot_list[[14]], plot_list[[15]], plot_list[[16]],
                 design = layout_design)

ggsave(file.path(output_dir, "Fig5.eps"), p5, width=12, height=10, device=cairo_ps)

cat("\nProcessing Complete. Table 2 and Figures 3-5 have been generated.\n")
