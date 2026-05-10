import numpy as np
from scipy import stats
import collections

# This is a custom implementation of Logistic Regression with Statistical Inference.
# Authors: Joan Acero, Mateja Zatezalo, Pawarit Jamjod

class CustomLogisticRegression:
    def __init__(self, max_iter=25, tol=1e-6, verbose=True):
        """
        Initialize Logistic Regression model
        
        Parameters:
        -----------
        max_iter : int
            Maximum number of IRLS iterations
        tol : float
            Convergence tolerance for the optimization process
        verbose : bool
            If True, print fitting progress
        """
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        
        self.weights = None
        self.bias = None
        self.coef_ = None  # Combined coefficients [bias, weights]
        self.losses = []
        self.converged = False
        
        # Statistical inference attributes
        self.vcov_matrix = None      # Variance-covariance matrix
        self.std_errors = None       # Standard errors
        self.z_scores = None         # Z-statistics 
        self.p_values = None         # P-values
        self.conf_intervals = None   # Confidence intervals
        self.odds_ratios = None      # Odds ratios
        self.odds_ratios_ci = None   # Odds ratios CI
        
        # Model fit statistics
        self.n_samples = None
        self.n_features = None
        self.log_likelihood_ = None
        self.null_log_likelihood = None
        self.deviance = None
        self.null_deviance = None
        self.aic = None
        self.bic = None
        self.pseudo_r2_mcfadden = None
        
        # Store final weights for inference
        self.final_W = None
        
        # Store data for step()
        self.X_fit_ = None                  # X data used for fitting
        self.y_fit_ = None                  # y data used for fitting
        self.feature_names_in_ = None       # List of column names
        self.feature_groups_ = None         # Dict of grouped features
        self.X_with_intercept_ = None

    
    def sigmoid(self, z):
        """
        Sigmoid function with numerical stability
        """
        z = np.array(z)
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_log_likelihood(self, y, p):
        """
        Compute log-likelihood
        """
        epsilon = 1e-15
        p = np.clip(p, epsilon, 1 - epsilon)
        return np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    
    def fit(self, X, y, feature_names=None, feature_groups=None):
        """
        Train the logistic regression model using IRLS
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            Training data
        y : array-like, shape (n_samples,)
            Target values (0 or 1)
        feature_names : list of str, optional
            Names of features (columns in X) for interpretation
        feature_groups : dict, optional
            Maps a 'group name' to a list of 'feature_names' in that group.
            Example: {'Age': ['Age'], 'Region': ['Region_B', 'Region_C']}
            Used by step() to drop all columns for a categorical variable at once.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        # Store dimensions
        self.n_samples, self.n_features = X.shape
        
        # Store data for step()
        self.X_fit_ = X
        self.y_fit_ = y
        
        # Store feature names
        if feature_names:
            self.feature_names_in_ = list(feature_names)
        else:
            self.feature_names_in_ = [f'X{i}' for i in range(self.n_features)]
            
        if len(self.feature_names_in_) != self.n_features:
            raise ValueError("Length of feature_names must match number of columns in X")

        # Store feature groups
        if feature_groups:
            self.feature_groups_ = feature_groups
        else:
            # If no groups, assume each feature is its own group
            self.feature_groups_ = {name: [name] for name in self.feature_names_in_}
        
        # Add intercept term (bias) to X
        X_with_intercept = np.column_stack([np.ones(self.n_samples), X])
        self.X_with_intercept_ = X_with_intercept
        
        # Initialize coefficients (bias + weights)
        self.coef_ = np.zeros(self.n_features + 1)
        
        prev_log_likelihood = -np.inf
        
        if self.verbose:
            print("Starting IRLS optimization...")
        
        for iteration in range(self.max_iter):
            # 1. Compute linear predictor: η = Xβ
            eta = np.dot(X_with_intercept, self.coef_)
            
            # 2. Compute predicted probabilities: p = σ(η)
            p = self.sigmoid(eta)
            
            # 3. Compute log-likelihood
            log_likelihood = self.compute_log_likelihood(y, p)
            self.losses.append(-log_likelihood)  # Store negative for loss
            
            # 4. Check convergence
            if abs(log_likelihood - prev_log_likelihood) < self.tol:
                if self.verbose:
                    print(f"Converged at iteration {iteration}")
                self.converged = True
                break
            
            prev_log_likelihood = log_likelihood
            
            # 5. Compute weights matrix W (diagonal matrix of p(1-p))
            W = p * (1 - p)
            W = np.maximum(W, 1e-10) # Add small value to prevent singularity
            self.final_W = W
            
            # 6. Compute working response (adjusted dependent variable)
            z = eta + (y - p) / W
            
            # 7. Solve weighted least squares: (X^T W X)β = X^T W z
            W_matrix = np.diag(W)
            XtWX = X_with_intercept.T @ W_matrix @ X_with_intercept
            XtWz = X_with_intercept.T @ (W * z)
            
            # Solve the system (with regularization for numerical stability)
            try:
                ridge = 1e-8 * np.eye(XtWX.shape[0])
                self.coef_ = np.linalg.solve(XtWX + ridge, XtWz)
            except np.linalg.LinAlgError:
                if self.verbose:
                    print(f"Singular matrix at iteration {iteration}, stopping.")
                break
            
            if self.verbose:
                print(f"Iteration {iteration}: Log-likelihood = {log_likelihood:.6f}")
        
        # Extract bias and weights
        self.bias = self.coef_[0]
        self.weights = self.coef_[1:]
        self.log_likelihood_ = prev_log_likelihood
        
        if self.verbose:
            print(f"\nFinal log-likelihood: {prev_log_likelihood:.6f}")
            print(f"Converged: {self.converged}")
        
        # Compute statistical inference
        self._compute_inference(X_with_intercept, y)
        
        return self
    
    def _compute_inference(self, X_with_intercept, y):
        W_matrix = np.diag(self.final_W)
        hessian = X_with_intercept.T @ W_matrix @ X_with_intercept
        
        try:
            self.vcov_matrix = np.linalg.inv(hessian)
        except np.linalg.LinAlgError:
            if self.verbose:
                print("Warning: Singular Hessian. Using pseudo-inverse.")
            self.vcov_matrix = np.linalg.pinv(hessian)
        
        self.std_errors = np.sqrt(np.diag(self.vcov_matrix))
        self.z_scores = self.coef_ / self.std_errors
        self.p_values = 2 * (1 - stats.norm.cdf(np.abs(self.z_scores)))
        
        z_critical = stats.norm.ppf(0.975)  # 1.96
        margin = z_critical * self.std_errors
        self.conf_intervals = np.column_stack([
            self.coef_ - margin,
            self.coef_ + margin
        ])
        
        self.odds_ratios = np.exp(self.coef_)
        self.odds_ratios_ci = np.exp(self.conf_intervals)
        
        self._compute_fit_statistics(y)
    
    def _compute_fit_statistics(self, y):
        n = self.n_samples
        k = self.n_features + 1
        
        self.deviance = -2 * self.log_likelihood_
        
        p_null = np.mean(y)
        epsilon = 1e-15
        p_null = np.clip(p_null, epsilon, 1 - epsilon)
        self.null_log_likelihood = np.sum(
            y * np.log(p_null) + (1 - y) * np.log(1 - p_null)
        )
        self.null_deviance = -2 * self.null_log_likelihood
        
        # AIC = 2k - 2*log-likelihood
        self.aic = 2 * k - 2 * self.log_likelihood_
        
        # BIC = k*log(n) - 2*log-likelihood
        self.bic = k * np.log(n) - 2 * self.log_likelihood_
        
        self.pseudo_r2_mcfadden = 1 - (self.log_likelihood_ / self.null_log_likelihood)
    
    def summary(self, alpha=0.05, odds_ratios=False):
        """
        Display comprehensive statistical summary
        """
        # Create feature labels
        if self.feature_names_in_:
            labels = ['Intercept'] + self.feature_names_in_
        else:
            labels = ['Intercept'] + [f'X{i}' for i in range(1, self.n_features + 1)]
        
        print("\n" + "="*85)
        print(" "*25 + "LOGISTIC REGRESSION SUMMARY")
        print("="*85)
        
        print(f"\nModel Information:")
        print("-"*85)
        print(f"  Number of observations: {self.n_samples:,}")
        print(f"  Number of predictors:   {self.n_features}")
        if self.feature_groups_:
             print(f"  Predictor Groups:       {', '.join(self.feature_groups_.keys())}")
        print(f"  Converged:              {self.converged}")
        print(f"  Iterations:             {len(self.losses)}")
        
        print("\nGoodness-of-Fit Statistics:")
        print("-"*85)
        print(f"  Log-Likelihood:         {self.log_likelihood_:>12.4f}")
        print(f"  Deviance:               {self.deviance:>12.4f}")
        print(f"  Null Deviance:          {self.null_deviance:>12.4f}")
        print(f"  AIC:                    {self.aic:>12.4f}")
        print(f"  BIC:                    {self.bic:>12.4f}")
        print(f"  McFadden's Pseudo R²:   {self.pseudo_r2_mcfadden:>12.4f}")
        
        lr_statistic = -2 * (self.null_log_likelihood - self.log_likelihood_)
        lr_df = self.n_features
        lr_pvalue = 1 - stats.chi2.cdf(lr_statistic, lr_df)
        print(f"\n  Likelihood Ratio Test:  χ²({lr_df}) = {lr_statistic:.4f}, p = {lr_pvalue:.4e}")
        
        ci_level = int((1 - alpha) * 100)
        print(f"\nCoefficients (with {ci_level}% Confidence Intervals):")
        print("="*85)
        print(f"{'Variable':<15} {'Coef':>10} {'Std.Err':>10} {'z':>8} "
              f"{'P>|z|':>10} {'[{:.3f}'.format(alpha/2):>10} {'{:.3f}]'.format(1-alpha/2):>10} {'':>3}")
        print("-"*85)
        
        for i, label in enumerate(labels):
            stars = self._get_significance_stars(self.p_values[i])
            print(f"{label:<15} {self.coef_[i]:>10.4f} {self.std_errors[i]:>10.4f} "
                  f"{self.z_scores[i]:>8.3f} {self.p_values[i]:>10.4f} "
                  f"{self.conf_intervals[i,0]:>10.4f} {self.conf_intervals[i,1]:>10.4f} {stars:>3}")
        
        print("-"*85)
        print("Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1")
        
        if odds_ratios:
            print(f"\nOdds Ratios (with {ci_level}% Confidence Intervals):")
            print("="*85)
            print(f"{'Variable':<15} {'OR':>10} {'[{:.3f}'.format(alpha/2):>10} {'{:.3f}]'.format(1-alpha/2):>10} {'Interpretation':<35}")
            print("-"*85)
            
            for i, label in enumerate(labels):
                interpretation = self._interpret_odds_ratio(self.odds_ratios[i], label)
                print(f"{label:<15} {self.odds_ratios[i]:>10.4f} "
                    f"{self.odds_ratios_ci[i,0]:>10.4f} {self.odds_ratios_ci[i,1]:>10.4f} "
                    f"{interpretation:<35}")
            
            print("="*85)
        
    def _get_significance_stars(self, p):
        if p < 0.001: return "***"
        elif p < 0.01: return "**"
        elif p < 0.05: return "*"
        elif p < 0.1: return "."
        return ""
    
    def _interpret_odds_ratio(self, or_value, var_name):
        if or_value > 1.5: return f"Strong positive effect"
        elif or_value > 1.1: return f"Moderate positive effect"
        elif or_value > 0.95: return f"Weak/no effect"
        elif or_value > 0.67: return f"Moderate negative effect"
        else: return f"Strong negative effect"
    
    def get_inference_dict(self):
        """
        Return inference results as dictionary
        """
        if self.feature_names_in_:
            labels = ['Intercept'] + self.feature_names_in_
        else:
            labels = ['Intercept'] + [f'X{i}' for i in range(1, self.n_features + 1)]
        
        return {
            'coefficients': dict(zip(labels, self.coef_)),
            'std_errors': dict(zip(labels, self.std_errors)),
            'z_scores': dict(zip(labels, self.z_scores)),
            'p_values': dict(zip(labels, self.p_values)),
            'conf_intervals_95': dict(zip(labels, self.conf_intervals.tolist())),
            'odds_ratios': dict(zip(labels, self.odds_ratios)),
            'odds_ratios_ci_95': dict(zip(labels, self.odds_ratios_ci.tolist())),
            'vcov_matrix': self.vcov_matrix,
            'model_fit': {
                'log_likelihood': self.log_likelihood_,
                'deviance': self.deviance,
                'null_deviance': self.null_deviance,
                'aic': self.aic,
                'bic': self.bic,
                'pseudo_r2_mcfadden': self.pseudo_r2_mcfadden
            }
        }
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_model)
    
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def get_coefficients(self):
        return {
            'bias': self.bias,
            'weights': self.weights,
            'all_coefficients': self.coef_
        }
    
    def step(self, direction="backward", criterion="AIC", single_step=False):
        """
        Performs variable selection based on AIC.
        """
        if direction != "backward":
            raise NotImplementedError("Only 'backward' selection is supported.")
        if criterion != "AIC":
            raise NotImplementedError(f"Only '{criterion}' criterion is supported.")
        if self.X_fit_ is None or self.y_fit_ is None:
            raise ValueError("Model must be fitted with X and y before calling step().")
        if not self.converged:
            print("Warning: Initial model did not converge. Stepwise selection may be unreliable.")

        def fit_model_subset(X_subset, y, feature_names_subset, feature_groups_subset):
            model = CustomLogisticRegression(max_iter=self.max_iter, tol=self.tol, verbose=False)
            model.fit(X_subset, y, 
                      feature_names=feature_names_subset, 
                      feature_groups=feature_groups_subset)
            return model

        current_model = self
        current_aic = self.aic
        current_feature_groups = self.feature_groups_.copy()

        print(f"Start:  AIC={current_aic:.4f}")
        print(f"        Variables: {'Intercept'}, {', '.join(current_feature_groups.keys())}")
        print("-" * 80)

        while True:
            best_candidate_aic = current_aic
            best_candidate_model = current_model
            group_to_drop = None

            # Iterate over the groups to drop
            for group_name in current_feature_groups.keys():
                
                # 1. Determine which groups remain
                candidate_groups = current_feature_groups.copy()
                del candidate_groups[group_name]
                
                # Check for intercept-only model constraint
                if not candidate_groups and len(current_feature_groups) == 1:
                     continue

                # 2. Identify ALL features belonging to the remaining groups
                features_to_keep_set = set()
                for cols in candidate_groups.values():
                    features_to_keep_set.update(cols)
                
                # 3. Reconstruct X and Names simultaneously to ensure alignment
                candidate_indices = []
                candidate_feature_names = []
                
                for i, col_name in enumerate(self.feature_names_in_):
                    if col_name in features_to_keep_set:
                        candidate_indices.append(i)
                        candidate_feature_names.append(col_name)

                # Create the subset matrix
                if not candidate_indices:
                    candidate_X = np.empty((self.n_samples, 0))
                else:
                    candidate_X = self.X_fit_[:, candidate_indices]

                # 4. Fit the candidate model
                candidate_model = fit_model_subset(candidate_X, self.y_fit_, 
                                                   candidate_feature_names, 
                                                   candidate_groups)

                if not candidate_model.converged:
                    continue
                
                candidate_aic = candidate_model.aic

                if candidate_aic < best_candidate_aic:
                    best_candidate_aic = candidate_aic
                    best_candidate_model = candidate_model
                    group_to_drop = group_name

            if group_to_drop:
                print("-" * 80)
                print(f"Step:   AIC={best_candidate_aic:.4f}  Dropping Group: {group_to_drop}")
                print("-" * 80)
                
                current_model = best_candidate_model
                current_aic = best_candidate_aic
                del current_feature_groups[group_to_drop]
                
                if single_step:
                    print("Single step requested. Returning new model.")
                    return current_model
                
            else:
                print("\n" + "=" * 80)
                print("Stepwise selection finished.")
                print(f"Final Model AIC: {current_aic:.4f}")
                print(f"Final Variables: {'Intercept'}, {', '.join(current_feature_groups.keys())}")
                print("=" * 80)
                return current_model
