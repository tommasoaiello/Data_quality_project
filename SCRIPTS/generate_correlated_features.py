import numpy as np
from scipy.stats import pearsonr

def inject_one_correlated_feature(X, existing_feature_index, correlation_strength):
    # Step 1: Calculate Pearson Correlation Coefficient
    existing_feature = X[:, existing_feature_index]
    target_variable = np.random.rand(X.shape[0])  # Replace with your actual target variable
    correlation_coefficient, _ = pearsonr(existing_feature, target_variable)

    # Step 2: Generate Random Data
    random_data = np.random.rand(X.shape[0])

    # Step 3: Adjust Random Data for Correlation
    new_feature = correlation_strength * random_data + np.sqrt(1 - correlation_strength**2) * existing_feature

    # Step 4: Combine Features
    X_with_new_feature = np.column_stack((X, new_feature.reshape(-1, 1)))

    return X_with_new_feature