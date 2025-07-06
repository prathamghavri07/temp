import numpy as np

# Function to compute reconstruction error for each sample
def reconstruction_error(X):
    reconstructed = autoencoder.predict(X)
    # Mean squared error per sample
    return np.mean((X - reconstructed) ** 2, axis=1)
import shap

# Randomly sample 100 rows for background
background_size = 100
background = X_train[np.random.choice(X_train.shape[0], background_size, replace=False)]
# Create the explainer
explainer = shap.KernelExplainer(reconstruction_error, background, feature_names=feature_names)

# Compute SHAP values for a subset (e.g., first 200 samples)
shap_values = explainer.shap_values(X_train[:200])
# Summary plot for feature importance
shap.summary_plot(shap_values, X_train[:200], feature_names=feature_names)
