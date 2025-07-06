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


import numpy as np

# If your data is a torch tensor, convert it to numpy
background_np = background.cpu().numpy() if hasattr(background, 'cpu') else np.array(background)
X_train_np = X_train.cpu().numpy() if hasattr(X_train, 'cpu') else np.array(X_train)

# Now use the numpy arrays in KernelExplainer
explainer = shap.KernelExplainer(reconstruction_error, background_np, feature_names=feature_names)

# When calling shap_values, also use numpy arrays
shap_values = explainer.shap_values(X_train_np[:200])


import numpy as np
import pandas as pd

# Assume shap_values is a NumPy array of shape (num_samples, num_features)
# and feature_names is a list of feature names

# Calculate mean absolute SHAP value for each feature
mean_abs_shap = np.abs(shap_values).mean(axis=0)

# Create a DataFrame for sorting
shap_df = pd.DataFrame({
    'feature': feature_names,
    'mean_abs_shap': mean_abs_shap
})

# Sort features from most to least important
shap_df_sorted = shap_df.sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

# Display the full sorted list
print(shap_df_sorted)






import shap
import torch
import numpy as np
import pandas as pd

# Assume you have:
# - model: your trained autoencoder (on GPU)
# - X_train: your training data as a NumPy array or torch.Tensor (shape: [10000, 900])
# - feature_names: list of 900 feature names

# Ensure data is a torch tensor and on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)

# Select background dataset (100 random samples)
background_size = 100
background_idx = np.random.choice(X_train.shape[0], background_size, replace=False)
background = X_train_tensor[background_idx]

# Define a function that outputs reconstruction error for each feature
def reconstruction_error(inputs):
    with torch.no_grad():
        outputs = model(inputs)
        # Per-feature squared error
        errors = (inputs - outputs) ** 2
    return errors.cpu().numpy()

# Use DeepExplainer (GPU-accelerated if model is on GPU)
explainer = shap.DeepExplainer(model, background)

# Choose a batch of samples to explain (e.g., first 100)
to_explain = X_train_tensor[:100]

# Compute SHAP values for the reconstruction error
shap_values = explainer.shap_values(to_explain)

# Aggregate SHAP values across samples (mean absolute value for each feature)
shap_importance = np.abs(shap_values).mean(axis=(0, 1))

# Create a DataFrame for sorting and display
shap_df = pd.DataFrame({
    'feature': feature_names,
    'shap_value': shap_importance
}).sort_values('shap_value', ascending=False)

# Display top features
print(shap_df.head(20))  # Top 20 features by SHAP value



import numpy as np
import pandas as pd
from sklearn.utils import shuffle

# Set background sample size
background_size = 1000
X_bg = shuffle(X_train, random_state=42)[:background_size]

# Compute baseline reconstruction error
recon = autoencoder.predict(X_bg)
baseline_error = np.mean(np.square(X_bg - recon), axis=1)
baseline_score = np.mean(baseline_error)

importances = []
for i, col in enumerate(feature_names):
    X_perm = X_bg.copy()
    # Permute the column
    if isinstance(X_perm, pd.DataFrame):
        X_perm[col] = np.random.permutation(X_perm[col].values)
    else:
        X_perm[:, i] = np.random.permutation(X_perm[:, i])
    recon_perm = autoencoder.predict(X_perm)
    perm_error = np.mean(np.square(X_bg - recon_perm), axis=1)
    perm_score = np.mean(perm_error)
    importances.append(perm_score - baseline_score)

# Create a DataFrame of importances
importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
importance_df = importance_df.sort_values('importance', ascending=False)

# Show top 10 features
print(importance_df.head(10))

