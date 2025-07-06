def get_shap_values(self, data, max_samples=100):  # Reduced default sample size
    # ... sampling code ...
    if self.explainer_type in ['deep', 'gradient']:
        sample_tensor = torch.FloatTensor(sample_data).to(self.device)
        shap_values = self.explainer.shap_values(sample_tensor)
        # Handle different return formats
    else:
        shap_values = self.explainer(sample_data)

def _initialize_explainer(self, background_data):
        """Initialize the appropriate SHAP explainer"""
        if self.explainer_type == 'deep':
            # Use DeepExplainer for neural networks
            if background_data.shape[0] > 50:
                bg_subset = background_data[:50]  # Smaller subset for DeepExplainer
            else:
                bg_subset = background_data
            
            bg_tensor = torch.FloatTensor(bg_subset).to(self.device)
            self.explainer = shap.DeepExplainer(self.model, bg_tensor)
            
        elif self.explainer_type == 'kernel':
            # Use KernelExplainer with smaller background
            if background_data.shape[0] > 25:
                bg_subset = background_data[:25]
            else:
                bg_subset = background_data
            
            self.explainer = shap.KernelExplainer(self.model_wrapper, bg_subset)
            
        elif self.explainer_type == 'permutation':
            # Use PermutationExplainer with sufficient max_evals
            if background_data.shape[0] > 100:
                bg_subset = background_data[:100]
            else:
                bg_subset = background_data
            
            num_features = background_data.shape[1]
            max_evals = max(2 * num_features + 1, 5000)  # Ensure sufficient evaluations
            
            self.explainer = shap.PermutationExplainer(
                self.model_wrapper, 
                bg_subset, 
                max_evals=max_evals
            )
        
        elif self.explainer_type == 'gradient':
            # Use GradientExplainer
            if background_data.shape[0] > 50:
                bg_subset = background_data[:50]
            else:
                bg_subset = background_data
            
            bg_tensor = torch.FloatTensor(bg_subset).to(self.device)
            self.explainer = shap.GradientExplainer(self.model, bg_tensor)


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

class AutoEncoder(nn.Module):
    """
    GPU-optimized Autoencoder for high-dimensional one-hot encoded data
    """
    def __init__(self, input_dim, encoding_dims=[512, 256, 128, 64]):
        super(AutoEncoder, self).__init__()
        
        # Encoder layers
        encoder_layers = []
        prev_dim = input_dim
        
        for dim in encoding_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Decoder layers (reverse of encoder)
        decoder_layers = []
        encoding_dims_reversed = list(reversed(encoding_dims[:-1])) + [input_dim]
        
        for dim in encoding_dims_reversed:
            decoder_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim) if dim != input_dim else nn.Identity(),
                nn.ReLU() if dim != input_dim else nn.Sigmoid(),
                nn.Dropout(0.2) if dim != input_dim else nn.Identity()
            ])
            prev_dim = dim
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Store latent dimension
        self.latent_dim = encoding_dims[-1]
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)

class AutoEncoderTrainer:
    """
    GPU-optimized trainer for the autoencoder
    """
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(self, train_loader, val_loader, epochs=100, lr=0.001):
        """
        Train the autoencoder
        """
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            
            for batch_data, _ in train_loader:
                batch_data = batch_data.to(self.device)
                
                optimizer.zero_grad()
                reconstructed = self.model(batch_data)
                loss = criterion(reconstructed, batch_data)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            
            with torch.no_grad():
                for batch_data, _ in val_loader:
                    batch_data = batch_data.to(self.device)
                    reconstructed = self.model(batch_data)
                    loss = criterion(reconstructed, batch_data)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_autoencoder.pth')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f'Epoch [{epoch}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Load best model
        self.model.load_state_dict(torch.load('best_autoencoder.pth'))
        return self.history

def calculate_reconstruction_error(model, data_loader, device='cuda'):
    """
    Calculate reconstruction error for each sample
    """
    model.eval()
    reconstruction_errors = []
    
    with torch.no_grad():
        for batch_data, _ in data_loader:
            batch_data = batch_data.to(device)
            reconstructed = model(batch_data)
            
            # Calculate MSE for each sample
            mse = torch.mean((batch_data - reconstructed) ** 2, dim=1)
            reconstruction_errors.extend(mse.cpu().numpy())
    
    return np.array(reconstruction_errors)

class SHAPAnalyzer:
    """
    SHAP analysis for autoencoder reconstruction error
    """
    def __init__(self, model, background_data, device='cuda'):
        self.model = model.to(device)
        self.device = device
        self.background_data = background_data
        
        # Create wrapper function for SHAP
        def model_wrapper(x):
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x).to(device)
            
            self.model.eval()
            with torch.no_grad():
                reconstructed = self.model(x)
                # Return reconstruction error
                mse = torch.mean((x - reconstructed) ** 2, dim=1)
                return mse.cpu().numpy()
        
        self.model_wrapper = model_wrapper
        
        # Initialize SHAP explainer
        if background_data.shape[0] > 100:
            # Use subset for efficiency
            bg_subset = background_data[:100]
        else:
            bg_subset = background_data
            
        self.explainer = shap.Explainer(self.model_wrapper, bg_subset)
    
    def get_shap_values(self, data, max_samples=500):
        """
        Calculate SHAP values for reconstruction error
        """
        if data.shape[0] > max_samples:
            # Sample for efficiency
            indices = np.random.choice(data.shape[0], max_samples, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
        
        print("Calculating SHAP values...")
        shap_values = self.explainer(sample_data)
        return shap_values
    
    def find_top_contributing_features(self, shap_values, feature_names, top_k=20):
        """
        Find features contributing most to reconstruction error
        """
        # Get mean absolute SHAP values for each feature
        mean_shap = np.mean(np.abs(shap_values.values), axis=0)
        
        # Get top k features
        top_indices = np.argsort(mean_shap)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'rank': i + 1,
                'feature_name': feature_names[idx] if feature_names is not None else f'feature_{idx}',
                'feature_index': idx,
                'mean_abs_shap': mean_shap[idx]
            })
        
        return results
    
    def plot_feature_importance(self, shap_values, feature_names=None, top_k=20):
        """
        Plot feature importance based on SHAP values
        """
        top_features = self.find_top_contributing_features(shap_values, feature_names, top_k)
        
        features = [f['feature_name'] for f in top_features]
        values = [f['mean_abs_shap'] for f in top_features]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Mean Absolute SHAP Value')
        plt.title('Top Features Contributing to Reconstruction Error')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return top_features

def prepare_data(df, test_size=0.2, batch_size=64):
    """
    Prepare data for training
    """
    # Convert to numpy if pandas DataFrame
    if isinstance(df, pd.DataFrame):
        data = df.values.astype(np.float32)
        feature_names = df.columns.tolist()
    else:
        data = df.astype(np.float32)
        feature_names = [f'flag{i+1}' for i in range(data.shape[1])]
    
    # Split data
    X_train, X_test = train_test_split(data, test_size=test_size, random_state=42)
    
    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(X_train))
    test_dataset = TensorDataset(torch.FloatTensor(X_test), torch.FloatTensor(X_test))
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader, X_train, X_test, feature_names

def main_pipeline(df):
    """
    Main pipeline for autoencoder training and SHAP analysis
    """
    # Check GPU availability
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_loader, test_loader, X_train, X_test, feature_names = prepare_data(df)
    
    # Initialize model
    input_dim = df.shape[1]
    model = AutoEncoder(input_dim)
    
    # Train model
    trainer = AutoEncoderTrainer(model, device)
    history = trainer.train(train_loader, test_loader, epochs=100)
    
    # Calculate reconstruction errors
    test_errors = calculate_reconstruction_error(trainer.model, test_loader, device)
    
    # SHAP Analysis
    shap_analyzer = SHAPAnalyzer(trainer.model, X_train, device)
    shap_values = shap_analyzer.get_shap_values(X_test)
    
    # Find top contributing features
    top_features = shap_analyzer.find_top_contributing_features(
        shap_values, feature_names, top_k=20
    )
    
    # Plot results
    shap_analyzer.plot_feature_importance(shap_values, feature_names, top_k=20)
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.hist(test_errors, bins=50, alpha=0.7)
    plt.title('Reconstruction Error Distribution')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()
    
    return {
        'model': trainer.model,
        'top_features': top_features,
        'reconstruction_errors': test_errors,
        'shap_values': shap_values,
        'training_history': history
    }

# Example usage:
if __name__ == "__main__":
    # Create sample data (replace with your actual data)
    # df = pd.read_csv('your_data.csv')  # Load your data
    
    # Create sample data for demonstration
    np.random.seed(42)
    sample_data = np.random.randint(0, 2, size=(10000, 1000)).astype(np.float32)
    df = pd.DataFrame(sample_data, columns=[f'flag{i+1}' for i in range(1000)])
    
    # Run pipeline
    results = main_pipeline(df)
    
    # Print top contributing features
    print("\nTop 10 features contributing to reconstruction error:")
    for feature in results['top_features'][:10]:
        print(f"{feature['rank']}. {feature['feature_name']}: {feature['mean_abs_shap']:.6f}")




"""
ACeDeC Model SHAP Analysis
==========================
Complete SHAP analysis implementation for ACeDeC clustering model
to identify features causing reconstruction errors.

Author: Generated for ACeDeC Model Investigation
Date: 2025
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class ACeDeCShapAnalyzer:
    """
    SHAP analyzer for ACeDeC clustering models
    """
    
    def __init__(self, model, feature_names=None):
        """
        Initialize SHAP analyzer
        
        Args:
            model: Trained ACeDeC model
            feature_names: List of feature names (optional)
        """
        self.model = model
        self.feature_names = feature_names
        self.shap_values = None
        self.explainer = None
        self.background_data = None
        self.sample_data = None
        
    def reconstruction_error_function(self, x):
        """
        Wrapper function that returns reconstruction error for SHAP analysis
        
        Args:
            x: Input data array
            
        Returns:
            Array of reconstruction errors per sample
        """
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        
        self.model.neural_network.eval()
        with torch.no_grad():
            reconstructed = self.model.neural_network(x)
            mse_loss = F.mse_loss(reconstructed, x, reduction='none')
            # Return per-sample reconstruction error
            return mse_loss.mean(dim=1).numpy()
    
    def prepare_data(self, X, background_size=100, sample_size=50):
        """
        Prepare data for SHAP analysis
        
        Args:
            X: Input data
            background_size: Size of background dataset for SHAP
            sample_size: Number of samples to explain
            
        Returns:
            Prepared background and sample data
        """
        # Convert to numpy if needed
        if isinstance(X, torch.Tensor):
            X = X.numpy()
        
        # Create background dataset
        if len(X) > background_size:
            background_indices = np.random.choice(len(X), background_size, replace=False)
            self.background_data = X[background_indices]
        else:
            self.background_data = X
        
        # Create sample dataset for explanation
        if len(X) > sample_size:
            sample_indices = np.random.choice(len(X), sample_size, replace=False)
            self.sample_data = X[sample_indices]
        else:
            self.sample_data = X
        
        print(f"✅ Data prepared:")
        print(f"   Background size: {len(self.background_data)}")
        print(f"   Sample size: {len(self.sample_data)}")
        
        return self.background_data, self.sample_data
    
    def calculate_shap_values(self, X, method='kernel', **kwargs):
        """
        Calculate SHAP values for the model
        
        Args:
            X: Input data
            method: SHAP method ('kernel', 'deep', 'linear')
            **kwargs: Additional arguments for SHAP explainer
            
        Returns:
            SHAP values and explainer
        """
        print("🔄 Calculating SHAP values...")
        
        # Prepare data
        self.prepare_data(X, **kwargs)
        
        try:
            if method == 'kernel':
                # Kernel SHAP (model-agnostic)
                self.explainer = shap.KernelExplainer(
                    self.reconstruction_error_function, 
                    self.background_data
                )
                
            elif method == 'deep':
                # Deep SHAP (for neural networks)
                self.explainer = shap.DeepExplainer(
                    self.model.neural_network,
                    torch.FloatTensor(self.background_data)
                )
                
            elif method == 'linear':
                # Linear SHAP (for linear models)
                self.explainer = shap.LinearExplainer(
                    self.reconstruction_error_function,
                    self.background_data
                )
            
            # Calculate SHAP values
            if method == 'deep':
                self.shap_values = self.explainer.shap_values(
                    torch.FloatTensor(self.sample_data)
                )
            else:
                self.shap_values = self.explainer.shap_values(self.sample_data)
            
            print(f"✅ SHAP values calculated successfully!")
            print(f"   Shape: {np.array(self.shap_values).shape}")
            
            return self.shap_values, self.explainer
            
        except Exception as e:
            print(f"❌ SHAP calculation failed: {e}")
            print("🔄 Trying fallback method...")
            
            # Fallback to kernel method
            try:
                self.explainer = shap.KernelExplainer(
                    self.reconstruction_error_function, 
                    self.background_data
                )
                self.shap_values = self.explainer.shap_values(self.sample_data)
                print(f"✅ Fallback successful!")
                return self.shap_values, self.explainer
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                return None, None
    
    def plot_shap_summary(self, plot_type='dot', max_display=20, figsize=(10, 8)):
        """
        Create SHAP summary plot
        
        Args:
            plot_type: Type of plot ('dot', 'bar', 'violin')
            max_display: Maximum number of features to display
            figsize: Figure size
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Run calculate_shap_values() first.")
            return
        
        plt.figure(figsize=figsize)
        
        if plot_type == 'dot':
            shap.summary_plot(
                self.shap_values, 
                self.sample_data,
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            plt.title("SHAP Summary Plot: Feature Impact on Reconstruction Error")
            
        elif plot_type == 'bar':
            shap.summary_plot(
                self.shap_values, 
                self.sample_data,
                plot_type="bar",
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            plt.title("SHAP Feature Importance for Reconstruction Error")
            
        elif plot_type == 'violin':
            shap.summary_plot(
                self.shap_values, 
                self.sample_data,
                plot_type="violin",
                feature_names=self.feature_names,
                max_display=max_display,
                show=False
            )
            plt.title("SHAP Violin Plot: Feature Impact Distribution")
        
        plt.tight_layout()
        plt.show()
    
    def plot_shap_waterfall(self, sample_idx=0, max_display=20):
        """
        Create SHAP waterfall plot for a specific sample
        
        Args:
            sample_idx: Index of sample to explain
            max_display: Maximum number of features to display
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Run calculate_shap_values() first.")
            return
        
        if sample_idx >= len(self.shap_values):
            print(f"❌ Sample index {sample_idx} out of range. Max: {len(self.shap_values)-1}")
            return
        
        # Create explanation object
        explanation = shap.Explanation(
            values=self.shap_values[sample_idx],
            base_values=self.explainer.expected_value,
            data=self.sample_data[sample_idx],
            feature_names=self.feature_names
        )
        
        shap.waterfall_plot(explanation, max_display=max_display, show=False)
        plt.title(f"SHAP Waterfall Plot - Sample {sample_idx}")
        plt.tight_layout()
        plt.show()
    
    def plot_shap_force(self, sample_idx=0):
        """
        Create SHAP force plot for a specific sample
        
        Args:
            sample_idx: Index of sample to explain
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Run calculate_shap_values() first.")
            return
        
        if sample_idx >= len(self.shap_values):
            print(f"❌ Sample index {sample_idx} out of range. Max: {len(self.shap_values)-1}")
            return
        
        # Create force plot
        force_plot = shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[sample_idx],
            self.sample_data[sample_idx],
            feature_names=self.feature_names,
            matplotlib=True,
            show=False
        )
        
        plt.title(f"SHAP Force Plot - Sample {sample_idx}")
        plt.tight_layout()
        plt.show()
        
        return force_plot
    
    def get_feature_importance_ranking(self, top_n=20):
        """
        Get feature importance ranking based on SHAP values
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importance ranking
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Run calculate_shap_values() first.")
            return None
        
        # Calculate mean absolute SHAP values
        mean_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # Create feature importance dataframe
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f"Feature_{i}" for i in range(len(mean_shap))]
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'SHAP_Importance': mean_shap,
            'Rank': range(1, len(mean_shap) + 1)
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('SHAP_Importance', ascending=False)
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        
        print(f"🏆 Top {top_n} Most Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df.head(top_n)
    
    def analyze_problematic_features(self, threshold_percentile=90):
        """
        Identify features that consistently cause high reconstruction errors
        
        Args:
            threshold_percentile: Percentile threshold for identifying problematic features
            
        Returns:
            Analysis results
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Run calculate_shap_values() first.")
            return None
        
        # Calculate reconstruction errors for sample data
        reconstruction_errors = self.reconstruction_error_function(self.sample_data)
        
        # Find high-error samples
        error_threshold = np.percentile(reconstruction_errors, threshold_percentile)
        high_error_mask = reconstruction_errors >= error_threshold
        
        print(f"📊 Problematic Feature Analysis:")
        print(f"   Error threshold ({threshold_percentile}th percentile): {error_threshold:.6f}")
        print(f"   High-error samples: {high_error_mask.sum()}/{len(reconstruction_errors)}")
        
        if high_error_mask.sum() == 0:
            print("   No samples above threshold found.")
            return None
        
        # Analyze SHAP values for high-error samples
        high_error_shap = self.shap_values[high_error_mask]
        mean_high_error_shap = np.mean(np.abs(high_error_shap), axis=0)
        
        # Compare with overall mean
        overall_mean_shap = np.mean(np.abs(self.shap_values), axis=0)
        
        # Calculate relative importance for problematic samples
        relative_importance = mean_high_error_shap / (overall_mean_shap + 1e-8)
        
        # Create results dataframe
        if self.feature_names is not None:
            feature_names = self.feature_names
        else:
            feature_names = [f"Feature_{i}" for i in range(len(mean_high_error_shap))]
        
        results_df = pd.DataFrame({
            'Feature': feature_names,
            'High_Error_SHAP': mean_high_error_shap,
            'Overall_SHAP': overall_mean_shap,
            'Relative_Importance': relative_importance
        })
        
        # Sort by relative importance
        results_df = results_df.sort_values('Relative_Importance', ascending=False)
        
        print(f"\n🔴 Most Problematic Features (Top 10):")
        print(results_df.head(10)[['Feature', 'Relative_Importance']].to_string(index=False))
        
        return results_df
    
    def save_results(self, filepath_prefix="shap_analysis"):
        """
        Save SHAP analysis results
        
        Args:
            filepath_prefix: Prefix for output files
        """
        if self.shap_values is None:
            print("❌ No SHAP values available. Run calculate_shap_values() first.")
            return
        
        # Save SHAP values
        np.save(f"{filepath_prefix}_shap_values.npy", self.shap_values)
        np.save(f"{filepath_prefix}_sample_data.npy", self.sample_data)
        
        # Save feature importance
        importance_df = self.get_feature_importance_ranking()
        if importance_df is not None:
            importance_df.to_csv(f"{filepath_prefix}_feature_importance.csv", index=False)
        
        # Save problematic features analysis
        problematic_df = self.analyze_problematic_features()
        if problematic_df is not None:
            problematic_df.to_csv(f"{filepath_prefix}_problematic_features.csv", index=False)
        
        print(f"✅ Results saved with prefix: {filepath_prefix}")


def main_analysis_pipeline(model, X, feature_names=None):
    """
    Complete SHAP analysis pipeline for ACeDeC model
    
    Args:
        model: Trained ACeDeC model
        X: Input data
        feature_names: List of feature names (optional)
    """
    print("🚀 Starting Complete SHAP Analysis Pipeline")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = ACeDeCShapAnalyzer(model, feature_names)
    
    # Calculate SHAP values
    shap_values, explainer = analyzer.calculate_shap_values(X, method='kernel')
    
    if shap_values is None:
        print("❌ SHAP analysis failed. Cannot proceed.")
        return None
    
    # Generate visualizations
    print("\n📊 Generating SHAP Visualizations...")
    
    # Summary plots
    analyzer.plot_shap_summary(plot_type='bar')
    analyzer.plot_shap_summary(plot_type='dot')
    
    # Individual sample analysis
    analyzer.plot_shap_waterfall(sample_idx=0)
    analyzer.plot_shap_force(sample_idx=0)
    
    # Feature importance analysis
    print("\n🔍 Analyzing Feature Importance...")
    importance_df = analyzer.get_feature_importance_ranking()
    
    # Problematic features analysis
    print("\n🚨 Identifying Problematic Features...")
    problematic_df = analyzer.analyze_problematic_features()
    
    # Save results
    print("\n💾 Saving Results...")
    analyzer.save_results("acedec_shap_analysis")
    
    print("\n✅ SHAP Analysis Complete!")
    
    return analyzer


# Example usage
if __name__ == "__main__":
    """
    Example usage of the SHAP analyzer
    Replace 'model' and 'X' with your actual ACeDeC model and data
    """
    
    # Example: Load your model and data
    # model = your_trained_acedec_model
    # X = your_input_data
    # feature_names = your_feature_names  # Optional
    
    # Run complete analysis
    # analyzer = main_analysis_pipeline(model, X, feature_names)
    
    # Or use individual components
    # analyzer = ACeDeCShapAnalyzer(model, feature_names)
    # analyzer.calculate_shap_values(X)
    # analyzer.plot_shap_summary()
    # analyzer.get_feature_importance_ranking()
    
    print("SHAP Analysis module loaded successfully!")
    print("Use main_analysis_pipeline(model, X, feature_names) to run complete analysis")



import torch
import pickle
import os
from datetime import datetime

import torch
import os

def save_neural_network_weights(model, save_path="acedec_nn_weights.pth"):
    state_dict = model.neural_network.state_dict()
    torch.save(state_dict, save_path)
    print(f"Neural network weights saved to {save_path}")

# Usage
save_neural_network_weights(model)
import pickle

def save_model_metadata(model, feature_names, save_path="acedec_metadata.pkl"):
    metadata = {
        "n_clusters": model.n_clusters,
        "embedding_size": model.embedding_size,
        "labels_": model.labels_,
        "cluster_centers_": model.cluster_centers_,
        "feature_names": feature_names
    }
    with open(save_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Model metadata saved to {save_path}")

# Usage
save_model_metadata(model, feature_names)
def load_model_metadata(save_path="acedec_metadata.pkl"):
    with open(save_path, "rb") as f:
        metadata = pickle.load(f)
    print("Model metadata loaded")
    return metadata

# Usage
metadata = load_model_metadata()
from clustpy.deep.enrc import ACeDeC
import torch

def recreate_acedec_model(metadata, device="cuda:0"):
    model = ACeDeC(
        n_clusters=metadata["n_clusters"],
        embedding_size=metadata["embedding_size"],
        device=device
    )
    # Dummy fit to initialize architecture (with minimal data)
    import numpy as np
    dummy_X = np.zeros((2, len(metadata["feature_names"])), dtype=np.float32)
    model.fit(dummy_X)
    print("Model architecture recreated")
    return model

# Usage
model_loaded = recreate_acedec_model(metadata)
def load_neural_network_weights(model, weights_path="acedec_nn_weights.pth", device="cuda:0"):
    state_dict = torch.load(weights_path, map_location=device)
    model.neural_network.load_state_dict(state_dict)
    model.neural_network.to(device)
    print("Neural network weights loaded into model")

# Usage
load_neural_network_weights(model_loaded)
model_loaded.labels_ = metadata["labels_"]
model_loaded.cluster_centers_ = metadata["cluster_centers_"]
def validate_weights(model):
    state_dict = model.neural_network.state_dict()
    print(f"Total tensors in state_dict: {len(state_dict)}")
    for name, tensor in state_dict.items():
        print(f"{name}: shape {tuple(tensor.shape)}, dtype {tensor.dtype}, device {tensor.device}")
    # Additional checks
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"Total parameters: {total_params}")
    return state_dict

# Usage
validate_weights(model_loaded)
import torch
def test_forward_pass(model, feature_names):
    device = next(model.neural_network.parameters()).device
    test_input = torch.zeros((1, len(feature_names)), dtype=torch.float32).to(device)
    model.neural_network.eval()
    with torch.no_grad():
        output = model.neural_network(test_input)
    print(f"Forward pass successful: input shape {test_input.shape} → output shape {output.shape}")

# Usage
test_forward_pass(model_loaded, metadata["feature_names"])
