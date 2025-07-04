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

def save_acedec_model_complete(model, X, feature_names, save_dir="saved_models"):
    """
    Save complete ACeDeC model with all necessary components
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"💾 Saving ACeDeC model to {save_dir}/")
    
    # 1. Save neural network state dict (most important)
    if hasattr(model, 'neural_network') and model.neural_network is not None:
        torch.save(
            model.neural_network.state_dict(), 
            f"{save_dir}/acedec_neural_network_{timestamp}.pth"
        )
        print("✅ Neural network weights saved")
    
    # 2. Save model parameters and metadata
    model_metadata = {
        'n_clusters': getattr(model, 'n_clusters', None),
        'embedding_size': getattr(model, 'embedding_size', None),
        'labels_': getattr(model, 'labels_', None),
        'cluster_centers_': getattr(model, 'cluster_centers_', None),
        'device': str(next(model.neural_network.parameters()).device),
        'input_shape': X.shape,
        'feature_names': feature_names,
        'timestamp': timestamp
    }
    
    with open(f"{save_dir}/acedec_metadata_{timestamp}.pkl", 'wb') as f:
        pickle.dump(model_metadata, f)
    print("✅ Model metadata saved")
    
    # 3. Save the complete model object (backup method)
    try:
        torch.save(model, f"{save_dir}/acedec_complete_model_{timestamp}.pth")
        print("✅ Complete model saved as backup")
    except Exception as e:
        print(f"⚠️  Complete model save failed: {e}")
    
    # 4. Save sample data for testing
    sample_data = {
        'X_sample': X[:10],  # First 10 samples for testing
        'feature_names': feature_names
    }
    
    with open(f"{save_dir}/sample_data_{timestamp}.pkl", 'wb') as f:
        pickle.dump(sample_data, f)
    print("✅ Sample data saved")
    
    print(f"🎉 Model saved successfully with timestamp: {timestamp}")
    return timestamp

# Save your model
timestamp = save_acedec_model_complete(model, X, feature_names)
def save_acedec_model_enhanced(model, X, feature_names, save_dir="saved_models"):
    """
    Enhanced saving that stores everything needed for reliable reconstruction
    """
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"💾 Enhanced saving to {save_dir}/")
    
    # 1. Save neural network architecture and weights
    if hasattr(model, 'neural_network') and model.neural_network is not None:
        # Save state dict
        torch.save(
            model.neural_network.state_dict(),
            f"{save_dir}/neural_network_weights_{timestamp}.pth"
        )
        
        # Save model architecture info
        architecture_info = {
            'model_class': type(model.neural_network).__name__,
            'model_structure': str(model.neural_network)
        }
        
        with open(f"{save_dir}/architecture_{timestamp}.pkl", 'wb') as f:
            pickle.dump(architecture_info, f)
        
        print("✅ Neural network saved with architecture info")
    
    # 2. Save complete ACeDeC model state
    acedec_state = {
        # Model parameters
        'n_clusters': getattr(model, 'n_clusters', None),
        'embedding_size': getattr(model, 'embedding_size', None),
        'pretrain_epochs': getattr(model, 'pretrain_epochs', None),
        'clustering_epochs': getattr(model, 'clustering_epochs', None),
        'batch_size': getattr(model, 'batch_size', None),
        
        # Training results
        'labels_': getattr(model, 'labels_', None),
        'cluster_centers_': getattr(model, 'cluster_centers_', None),
        
        # Device and data info
        'device': str(next(model.neural_network.parameters()).device),
        'input_shape': X.shape,
        'feature_names': feature_names,
        
        # Additional attributes
        'random_state': getattr(model, 'random_state', None),
        'timestamp': timestamp
    }
    
    with open(f"{save_dir}/acedec_state_{timestamp}.pkl", 'wb') as f:
        pickle.dump(acedec_state, f)
    print("✅ ACeDeC state saved")
    
    # 3. Save training data (for model reconstruction)
    training_data = {
        'X': X,
        'feature_names': feature_names
    }
    
    with open(f"{save_dir}/training_data_{timestamp}.pkl", 'wb') as f:
        pickle.dump(training_data, f)
    print("✅ Training data saved")
    
    print(f"🎉 Enhanced save completed: {timestamp}")
    return timestamp

# Enhanced save
timestamp = save_acedec_model_enhanced(model, X, feature_names)
def load_acedec_model_enhanced(timestamp, save_dir="saved_models", device=None, retrain=True):
    """
    Enhanced loading that recreates the model reliably
    """
    print(f"📂 Enhanced loading from {save_dir}/ with timestamp {timestamp}")
    
    # Auto-detect device
    if device is None:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"🎯 Target device: {device}")
    
    try:
        # 1. Load ACeDeC state
        with open(f"{save_dir}/acedec_state_{timestamp}.pkl", 'rb') as f:
            acedec_state = pickle.load(f)
        print("✅ ACeDeC state loaded")
        
        # 2. Load training data
        with open(f"{save_dir}/training_data_{timestamp}.pkl", 'rb') as f:
            training_data = pickle.load(f)
        print("✅ Training data loaded")
        
        # 3. Recreate the model
        from clustpy.deep.enrc import ACeDeC
        
        print("🔄 Recreating ACeDeC model...")
        recreated_model = ACeDeC(
            n_clusters=acedec_state['n_clusters'],
            embedding_size=acedec_state['embedding_size'],
            pretrain_epochs=acedec_state.get('pretrain_epochs', 50),
            clustering_epochs=acedec_state.get('clustering_epochs', 100),
            batch_size=acedec_state.get('batch_size', 64),
            random_state=acedec_state.get('random_state', 42),
            device=device
        )
        
        if retrain:
            # 4. Retrain the model (this recreates the neural network architecture)
            print("🔄 Retraining model to recreate architecture...")
            recreated_model.fit(training_data['X'])
            print("✅ Model retrained")
            
            # 5. Load the saved weights
            try:
                saved_weights = torch.load(
                    f"{save_dir}/neural_network_weights_{timestamp}.pth",
                    map_location=device
                )
                recreated_model.neural_network.load_state_dict(saved_weights)
                print("✅ Saved weights loaded into retrained model")
            except Exception as e:
                print(f"⚠️  Weight loading failed: {e}")
                print("   Using retrained weights instead")
        
        # 6. Restore other attributes
        if acedec_state['labels_'] is not None:
            recreated_model.labels_ = acedec_state['labels_']
        if acedec_state['cluster_centers_'] is not None:
            recreated_model.cluster_centers_ = acedec_state['cluster_centers_']
        
        print("🎉 Model successfully recreated!")
        return recreated_model, acedec_state, training_data
        
    except Exception as e:
        print(f"❌ Enhanced loading failed: {e}")
        return None, None, None

# Enhanced load
loaded_model, loaded_state, loaded_data = load_acedec_model_enhanced(timestamp, retrain=True)
def test_loaded_model(loaded_model, original_X, feature_names):
    """
    Test that the loaded model works correctly
    """
    print("🧪 Testing loaded model...")
    
    if loaded_model is None:
        print("❌ No model to test")
        return False
    
    try:
        # Test 1: Check model attributes
        print("1. Checking model attributes...")
        required_attrs = ['labels_', 'cluster_centers_', 'neural_network']
        for attr in required_attrs:
            if hasattr(loaded_model, attr) and getattr(loaded_model, attr) is not None:
                print(f"   ✅ {attr}: Available")
            else:
                print(f"   ⚠️  {attr}: Missing or None")
        
        # Test 2: Test neural network forward pass
        print("2. Testing neural network...")
        loaded_model.neural_network.eval()
        with torch.no_grad():
            # Get device
            device = next(loaded_model.neural_network.parameters()).device
            test_input = torch.FloatTensor(original_X[:5]).to(device)
            output = loaded_model.neural_network(test_input)
            print(f"   ✅ Forward pass successful: {test_input.shape} → {output.shape}")
        
        # Test 3: Test reconstruction loss calculation
        print("3. Testing reconstruction loss...")
        reconstruction_results = calculate_reconstruction_loss_gpu(loaded_model, original_X[:100])
        print(f"   ✅ Reconstruction loss: {reconstruction_results['total_loss']:.6f}")
        
        # Test 4: Test SHAP analyzer compatibility
        print("4. Testing SHAP analyzer compatibility...")
        analyzer_test = ACeDeCShapAnalyzerGPU_Fixed(loaded_model, feature_names)
        print("   ✅ SHAP analyzer initialized successfully")
        
        print("🎉 All tests passed! Model loaded successfully.")
        return True
        
    except Exception as e:
        print(f"❌ Testing failed: {e}")
        return False

# Test your loaded model
if loaded_model is not None:
    test_success = test_loaded_model(loaded_model, X, feature_names)
else:
    print("❌ No model loaded to test")
# Save your trained model
print("Saving your ACeDeC model...")
timestamp = save_acedec_model_enhanced(model, X, feature_names)
print(f"Model saved with timestamp: {timestamp}")
# In a new session or script, load your model
print("Loading your saved ACeDeC model...")

# Option 1: Load with retraining (most reliable)
loaded_model, loaded_state, loaded_data = load_acedec_model_enhanced(
    timestamp, 
    retrain=True,
    device=torch.device('cuda:0')  # Specify your target device
)

# Option 2: Load without retraining (faster but less reliable)
# loaded_model, loaded_state, loaded_data = load_acedec_model_enhanced(
#     timestamp, 
#     retrain=False,
#     device=torch.device('cuda:0')
# )

# Test the loaded model
if loaded_model is not None:
    test_success = test_loaded_model(loaded_model, loaded_data['X'], loaded_data['feature_names'])
    
    if test_success:
        print("🎉 Model ready for analysis!")
        
        # Continue with your analysis
        analyzer_gpu = ACeDeCShapAnalyzerGPU_Fixed(loaded_model, loaded_data['feature_names'])
        
        # Run reconstruction loss analysis
        reconstruction_results = calculate_reconstruction_loss_gpu(loaded_model, loaded_data['X'])
        
        # Run SHAP analysis
        shap_values, explainer = analyzer_gpu.calculate_shap_values_numpy_fixed(
            loaded_data['X'], background_size=20, sample_size=5
        )

