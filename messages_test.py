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
