import torch
import torch.nn as nn
import numpy as np
import shap
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset

class GPUOptimizedSHAPAnalyzer:
    """
    GPU-optimized SHAP analysis for autoencoder reconstruction error
    """
    def __init__(self, model, background_data, device='cuda'):
        self.model = model.to(device)
        self.device = device
        
        # Keep background data on GPU
        if isinstance(background_data, np.ndarray):
            self.bg_data_gpu = torch.FloatTensor(background_data).to(device)
        else:
            self.bg_data_gpu = background_data.to(device)
        
        # Also keep CPU version for SHAP explainers that need it
        self.bg_data_cpu = self.bg_data_gpu.cpu().numpy()
        
        # Reduce background size for GPU memory efficiency
        if self.bg_data_gpu.shape[0] > 50:
            self.bg_data_gpu = self.bg_data_gpu[:50]
            self.bg_data_cpu = self.bg_data_cpu[:50]
        
        print(f"Background data shape: {self.bg_data_gpu.shape}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated(device) / 1024**2:.2f} MB")
        
    def gpu_reconstruction_error_wrapper(self, x):
        """
        GPU-optimized wrapper for reconstruction error
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        elif not x.is_cuda:
            x = x.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(x)
            mse = torch.mean((x - reconstructed) ** 2, dim=1)
            return mse.cpu().numpy()
    
    def gpu_feature_reconstruction_wrapper(self, x):
        """
        GPU-optimized wrapper for feature-wise reconstruction error
        """
        if isinstance(x, np.ndarray):
            x = torch.FloatTensor(x).to(self.device)
        elif not x.is_cuda:
            x = x.to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            reconstructed = self.model(x)
            feature_errors = (x - reconstructed) ** 2
            return feature_errors.cpu().numpy()
    
    def get_shap_values_gpu_batched(self, data, max_samples=100, batch_size=32):
        """
        Method 1: GPU-batched KernelExplainer
        Process data in batches to avoid GPU memory issues
        """
        # Reduce sample size for GPU efficiency
        if data.shape[0] > max_samples:
            indices = np.random.choice(data.shape[0], max_samples, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
        
        print(f"Method 1: GPU-batched KernelExplainer")
        print(f"Sample size: {sample_data.shape[0]}, Features: {sample_data.shape[1]}")
        print(f"Processing in batches of {batch_size}")
        
        # Use smaller background for KernelExplainer
        bg_small = self.bg_data_cpu[:min(20, len(self.bg_data_cpu))]
        
        try:
            explainer = shap.KernelExplainer(
                self.gpu_reconstruction_error_wrapper, 
                bg_small
            )
            
            # Process in smaller batches to avoid memory issues
            batch_size = min(batch_size, 10)  # Smaller batches for KernelExplainer
            shap_values_list = []
            
            for i in range(0, len(sample_data), batch_size):
                batch = sample_data[i:i+batch_size]
                print(f"Processing batch {i//batch_size + 1}/{(len(sample_data)-1)//batch_size + 1}")
                
                batch_shap = explainer.shap_values(batch)
                shap_values_list.append(batch_shap)
                
                # Clear GPU cache
                torch.cuda.empty_cache()
            
            # Concatenate results
            shap_values = np.concatenate(shap_values_list, axis=0)
            return shap_values
            
        except Exception as e:
            print(f"KernelExplainer failed: {e}")
            return None
    
    def get_shap_values_gpu_gradient_batched(self, data, max_samples=200, batch_size=64):
        """
        Method 2: GPU-batched gradient-based importance (RECOMMENDED)
        Most efficient for GPU with large datasets
        """
        # Can handle more samples with gradient method
        if data.shape[0] > max_samples:
            indices = np.random.choice(data.shape[0], max_samples, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
        
        print(f"Method 2: GPU-batched gradient-based importance")
        print(f"Sample size: {sample_data.shape[0]}, Features: {sample_data.shape[1]}")
        print(f"Processing in batches of {batch_size}")
        
        self.model.eval()
        
        # Convert to tensor on GPU
        if isinstance(sample_data, np.ndarray):
            sample_tensor = torch.FloatTensor(sample_data).to(self.device)
        else:
            sample_tensor = sample_data.to(self.device)
        
        # Process in batches
        gradients_list = []
        
        for i in range(0, len(sample_tensor), batch_size):
            batch = sample_tensor[i:i+batch_size]
            batch.requires_grad_(True)
            
            # Forward pass
            reconstructed = self.model(batch)
            reconstruction_error = torch.mean((batch - reconstructed) ** 2)
            
            # Backward pass
            reconstruction_error.backward()
            
            # Store gradients
            batch_gradients = batch.grad.abs().cpu().numpy()
            gradients_list.append(batch_gradients)
            
            # Clear gradients and cache
            batch.grad = None
            torch.cuda.empty_cache()
            
            if (i // batch_size + 1) % 10 == 0:
                print(f"Processed {i//batch_size + 1}/{(len(sample_tensor)-1)//batch_size + 1} batches")
        
        # Concatenate results
        gradients = np.concatenate(gradients_list, axis=0)
        return gradients
    
    def get_shap_values_gpu_feature_ablation(self, data, max_samples=100, max_features=200):
        """
        Method 3: GPU-optimized feature ablation
        Sample features for computational efficiency
        """
        if data.shape[0] > max_samples:
            indices = np.random.choice(data.shape[0], max_samples, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
        
        print(f"Method 3: GPU-optimized feature ablation")
        print(f"Sample size: {sample_data.shape[0]}, Features: {sample_data.shape[1]}")
        
        # Convert to GPU tensor
        if isinstance(sample_data, np.ndarray):
            sample_tensor = torch.FloatTensor(sample_data).to(self.device)
        else:
            sample_tensor = sample_data.to(self.device)
        
        # Calculate baseline reconstruction error
        with torch.no_grad():
            baseline_recon = self.model(sample_tensor)
            baseline_error = torch.mean((sample_tensor - baseline_recon) ** 2, dim=1)
            baseline_mean = torch.mean(baseline_error)
        
        # Sample features for ablation
        n_features = sample_tensor.shape[1]
        if n_features > max_features:
            feature_indices = np.random.choice(n_features, max_features, replace=False)
        else:
            feature_indices = np.arange(n_features)
        
        print(f"Testing {len(feature_indices)} features")
        
        # Feature importance scores
        feature_importance = torch.zeros(n_features, device=self.device)
        
        for i, feat_idx in enumerate(feature_indices):
            # Create ablated version
            ablated_tensor = sample_tensor.clone()
            ablated_tensor[:, feat_idx] = 0
            
            # Calculate reconstruction error
            with torch.no_grad():
                ablated_recon = self.model(ablated_tensor)
                ablated_error = torch.mean((ablated_tensor - ablated_recon) ** 2, dim=1)
                ablated_mean = torch.mean(ablated_error)
            
            # Feature importance = increase in reconstruction error
            feature_importance[feat_idx] = ablated_mean - baseline_mean
            
            if (i + 1) % 50 == 0:
                print(f"Processed {i+1}/{len(feature_indices)} features")
        
        # Create SHAP-like values (same importance for all samples)
        shap_values = feature_importance.cpu().numpy()
        shap_values = np.tile(shap_values, (sample_data.shape[0], 1))
        
        return shap_values
    
    def get_shap_values_gpu_integrated_gradients(self, data, max_samples=100, steps=50):
        """
        Method 4: GPU-optimized Integrated Gradients
        More accurate than simple gradients
        """
        if data.shape[0] > max_samples:
            indices = np.random.choice(data.shape[0], max_samples, replace=False)
            sample_data = data[indices]
        else:
            sample_data = data
        
        print(f"Method 4: GPU-optimized Integrated Gradients")
        print(f"Sample size: {sample_data.shape[0]}, Features: {sample_data.shape[1]}")
        print(f"Integration steps: {steps}")
        
        # Convert to GPU tensor
        if isinstance(sample_data, np.ndarray):
            sample_tensor = torch.FloatTensor(sample_data).to(self.device)
        else:
            sample_tensor = sample_data.to(self.device)
        
        # Use background mean as baseline
        baseline = torch.mean(self.bg_data_gpu, dim=0, keepdim=True)
        
        self.model.eval()
        
        # Calculate integrated gradients
        integrated_gradients = []
        
        for i in range(sample_tensor.shape[0]):
            sample = sample_tensor[i:i+1]
            
            # Create interpolated inputs
            alphas = torch.linspace(0, 1, steps, device=self.device)
            interpolated_inputs = []
            
            for alpha in alphas:
                interpolated = baseline + alpha * (sample - baseline)
                interpolated_inputs.append(interpolated)
            
            # Stack all interpolated inputs
            interpolated_batch = torch.cat(interpolated_inputs, dim=0)
            interpolated_batch.requires_grad_(True)
            
            # Forward pass
            reconstructed = self.model(interpolated_batch)
            reconstruction_errors = torch.mean((interpolated_batch - reconstructed) ** 2, dim=1)
            total_error = torch.sum(reconstruction_errors)
            
            # Backward pass
            total_error.backward()
            
            # Get gradients
            gradients = interpolated_batch.grad
            
            # Integrate gradients
            integrated_grad = torch.mean(gradients, dim=0) * (sample - baseline).squeeze()
            integrated_gradients.append(integrated_grad.abs().cpu().numpy())
            
            # Clear gradients
            interpolated_batch.grad = None
            
            if (i + 1) % 10 == 0:
                print(f"Processed {i+1}/{sample_tensor.shape[0]} samples")
                torch.cuda.empty_cache()
        
        return np.array(integrated_gradients)
    
    def find_top_contributing_features(self, shap_values, feature_names, top_k=20):
        """
        Find features contributing most to reconstruction error
        """
        # Handle different input shapes
        if len(shap_values.shape) == 3:
            # Take diagonal if shape is (samples, features, features)
            shap_values = np.array([shap_values[i, :, i] for i in range(shap_values.shape[0])])
        
        # Calculate mean absolute importance
        mean_shap = np.mean(np.abs(shap_values), axis=0)
        
        # Get top k features
        top_indices = np.argsort(mean_shap)[-top_k:][::-1]
        
        results = []
        for i, idx in enumerate(top_indices):
            results.append({
                'rank': i + 1,
                'feature_name': feature_names[idx] if feature_names is not None else f'feature_{idx}',
                'feature_index': idx,
                'mean_abs_shap': mean_shap[idx],
                'std_shap': np.std(shap_values[:, idx]) if len(shap_values.shape) > 1 else 0
            })
        
        return results
    
    def plot_feature_importance(self, shap_values, feature_names=None, top_k=20, method_name=""):
        """
        Plot feature importance based on SHAP values
        """
        top_features = self.find_top_contributing_features(shap_values, feature_names, top_k)
        
        features = [f['feature_name'] for f in top_features]
        values = [f['mean_abs_shap'] for f in top_features]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(features)), values)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Mean Absolute Importance Score')
        plt.title(f'Top Features Contributing to Reconstruction Error ({method_name})')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()
        
        return top_features
    
    def monitor_gpu_memory(self):
        """
        Monitor GPU memory usage
        """
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated(self.device) / 1024**2
            reserved = torch.cuda.memory_reserved(self.device) / 1024**2
            print(f"GPU Memory - Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")
        
    def clear_gpu_cache(self):
        """
        Clear GPU cache
        """
        torch.cuda.empty_cache()
        print("GPU cache cleared")

def run_gpu_optimized_analysis(model, X_train, X_test, feature_names, device='cuda'):
    """
    Run GPU-optimized SHAP analysis with all methods
    """
    print("=== GPU-Optimized SHAP Analysis ===")
    print(f"Using device: {device}")
    print(f"GPU available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU name: {torch.cuda.get_device_name(device)}")
        print(f"GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1024**2:.2f} MB")
    
    # Initialize GPU-optimized analyzer
    analyzer = GPUOptimizedSHAPAnalyzer(model, X_train, device)
    analyzer.monitor_gpu_memory()
    
    results = {}
    
    # Method 1: Gradient-based (RECOMMENDED - fastest and most reliable)
    print("\n" + "="*50)
    print("METHOD 1: GRADIENT-BASED (RECOMMENDED)")
    print("="*50)
    try:
        shap_values_grad = analyzer.get_shap_values_gpu_gradient_batched(
            X_test, max_samples=200, batch_size=64
        )
        
        if shap_values_grad is not None:
            print(f"Gradient SHAP values shape: {shap_values_grad.shape}")
            
            top_features_grad = analyzer.find_top_contributing_features(
                shap_values_grad, feature_names, top_k=20
            )
            
            results['gradient'] = {
                'shap_values': shap_values_grad,
                'top_features': top_features_grad
            }
            
            analyzer.plot_feature_importance(
                shap_values_grad, feature_names, top_k=20, method_name="Gradient-based"
            )
            
            print("✓ Gradient-based method completed successfully")
        
    except Exception as e:
        print(f"✗ Gradient-based method failed: {e}")
        results['gradient'] = None
    
    analyzer.clear_gpu_cache()
    
    # Method 2: Integrated Gradients (More accurate)
    print("\n" + "="*50)
    print("METHOD 2: INTEGRATED GRADIENTS")
    print("="*50)
    try:
        shap_values_ig = analyzer.get_shap_values_gpu_integrated_gradients(
            X_test, max_samples=50, steps=30
        )
        
        if shap_values_ig is not None:
            print(f"Integrated Gradients SHAP values shape: {shap_values_ig.shape}")
            
            top_features_ig = analyzer.find_top_contributing_features(
                shap_values_ig, feature_names, top_k=20
            )
            
            results['integrated_gradients'] = {
                'shap_values': shap_values_ig,
                'top_features': top_features_ig
            }
            
            analyzer.plot_feature_importance(
                shap_values_ig, feature_names, top_k=20, method_name="Integrated Gradients"
            )
            
            print("✓ Integrated Gradients method completed successfully")
        
    except Exception as e:
        print(f"✗ Integrated Gradients method failed: {e}")
        results['integrated_gradients'] = None
    
    analyzer.clear_gpu_cache()
    
    # Method 3: Feature Ablation (Most interpretable)
    print("\n" + "="*50)
    print("METHOD 3: FEATURE ABLATION")
    print("="*50)
    try:
        shap_values_ablation = analyzer.get_shap_values_gpu_feature_ablation(
            X_test, max_samples=100, max_features=200
        )
        
        if shap_values_ablation is not None:
            print(f"Feature Ablation SHAP values shape: {shap_values_ablation.shape}")
            
            top_features_ablation = analyzer.find_top_contributing_features(
                shap_values_ablation, feature_names, top_k=20
            )
            
            results['feature_ablation'] = {
                'shap_values': shap_values_ablation,
                'top_features': top_features_ablation
            }
            
            analyzer.plot_feature_importance(
                shap_values_ablation, feature_names, top_k=20, method_name="Feature Ablation"
            )
            
            print("✓ Feature Ablation method completed successfully")
        
    except Exception as e:
        print(f"✗ Feature Ablation method failed: {e}")
        results['feature_ablation'] = None
    
    analyzer.clear_gpu_cache()
    
    # Method 4: KernelExplainer (Most accurate but slowest)
    print("\n" + "="*50)
    print("METHOD 4: KERNEL EXPLAINER (SLOW)")
    print("="*50)
    try:
        shap_values_kernel = analyzer.get_shap_values_gpu_batched(
            X_test, max_samples=30, batch_size=5  # Very small for KernelExplainer
        )
        
        if shap_values_kernel is not None:
            print(f"Kernel SHAP values shape: {shap_values_kernel.shape}")
            
            top_features_kernel = analyzer.find_top_contributing_features(
                shap_values_kernel, feature_names, top_k=20
            )
            
            results['kernel'] = {
                'shap_values': shap_values_kernel,
                'top_features': top_features_kernel
            }
            
            analyzer.plot_feature_importance(
                shap_values_kernel, feature_names, top_k=20, method_name="Kernel Explainer"
            )
            
            print("✓ Kernel Explainer method completed successfully")
        
    except Exception as e:
        print(f"✗ Kernel Explainer method failed: {e}")
        results['kernel'] = None
    
    analyzer.clear_gpu_cache()
    analyzer.monitor_gpu_memory()
    
    # Print summary
    print("\n" + "="*50)
    print("ANALYSIS SUMMARY")
    print("="*50)
    
    for method_name, method_results in results.items():
        if method_results is not None:
            print(f"\n{method_name.upper().replace('_', ' ')} - Top 5 features:")
            for feature in method_results['top_features'][:5]:
                print(f"  {feature['rank']}. {feature['feature_name']}: {feature['mean_abs_shap']:.6f}")
        else:
            print(f"\n{method_name.upper().replace('_', ' ')}: Failed")
    
    print(f"\nRecommendation: Use GRADIENT-BASED method for best speed/accuracy trade-off")
    print(f"Use INTEGRATED GRADIENTS for highest accuracy")
    print(f"Use FEATURE ABLATION for most interpretable results")
    
    return results

# Example usage
if __name__ == "__main__":
    # Ensure you have your trained model and data
    # model = your_trained_autoencoder_model
    # X_train, X_test = your_training_and_test_data
    # feature_names = your_feature_names
    
    # Run GPU-optimized analysis
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # results = run_gpu_optimized_analysis(model, X_train, X_test, feature_names, device)
    
    print("\nTo use this code:")
    print("1. results = run_gpu_optimized_analysis(model, X_train, X_test, feature_names)")
    print("2. Best method is usually 'gradient' for speed and 'integrated_gradients' for accuracy")
    print("3. Monitor GPU memory usage with the built-in monitoring functions")
