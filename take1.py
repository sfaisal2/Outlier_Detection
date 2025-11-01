import pandas as pd
import numpy as np
from scipy.stats import chi2
import matplotlib.pyplot as plt
import seaborn as sns

class MahalanobisOutlierDetector:
    def __init__(self):
        self.df = None
        self.df_analysis = None
        self.cov_matrix = None
        self.inv_cov_matrix = None
        self.centerpoint = None
        
    def load_and_prepare_data(self, file_path='HW2023.csv'):
        """Load and preprocess the weather dataset - using your approach"""
        self.df = pd.read_csv(file_path, sep=",", decimal='.')
        
        # Keep the DATE column for reference
        if 'DATE' in self.df.columns:
            self.dates = self.df['DATE'].copy()
        
        # Select numeric columns for analysis
        self.numeric_columns = ['cloudiness', 'rain', 'temp_max', 'wind_speed', 'visibility', 'humidity']
        self.df_analysis = self.df[self.numeric_columns].copy()
        
        # Handle missing values in rain (no entry means 'no rain')
        self.df_analysis['rain'] = self.df_analysis['rain'].fillna(0)
        
        # Remove any remaining rows with missing values
        self.df_analysis = self.df_analysis.dropna()
        
        print(f"Dataset shape: {self.df_analysis.shape}")
        print(f"Columns: {list(self.df_analysis.columns)}")
        
        return self.df_analysis
    
    def calculate_mahalanobis_components(self):
        """Calculate covariance matrix and center point - using your approach"""
        # Convert to numpy array like your code
        data_array = self.df_analysis.to_numpy()
        
        # Covariance matrix (your approach)
        self.cov_matrix = np.cov(data_array, rowvar=False)
        
        # Covariance matrix power of -1 (your approach)
        self.inv_cov_matrix = np.linalg.pinv(self.cov_matrix)
        
        # Center point (your approach)
        self.centerpoint = np.mean(data_array, axis=0)
        
        return self.cov_matrix, self.inv_cov_matrix, self.centerpoint
    
    def calculate_outlier_scores(self):
        """Calculate Mahalanobis distance for each point - enhanced from your code"""
        data_array = self.df_analysis.to_numpy()
        distances = []
        
        # Calculate distances using your method
        for i, val in enumerate(data_array):
            p1 = val
            p2 = self.centerpoint
            distance = np.sqrt((p1-p2).T.dot(self.inv_cov_matrix).dot(p1-p2))
            distances.append(distance)
        
        distances = np.array(distances)
        return distances
    
    def detect_outliers_multiple_configs(self):
        """Apply Mahalanobis with 3 different cutoff thresholds"""
        distances = self.calculate_outlier_scores()
        
        results = {}
        
        # Configuration 1: 95% confidence cutoff (your original approach)
        cutoff1 = chi2.ppf(0.95, self.df_analysis.shape[1])
        df1 = self.df_analysis.copy()
        df1['OLS'] = distances
        df1['is_outlier'] = distances > cutoff1
        df1['DATE'] = self.dates.iloc[:len(df1)] if hasattr(self, 'dates') else range(len(df1))
        results['config1'] = {'df': df1, 'cutoff': cutoff1, 'confidence': 0.95}
        
        # Configuration 2: 90% confidence cutoff
        cutoff2 = chi2.ppf(0.90, self.df_analysis.shape[1])
        df2 = self.df_analysis.copy()
        df2['OLS'] = distances
        df2['is_outlier'] = distances > cutoff2
        df2['DATE'] = self.dates.iloc[:len(df2)] if hasattr(self, 'dates') else range(len(df2))
        results['config2'] = {'df': df2, 'cutoff': cutoff2, 'confidence': 0.90}
        
        # Configuration 3: 99% confidence cutoff
        cutoff3 = chi2.ppf(0.99, self.df_analysis.shape[1])
        df3 = self.df_analysis.copy()
        df3['OLS'] = distances
        df3['is_outlier'] = distances > cutoff3
        df3['DATE'] = self.dates.iloc[:len(df3)] if hasattr(self, 'dates') else range(len(df3))
        results['config3'] = {'df': df3, 'cutoff': cutoff3, 'confidence': 0.99}
        
        return results, distances
    
    def analyze_results(self, results):
        """Analyze and display the results"""
        print("MAHALANOBIS OUTLIER DETECTION RESULTS")
        print("=" * 60)
        
        for config_name, config_data in results.items():
            df = config_data['df']
            cutoff = config_data['cutoff']
            confidence = config_data['confidence']
            
            print(f"\n{config_name.upper()} (Confidence: {confidence:.0%}, Cutoff: {cutoff:.4f})")
            print("-" * 50)
            
            # Sort by outlier score (descending)
            df_sorted = df.sort_values('OLS', ascending=False)
            
            # Number of outliers
            num_outliers = df['is_outlier'].sum()
            print(f"Number of outliers detected: {num_outliers}")
            
            # Top 4 outliers
            print(f"\nTop 4 Outliers:")
            top_outliers = df_sorted[df_sorted['is_outlier'] == True].head(4)
            if len(top_outliers) > 0:
                for i, (idx, row) in enumerate(top_outliers.iterrows()):
                    print(f"Rank {i+1}: OLS = {row['OLS']:.4f}")
                    if 'DATE' in row:
                        print(f"  Date: {row['DATE']}")
                    for col in self.numeric_columns:
                        print(f"  {col}: {row[col]}")
                    print()
            else:
                print("No outliers detected at this confidence level")
            
            # Most normal example (lowest outlier score)
            most_normal = df_sorted.iloc[-1]
            print(f"Most Normal Example (Lowest OLS):")
            print(f"OLS = {most_normal['OLS']:.4f}")
            if 'DATE' in most_normal:
                print(f"Date: {most_normal['DATE']}")
            for col in self.numeric_columns:
                print(f"  {col}: {most_normal[col]}")
            print()
    
    def visualize_results(self, results, distances):
        """Create visualizations for the results - enhanced from your ellipse plot"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Distribution of Mahalanobis distances
        axes[0,0].hist(distances, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,0].axvline(x=results['config1']['cutoff'], color='red', linestyle='--', label='95% cutoff')
        axes[0,0].axvline(x=results['config2']['cutoff'], color='orange', linestyle='--', label='90% cutoff')
        axes[0,0].axvline(x=results['config3']['cutoff'], color='green', linestyle='--', label='99% cutoff')
        axes[0,0].set_xlabel('Mahalanobis Distance')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].set_title('Distribution of Mahalanobis Distances')
        axes[0,0].legend()
        
        # Plot 2: Outlier scores vs temperature and rain
        config1_df = results['config1']['df']
        scatter = axes[0,1].scatter(config1_df['temp_max'], config1_df['rain'], 
                                   c=config1_df['OLS'], cmap='viridis', alpha=0.7)
        axes[0,1].set_xlabel('Max Temperature')
        axes[0,1].set_ylabel('Rain')
        axes[0,1].set_title('Mahalanobis Distance: Temperature vs Rain')
        plt.colorbar(scatter, ax=axes[0,1], label='Mahalanobis Distance')
        
        # Plot 3: Number of outliers per configuration
        outlier_counts = [config['df']['is_outlier'].sum() for config in results.values()]
        config_names = [f'Config{i+1}\n({config["confidence"]:.0%})' 
                       for i, config in enumerate(results.values())]
        axes[1,0].bar(config_names, outlier_counts, color=['lightcoral', 'lightblue', 'lightgreen'])
        axes[1,0].set_ylabel('Number of Outliers Detected')
        axes[1,0].set_title('Outliers Detected per Configuration')
        
        # Plot 4: Your ellipse plot (modified)
        self.create_ellipse_plot(axes[1,1], results['config1']['cutoff'])
        
        plt.tight_layout()
        plt.show()
    
    def create_ellipse_plot(self, ax, cutoff):
        """Your original ellipse plot code, adapted"""
        data_array = self.df_analysis.to_numpy()
        
        # Use first two features for 2D visualization (like your code)
        x_data = data_array[:, 0]  # cloudiness
        y_data = data_array[:, 1]  # rain
        
        # Calculate correlation (your approach)
        pearson = self.cov_matrix[0, 1] / np.sqrt(self.cov_matrix[0, 0] * self.cov_matrix[1, 1])
        
        # Eigenvalues and eigenvectors (your approach)
        lambda_, v = np.linalg.eig(self.cov_matrix[:2, :2])
        lambda_ = np.sqrt(lambda_)
        
        # Create ellipse (your approach)
        from matplotlib.patches import Ellipse
        ellipse = Ellipse(xy=(self.centerpoint[0], self.centerpoint[1]),
                         width=lambda_[0] * np.sqrt(cutoff) * 2, 
                         height=lambda_[1] * np.sqrt(cutoff) * 2,
                         angle=np.rad2deg(np.arccos(v[0, 0])), 
                         edgecolor='red', facecolor='lightblue', alpha=0.3)
        
        ax.add_patch(ellipse)
        ax.scatter(x_data, y_data, alpha=0.6)
        ax.set_xlabel('Cloudiness')
        ax.set_ylabel('Rain')
        ax.set_title('Mahalanobis Distance Ellipse (95% Confidence)')
        
        # Mark outliers
        config1_df = self.df_analysis.copy()
        config1_df['OLS'] = self.calculate_outlier_scores()
        outliers = config1_df[config1_df['OLS'] > cutoff]
        if len(outliers) > 0:
            ax.scatter(outliers['cloudiness'], outliers['rain'], 
                      color='red', s=50, label='Outliers')
            ax.legend()
    
    def assess_performance(self, results):
        """Assess how well the outlier detection technique worked"""
        print("\n" + "="*60)
        print("PERFORMANCE ASSESSMENT")
        print("="*60)
        
        print("Mahalanobis Distance Advantages:")
        print("✓ Accounts for correlations between weather variables")
        print("✓ Scale-invariant (handles different units automatically)")
        print("✓ Statistically sound for multivariate normal distributions")
        print("✓ Provides probabilistic interpretation via Chi-square distribution")
        
        print("\nConfiguration Comparison:")
        for config_name, config_data in results.items():
            df = config_data['df']
            cutoff = config_data['cutoff']
            confidence = config_data['confidence']
            
            num_outliers = df['is_outlier'].sum()
            percent_outliers = (num_outliers / len(df)) * 100
            
            print(f"\n{config_name}: {confidence:.0%} confidence level")
            print(f"  Outliers detected: {num_outliers} ({percent_outliers:.1f}% of data)")
            print(f"  Cutoff value: {cutoff:.4f}")
    
    def save_results(self, results):
        """Save the augmented datasets with OLS column"""
        for config_name, config_data in results.items():
            filename = f"HW2023_OLS_{config_name}.csv"
            config_data['df'].to_csv(filename, index=False)
            print(f"Saved: {filename}")

def main():
    """Main function to run the complete Mahalanobis outlier detection"""
    # Initialize detector
    detector = MahalanobisOutlierDetector()
    
    # Load and prepare data
    print("Loading and preparing Houston 2023 weather data...")
    df = detector.load_and_prepare_data('HW2023.csv')
    
    # Calculate Mahalanobis components
    print("\nCalculating Mahalanobis distance components...")
    cov_matrix, inv_cov_matrix, centerpoint = detector.calculate_mahalanobis_components()
    print(f"Covariance matrix shape: {cov_matrix.shape}")
    print(f"Center point: {centerpoint}")
    
    # Detect outliers with multiple configurations
    print("\nDetecting outliers with Mahalanobis distance...")
    results, distances = detector.detect_outliers_multiple_configs()
    
    # Analyze results
    print("\nAnalyzing results...")
    detector.analyze_results(results)
    
    # Visualize results
    print("\nCreating visualizations...")
    detector.visualize_results(results, distances)
    
    # Assess performance
    print("\nAssessing performance...")
    detector.assess_performance(results)
    
    # Save results
    print("\nSaving results...")
    detector.save_results(results)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()