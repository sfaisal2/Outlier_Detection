import pandas as pd
import numpy as np
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
        self.df = pd.read_csv(file_path, sep=",", decimal='.')
        
        #categorical column for reference
        if 'date' in self.df.columns:
            self.dates = self.df['date'].copy()
        
        #select numeric columns
        self.numeric_columns = ['temp_max', 'humidity', 'visibility', 'cloudiness', 'wind_speed', 'rain']
        self.df_analysis = self.df[self.numeric_columns].copy()
        
        # Handle missing values
        self.df_analysis['rain'] = self.df_analysis['rain'].fillna(0)
        self.df_analysis = self.df_analysis.dropna()
        
        return self.df_analysis
    
    def calculate_mahalanobis_components(self):
        data_array = self.df_analysis.to_numpy()
        
        self.cov_matrix = np.cov(data_array, rowvar=False)
        self.inv_cov_matrix = np.linalg.pinv(self.cov_matrix)
        self.centerpoint = np.mean(data_array, axis=0)
        
        return self.cov_matrix, self.inv_cov_matrix, self.centerpoint
    
    def calculate_mahalanobis_distance(self, point1, point2):
        diff = point1 - point2
        return np.sqrt(diff.T.dot(self.inv_cov_matrix).dot(diff))
    
    def calculate_outlier_scores(self, method='centroid', alpha=1.0):
        """
        Parameters:
        - method: 'centroid', 'neighbors', 'robust'
        - alpha: weight parameter for robust method
        """
        data_array = self.df_analysis.to_numpy()
        distances = []
        
        if method == 'centroid':
            #basic Mahalanobis distance from center
            for val in data_array:
                distance = self.calculate_mahalanobis_distance(val, self.centerpoint)
                distances.append(distance)
                
        elif method == 'neighbors':
            #average distance to all other points
            for i, val in enumerate(data_array):
                point_distances = []
                for j, other_val in enumerate(data_array):
                    if i != j:
                        distance = self.calculate_mahalanobis_distance(val, other_val)
                        point_distances.append(distance)
                avg_distance = np.mean(point_distances)
                distances.append(avg_distance)
                
        elif method == 'robust':
            # Weighted combination of centroid and neighbor distances
            for i, val in enumerate(data_array):
                # Distance from centroid
                centroid_dist = self.calculate_mahalanobis_distance(val, self.centerpoint)
                
                # Average distance to other points
                neighbor_dists = []
                for j, other_val in enumerate(data_array):
                    if i != j:
                        distance = self.calculate_mahalanobis_distance(val, other_val)
                        neighbor_dists.append(distance)
                avg_neighbor_dist = np.mean(neighbor_dists)
                
                # Combined score
                combined = alpha * centroid_dist + (1 - alpha) * avg_neighbor_dist
                distances.append(combined)
        
        return np.array(distances)
    
    def apply_detection_technique(self):
        results = {}
        
        # Configuration 1
        ols1 = self.calculate_outlier_scores(method='centroid')
        df1 = self.df_analysis.copy()
        df1['OLS'] = ols1
        if hasattr(self, 'dates'):
            df1['date'] = self.dates.iloc[:len(df1)]
        results['config1'] = {'df': df1, 'method': 'centroid', 'params': 'basic'}
        
        # Configuration 2
        ols2 = self.calculate_outlier_scores(method='neighbors')
        df2 = self.df_analysis.copy()
        df2['OLS'] = ols2
        if hasattr(self, 'dates'):
            df2['date'] = self.dates.iloc[:len(df2)]
        results['config2'] = {'df': df2, 'method': 'neighbors', 'params': 'average_all'}
        
        # Configuration 3
        ols3 = self.calculate_outlier_scores(method='robust', alpha=0.7)
        df3 = self.df_analysis.copy()
        df3['OLS'] = ols3
        if hasattr(self, 'dates'):
            df3['date'] = self.dates.iloc[:len(df3)]
        results['config3'] = {'df': df3, 'method': 'robust', 'params': 'alpha=0.7'}
        
        return results
    
    def analyze_results(self, results):    
      for config_name, config_data in results.items():
            df = config_data['df']
            method = config_data['method']
            
            # Sort by OLS (descending)
            df_sorted = df.sort_values('OLS', ascending=False)
            
            print(f"\n{config_name.upper()} - Method: {method}")
            print("-" * 40)
            
            # Top 4 examples (highest OLS - potential outliers)
            print("TOP 4 DATES (Highest OLS - Potential Outliers):")
            top_dates = []
            for i in range(min(4, len(df_sorted))):
                  row = df_sorted.iloc[i]
                  if 'date' in row:
                        top_dates.append(f"{row['date']} (OLS: {row['OLS']:.4f})")
            
            for i, date_info in enumerate(top_dates, 1):
                  print(f"Rank {i}: {date_info}")
            
            # Bottom example (lowest OLS - most normal)
            bottom_row = df_sorted.iloc[-1]
            bottom_date = bottom_row['date'] if 'date' in bottom_row else "N/A"
            print(f"\nMOST NORMAL DATE (Lowest OLS):")
            print(f"{bottom_date} (OLS: {bottom_row['OLS']:.4f})")
    
    def save_results(self, results):
        for config_name, config_data in results.items():
            df = config_data['df']
            df_sorted = df.sort_values('OLS', ascending=False)
            filename = f"HW2023_OLS_{config_name}.csv"
            df_sorted.to_csv(filename, index=False)

def main():
    #detector
    detector = MahalanobisOutlierDetector()
    
    # Load and prepare data
    df = detector.load_and_prepare_data('HW2023.csv')
    
    # Calculate Mahalanobis components
    detector.calculate_mahalanobis_components()
    
    # Apply detection technique with 3 configurations
    results = detector.apply_detection_technique()
    
    # Analyze results
    detector.analyze_results(results)
    
    # Save results
    detector.save_results(results)

if __name__ == "__main__":
    main()