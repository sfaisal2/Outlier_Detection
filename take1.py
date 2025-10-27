import pandas as pd
import numpy as np
from scipy.stats import chi2
from matplotlib import patches
import matplotlib.pyplot as plt

#import dataset and clean it 
df = pd.read_csv('HW2023.csv', sep = ",", decimal ='.')
df.head()

df = df.iloc[:, 1:]
df = df.dropna()
df = df.to_numpy()

#covariance matirx
cov  = np.cov(df , rowvar=False)

#covariance matrix power of -1
cov_pm1 = np.linalg.matrix_power(cov, -1)

#center point
centerpoint = np.mean(df , axis=0)

#distances between center point and each obs.
distances = []
for i, val in enumerate(df):
      p1 = val
      p2 = centerpoint
      distance = (p1-p2).T.dot(cov_pm1).dot(p1-p2)
      distances.append(distance)
distances = np.array(distances)

#cutoff (threshold) value from Chi-Sqaure Distribution for detecting outliers 
cutoff = chi2.ppf(0.95, df.shape[1])

#index of outliers
outlierIndexes = np.where(distances > cutoff )

print('--- Index of Outliers ----')
print(outlierIndexes)
# array([24, 35, 67, 81])

print('--- Observations found as outlier -----')
print(df[ distances > cutoff , :])
# [[115.  79.], [135.  84.], [122.  89.], [168.  81.]]