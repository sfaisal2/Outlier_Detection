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

## Finding ellipse dimensions 
pearson = cov[0, 1]/np.sqrt(cov[0, 0] * cov[1, 1])
ell_radius_x = np.sqrt(1 + pearson)
ell_radius_y = np.sqrt(1 - pearson)
lambda_, v = np.linalg.eig(cov)
lambda_ = np.sqrt(lambda_)

# Ellipse patch
ellipse = patches.Ellipse(xy=(centerpoint[0], centerpoint[1]),
                  width=lambda_[0]*np.sqrt(cutoff)*2, height=lambda_[1]*np.sqrt(cutoff)*2,
                  angle=np.rad2deg(np.arccos(v[0, 0])), edgecolor='#fab1a0')
ellipse.set_facecolor('#0984e3')
ellipse.set_alpha(0.5)
fig = plt.figure()
ax = plt.subplot()
ax.add_artist(ellipse)
plt.scatter(df[: , 0], df[ : , 1])
plt.show()