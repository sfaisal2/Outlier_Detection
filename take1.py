import pandas as pd
import numpy as np
from scipy.stats import chi2
from matplotlib import patches
import matplotlib.pyplot as plt

#import dataset and clean it 
df = pd.read_csv('HW2023.csv', sep = ",", decimal ='.')
df.head()
