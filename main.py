import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


dataset = pd.read_csv('E:\\Web\\IA\\03_Energy-Efficiency_Dataset.csv')
print("HOLA")

print('ENERGY EFFICIENCY DATASET - HEAD: \n', dataset.head(5))

print('ENERGY EFFICIENCY DATASET - SHAPE: ', dataset.shape)

print('ENERGY EFFICIENCY DATASET - INFO: \n', dataset.info())

print('ENERGY EFFICIENCY DATASET - STATISTICS: \n', dataset.describe())

cormat = dataset.corr()
print('ENERGY EFFICIENCY DATASET - CORRELATION MATRIX: \n', round(cormat,2))
sns.heatmap(cormat);


sns.pairplot(dataset[[
'Relative_Compactness',
'Surface_Area',
'Wall_Area',
'Roof_Area',
'Overall_Height',
'Orientation',
'Glazing_Area',
'Glazing_Area_Distribution',
'Heating_Load',
'Cooling_Load',
 ]]);

