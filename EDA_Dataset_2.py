from datetime import date
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

path = 'Zara_sales_data.csv'
"""Preprocesses the dataset by handling missing values, encoding categorical variables, and scaling numerical features."""
df = pd.read_csv(path)
target_col = 'Sales Amount'

print("\n First 5 rows:")
print(df.head())

print("\n Dataset Info:")
print(df.info())

print("\n Statistical Summary:")
print(df.describe())

print("\n Missing values in dataset:\n", df.isnull().sum())

sns.countplot(data=df, x='Product Category')
plt.title('Product Category Distribution')
plt.show()