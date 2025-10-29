import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import kagglehub


# df = pd.read_csv('zara-sales-for-eda.csv')
path = kagglehub.dataset_download("marixe/zara-sales-for-eda")

print("Path to dataset files:", path)

df = pd.read_csv(path)

print("\n🔹 First 5 rows:")
print(df.head())

print("\n🔹 Dataset Info:")
print(df.info())

print("\n🔹 Statistical Summary:")
print(df.describe())


numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Numeric Columns Only)")
plt.show()