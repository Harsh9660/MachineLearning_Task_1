import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder


def load_data(path):
    """Loads dataset from a given path"""
    return pd.read_csv(path)

df = load_data('StudentPerformance.csv')


print("\n🔹 Statistical Summary:")
print(df.describe())


numeric_df = df.select_dtypes(include=['int64', 'float64'])
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap (Numeric Columns Only)")
plt.show()


df.fillna(df.median(numeric_only=True), inplace=True)


for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])


scaler = StandardScaler()
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = scaler.fit_transform(df[num_cols])


target_col = 'Performance Index'  

if target_col not in df.columns:
    raise ValueError(f" Target column '{target_col}' not found in dataset!")

X = df.drop(target_col, axis=1)
y = df[target_col]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n Training set shape:", X_train.shape)
print(" Test set shape:", X_test.shape)
print("\n Missing values in dataset:\n", df.isnull().sum())


def load_data(path):
    """Loads dataset from a given path"""
    return pd.read_csv(path)

def preprocess_data(data, target_col):
    """Preprocess dataset: handle NaNs, encode, scale, and split"""
    data = data.copy()
    
   
    data.fillna(data.median(numeric_only=True), inplace=True)
    
    
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = LabelEncoder().fit_transform(data[col])
    
    
    num_cols = data.select_dtypes(include=['int64', 'float64']).columns
    data[num_cols] = StandardScaler().fit_transform(data[num_cols])
    
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    return X, y
