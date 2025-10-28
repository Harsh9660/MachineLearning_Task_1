import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import StandardScaler


df = pd.read_csv("StudentPerformance.csv")
df.head()
df.info()

df.drop_duplicates(inplace=True)
df.dropna(inplace=True)

# df.drop(['student_id'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)
df.head()
