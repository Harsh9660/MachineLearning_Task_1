from datetime import date
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

df = pd.read_csv('Zara_sales_EDA.csv', sep=';')
target_col = 'Sales Volume'

print("\n First 5 rows:")
print(df.head())

print("\n Dataset Info:")
print(df.info())

print("\n Statistical Summary:")
print(df.describe())

print("\n Missing values in dataset:\n", df.isnull().sum())

for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

df.fillna(df.median(numeric_only=True), inplace=True)
scaler = StandardScaler()
num_cols = df.select_dtypes(include=[np.number]).columns
df[num_cols] = scaler.fit_transform(df[num_cols])

X = df.drop(target_col, axis=1)
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\n Dataset shape:", df.shape)
print(" Train set shape:", X_train.shape)
print(" Test set shape:", X_test.shape)

if 'name' in df.columns:
    name_sales = df.groupby('name')[target_col].sum().reset_index()
    top_names = name_sales.nlargest(10, target_col)
    fig1 = px.bar(top_names, x='name', y=target_col, color='name', title='Top 10 Products by Sales Volume', text_auto=True)
    fig1.update_layout(xaxis_title="Product Name", yaxis_title="Sales Volume", title_x=0.5, showlegend=False)
    fig1.show()

fig2 = px.histogram(df, x=target_col, nbins=20, title='Distribution of Sales Volume', color_discrete_sequence=['teal'])
fig2.update_layout(title_x=0.5)
fig2.show()

corr = df.corr(numeric_only=True)
fig3 = px.imshow(corr, text_auto=True, color_continuous_scale='Viridis', title='Correlation Heatmap')
fig3.update_layout(title_x=0.5)
fig3.show()

if 'Quantity' in df.columns:
    fig4 = px.scatter(df, x='Quantity', y=target_col, color='name', title='Quantity vs Sales Volume', hover_data=['name'])
    fig4.update_traces(marker=dict(size=10, opacity=0.7))
    fig4.update_layout(title_x=0.5)
    fig4.show()

if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    fig5 = px.line(df.sort_values('Date'), x='Date', y=target_col, title='Sales Trend Over Time', markers=True, color_discrete_sequence=['orange'])
    fig5.update_layout(title_x=0.5)
    fig5.show()

sizes = [len(X_train), len(X_test)]
labels = ['Train', 'Test']
fig6 = px.pie(names=labels, values=sizes, title='Train vs Test Data Split', hole=0.4)
fig6.update_traces(textinfo='label+percent')
fig6.update_layout(title_x=0.5)
fig6.show()

train_labels = pd.DataFrame({'Set': 'Train', 'Target': y_train})
test_labels = pd.DataFrame({'Set': 'Test', 'Target': y_test})
split_df = pd.concat([train_labels, test_labels])
fig7 = px.histogram(split_df, x='Target', color='Set', barmode='group', title='Target Distribution in Train vs Test Sets')
fig7.update_layout(title_x=0.5)
fig7.show()
