# 1. Importing Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D

# 2. Load Dataset
data = pd.read_csv("C:\\Users\\abhin\\OneDrive\\Desktop\\project\\i have workig dataset.csv")

# 3. Data Cleaning
# Drop unnecessary columns
if 'LSTREET2' in data.columns:
    data.drop(columns=['LSTREET2'], inplace=True)

# Fill missing numerical values with mean
numeric_cols = data.select_dtypes(include=[np.number]).columns
for column in numeric_cols:
    data[column].fillna(data[column].mean(), inplace=True)

# Fill missing categorical values with mode
categorical_cols = data.select_dtypes(include=['object']).columns
for column in categorical_cols:
    data[column].fillna(data[column].mode()[0], inplace=True)

# 4. Descriptive Statistics
print("\nBasic Dataset Information:")
print(data.info())
print("\nStatistical Summary:")
print(data.describe())

# 5. Visualization Section
# Bar Chart - School Level
plt.figure(figsize=(10, 6))
sns.countplot(x='SCHOOL_LEVEL', data=data)
plt.title('Distribution of School Levels')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Histogram - Total Enrollments
plt.figure(figsize=(8, 5))
sns.histplot(data['TOTAL'], kde=True, bins=30, color='skyblue')
plt.title('Histogram of Total Enrollment')
plt.xlabel('Total Students')
plt.ylabel('Frequency')
plt.show()

# Pie Chart - Charter vs Non-charter
plt.figure(figsize=(6, 6))
charter_counts = data['CHARTER_TEXT'].value_counts()
plt.pie(charter_counts, labels=charter_counts.index, autopct='%1.1f%%', colors=['#ff9999','#66b3ff'])
plt.title('Charter vs Non-Charter Schools')
plt.show()

# Boxplot - Student to Teacher Ratio
plt.figure(figsize=(8, 5))
sns.boxplot(y=data['STUTERATIO'], color='lightgreen')
plt.title('Boxplot of Student-Teacher Ratio')
plt.ylabel('Ratio')
plt.show()

# Line Graph - Total Enrollment of Sample Schools
sample_data = data.sample(10).sort_values('TOTAL')
plt.figure(figsize=(10, 6))
plt.plot(sample_data['SCH_NAME'], sample_data['TOTAL'], marker='o')
plt.title('Total Enrollment in 10 Sample Schools')
plt.xticks(rotation=90)
plt.ylabel('Total Enrollment')
plt.tight_layout()
plt.show()

# Heatmap - Correlation Matrix
plt.figure(figsize=(15, 10))
corr_matrix = data.corr(numeric_only=True)
sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# 6. Correlation and Linear Regression Model
# Predicting total enrollment using male and female enrollment
features = ['TOTMENROL', 'TOTFENROL']
target = 'TOTAL'

X = data[features]
y = data[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Model Evaluation
print("\nLinear Regression Results:")
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared Score:", r2_score(y_test, y_pred))

# Regression Plot
plt.figure(figsize=(8, 5))
sns.regplot(x=data['TOTMENROL'], y=data['TOTAL'], line_kws={"color":"red"})
plt.title('Regression Line: Male Enrollment vs Total Enrollment')
plt.xlabel('Male Enrollment')
plt.ylabel('Total Enrollment')
plt.show()

# 7. 3D Scatter Plot - Male vs Female Enrollment vs Total Enrollment
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data['TOTMENROL'], data['TOTFENROL'], data['TOTAL'], c=data['TOTAL'], cmap='viridis')
ax.set_xlabel('Male Enrollment')
ax.set_ylabel('Female Enrollment')
ax.set_zlabel('Total Enrollment')
ax.set_title('3D Scatter Plot: Male vs Female vs Total Enrollment')
plt.show()
