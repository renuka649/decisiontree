# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:18:02 2024

@author: renuk
"""


'''
Business Understanding:
The provided dataset appears to contain information related to individuals,
 including their marital status, taxable income, city population, work
 experience, and whether they live in an urban area. The business context 
 seems to revolve around assessing the risk of fraud, possibly for financial
 or insurance-related purposes.

Business Objective:
The primary objective is likely to develop a fraud detection system that
 can identify individuals who may be engaging in fraudulent activities based
 on the given attributes. This could be crucial for businesses or organizations
 to minimize financial losses, ensure compliance, and maintain the integrity 
 of their operations.
'''

import pandas as pd

# Load the dataset
fraud_data = pd.read_csv('Fraud_check.csv.xls')

# Display the first few rows of the dataframe
print(fraud_data.head())

# Create a dataframe from the dataset
df = pd.DataFrame(fraud_data)

# Rename columns for clarity
df.rename(columns={'Undergrad': 'Education', 
                   'Marital.Status': 'MaritalStatus', 
                   'Taxable.Income': 'TaxableIncome', 
                   'City.Population': 'CityPopulation',
                   'Work.Experience': 'WorkExperience', 
                   'Urban': 'IsUrban'}, inplace=True)

# Display the first 12 rows of the dataframe
print(df[:12])

# One-hot encode the categorical columns
df = pd.get_dummies(df, columns=['Education', 'MaritalStatus', 'IsUrban'], drop_first=True)

# Separate features (X) and target variable (y)
X = df.drop('TaxableIncome', axis=1)
y = df['TaxableIncome']

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train a Decision Tree Classifier model
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Evaluate the model
model_score = model.score(X_test, y_test)
print("Model Accuracy:", model_score)

# Predict using the trained model
y_predicted = model.predict(X_test)

# Compute confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:\n", cm)

# Visualize confusion matrix
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10, 7))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')
plt.show()