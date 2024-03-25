# Add lag features to df
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import platform
import sys
import statsmodels.api  as sm
import tensorflow       as tf
import tensorflow.keras as keras
import pmdarima as pm
from pmdarima import utils
from pmdarima import arima
from pmdarima import model_selection
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from statsmodels.tsa.seasonal import STL
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
file_path1 = "0313data.csv"
df = pd.read_csv(file_path1)
df.set_index("DateTime", inplace=True)
df.index.name = 'DateTime'
df['co2_lag'] = df['co2'].diff()
df = df.dropna()

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

X = df[['co2', 'sensor', 'co2_lag']]
y = df['y']
X_train_Dt, X_test_Dt, y_train_Dt, y_test_Dt = train_test_split(X, y, test_size=0.2, random_state=42)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_Dt, y_train_Dt)

cv_scores = cross_val_score(dt_model, X, y, cv=5)

print("Cross-validation scores:", cv_scores)
print("Average cross-validation score:", cv_scores.mean())

importances = dt_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45)
plt.xlim([-1, X.shape[1]])
plt.savefig('feature importance_0313data+lag.png')
plt.show()

from sklearn.tree import plot_tree
plt.figure(figsize=(12, 8), dpi=600)
plot_tree(dt_model, filled=True, feature_names=X.columns, class_names=['No', 'Yes'])
plt.savefig('DecisionTree_0313data+lag.png')
plt.show()

# Random Forest
from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier()

scores = cross_val_score(random_forest_model, X, y, cv=5)

print("Cross-Validation Scores:", scores)
print("Mean Score:", scores.mean())

# Performance of logistic regression when a penalty term is added (type of penalty term and coefficients are varied accordingly)
from sklearn.linear_model import LogisticRegression

log_models = {'Lasso': LogisticRegression(penalty='l1', solver='liblinear'),
          'Ridge': LogisticRegression(penalty='l2')}

# Evaluate performance by varying the strength of penalties (C)
for name, model in log_models.items():
    for C in [0.001, 0.01, 0.1, 1, 10]:
        model.C = C
        scores = cross_val_score(model, X, y, cv=5)
        print(f"{name} (C={C}): Mean Score:", np.mean(scores))
