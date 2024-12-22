#**Task-1**
"""

!pip install ucimlrepo

# importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import svm
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVR
from sklearn.decomposition import PCA

from ucimlrepo import fetch_ucirepo

# fetch dataset
real_estate_valuation = fetch_ucirepo(id=477)

# data (as pandas dataframes)
X = real_estate_valuation.data.features
y = real_estate_valuation.data.targets

# metadata
print(real_estate_valuation.metadata)

# variable information
print(real_estate_valuation.variables)

# Scale the features to bring them to a similar scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Plot all features against the target with different colors
plt.figure(figsize=(10, 8))

# Assign colors to each feature
colors = ['b', 'g', 'r', 'c', 'm', 'y']

for i, feature in enumerate(X.columns):
    plt.scatter(X_scaled[:, i], y, alpha=0.5, color=colors[i], label=feature)

# Add labels, title, and legend
plt.xlabel('Scaled Features')
plt.ylabel('House Price per Unit Area (10000 NT$/Ping)')
plt.title('Scatter Plot: All Features vs House Price per Unit Area')
plt.legend(loc='upper right')
plt.show()

plt.figure(figsize=(6, 3))
sns.heatmap(X.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

df=pd.DataFrame(X)
df['target']=y
display(df)

"""# **Task-2**"""

print(df.isnull().sum())  # Checks for NaN values in each column
print(np.isinf(df).sum())  # Checks for infinite values in each column

df.duplicated().sum()

df.info()

"""# **Task-3**"""

# Splitting the dataset
X=df.drop('target',axis=1)
y=df['target']
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# shape of all sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_val:", X_val.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_val:", y_val.shape)
print("Shape of y_test:", y_test.shape)

# Define hyperparameter grid
param_grid = {
    'svr__kernel': ['poly', 'rbf'],
    'svr__degree': [1, 2, 3, 4],
    'svr__gamma': ['scale', 'auto'],
    'svr__C': [1, 10, 100]
}

print(param_grid)

"""# **Custom Implimentation of Grid search**"""

def custom_grid_search(param_grid, model, X_train, y_train, X_val, y_val):
    best_param = None
    best_mse = float('inf')
    results = []

    for param in param_grid:
        # Set individual parameters correctly, not as lists
        model.set_params(**param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        results.append((param, mse))
        if mse < best_mse:
            best_mse = mse
            best_param = param

    return best_param, results

# Polynomial Kernel SVR
poly_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
    ('scaler', StandardScaler()),  # Standardize the features
    ('svr', SVR(kernel='poly'))
])
print(poly_pipeline)

# RBF Kernel SVR
rbf_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Impute missing values with the mean
    ('scaler', StandardScaler()),  # Standardize the features
    ('svr', SVR(kernel='rbf'))
])

print(rbf_pipeline)

"""# **Task-4**"""

# Define the range for polynomial degrees correctly (each degree is checked individually)
poly_param_grid = [{'svr__degree': d} for d in [2, 3, 4, 5]]

# Call custom grid search function
best_poly_param, poly_results = custom_grid_search(poly_param_grid, poly_pipeline, X_train, y_train, X_val, y_val)

# Define the range for gamma (each gamma value is tested individually)
rbf_param_grid = [{'svr__gamma': g} for g in [0.001, 0.01, 0.1, 1]]

# Call custom grid search function
best_rbf_param, rbf_results = custom_grid_search(rbf_param_grid, rbf_pipeline, X_train, y_train, X_val, y_val)

"""# **Task-6**"""

# Plot MSE vs Degree
degrees = [param['svr__degree'] for param, _ in poly_results]
mse_values_poly = [mse for _, mse in poly_results]

plt.plot(degrees, mse_values_poly)
plt.xlabel('Degree')
plt.ylabel('MSE')
plt.title('MSE vs Degree (Polynomial Kernel)')
plt.show()

"""# **Task-7**"""

# Plot MSE vs Gamma
gammas = [param['svr__gamma'] for param, _ in rbf_results]
mse_values_rbf = [mse for _, mse in rbf_results]

plt.plot(gammas, mse_values_rbf)
plt.xlabel('Gamma')
plt.ylabel('MSE')
plt.title('MSE vs Gamma (RBF Kernel)')
plt.show()

print(f"Best parameters for Polynomial Kernel: {best_poly_param}")
print(f"Best parameters for RBF Kernel: {best_rbf_param}")

"""# **Task-8**"""

# Evaluate the Best Model on the Test Set
# Select the best model (compare results from Polynomial and RBF)
best_model = poly_pipeline if min(mse_values_poly) < min(mse_values_rbf) else rbf_pipeline
best_model.set_params(**(best_poly_param if best_model == poly_pipeline else best_rbf_param))
best_model.fit(X_train, y_train)

y_test_pred = best_model.predict(X_test)

# Find the minimum MSE for Polynomial and RBF kernels
min_mse_poly = min(mse_values_poly)
min_mse_rbf = min(mse_values_rbf)

# Compare the MSE values to select the best model
if min_mse_poly < min_mse_rbf:
    print("The best model is the SVR with Polynomial Kernel.")
    best_model_name = "Polynomial Kernel SVR"
    best_mse = min_mse_poly
    best_param = best_poly_param
else:
    print("The best model is the SVR with RBF Kernel.")
    best_model_name = "RBF Kernel SVR"
    best_mse = min_mse_rbf
    best_param = best_rbf_param

print(f"Best MSE on validation set: {best_mse}")
print(f"Best parameters: {best_param}")

# Plot scatter of Predictions vs Ground Truth
plt.scatter(y_test, y_test_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Ground Truth')
plt.ylabel('Predictions')
plt.title('Predictions vs Ground Truth')
plt.show()

# Evaluate the model
mae = mean_absolute_error(y_test, y_test_pred)
mse = mean_squared_error(y_test, y_test_pred)
r2 = r2_score(y_test, y_test_pred)

print(f'MAE: {mae:.3f}')
print(f'MSE: {mse:.3f}')
print(f'R2: {r2:.3f}')
