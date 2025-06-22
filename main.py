# Import Libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from scipy.stats import pearsonr
from scipy.stats import shapiro
from scipy.stats import skew

from google.colab import drive
drive.mount('/content/drive')

# Load Dataset
data = pd.read_csv("/content/drive/MyDrive/Colab Notebooks/data_YesBank_StockPrices.csv2edited.csv")

# Dataset First Look
data

# Dataset Rows & Columns count
data.shape

# Dataset Info
data.info()

# Dataset Duplicate Value Count
data.duplicated()

# Missing Values/Null Values Count
data.isnull().sum()

# Dataset Columns
data.columns

# Dataset Describe
data.describe()

# Check Unique Values for each variable.
data.nunique()

# Chart - 1 visualization code (Box Plot)
plt.figure(figsize=(8,6))
sns.boxplot(data['Close'])
plt.title('Box Plot Of Closing Price')
plt.show()

# Chart - 2 visualization code (Dist Plot)
sns.distplot(data['Close'])

# Chart - 3 visualization code (Histogram)
sns.histplot(data["Close"])

# Chart - 4 visualization code (Scatter plot)
plt.scatter(data['Open'],data['Close'])
plt.xlabel('Open')
plt.ylabel('Close')
plt.title('Scatter Plot of Opening and Closing Prices')
plt.show()

# Correlation Heatmap visualization code
corelation = data.drop('Months', axis=1).corr()
sns.heatmap(corelation, annot=True)
plt.show()

# Pair Plot visualization code
sns.pairplot(data)
plt.show()

# Perform Statistical Test to obtain P-Value (paired t-test)
t_stat, p_value = ttest_rel(data['Open'], data['Close'])
print("T-statistic:", t_stat)
print("Two-tailed P-value:", p_value)

# Perform Statistical Test to obtain P-Value (Pearson correlation coefficient test)
columns = ['Open', 'High', 'Low', 'Close']

for i in range(len(columns)):
    for j in range(i+1, len(columns)):
        col1, col2 = columns[i], columns[j]
        r, p = pearsonr(data[col1], data[col2])
        print(f"{col1} vs {col2} → Correlation: {r:.3f}, P-value: {p:.5f}")

# Perform Statistical Test to obtain P-Value (shapiro wilk test and skewness coefficient test)
stat, p = shapiro(data['Close'])

print("Shapiro-Wilk Test Statistic:", stat)
print("P-value:", p)

skew_val = skew(data['Close'])
print("Skewness Coefficient:", skew_val)

# Handling Missing Values & Missing Value Imputation

# Check missing values
data.isnull().sum()

# Fill with median
data.fillna(data.median(numeric_only=True), inplace=True)

# Handling Outliers & Outlier treatments

# Using IQR for 'Close'
Q1 = data['Close'].quantile(0.25)
Q3 = data['Close'].quantile(0.75)
IQR = Q3 - Q1

# Filtering outliers and creating a copy to avoid SettingWithCopyWarning
data = data[(data['Close'] >= Q1 - 1.5 * IQR) & (data['Close'] <= Q3 + 1.5 * IQR)].copy()

# Encode your categorical columns
data['Months'] = pd.to_datetime(data['Months'], format='%b-%y')
data['Month'] = data['Months'].dt.month
data = pd.get_dummies(data, columns=['Month'])

# Scaling your data
from sklearn.preprocessing import StandardScaler

# Example: Scaling Open, High, Low, Close
features = ['Open', 'High', 'Low', 'Close']
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[features])

# Convert back to DataFrame
import pandas as pd
data_scaled = pd.DataFrame(data_scaled, columns=[f'{col}_scaled' for col in features])

data_scaled.head()

# DImensionality Reduction (If needed)
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Step 1: Standardize the data
features = ['Open', 'High', 'Low', 'Close']
X = data[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 2: Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

# Step 3: View explained variance
print("Explained variance ratio:", pca.explained_variance_ratio_)

# Split your data to train and test. Choose Splitting ratio wisely.

import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming my DataFrame is already sorted by date
# Features and target
features = ['Open', 'High', 'Low']   # Using these to predict 'Close'
target = 'Close'

X = data[features]
y = data[target]

# Split data: e.g., 80% for training, 20% for testing
split_index = int(len(data) * 0.8)

X_train = X.iloc[:split_index]
X_test  = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test  = y.iloc[split_index:]

# ML Model - 1 Implementation
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Fit the Algorithm
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the model
y_pred = model.predict(X_test)

# Visualizing evaluation Metric Score chart
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# ML Model - 1 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

# Define the model
lr = LinearRegression()

# Define hyperparameter grid
param_grid = {
    'fit_intercept': [True, False],
    'positive': [True, False]  # Ensures coefficients stay positive (newer versions)
}

# Apply GridSearchCV
grid_search = GridSearchCV(estimator=lr, param_grid=param_grid,
                           cv=5, scoring='r2', n_jobs=-1, verbose=1)

# Fit the model on training data
grid_search.fit(X_train, y_train)

# Best model after hyperparameter tuning
best_lr = grid_search.best_estimator_

# Predict using the best model
y_pred = best_lr.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

# Display results
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# Visualizing evaluation Metric Score chart
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Initialize the model
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)

# Predict on test set
y_pred = rf.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")

# ML Model - 1 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

# GridSearchCV setup
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1
)

# Fit model
grid_search.fit(X_train, y_train)

# Best estimator
best_rf = grid_search.best_estimator_
y_pred_tuned = best_rf.predict(X_test)

# New evaluation metrics
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = mse_tuned ** 0.5
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"MAE: {mae_tuned:.2f}")
print(f"MSE: {mse_tuned:.2f}")
print(f"RMSE: {rmse_tuned:.2f}")
print(f"R² Score: {r2_tuned:.2f}")

# Visualizing evaluation Metric Score chart

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Base model
xgb = XGBRegressor(random_state=42, verbosity=0)
xgb.fit(X_train, y_train)

# Predictions
y_pred = xgb.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.3f}")

# ML Model - 3 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)

from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

# Grid Search with 5-fold CV
grid_search = GridSearchCV(
    estimator=XGBRegressor(random_state=42, verbosity=0),
    param_grid=param_grid,
    cv=5,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_xgb = grid_search.best_estimator_
y_pred_tuned = best_xgb.predict(X_test)

# Evaluation after tuning
mae_tuned = mean_absolute_error(y_test, y_pred_tuned)
mse_tuned = mean_squared_error(y_test, y_pred_tuned)
rmse_tuned = mse_tuned ** 0.5
r2_tuned = r2_score(y_test, y_pred_tuned)

print(f"MAE: {mae_tuned:.2f}")
print(f"MSE: {mse_tuned:.2f}")
print(f"RMSE: {rmse_tuned:.2f}")
print(f"R² Score: {r2_tuned:.3f}")

