import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
import os
print(os.listdir())
import warnings
warnings.filterwarnings('ignore')

from google.colab import drive
drive.mount('/content/drive')


dataset = pd.read_csv("/content/Weather_Data.csv")


type(dataset)


dataset.shape


dataset.head(5)


dataset.sample(5)


dataset.describe()



dataset.info()


info = ["RainToday","1: yes, 0: no",
"resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"]
for i in range(len(info)):
  pass # You likely want to do something with info here.
["oldpeak = ST depression induced by exercise relative to rest",
"number of major vessels (0-3) colored by flourosopy",
"chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect",
" maximum heart rate achieved","exercise induced angina",
"the slope of the peak exercise ST segment"]
print(dataset.columns[i]+":\t\t\t"+info[i])


# Try printing all column names to verify the correct name
print(dataset.columns)

# Assuming the target variable is present and there was just a typo
dataset["Date"].describe()

# If the target variable is not present, you will need to investigate further
# and determine how to add it to your DataFrame




dataset["RainToday"].unique()



import pandas as pd

# Assuming 'dataset' is your DataFrame
# Convert the 'Date' column to datetime objects
dataset['Date'] = pd.to_datetime(dataset['Date'])

# Convert 'RainToday' to numerical representation (0 and 1)
dataset['RainToday'] = dataset['RainToday'].map({'Yes': 1, 'No': 0})

# Now calculate correlations
# (excluding 'Date' as it's datetime, not numeric for correlation)
# Select numeric features and drop 'Date' if it's present
numeric_features = dataset.select_dtypes(include=['number'])
if 'Date' in numeric_features.columns:
    numeric_features = numeric_features.drop(columns=['Date'])

correlations = numeric_features.corr()['Temp9am'].abs().sort_values(ascending=False)
print(correlations)





# Data Cleaning
# 1. Handling Missing Values
# Replace missing values with the mean or median (depending on the distribution)
for column in dataset.columns:
    if dataset[column].isnull().any():
        if pd.api.types.is_numeric_dtype(dataset[column]):
            dataset[column].fillna(dataset[column].mean(), inplace=True)
        else:
            dataset[column].fillna(dataset[column].mode()[0], inplace=True)

# 2. Handling Outliers
# Remove or replace outliers based on domain knowledge or statistical methods (e.g., IQR)
# For example, if 'MaxTemp' has outliers:
# Q1 = dataset['MaxTemp'].quantile(0.25)
# Q3 = dataset['MaxTemp'].quantile(0.75)
# IQR = Q3 - Q1
# lower_bound = Q1 - 1.5 * IQR
# upper_bound = Q3 + 1.5 * IQR
# dataset = dataset[(dataset['MaxTemp'] >= lower_bound) & (dataset['MaxTemp'] <= upper_bound)]

# 3. Handling Inconsistent Data
# Check for inconsistencies in categorical features and correct them
# Example: If 'RainToday' has values like 'yes', 'Yes', 'y', etc., convert them to a uniform format
# dataset['RainToday'] = dataset['RainToday'].str.lower().replace({'yes': 1, 'no': 0})

# 4. Handling Duplicates
# Remove duplicate rows
dataset.drop_duplicates(inplace=True)

# 5. Feature Engineering (if applicable)
# Create new features that might be useful for your analysis
# Example: Combine 'MinTemp' and 'MaxTemp' to create 'TemperatureRange'
# dataset['TemperatureRange'] = dataset['MaxTemp'] - dataset['MinTemp']

# After applying data cleaning techniques
dataset.info()

info = ["RainToday","1: yes, 0: no",
"resting blood pressure"," serum cholestoral in mg/dl","fasting blood sugar > 120 mg/dl","resting electrocardiographic results (values 0,1,2)"]
for i in range(len(info)):
  pass # You likely want to do something with info here.
["oldpeak = ST depression induced by exercise relative to rest",
"number of major vessels (0-3) colored by flourosopy",
"chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","thal: 3 = normal; 6 = fixed defect; 7 = reversable defect",
" maximum heart rate achieved","exercise induced angina",
"the slope of the peak exercise ST segment"]
print(dataset.columns[i]+":\t\t\t"+info[i])
# Try printing all column names to verify the correct name
print(dataset.columns)

# Assuming the target variable is present and there was just a typo
dataset["Date"].describe()

# If the target variable is not present, you will need to investigate further
# and determine how to add it to your DataFrame
dataset["RainToday"].unique()

# Assuming 'dataset' is your DataFrame
# Convert the 'Date' column to datetime objects
dataset['Date'] = pd.to_datetime(dataset['Date'])

# Convert 'RainToday' to numerical representation (0 and 1)
dataset['RainToday'] = dataset['RainToday'].map({'Yes': 1, 'No': 0})

# Now calculate correlations
# (excluding 'Date' as it's datetime, not numeric for correlation)
# Select numeric features and drop 'Date' if it's present
numeric_features = dataset.select_dtypes(include=['number'])
if 'Date' in numeric_features.columns:
    numeric_features = numeric_features.drop(columns=['Date'])

correlations = numeric_features.corr()['Temp9am'].abs().sort_values(ascending=False)
print(correlations)


#analysing the target variable
y = dataset["RainToday"]
sns.countplot(y)
RainToday = dataset.RainToday.value_counts()
print(RainToday)


#analyzing the each feature
dataset["Temp9am"].unique()
sns.distplot(dataset["Temp9am"],y)


dataset["Temp3pm"].unique()
sns.distplot(dataset["Temp3pm"],y)


dataset["MinTemp"].unique()
sns.distplot(dataset["MinTemp"],y)


dataset["MaxTemp"].unique()
sns.distplot(dataset["MaxTemp"],y)



dataset["Rainfall"].unique()
sns.distplot(dataset["Rainfall"],y)



dataset["Evaporation"].unique()
sns.distplot(dataset["Evaporation"],y)



dataset["Sunshine"].unique()
sns.distplot(dataset["Sunshine"],y)


dataset["WindGustSpeed"].unique()

sns.distplot(dataset["WindGustSpeed"],y)




dataset["WindSpeed3pm"].unique()

sns.distplot(dataset["WindSpeed3pm"],y)





dataset["WindSpeed9am"].unique()

sns.distplot(dataset["WindSpeed9am"],y)




dataset["Humidity9am"].unique()
sns.distplot(dataset["Humidity9am"],y)




dataset["Humidity3pm"].unique()
sns.distplot(dataset["Humidity3pm"],y)




dataset["Pressure3pm"].unique()
sns.distplot(dataset["Pressure3pm"],y)



dataset["Cloud9am"].unique()
sns.distplot(dataset["Cloud9am"],y)




dataset["Cloud3pm"].unique()
sns.distplot(dataset["Cloud3pm"],y)




plt.figure(figsize=(10, 6))  # Adjust figure size as needed
sns.countplot(x='WindGustDir', data=dataset)
plt.title('Distribution of Wind Gust Direction')
plt.xlabel('Wind Gust Direction')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()



# Assuming 'dataset' is your DataFrame and 'WindGustDir' is the column you want to visualize
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
sns.countplot(x='WindDir9am', data=dataset)
plt.title('Distribution of Wind  Direction at 9 am')
plt.xlabel('Wind  Direction')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()



# Assuming 'dataset' is your DataFrame and 'WindGustDir' is the column you want to visualize
plt.figure(figsize=(10, 6))  # Adjust figure size as needed
sns.countplot(x='WindDir3pm', data=dataset)
plt.title('Distribution of Wind  Direction at 3pm')
plt.xlabel('Wind  Direction')
plt.ylabel('Count')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()




#correlation matrix
# Convert date columns to datetime objects
# Replace 'Date' with the actual name of your date column(s) if different
date_columns = ['Date']
for col in date_columns:
    df[col] = pd.to_datetime(df[col], errors='coerce')  # Handle potential errors

# Select only numeric features for correlation calculation
numeric_df = df.select_dtypes(include=['number'])

# Compute the correlation matrix
corr_matrix = numeric_df.corr()

# Set the size of the plot
plt.figure(figsize=(10, 8))

# Generate the heatmap with a color bar
sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, cbar=True)

# Show the plot
plt.show()










#feature or model selection
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2, RFE

# Load dataset
data = pd.read_csv('Weather_Data.csv')

# Drop 'Date' as it's irrelevant for prediction
data = data.drop(columns=['Date'])

# Target variable and features
X = data.drop(columns=['RainToday'])
y = data['RainToday']

# Separate numerical and categorical features
numerical_cols = X.select_dtypes(include=['number']).columns
categorical_cols = X.select_dtypes(include=['object']).columns

# Handling missing values for numerical features
imputer_num = SimpleImputer(strategy='mean')  # Imputer for numerical features
X[numerical_cols] = imputer_num.fit_transform(X[numerical_cols])

# Handling missing values for categorical features
imputer_cat = SimpleImputer(strategy='most_frequent')  # Imputer for categorical features
X[categorical_cols] = imputer_cat.fit_transform(X[categorical_cols])

# Encoding categorical variables
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)
X[categorical_cols] = X[categorical_cols].apply(label_encoder.fit_transform)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Feature selection using SelectKBest or RFE for tree-based classifiers
selector = SelectKBest(score_func=chi2, k=15)  # Select top 15 features
X_train = selector.fit_transform(X_train, y_train)
X_test = selector.transform(X_test)

# Apply Recursive Feature Elimination (RFE) for Random Forest
rfe = RFE(estimator=RandomForestClassifier(), n_features_to_select=15)
X_train = rfe.fit_transform(X_train, y_train)
X_test = rfe.transform(X_test)

# Standardize the features after feature selection
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning with expanded grid
def tune_model(clf, params, X_train, y_train):
    grid_search = GridSearchCV(clf, param_grid=params, cv=StratifiedKFold(n_splits=5), scoring='accuracy')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Define hyperparameters for each classifier
param_grids = {
    'SVM': {'C': [0.1, 1, 10, 100], 'gamma': [1, 0.1, 0.01, 0.001], 'kernel': ['rbf'], 'class_weight': ['balanced']},
    'Logistic Regression': {'C': [0.01, 0.1, 1, 10], 'solver': ['liblinear', 'lbfgs'], 'penalty': ['l2'], 'class_weight': ['balanced']},
    'Random Forest': {'n_estimators': [100, 300, 500], 'max_depth': [10, 20, 30], 'min_samples_split': [2, 5, 10], 'class_weight': ['balanced']},
    'KNN': {'n_neighbors': [3, 5, 7, 10], 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
    'XGBoost': {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7], 'n_estimators': [100, 200, 500], 'scale_pos_weight': [1, 2, 5]},
    'ANN': {'hidden_layer_sizes': [(100,), (100, 100)], 'alpha': [0.0001, 0.001], 'max_iter': [500, 1000]}
}

# Initialize classifiers with hyperparameter tuning
classifiers = {
    'Naive Bayes': GaussianNB(),
    'SVM': tune_model(SVC(), param_grids['SVM'], X_train_scaled, y_train),
    'Logistic Regression': tune_model(LogisticRegression(), param_grids['Logistic Regression'], X_train_scaled, y_train),
    'Random Forest': tune_model(RandomForestClassifier(), param_grids['Random Forest'], X_train, y_train),
    'KNN': tune_model(KNeighborsClassifier(), param_grids['KNN'], X_train_scaled, y_train),
    'XGBoost': tune_model(xgb.XGBClassifier(), param_grids['XGBoost'], X_train, y_train),
    'ANN': tune_model(MLPClassifier(), param_grids['ANN'], X_train_scaled, y_train)
}

# Cross-validation strategy
skf = StratifiedKFold(n_splits=5)

# Train and evaluate each classifier using cross-validation
for name, clf in classifiers.items():
    if name in ['SVM', 'KNN', 'ANN']:
        # Use scaled features for models that benefit from scaling
        X_train_cv, X_test_cv = X_train_scaled, X_test_scaled
    else:
        # Use unscaled features for models like Naive Bayes, Random Forest
        X_train_cv, X_test_cv = X_train, X_test

    # Perform cross-validation and calculate mean accuracy
    cv_scores = cross_val_score(clf, X_train_cv, y_train, cv=skf, scoring='accuracy')
    clf.fit(X_train_cv, y_train)
    y_pred = clf.predict(X_test_cv)

    print(f"{name} Classifier:")
    print(f"Cross-Validation Accuracy: {cv_scores.mean():.2f}")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    print("-" * 30)





#plotting graph
import matplotlib.pyplot as plt
import seaborn as sns

# Define accuracy scores for each algorithm (replace with your actual scores)
score_lr = 78  # Example score for Logistic Regression
score_nb = 77  # Example score for Naive Bayes
score_svm = 93  # Example score for Support Vector Machine
score_knn = 88  # Example score for K-Nearest Neighbors

score_rf = 89  # Example score for Random Forest
score_xgb = 89  # Example score for XGBoost
score_nn = 89  # Example score for Neural Network

# Assuming you have a list of accuracy scores for different algorithms
scores = [score_lr, score_nb, score_svm, score_knn, score_rf, score_xgb, score_nn]
algorithms = ["Logistic Regression", "Naive Bayes", "Support Vector Machine", "K-Nearest Neighbors", "Random Forest", "XGBoost", "Neural Network"]

# Create the bar plot using seaborn
plt.figure(figsize=(12, 6))  # Adjust figure size as needed
sns.barplot(x=algorithms, y=scores)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy Score (%)")
plt.title("Classifier Accuracy Comparison")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()