# Project_3_Machine_learning
EcoType is a machine learning project that predicts forest cover types in a given geographical area using environmental features such as elevation, slope, soil type, and wilderness area. The project aims to support forestry management, environmental monitoring, and land-use planning through automated and reliable cover type classification.

🌿 EcoType: Forest Cover Type Prediction Using Machine Learning
📌 Project Overview

EcoType is a machine learning project that predicts forest cover types in a given geographical area using environmental features such as elevation, slope, soil type, and wilderness area. The project aims to support forestry management, environmental monitoring, and land-use planning through automated and reliable cover type classification.

🚀 Skills & Concepts Applied

Exploratory Data Analysis (EDA)

Data Cleaning & Preprocessing

Feature Engineering

Classification Model Building (Random Forest, Gradient Boosting, XGBoost)

Model Evaluation & Hyperparameter Tuning

Streamlit App Development

📂 Repository Structure

notebooks/ → EDA, preprocessing, modeling, tuning, and testing

app/ → Streamlit application code

models/ → Saved trained models

data/ → Dataset (if not public, provide a download link instead)


import pandas as pd
import io
from google.colab import files

# Upload
uploaded = files.upload()


# Get the actual filename key
filename = list(uploaded.keys())[0]

# Read CSV
df = pd.read_csv(io.BytesIO(uploaded[filename]))
df.head()
df.shape
df.info()
df.describe().T
df.duplicated().sum()
Handling Outliers
import seaborn as sns
import matplotlib.pyplot as plt
numerical_cols=['Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
       'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
       'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
       'Horizontal_Distance_To_Fire_Points']

# Loop through numerical columns and create boxplots

plt.figure(figsize=(15, 7))
plt.suptitle("Boxplots before Outliers removal")
for i in range(0, len(numerical_cols)):
    plt.subplot(1, 10, i+1)
    sns.boxplot(y=df[numerical_cols[i]],color='purple',)
    plt.tight_layout()
IQR method for outlayer handling
import numpy as np

for col in numerical_cols:

    Q1 = df[col].quantile(0.25)  # 25th percentile
    Q3 = df[col].quantile(0.75)  # 75th percentile
    IQR = Q3 - Q1  # Interquartile range

    # Define lower and upper bound
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Cap values at lower and upper bound
    df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
    df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
  checking skewinwss
  skewness = df[numerical_cols].skew()
skewness
# Select numerical columns with high skewness (absolute skewness > 0.5)
skewed_cols = skewness[abs(skewness) > 0.5].index

# Plot histograms before transformation
plt.figure(figsize=(10, 4))
for i, col in enumerate(skewed_cols, 1):
    plt.subplot(3, 3, i)  # Adjust grid size as needed
    plt.hist(df[col], bins=30, color='skyblue', edgecolor='black')
    plt.title(f"{col}")
plt.tight_layout()
plt.show()
import pandas as pd
import numpy as np

numeric_cols = [
    'Elevation','Aspect','Slope',
    'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways','Hillshade_9am',
    'Hillshade_Noon','Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

def handle_moderate_skew(df, numeric_cols, pos_thresh=0.5, neg_thresh=-0.5):
    """
    Handles skew for linear models:
    - Positive skew > pos_thresh → log1p (if all values positive)
    - Negative skew < neg_thresh → square
    - Mild skew → no transformation
    """
    skew_vals = df[numeric_cols].skew()
    applied_transforms = {}

    for col, skew in skew_vals.items():
        if skew > pos_thresh and (df[col] > 0).all():
            df[col] = np.log1p(df[col])
            applied_transforms[col] = 'log1p'
            print(f"{col}: log1p applied (skew={skew:.3f})")
        elif skew < neg_thresh:
            df[col] = np.square(df[col])
            applied_transforms[col] = 'square'
            print(f"{col}: square applied (skew={skew:.3f})")
        else:
            applied_transforms[col] = None
            print(f"{col}: no transform (skew={skew:.3f})")

    new_skew = df[numeric_cols].skew()
    return df, new_skew, applied_transforms

# Apply transformation
df, new_skew, applied_transforms = handle_moderate_skew(df, numeric_cols)

print("\n✅ Skewness after processing:\n", new_skew)

from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(method='yeo-johnson', standardize=False)

# Example for a single column
df['Horizontal_Distance_To_Hydrology'] = pt.fit_transform(df[['Horizontal_Distance_To_Hydrology']])
pos_skewed_cols = ['Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                    'Horizontal_Distance_To_Roadways', 'Horizontal_Distance_To_Fire_Points']

for col in pos_skewed_cols:
    df[col] = pt.fit_transform(df[[col]])
Feature importances using Random Forest(Classification)
It’s a score assigned to each feature to reflect how useful or valuable it was in building the model — i.e., how much it contributed to reducing prediction error.
from sklearn.ensemble import RandomForestClassifier
X = df.drop(columns=['Cover_Type'])
y = df['Cover_Type']
# Train a Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X, y)

# Create a DataFrame
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

# Sort by importance in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Display the top features
feature_importance_df
# Imports
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb  # ✅ Highlight: XGBoost import

from imblearn.over_sampling import SMOTE

# Assuming df is your DataFrame and 'Cover_Type' is target
X = df.drop(columns=['Cover_Type'])
y = df['Cover_Type']

# ✅ Highlight: Shift labels for XGBoost (0-based)
y_xgb = y - 1

# Stratified train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
# Corresponding split for XGBoost
_, _, y_train_xgb, y_test_xgb = train_test_split(
    X, y_xgb, test_size=0.2, random_state=42, stratify=y_xgb
)

# Numeric features for scaling
numeric_cols = [
    'Elevation','Aspect','Slope',
    'Horizontal_Distance_To_Hydrology','Vertical_Distance_To_Hydrology',
    'Horizontal_Distance_To_Roadways','Hillshade_9am',
    'Hillshade_Noon','Hillshade_3pm',
    'Horizontal_Distance_To_Fire_Points'
]

# Scale numeric features (important for Logistic Regression & KNN)
scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[numeric_cols] = scaler.fit_transform(X_train[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test[numeric_cols])

# Dictionary of models
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced', max_iter=500, random_state=42),
    "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(class_weight='balanced', random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42),
    "XGBoost": xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False, random_state=42)
}

# Train and evaluate
for name, model in models.items():
    # Use scaled data for Logistic Regression & KNN, raw for trees
    if name in ["Logistic Regression", "K-Nearest Neighbors"]:
        X_tr, X_te = X_train_scaled.copy(), X_test_scaled.copy()
    else:
        X_tr, X_te = X_train.copy(), X_test.copy()

    # Apply SMOTE only for KNN
    if name == "K-Nearest Neighbors":
        smote = SMOTE(random_state=42)
        X_tr, y_tr = smote.fit_resample(X_tr, y_train)
    elif name == "XGBoost":
        # ✅ Highlight: Use shifted labels for XGBoost
        y_tr = y_train_xgb
        X_te = X_test.copy()  # Keep original features
        y_te = y_test_xgb
    else:
        y_tr = y_train
        y_te = y_test

    # Fit the model
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)

    # ✅ Highlight: Shift predictions back for XGBoost
    if name == "XGBoost":
        y_pred = y_pred + 1

    print(f"\n=== {name} ===")
    print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
    print("Macro F1 Score:", round(f1_score(y_test, y_pred, average='macro'), 4))
    print("Classification Report:\n", classification_report(y_test, y_pred, digits=4))
HYPER PARAMETER
    
from sklearn.model_selection import RandomizedSearchCV
# ==============================
# Imports
# ==============================
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import numpy as np

# ==============================
# Optional: Sample your training data for faster tuning
# ==============================
sample_frac = 0.8  # 80% of training data for faster tuning
X_sample = X_train.sample(frac=sample_frac, random_state=42)
y_sample = y_train.loc[X_sample.index]

# ==============================
# Define Random Forest (memory-efficient)
# ==============================
rf = RandomForestClassifier(
    random_state=42,
    class_weight='balanced',
    n_jobs=1,        # safe to avoid worker crash
    max_samples=0.8   # each tree uses 80% of sampled data
)

# ==============================
# Hyperparameter distributions
# ==============================
param_dist = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 15, 25, 35],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

# ==============================
# Stratified CV for imbalanced data
# ==============================
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# ==============================
# Randomized Search setup (total 45 fits)
# ==============================
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=15,            # 15 combinations × 3 folds = 45 fits
    scoring='f1_macro',
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=1              # safe single-threaded
)

# ==============================
# Fit the RandomizedSearchCV
# ==============================
random_search.fit(X_sample, y_sample)

# ==============================
# Best hyperparameters
# ==============================
best_params = random_search.best_params_
print("✅ Best Hyperparameters:\n", best_params)

# ==============================
# Train final model on full training data
# ==============================
best_rf = RandomForestClassifier(
    **best_params,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1       # now safely use all cores for final training
)
best_rf.fit(X_train, y_train)

# ==============================
# Evaluate on test set
# ==============================
y_pred = best_rf.predict(X_test)

print("\n=== Random Forest (Final) ===")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Macro F1 Score:", round(f1_score(y_test, y_pred, average='macro'), 4))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=4))
import streamlit as st
import pickle
import numpy as np

# Load the trained Random Forest model
with open("best_random_forest.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Forest Cover Type Prediction")
st.write("Enter the feature values to predict the forest cover type.")

# -----------------------------
# Example: suppose your dataset has 10 features
# Replace feature names with your actual columns
# -----------------------------
feature_names = [
    'Elevation', 'Aspect', 'Slope', 'Horizontal_Distance_To_Hydrology',
    'Vertical_Distance_To_Hydrology', 'Horizontal_Distance_To_Roadways',
    'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points'
]

# Collect inputs from user
user_input = []
for feature in feature_names:
    val = st.number_input(f"Enter {feature}", value=0.0)
    user_input.append(val)

# Convert to 2D array for prediction
input_array = np.array(user_input).reshape(1, -1)

# Predict button
if st.button("Predict Cover Type"):
    prediction = model.predict(input_array)[0]
    st.success(f"Predicted Forest Cover Type: {prediction}")
    
    
