import numpy as np
import pandas as pd
import pickle
import os
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

# Metrics
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

# Load Dataset
print("Loading Breast Cancer Dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(f"Dataset Shape: {X.shape}")
print(f"Features: {X.shape[1]}")
print(f"Instances: {X.shape[0]}")
print(f"Classes: {np.unique(y)}")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize Features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining Set Size: {X_train_scaled.shape[0]}")
print(f"Test Set Size: {X_test_scaled.shape[0]}")

# Define Models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
    'Naive Bayes': GaussianNB(),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'XGBoost': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train Models & Calculate Metrics
results = {}
trained_models = {}

print("\n" + "="*80)
print("TRAINING MODELS AND CALCULATING METRICS")
print("="*80)

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train
    model.fit(X_train_scaled, y_train)
    trained_models[model_name] = model
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Calculate Metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    
    results[model_name] = {
        'Accuracy': accuracy,
        'AUC Score': auc,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'MCC Score': mcc
    }
    
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  AUC Score: {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  MCC Score: {mcc:.4f}")

# Create Model Directory
model_dir = os.path.dirname(__file__)
os.makedirs(model_dir, exist_ok=True)

# Save Models
print("\n" + "="*80)
print("SAVING MODELS AND RESULTS")
print("="*80)

for model_name, model in trained_models.items():
    model_path = os.path.join(model_dir, f"{model_name.replace(' ', '_').lower()}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Saved: {model_path}")

# Save Scaler
scaler_path = os.path.join(model_dir, "scaler.pkl")
with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"Saved: {scaler_path}")

# Save Results
results_df = pd.DataFrame(results).T
results_path = os.path.join(model_dir, "results.csv")
results_df.to_csv(results_path)
print(f"Saved: {results_path}")

# Save Feature Names
feature_names_path = os.path.join(model_dir, "feature_names.pkl")
with open(feature_names_path, 'wb') as f:
    pickle.dump(list(X.columns), f)
print(f"Saved: {feature_names_path}")

print("\n" + "="*80)
print("SUMMARY OF RESULTS")
print("="*80)
print(results_df.round(4))
print("\nAverage F1 Score per Model:")
print(results_df['F1 Score'].round(4))

print("\nTraining Complete! All models and results saved.")
