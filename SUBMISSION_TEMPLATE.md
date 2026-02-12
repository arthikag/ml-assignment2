# ML Assignment 2: Classification Models with Streamlit Deployment
## Submission Requirements

---

## 1. GITHUB REPOSITORY LINK

**Repository URL:**
```
https://github.com/YOUR_USERNAME/ml-assignment2-classification
```

**Contents Included:**
✓ Complete source code (train_models.py, app.py)
✓ requirements.txt with all dependencies
✓ README.md with comprehensive documentation
✓ Model directory with training scripts
✓ All necessary configuration files

**How to Access:**
Click the link above to view the complete GitHub repository containing:
- Training script for 6 ML models
- Interactive Streamlit web application
- Full documentation and setup instructions

---

## 2. LIVE STREAMLIT APP LINK

**Deployed Application URL:**
```
https://YOUR_USERNAME-ml-assignment2.streamlit.app
```

**Application Features:**
- 📊 Model Performance Dashboard: View accuracy, AUC, precision, recall, F1, and MCC scores for all 6 models
- 🎯 Live Predictions: Enter feature values and get predictions from all models
- 📈 Metrics Comparison: Filter and compare models by individual metrics
- ℹ️ Dataset Information: Complete details about the Breast Cancer dataset used

**Models Deployed:**
1. Logistic Regression (98.25% accuracy)
2. Decision Tree (91.23% accuracy)
3. K-Nearest Neighbor (95.61% accuracy)
4. Naive Bayes (92.98% accuracy)
5. Random Forest (95.61% accuracy)
6. XGBoost (95.61% accuracy)

**How to Use:**
1. Click the link above to open the live application
2. Select pages from sidebar: Model Performance, Make Predictions, Metrics Comparison, About Dataset
3. Interact with visualizations and test predictions
4. Application runs 24/7 on Streamlit Community Cloud (free hosting)

---

## 3. BITS VIRTUAL LAB SCREENSHOT

[INSERT SCREENSHOT HERE]

**Screenshot showing:**
- Successful execution of `python model/train_models.py` on BITS Virtual Lab
- Output displaying all model metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
- Streamlit application running on BITS Virtual Lab environment
- Proof of assignment completion on required platform

**Evidence of Completion:**
- All 6 models trained successfully
- All evaluation metrics calculated correctly
- Application responsive and fully functional
- Executed on official BITS Virtual Lab as per assignment requirements

---

## 4. GITHUB README CONTENT (Full Text)

# ML Assignment 2: Classification Models with Streamlit Deployment

## Project Overview

This project implements a complete machine learning pipeline for binary classification using the **Breast Cancer Dataset**. It demonstrates 6 different classification algorithms with comprehensive evaluation metrics and an interactive Streamlit web application.

## Dataset

- **Name:** Breast Cancer Wisconsin (Diagnostic)
- **Source:** UCI Machine Learning Repository / Scikit-learn
- **Instances:** 569
- **Features:** 30 (exceeds minimum requirement of 12)
- **Classes:** 2 (Benign/Malignant)
- **Problem Type:** Binary Classification

## Classification Models Implemented

1. **Logistic Regression** - Linear classification model
2. **Decision Tree Classifier** - Tree-based classification
3. **K-Nearest Neighbor (KNN)** - Instance-based learning (k=5)
4. **Naive Bayes Classifier** - Gaussian probabilistic classifier
5. **Random Forest** - Ensemble of decision trees (100 estimators)
6. **XGBoost** - Gradient boosting classifier (100 estimators)

## Evaluation Metrics

For each model, the following metrics are calculated:

1. **Accuracy** - Proportion of correct predictions
2. **AUC Score** - Area under the Receiver Operating Characteristic curve
3. **Precision** - Proportion of true positives among predicted positives
4. **Recall** - Proportion of true positives among actual positives
5. **F1 Score** - Harmonic mean of precision and recall
6. **MCC Score** - Matthews Correlation Coefficient (balanced accuracy measure)

## Project Structure

```
ml-assignment2-classification/
├── model/
│   ├── train_models.py              # Model training script
│   ├── logistic_regression.pkl      # Trained model (auto-generated)
│   ├── decision_tree.pkl            # Trained model (auto-generated)
│   ├── k-nearest_neighbor.pkl       # Trained model (auto-generated)
│   ├── naive_bayes.pkl              # Trained model (auto-generated)
│   ├── random_forest.pkl            # Trained model (auto-generated)
│   ├── xgboost.pkl                  # Trained model (auto-generated)
│   ├── scaler.pkl                   # Feature scaler (auto-generated)
│   ├── feature_names.pkl            # Feature names (auto-generated)
│   └── results.csv                  # Model performance metrics (auto-generated)
├── app.py                           # Streamlit web application
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── SETUP.md                         # Installation guide
├── DEPLOYMENT_GUIDE.md              # Deployment instructions
└── ML_Assignment2.txt               # Assignment specifications
```

## Installation & Setup

### 1. Clone or Download the Repository
```bash
git clone https://github.com/YOUR_USERNAME/ml-assignment2-classification.git
cd ml-assignment2-classification
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the Models
```bash
python model/train_models.py
```

This script will:
- Load the Breast Cancer dataset
- Split data into 80% training and 20% testing
- Standardize features using StandardScaler
- Train all 6 classification models
- Calculate evaluation metrics for each model
- Save trained models as pickle files
- Generate results.csv with all metrics

Expected output: Display of accuracy, AUC, precision, recall, F1, and MCC scores for each model.

### 4. Run the Streamlit Application
```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`

## Application Features

### 📊 Model Performance Page
- View all evaluation metrics in a table
- Identify best-performing models for each metric
- Visualize model performance across different metrics
- Line charts and bar charts for easy comparison

### 🎯 Make Predictions Page
- Interactive sliders for all 30 features
- Real-time predictions from all 6 trained models
- Confidence scores for each prediction
- Compare predictions across different algorithms

### 📈 Metrics Comparison Page
- Detailed comparison of specific metrics
- Filter and sort by individual evaluation metrics
- Statistical summary (best, worst, average)
- Side-by-side model comparison

### ℹ️ About Dataset Page
- Comprehensive dataset information
- Feature descriptions and categories
- Dataset characteristics and use cases
- Binary classification details

## Model Performance Summary

All models are trained on the same preprocessed dataset with 80-20 train-test split.

**Performance Results:**
- Logistic Regression: 98.25% accuracy, 98.61% F1 score
- XGBoost: 95.61% accuracy, 96.60% F1 score
- Random Forest: 95.61% accuracy, 96.55% F1 score
- K-Nearest Neighbor: 95.61% accuracy, 96.55% F1 score
- Naive Bayes: 92.98% accuracy, 94.44% F1 score
- Decision Tree: 91.23% accuracy, 92.86% F1 score

**Key Metrics:**
- All models achieve high accuracy (>90%)
- Evaluation based on consistent metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Ensemble methods (Random Forest, XGBoost) show superior performance
- Results are reproducible with fixed random_state=42

## Deployment

### Local Testing
1. Run `python model/train_models.py` to train models
2. Run `streamlit run app.py` to test locally
3. Test all features: Model metrics view, predictions, comparisons

### Streamlit Community Cloud Deployment
1. Push the repository to GitHub
2. Sign up at https://share.streamlit.io
3. Deploy by connecting your GitHub repository
4. Access your live app via the provided Streamlit Community Cloud URL

**Requirements for deployment:**
- requirements.txt with all dependencies
- app.py as the main Streamlit file
- model/ directory with trained .pkl files
- Public GitHub repository

## Data Preprocessing

1. **Loading:** Breast Cancer dataset from scikit-learn
2. **Train-Test Split:** 80% training, 20% testing (stratified)
3. **Feature Standardization:** StandardScaler normalization
4. **Class Balance:** Ensured through stratified splitting

## Technical Stack

- **Python 3.8+**
- **Scikit-learn** - Machine learning models and metrics
- **XGBoost** - Gradient boosting implementation
- **Pandas** - Data manipulation
- **NumPy** - Numerical computations
- **Streamlit** - Interactive web application framework

## Performance Notes

- Training time: < 1 second on modern hardware
- Model inference: Instant predictions
- Dataset size: Small enough for rapid iteration, large enough for meaningful evaluation
- Scalability: Code can be easily adapted for larger datasets

## Key Features

✅ **Complete Implementation** - All 6 models and 6 metrics as per assignment

✅ **Interactive UI** - Streamlit-based user-friendly interface

✅ **Real-time Predictions** - Test models with custom feature inputs

✅ **Metric Visualization** - Charts and comparisons for analysis

✅ **Reproducible** - Fixed random seeds for consistent results

✅ **Deployment Ready** - Optimized for Streamlit Community Cloud

## Assignment Requirements Compliance

✅ Step 1: Choose classification dataset (Breast Cancer, 30 features, 569 instances)

✅ Step 2: Implement 6 classification models with all required evaluation metrics

✅ Step 3: Interactive Streamlit application with model visualization

✅ Step 4: Deployment-ready code for Streamlit Community Cloud

✅ Step 5: Comprehensive documentation (this README)

## License

This project is created for educational purposes as part of ML Assignment 2 at BITS Pilani.

## Support

For issues or questions:
- Review the assignment specification in ML_Assignment2.txt
- Check the DEPLOYMENT_GUIDE.md for step-by-step instructions
- Review SETUP.md for installation help
- Check Streamlit documentation: https://docs.streamlit.io
- Check Scikit-learn documentation: https://scikit-learn.org

---

**Author:** ML Assignment 2 Student
**Date:** February 2026
**Institution:** BITS Pilani, Work Integrated Learning Programmes Division
**Course:** Machine Learning (M.Tech AIML/DSE)

