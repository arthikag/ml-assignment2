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
│   ├── train_models.py
│   ├── logistic_regression.pkl
│   ├── decision_tree.pkl
│   ├── k-nearest_neighbor.pkl
│   ├── naive_bayes.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   ├── scaler.pkl
│   ├── feature_names.pkl
│   └── results.csv
├── app.py
├── requirements.txt
├── README.md 
└── ML_Assignment2.txt
```

## Installation & Setup

### 1. Clone or Download the Repository
```bash
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

### Model Performance Page
- View all evaluation metrics in a table
- Identify best-performing models for each metric
- Visualize model performance across different metrics
- Line charts and bar charts for easy comparison

### Make Predictions Page
- Interactive sliders for all 30 features
- Real-time predictions from all 6 trained models
- Confidence scores for each prediction
- Compare predictions across different algorithms

### Metrics Comparison Page
- Detailed comparison of specific metrics
- Filter and sort by individual evaluation metrics
- Statistical summary (best, worst, average)
- Side-by-side model comparison

### About Dataset Page
- Comprehensive dataset information
- Feature descriptions and categories
- Dataset characteristics and use cases
- Binary classification details

## Model Performance Summary

All models are trained on the same preprocessed dataset with 80-20 train-test split.

**Key Metrics:**
- All models achieve high accuracy (>90%)
- Evaluation based on consistent metrics: Accuracy, AUC, Precision, Recall, F1, MCC
- Ensemble methods (Random Forest, XGBoost) typically show superior performance
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

 **Complete Implementation** - All 6 models and 6 metrics as per assignment

 **Interactive UI** - Streamlit-based user-friendly interface

 **Real-time Predictions** - Test models with custom feature inputs

 **Metric Visualization** - Charts and comparisons for analysis

 **Reproducible** - Fixed random seeds for consistent results

 **Deployment Ready** - Optimized for Streamlit Community Cloud

## Author Notes

This project demonstrates:
- End-to-end ML pipeline development
- Multiple classification algorithm implementation
- Comprehensive model evaluation framework
- Interactive data science application design
- Enterprise-grade code organization

## Assignment Requirements Compliance

 Step 1: Choose classification dataset (Breast Cancer, 30 features, 569 instances)

 Step 2: Implement 6 classification models with all required evaluation metrics

 Step 3: Interactive Streamlit application with model visualization

 Step 4: Deployment-ready code for Streamlit Community Cloud

 Step 5: Comprehensive documentation (this README)

## License

This project is created for educational purposes as part of ML Assignment 2 at BITS Pilani.

## Support

For issues or questions:
- Review the assignment specification in ML_Assignment2.txt
- Check Streamlit documentation: https://docs.streamlit.io
- Check Scikit-learn documentation: https://scikit-learn.org
