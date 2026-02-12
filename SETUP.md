# Setup Instructions for ML Assignment 2

## Prerequisites

You need Python 3.8 or higher installed. 

## Step 1: Install Python (if not already installed)

### Option A: Download from python.org
1. Go to https://python.org/downloads
2. Download Python 3.11 or 3.12
3. During installation, **CHECK the box "Add Python to PATH"**
4. Complete the installation

### Option B: Download from Microsoft Store
1. Open Microsoft Store
2. Search for "Python 3.11" or "Python 3.12"
3. Click "Get" to install

### Verify Installation
Open Command Prompt and run:
```
python --version
```

You should see output like: `Python 3.11.0` or similar

## Step 2: Project Setup

1. Navigate to the project directory:
```
cd c:\Users\aganesan2\Downloads\ml-assignment2-classification
```

2. Install required packages:
```
pip install -r requirements.txt
```

## Step 3: Train the Models

Run the training script:
```
python model/train_models.py
```

This will:
- Load the Breast Cancer dataset
- Train 6 classification models
- Calculate all evaluation metrics
- Save trained models and results

Expected output: Model metrics (Accuracy, AUC, Precision, Recall, F1, MCC) for each model

## Step 4: Run the Streamlit App

```
streamlit run app.py
```

The app will open at: http://localhost:8501

## IMPORTANT: BITS Virtual Lab

As per the assignment requirements, you MUST also:
1. Complete this assignment on BITS Virtual Lab
2. Take a screenshot of the application running on BITS Virtual Lab
3. Submit the screenshot as part of the assignment

To run on BITS Virtual Lab:
1. Log in to BITS Virtual Lab
2. Upload or clone this project
3. Follow the same steps above (Step 2-4)
4. Take a screenshot showing the Streamlit app running
5. Save this screenshot for submission

## Troubleshooting

### "Python not found"
- Make sure Python was added to PATH during installation
- Restart Command Prompt after installing Python
- Try restarting your computer

### "Module not found" errors
- Make sure you ran: `pip install -r requirements.txt`
- Check that all dependencies installed successfully

### Streamlit won't start
- Make sure Streamlit is installed: `pip install streamlit==1.29.0`
- Check that app.py is in the correct location

## Files Generated After Training

After running train_models.py, you'll have:
- model/logistic_regression.pkl
- model/decision_tree.pkl
- model/k-nearest_neighbor.pkl
- model/naive_bayes.pkl
- model/random_forest.pkl
- model/xgboost.pkl
- model/scaler.pkl
- model/feature_names.pkl
- model/results.csv

These are required for the Streamlit app to work.
