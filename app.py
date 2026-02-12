import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="ML Classification Models",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.title("ML Classification Models - Evaluation Dashboard")
st.markdown("Professional Machine Learning Classification Pipeline")

# Get model directory
model_dir = Path(__file__).parent / "model"

# Load trained models and data
@st.cache_resource
def load_models_and_data():
    models = {}
    model_files = [
        "logistic_regression.pkl",
        "decision_tree.pkl",
        "k-nearest_neighbor.pkl",
        "naive_bayes.pkl",
        "random_forest.pkl",
        "xgboost.pkl"
    ]
    
    for file in model_files:
        path = model_dir / file
        if path.exists():
            with open(path, 'rb') as f:
                model_name = file.replace('.pkl', '').replace('_', ' ').title()
                models[model_name] = pickle.load(f)
    
    # Load scaler
    scaler_path = model_dir / "scaler.pkl"
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Load results
    results_path = model_dir / "results.csv"
    results_df = pd.read_csv(results_path, index_col=0)
    
    # Load feature names
    feature_names_path = model_dir / "feature_names.pkl"
    with open(feature_names_path, 'rb') as f:
        feature_names = pickle.load(f)
    
    return models, scaler, results_df, feature_names

try:
    models, scaler, results_df, feature_names = load_models_and_data()
    models_loaded = True
except Exception as e:
    st.error(f"Error loading models: {str(e)}")
    st.info("Please run `python model/train_models.py` first to train and save the models.")
    models_loaded = False

if models_loaded:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("Model Performance", use_container_width=True):
            st.session_state.page = "performance"
    
    with col2:
        if st.button("Make Predictions", use_container_width=True):
            st.session_state.page = "predictions"
    
    with col3:
        if st.button("Metrics Comparison", use_container_width=True):
            st.session_state.page = "comparison"
    
    with col4:
        if st.button("About Dataset", use_container_width=True):
            st.session_state.page = "dataset"
    
    if "page" not in st.session_state:
        st.session_state.page = "performance"
    
    st.markdown("---")
    
    if st.session_state.page == "performance":
        st.header("Model Performance Metrics")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Metrics Table")
            st.dataframe(results_df.round(4), width='stretch')
        
        with col2:
            st.subheader("Best Models by Metric")
            for metric in results_df.columns:
                best_model = results_df[metric].idxmax()
                best_score = results_df[metric].max()
                st.metric(metric, f"{best_score:.4f}", f"({best_model})")
        
        st.markdown("---")
        st.subheader("Metric Visualizations")
        
        # Line chart
        st.line_chart(results_df)
        
        # Bar chart for each metric
        col1, col2 = st.columns(2)
        
        with col1:
            st.bar_chart(results_df['Accuracy'])
            st.caption("Accuracy by Model")
            
            st.bar_chart(results_df['F1 Score'])
            st.caption("F1 Score by Model")
        
        with col2:
            st.bar_chart(results_df['AUC Score'])
            st.caption("AUC Score by Model")
            
            st.bar_chart(results_df['MCC Score'])
            st.caption("MCC Score by Model")
    
    elif st.session_state.page == "predictions":
        st.header("Make Predictions with Trained Models")
        st.markdown("---")
        
        st.info("Enter feature values below to make predictions using all 6 trained models.")
        
        # Create input sliders for all 30 features
        st.subheader("Feature Values")
        feature_values = []
        
        cols = st.columns(5)
        for idx, feature in enumerate(feature_names):
            col = cols[idx % 5]
            with col:
                value = st.slider(
                    feature,
                    min_value=0.0,
                    max_value=100.0,
                    value=50.0,
                    step=0.1,
                    key=feature
                )
                feature_values.append(value)
        
        # Make predictions
        if st.button("🔮 Predict", width='stretch'):
            X_input = np.array(feature_values).reshape(1, -1)
            X_scaled = scaler.transform(X_input)
            
            st.markdown("---")
            st.subheader("Predictions")
            
            predictions = {}
            for model_name, model in models.items():
                pred = model.predict(X_scaled)[0]
                pred_proba = model.predict_proba(X_scaled)[0]
                
                predictions[model_name] = {
                    'Prediction': 'Malignant (1)' if pred == 1 else 'Benign (0)',
                    'Confidence (Class 0)': pred_proba[0],
                    'Confidence (Class 1)': pred_proba[1]
                }
            
            for model_name, pred_data in predictions.items():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.write(f"**{model_name}**")
                with col2:
                    prediction = pred_data['Prediction']
                    st.write(f"Prediction: {prediction}")
                with col3:
                    confidence = pred_data['Confidence (Class 1)']
                    st.write(f"Confidence: {confidence:.4f}")
    
    elif st.session_state.page == "comparison":
        st.header("Detailed Metrics Comparison")
        st.markdown("---")
        
        # Selectbox for metric
        selected_metric = st.selectbox("Select Metric:", results_df.columns)
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.bar_chart(results_df[selected_metric].sort_values(ascending=False))
        
        with col2:
            st.metric(f"Best {selected_metric}", 
                     f"{results_df[selected_metric].max():.4f}",
                     f"{results_df[selected_metric].idxmax()}")
            
            st.metric(f"Worst {selected_metric}", 
                     f"{results_df[selected_metric].min():.4f}",
                     f"{results_df[selected_metric].idxmin()}")
            
            st.metric(f"Average {selected_metric}", 
                     f"{results_df[selected_metric].mean():.4f}")
        
        st.markdown("---")
        st.subheader(f"All Models - {selected_metric}")
        st.dataframe(results_df[[selected_metric]].round(4), width='stretch')
    
    elif st.session_state.page == "dataset":
        st.header("About the Dataset")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.markdown("""
            **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
            
            **Source:** UCI Machine Learning Repository
            
            **Number of Instances:** 569
            
            **Number of Features:** 30
            
            **Classes:** 2 (Binary Classification)
            - Benign (0)
            - Malignant (1)
            
            **Feature Types:** Numeric (computed from digitized images)
            """)
        
        with col2:
            st.subheader("Feature Categories")
            st.markdown("""
            Features are computed for each cell nucleus:
            
            1. **Radius** - mean distance from center to perimeter
            2. **Texture** - standard deviation of gray-scale values
            3. **Perimeter** - size of the core tumor
            4. **Area** - size of the core tumor
            5. **Smoothness** - local variation in radius lengths
            6. **Compactness** - perimeter² / area - 1.0
            7. **Concavity** - severity of concave portions
            8. **Concave Points** - number of concave portions
            9. **Symmetry** - symmetry of the nucleus
            10. **Fractal Dimension** - coastline approximation
            
            *For each feature, the dataset provides:*
            - Mean
            - Standard Error
            - Worst (largest)
            """)
        
        st.markdown("---")
        st.subheader("Class Distribution")
        
        st.info("""
        This is a standard benchmark dataset used for:
        - Binary classification (Cancer vs. Normal)
        - Evaluating classification algorithms
        - Model comparison and selection
        
        The dataset is well-balanced and suitable for demonstrating
        various ML classification techniques.
        """)

else:
    st.error("Models not loaded. Please ensure the model files are trained and saved.")
