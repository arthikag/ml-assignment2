import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import random
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import io

st.set_page_config(
    page_title="ML Classification Models",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    * {
        font-family: 'Poppins', 'Inter', sans-serif !important;
    }
    
    h1 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 700 !important;
        color: #2C3E50 !important;
        letter-spacing: -0.5px !important;
        margin: -20px 0 5px 0 !important;
        padding: 0 !important;
    }
    
    h2, h3 {
        font-family: 'Poppins', sans-serif !important;
        font-weight: 600 !important;
        color: #2C3E50 !important;
    }
    
    p {
        font-family: 'Inter', sans-serif !important;
        font-weight: 400 !important;
        color: #555555 !important;
    }
    
    .stButton > button {
        background-color: rgb(255, 140, 66) !important;
        color: white !important;
        font-weight: 600 !important;
        font-family: 'Poppins', sans-serif !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 12px 24px !important;
        font-size: 16px !important;
        transition: all 0.3s ease !important;
    }
    .stButton > button:hover {
        background-color: rgb(255, 120, 40) !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 12px rgba(255, 140, 66, 0.3) !important;
    }
    
    .metric-value {
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    .footer {
        text-align: center;
        font-size: 12px;
        color: #888;
        margin-top: 40px;
        padding: 20px 0;
        border-top: 1px solid #ddd;
    }
    
    /* Styled Table CSS */
    .styled-table {
        border-collapse: collapse;
        width: 100%;
        border-radius: 8px;
        overflow: hidden;
        border: 2px solid rgb(255, 140, 66);
        font-family: 'Inter', sans-serif !important;
    }
    
    .styled-table thead tr {
        background-color: rgb(255, 140, 66);
        color: white;
        font-weight: 600;
        text-align: center;
    }
    
    .styled-table thead th {
        padding: 12px;
        border: 1px solid rgba(255, 140, 66, 0.8);
        font-weight: 600;
    }
    
    .styled-table tbody tr {
        background-color: #FFFEF5;
    }
    
    .styled-table tbody tr:nth-child(even) {
        background-color: #FFE4C4;
    }
    
    .styled-table tbody tr:hover {
        background-color: rgba(255, 140, 66, 0.15);
    }
    
    .styled-table tbody td {
        padding: 10px;
        border: 1px solid rgba(255, 140, 66, 0.3);
        text-align: center;
        color: #2C3E50;
    }
    </style>
""", unsafe_allow_html=True)

def display_styled_dataframe(df, title=""):
    """Display a dataframe with custom orange and cream styling"""
    html = df.to_html(classes='styled-table', border=0, index=True)
    if title:
        st.markdown(f"<h3 style='color: rgb(255, 140, 66);'>{title}</h3>", unsafe_allow_html=True)
    st.markdown(html, unsafe_allow_html=True)

def add_footer():
    """Add footer with user info to every page"""
    st.markdown("<div class='footer'>Arthika G | BITS ID: 2025AB05180</div>", unsafe_allow_html=True)

st.markdown("### ML Classification Models - Breast Cancer Dataset")

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
    col1, col2, col3, col4, col5 = st.columns(5)
    
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
        if st.button("Upload Dataset", use_container_width=True):
            st.session_state.page = "upload"
    
    with col5:
        if st.button("About Dataset", use_container_width=True):
            st.session_state.page = "dataset"
    
    st.markdown("---")
    
    if "page" not in st.session_state:
        st.session_state.page = "performance"
    
    if st.session_state.page == "performance":
        st.header("Model Performance Metrics")
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Metrics Table")
            display_styled_dataframe(results_df.round(4), "Evaluation Metrics")
        
        with col2:
            st.subheader("Best Models by Metric")
            metrics_list = results_df.columns.tolist()
            for i in range(0, len(metrics_list), 2):
                metric_cols = st.columns(2)
                for j, col in enumerate(metric_cols):
                    if i+j < len(metrics_list):
                        metric = metrics_list[i+j]
                        with col:
                            best_model = results_df[metric].idxmax()
                            best_score = results_df[metric].max()
                            st.metric(metric[:8], f"{best_score:.4f}")
        
        st.markdown("---")
        st.subheader("Metric Visualizations")
        
        # Line chart with responsive sizing
        fig = go.Figure()
        for model in results_df.index:
            fig.add_trace(go.Scatter(x=results_df.columns, y=results_df.loc[model], mode='lines+markers', name=model))
        fig.update_layout(height=400, hovermode='x unified', margin=dict(l=0, r=0, t=0, b=0))
        fig.update_xaxes(autorange=True)
        fig.update_yaxes(autorange=True)
        st.plotly_chart(fig, use_container_width=True)
        
        # Bar charts in 2x2 grid
        col1, col2 = st.columns(2)
        
        with col1:
            fig1 = px.bar(results_df, y='Accuracy', title='Accuracy by Model', height=350)
            fig1.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            fig1.update_yaxes(autorange=True)
            st.plotly_chart(fig1, use_container_width=True)
            
            fig3 = px.bar(results_df, y='F1 Score', title='F1 Score by Model', height=350)
            fig3.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            fig3.update_yaxes(autorange=True)
            st.plotly_chart(fig3, use_container_width=True)
        
        with col2:
            fig2 = px.bar(results_df, y='AUC Score', title='AUC Score by Model', height=350)
            fig2.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            fig2.update_yaxes(autorange=True)
            st.plotly_chart(fig2, use_container_width=True)
            
            fig4 = px.bar(results_df, y='MCC Score', title='MCC Score by Model', height=350)
            fig4.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            fig4.update_yaxes(autorange=True)
            st.plotly_chart(fig4, use_container_width=True)
        
        add_footer()
    
    elif st.session_state.page == "predictions":
        st.header("Make Predictions with Trained Models")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Prediction Method")
            prediction_method = st.radio(
                "Choose prediction method:",
                ["Use Sliders", "Manual Input"],
                key="pred_method"
            )
        
        with col2:
            st.subheader("Model Selection")
            selected_models = st.multiselect(
                "Select models to use:",
                list(models.keys()),
                default=list(models.keys())
            )
            
            if not selected_models:
                st.warning("Please select at least one model")
                selected_models = list(models.keys())[:1]
        
        st.markdown("---")
        st.subheader("Feature Values")
        
        # Create input sliders for all 30 features
        feature_values = []
        
        if prediction_method == "Use Sliders":
            cols = st.columns(5)
            for idx, feature in enumerate(feature_names):
                col = cols[idx % 5]
                with col:
                    random_value = round(random.uniform(0, 100), 1)
                    value = st.slider(
                        feature,
                        min_value=0.0,
                        max_value=100.0,
                        value=random_value,
                        step=0.1,
                        key=f"slider_{feature}"
                    )
                    feature_values.append(value)
        else:
            input_text = st.text_area(
                "Enter 30 feature values separated by commas:",
                placeholder="10.5, 20.3, 15.2, ..."
            )
            if input_text:
                try:
                    feature_values = list(map(float, input_text.split(',')))
                    if len(feature_values) != 30:
                        st.error(f"Please enter exactly 30 values. You entered {len(feature_values)}")
                        feature_values = []
                except:
                    st.error("Invalid input. Please enter numeric values separated by commas")
                    feature_values = []
        
        # Make predictions
        if st.button("Predict", use_container_width=True, key="predict_sliders"):
            if len(feature_values) == 30:
                X_input = np.array(feature_values).reshape(1, -1)
                X_scaled = scaler.transform(X_input)
                
                st.markdown("---")
                st.subheader("Predictions from Selected Models")
                
                predictions_list = []
                for model_name in selected_models:
                    model = models[model_name]
                    pred = model.predict(X_scaled)[0]
                    pred_proba = model.predict_proba(X_scaled)[0]
                    
                    predictions_list.append({
                        'Model': model_name,
                        'Prediction': 'Malignant (1)' if pred == 1 else 'Benign (0)',
                        'Class 0 Probability': f"{pred_proba[0]:.4f}",
                        'Class 1 Probability': f"{pred_proba[1]:.4f}"
                    })
                
                predictions_df = pd.DataFrame(predictions_list)
                display_styled_dataframe(predictions_df, "Model Predictions")
            else:
                st.error("Please ensure all 30 feature values are provided")
        
        add_footer()
    
    elif st.session_state.page == "comparison":
        st.header("Detailed Metrics Comparison")
        st.markdown("---")
        
        # Multiselect for metrics - all by default
        selected_metrics = st.multiselect(
            "Select Metrics:",
            results_df.columns.tolist(),
            default=results_df.columns.tolist()
        )
        
        if selected_metrics:
            for metric in selected_metrics:
                fig = px.bar(results_df[[metric]], title=f'{metric} by Model', height=350, color=metric)
                fig.update_layout(margin=dict(l=0, r=0, t=30, b=0))
                fig.update_yaxes(autorange=True)
                st.plotly_chart(fig, use_container_width=True)
                
                st.metric(f"Best {metric}", 
                         f"{results_df[metric].max():.4f}",
                         f"{results_df[metric].idxmax()}")
            
            st.markdown("---")
            st.subheader("All Models - Selected Metrics")
            display_styled_dataframe(results_df[selected_metrics].round(4))
        
        add_footer()
    
    elif st.session_state.page == "upload":
        st.header("Upload & Test Dataset")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Step 1: Download Sample Data")
            
            # Load test data from breast cancer dataset
            cancer_data = load_breast_cancer()
            X = cancer_data.data
            y = cancer_data.target
            
            # Use last 20 samples as test example
            sample_test_df = pd.DataFrame(
                X[-20:],
                columns=cancer_data.feature_names
            )
            sample_test_df['Target'] = y[-20:]
            
            # Create CSV for download
            csv_sample = sample_test_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample Test Data (CSV)",
                data=csv_sample,
                file_name="sample_test_data.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            st.info("Sample file contains 20 test instances with all 30 features + Target column")
        
        with col2:
            st.subheader("Step 2: Select Model")
            upload_model = st.selectbox(
                "Choose model for predictions:",
                list(models.keys()),
                key="upload_model_select"
            )
            
            st.info(f"Selected: **{upload_model}**")
        
        st.markdown("---")
        st.subheader("Step 3: Upload Your Dataset")
        
        # Initialize session state for default upload
        if "default_uploaded" not in st.session_state:
            st.session_state.default_uploaded = True
            st.session_state.default_df = sample_test_df
            # Note: do NOT set a random default model here; the selected model in the dropdown
            # should always control which model is used for predictions.

        uploaded_file = st.file_uploader(
            "Upload CSV file (first 30 columns should be features):",
            type=['csv'],
            help="CSV file with features. Optional: include 'Target' column for evaluation"
        )
        
        # Priority: uploaded file > default
        processing_df = None
        processing_model = None
        
        if uploaded_file is not None:
            processing_df = pd.read_csv(uploaded_file)
            processing_model = upload_model
        elif st.session_state.default_uploaded:
            processing_df = st.session_state.default_df
            # Use the model selected in the dropdown even when using the default sample data
            processing_model = upload_model
            st.info(f"📊 Using default sample data with **{processing_model}** model (upload your own file to replace)")

        if processing_df is not None and processing_model is not None:
            try:
                uploaded_df = processing_df
                
                st.markdown("---")
                st.subheader("Data Preview")
                st.info(f"Dataset shape: {uploaded_df.shape[0]} rows × {uploaded_df.shape[1]} columns")
                
                # Show preview
                display_styled_dataframe(uploaded_df.head(10), "First 10 Rows")
                
                # Prepare features
                has_target = 'Target' in uploaded_df.columns
                
                if has_target:
                    X_upload = uploaded_df.iloc[:, :30].values
                    y_upload = uploaded_df['Target'].values
                else:
                    X_upload = uploaded_df.iloc[:, :30].values
                    y_upload = None
                
                st.markdown("---")
                st.subheader("Predictions")
                
                # Make predictions
                X_scaled_upload = scaler.transform(X_upload)
                selected_model = models[processing_model]
                predictions = selected_model.predict(X_scaled_upload)
                pred_proba = selected_model.predict_proba(X_scaled_upload)
                
                # Create results dataframe
                results_upload = pd.DataFrame({
                    'Prediction': ['Malignant (1)' if p == 1 else 'Benign (0)' for p in predictions],
                    'Class 0 Prob': pred_proba[:, 0].round(4),
                    'Class 1 Prob': pred_proba[:, 1].round(4)
                })
                
                if y_upload is not None:
                    results_upload['Actual'] = ['Malignant (1)' if p == 1 else 'Benign (0)' for p in y_upload]
                    results_upload['Correct'] = predictions == y_upload
                
                display_styled_dataframe(results_upload, f"Predictions using {processing_model}")

                # Metrics if target is available
                if y_upload is not None:
                    st.markdown("---")
                    st.subheader("Evaluation Metrics")

                    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

                    accuracy = accuracy_score(y_upload, predictions)
                    precision = precision_score(y_upload, predictions, zero_division=0)
                    recall = recall_score(y_upload, predictions, zero_division=0)
                    f1 = f1_score(y_upload, predictions, zero_division=0)

                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Accuracy", f"{accuracy:.4f}")
                    col2.metric("Precision", f"{precision:.4f}")
                    col3.metric("Recall", f"{recall:.4f}")
                    col4.metric("F1 Score", f"{f1:.4f}")

                    st.markdown("---")
                    st.subheader("Confusion Matrix")

                    cm = confusion_matrix(y_upload, predictions)

                    col_left, col_center, col_right = st.columns([1, 2, 1])
                    with col_center:
                        fig, ax = plt.subplots(figsize=(5, 4))
                        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])
                        disp.plot(ax=ax, cmap='Blues')
                        plt.title(f"Confusion Matrix - {processing_model}", fontsize=11, fontweight='bold', color='#FF8C42')
                        plt.tight_layout()
                        st.pyplot(fig)

                    st.markdown("---")

                    report = classification_report(y_upload, predictions, target_names=['Benign', 'Malignant'], output_dict=True)
                    report_df = pd.DataFrame(report).transpose()

                    display_styled_dataframe(report_df.round(4), "Classification Report")
                
                # Download predictions
                csv_predictions = results_upload.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions (CSV)",
                    data=csv_predictions,
                    file_name=f"predictions_{processing_model.replace(' ', '_')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
        
        add_footer()

    elif st.session_state.page == "dataset":
        st.header("About the Dataset")
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Dataset Information")
            st.markdown("""
            **Dataset Name:** Breast Cancer Wisconsin (Diagnostic)
            
            **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
            
            **Original Paper:** [Wolberg et al., 1995](https://pubmed.ncbi.nlm.nih.gov/9144026/)
            
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
        
        add_footer()

else:
    st.error("Models not loaded. Please ensure the model files are trained and saved.")
    add_footer()
