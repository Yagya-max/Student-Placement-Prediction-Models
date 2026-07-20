# ══════════════════════════════════════════════════════════════════════════
# PART 1: IMPORTS & PAGE SETUP
# ------------------------------------------------------------------------
# Loads every library the app needs, then configures the Streamlit page
# (browser tab title, wide layout) and renders the top header.
# Nothing computational happens here — this is pure setup.
# ══════════════════════════════════════════════════════════════════════════

import streamlit as st                      # Turns this script into a web app
import pandas as pd                          # Tabular data (DataFrames)
import numpy as np                           # Numerical arrays, random generation
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve, auc)
import matplotlib.pyplot as plt              # Static charts (bar, line)
import seaborn as sns                        # Confusion matrix heatmap
import pickle                                # (Reserved for model saving)
import io

# Configure Streamlit
st.set_page_config(page_title="Student Placement Predictor", layout="wide")
st.title("🎓 Student Placement Prediction System")
st.markdown("Train classification models to predict student placement success")


# ══════════════════════════════════════════════════════════════════════════
# PART 2: DATA GENERATION & DATA SOURCE SELECTION
# ------------------------------------------------------------------------
# Defines how the dataset is created (synthetic generator), then lets the
# user choose in the sidebar whether to use generated data or upload a
# real CSV. Ends with `df` holding whichever dataset was chosen.
# ══════════════════════════════════════════════════════════════════════════

# Generate or load sample dataset
@st.cache_data
def generate_placement_data(n_samples=500):
    """Generate realistic student placement dataset"""
    np.random.seed(42)
    
    # Generate features
    data = {
        'cgpa': np.random.uniform(6.0, 9.5, n_samples),
        'internships': np.random.randint(0, 4, n_samples),
        'projects': np.random.randint(0, 8, n_samples),
        'coding_score': np.random.randint(40, 100, n_samples),
        'aptitude_score': np.random.randint(40, 100, n_samples),
        'communication_score': np.random.randint(40, 100, n_samples),
        'extra_curricular': np.random.randint(0, 2, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Generate placement status based on features
    # Students with better scores are more likely to be placed
    placement_probability = (
        (df['cgpa'] / 10) * 0.3 +
        (df['internships'] / 3) * 0.2 +
        (df['projects'] / 8) * 0.15 +
        (df['coding_score'] / 100) * 0.15 +
        (df['aptitude_score'] / 100) * 0.1 +
        (df['communication_score'] / 100) * 0.1
    )
    
    df['placed'] = (placement_probability + np.random.normal(0, 0.1, n_samples)) > 0.5
    df['placed'] = df['placed'].astype(int)
    
    return df

# Sidebar controls
st.sidebar.header("⚙️ Configuration")
dataset_option = st.sidebar.radio("Data Source", ["Generate Sample Data", "Upload CSV"])

if dataset_option == "Generate Sample Data":
    n_samples = st.sidebar.slider("Number of samples", 100, 1000, 500)
    df = generate_placement_data(n_samples)
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        st.sidebar.info("Please upload a CSV file")
        df = None

# ══════════════════════════════════════════════════════════════════════════
# PART 3: DATASET OVERVIEW & PREPROCESSING
# ------------------------------------------------------------------------
# Shows summary stats/preview of the data, then prepares it for modeling:
#   - Splits features (X) from target (y)
#   - Splits into train (80%) / test (20%) sets, stratified by class
#   - Scales features with StandardScaler (fit on train, applied to test)
# Everything below this point assumes X_train_scaled / X_test_scaled exist.
# ══════════════════════════════════════════════════════════════════════════

# Display dataset info
if df is not None:
    st.header("📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Students", len(df))
    with col2:
        placed_count = (df['placed'] == 1).sum()
        st.metric("Placed", f"{placed_count} ({placed_count/len(df)*100:.1f}%)")
    with col3:
        not_placed = (df['placed'] == 0).sum()
        st.metric("Not Placed", f"{not_placed} ({not_placed/len(df)*100:.1f}%)")
    with col4:
        st.metric("Features", df.shape[1] - 1)
    
    st.write(df.head(10))
    
    # Display statistics
    st.subheader("📈 Statistical Summary")
    st.write(df.describe())
    
    # Prepare data for modeling
    X = df.drop('placed', axis=1)
    y = df['placed']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # ══════════════════════════════════════════════════════════════════════
    # PART 4: MODEL TRAINING & EVALUATION
    # ----------------------------------------------------------------------
    # Triggered by the "Train All Models" button. Defines the 3 classifiers,
    # then for each one: trains it (.fit), predicts on the held-out test
    # set (.predict / .predict_proba), and computes all 5 evaluation
    # metrics + confusion matrix. Everything is stored in `results` dict,
    # keyed by model name, for use in the next part.
    # ══════════════════════════════════════════════════════════════════════

    # Model training section
    st.header("🤖 Model Training & Comparison")
    
    if st.button("🚀 Train All Models", key="train_models"):
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        }
        
        results = {}
        progress_bar = st.progress(0)
        
        for idx, (name, model) in enumerate(models.items()):
            with st.spinner(f"Training {name}..."):
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                # Calculate metrics
                results[name] = {
                    'model': model,
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba),
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'classification_report': classification_report(y_test, y_pred)
                }
            
            progress_bar.progress((idx + 1) / len(models))
        
        # ══════════════════════════════════════════════════════════════════
        # PART 5: RESULTS VISUALIZATION & MODEL COMPARISON DASHBOARD
        # ----------------------------------------------------------------
        # Takes the `results` dict from Part 4 and renders it visually:
        #   - Comparison table (all models, all metrics)
        #   - Grouped bar chart + best-model-by-F1 highlight chart
        #   - Deep dive on the best model: confusion matrix, classification
        #     report, ROC curves, feature importance (if tree-based)
        # Ends by saving the best model + scaler into st.session_state so
        # Part 6 (prediction form) can use them after this block reruns.
        # ══════════════════════════════════════════════════════════════════

        # Display model comparison
        st.subheader("📋 Model Performance Comparison")
        
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results.keys()],
            'Precision': [results[m]['precision'] for m in results.keys()],
            'Recall': [results[m]['recall'] for m in results.keys()],
            'F1-Score': [results[m]['f1'] for m in results.keys()],
            'ROC-AUC': [results[m]['roc_auc'] for m in results.keys()]
        })
        
        # Format only numeric columns, keep 'Model' as text
        formatter_dict = {col: "{:.4f}" for col in comparison_df.columns if col != 'Model'}
        st.dataframe(comparison_df.style.format(formatter_dict), use_container_width=True)
        
        # Visualize metrics comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Metrics comparison bar chart
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, model_name in enumerate(results.keys()):
            values = [
                results[model_name]['accuracy'],
                results[model_name]['precision'],
                results[model_name]['recall'],
                results[model_name]['f1'],
                results[model_name]['roc_auc']
            ]
            axes[0].bar(x + i*width, values, width, label=model_name)
        
        axes[0].set_xlabel('Metrics')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Model Performance Comparison')
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(metrics, rotation=45)
        axes[0].legend()
        axes[0].set_ylim([0, 1.1])
        axes[0].grid(axis='y', alpha=0.3)
        
        # Best model highlight
        best_model = max(results.keys(), key=lambda x: results[x]['f1'])
        colors = ['#2ecc71' if m == best_model else '#3498db' for m in results.keys()]
        axes[1].barh(list(results.keys()), 
                     [results[m]['f1'] for m in results.keys()],
                     color=colors)
        axes[1].set_xlabel('F1-Score')
        axes[1].set_title('Best Model by F1-Score')
        axes[1].set_xlim([0, 1])
        
        for i, v in enumerate([results[m]['f1'] for m in results.keys()]):
            axes[1].text(v + 0.02, i, f'{v:.4f}', va='center')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Detailed analysis for best model
        st.subheader(f"🏆 Best Model: {best_model}")
        
        col1, col2, col3, col4 = st.columns(4)
        best_results = results[best_model]
        
        with col1:
            st.metric("Accuracy", f"{best_results['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{best_results['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{best_results['recall']:.4f}")
        with col4:
            st.metric("ROC-AUC", f"{best_results['roc_auc']:.4f}")
        
        # Confusion Matrix for best model
        st.markdown("#### Confusion Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(best_results['confusion_matrix'], annot=True, fmt='d',
                   cmap='Blues', cbar=False, ax=ax,
                   xticklabels=['Not Placed', 'Placed'],
                   yticklabels=['Not Placed', 'Placed'])
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        st.pyplot(fig)
        
        # Classification Report
        st.markdown("#### Classification Report")
        st.text(best_results['classification_report'])
        
        # ROC Curves
        st.markdown("#### ROC Curves")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for model_name in results.keys():
            fpr, tpr, _ = roc_curve(y_test, results[model_name]['y_pred_proba'])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})', linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curves')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
        
        # Feature importance (for tree-based models)
        if best_model in ['Decision Tree', 'Random Forest']:
            st.markdown("#### Feature Importance")
            feature_importance = pd.DataFrame({
                'Feature': X.columns,
                'Importance': best_results['model'].feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(feature_importance['Feature'], feature_importance['Importance'])
            ax.set_xlabel('Importance')
            ax.set_title('Feature Importance')
            plt.tight_layout()
            st.pyplot(fig)
            
            st.dataframe(feature_importance, use_container_width=True)
        
        # Store best model for prediction
        st.session_state.best_model = best_results['model']
        st.session_state.scaler = scaler
        st.session_state.feature_names = X.columns
    
    # ══════════════════════════════════════════════════════════════════════
    # PART 6: PREDICTION INTERFACE (LIVE INFERENCE)
    # ----------------------------------------------------------------------
    # Renders an input form for a NEW student's details. On button click,
    # it takes those 7 values, scales them with the SAME fitted scaler
    # used in training (critical — must match training's transformation),
    # then calls .predict() and .predict_proba() on the best model stored
    # in session_state (from Part 5) to show a placement verdict + chart.
    # ══════════════════════════════════════════════════════════════════════

    # Prediction section
    st.header("🔮 Make Predictions")
    
    st.markdown("Enter student details to predict placement chances:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        cgpa = st.number_input("CGPA", min_value=6.0, max_value=10.0, value=7.5, step=0.1)
        internships = st.number_input("Number of Internships", min_value=0, max_value=5, value=1)
        projects = st.number_input("Number of Projects", min_value=0, max_value=10, value=3)
    
    with col2:
        coding_score = st.number_input("Coding Score (0-100)", min_value=0, max_value=100, value=75)
        aptitude_score = st.number_input("Aptitude Score (0-100)", min_value=0, max_value=100, value=70)
    
    with col3:
        communication_score = st.number_input("Communication Score (0-100)", min_value=0, max_value=100, value=80)
        extra_curricular = st.selectbox("Extra-curricular Activities", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    
    if st.button("📊 Predict Placement Status", key="predict"):
        if 'best_model' not in st.session_state:
            st.warning("⚠️ Please train models first!")
        else:
            input_data = np.array([[
                cgpa, internships, projects, coding_score, 
                aptitude_score, communication_score, extra_curricular
            ]])
            
            input_scaled = st.session_state.scaler.transform(input_data)
            prediction = st.session_state.best_model.predict(input_scaled)[0]
            probability = st.session_state.best_model.predict_proba(input_scaled)[0]
            
            col1, col2 = st.columns(2)
            
            with col1:
                if prediction == 1:
                    st.success("✅ **LIKELY TO BE PLACED**")
                    st.metric("Placement Probability", f"{probability[1]*100:.2f}%")
                else:
                    st.error("❌ **UNLIKELY TO BE PLACED**")
                    st.metric("Placement Probability", f"{probability[1]*100:.2f}%")
            
            with col2:
                fig, ax = plt.subplots(figsize=(8, 6))
                categories = ['Not Placed', 'Placed']
                colors = ['#e74c3c', '#2ecc71']
                bars = ax.bar(categories, probability, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
                ax.set_ylabel('Probability')
                ax.set_title('Placement Prediction Probability')
                ax.set_ylim([0, 1])
                
                for bar, prob in zip(bars, probability):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                           f'{prob*100:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
                
                st.pyplot(fig)

else:
    st.info("👆 Select data source in the sidebar to get started!")