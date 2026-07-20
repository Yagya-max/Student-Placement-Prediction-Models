"""
Standalone model training script for Student Placement Prediction
Trains and saves the best model to disk for later use
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)

def load_data(filepath):
    """Load and return placement dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"Classes distribution:\n{df['placed'].value_counts()}\n")
    return df

def prepare_data(df, test_size=0.2, random_state=42):
    """Prepare and split data"""
    X = df.drop('placed', axis=1)
    y = df['placed']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training set size: {X_train_scaled.shape[0]}")
    print(f"Test set size: {X_test_scaled.shape[0]}\n")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns

def train_models(X_train, X_test, y_train, y_test):
    """Train multiple classification models"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation score
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
        
        results[name] = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred)
        }
        
        print(f"  ✓ Accuracy: {accuracy:.4f}")
        print(f"  ✓ Precision: {precision:.4f}")
        print(f"  ✓ Recall: {recall:.4f}")
        print(f"  ✓ F1-Score: {f1:.4f}")
        print(f"  ✓ ROC-AUC: {roc_auc:.4f}")
        print(f"  ✓ Cross-Validation (5-fold): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})\n")
    
    return results

def print_comparison(results):
    """Print detailed model comparison"""
    print("=" * 80)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 80)
    
    comparison = []
    for model_name, metrics in results.items():
        comparison.append({
            'Model': model_name,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1-Score': metrics['f1'],
            'ROC-AUC': metrics['roc_auc'],
            'CV Mean': metrics['cv_mean']
        })
    
    df_comparison = pd.DataFrame(comparison)
    print(df_comparison.to_string(index=False))
    print()
    
    best_model = max(results.keys(), key=lambda x: results[x]['f1'])
    print(f"🏆 BEST MODEL (by F1-Score): {best_model}")
    print(f"   F1-Score: {results[best_model]['f1']:.4f}")
    print()
    
    return best_model

def save_models(best_model_name, results, scaler, feature_names, output_dir='models'):
    """Save trained models and artifacts"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("SAVING MODELS AND ARTIFACTS")
    print("=" * 80)
    
    # Save best model
    best_model = results[best_model_name]['model']
    best_model_path = output_path / f'best_model_{best_model_name.replace(" ", "_")}.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    print(f"✓ Best model saved: {best_model_path}")
    
    # Save scaler
    scaler_path = output_path / 'scaler.pkl'
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✓ Scaler saved: {scaler_path}")
    
    # Save feature names
    feature_names_path = output_path / 'feature_names.pkl'
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"✓ Feature names saved: {feature_names_path}")
    
    # Save metrics as JSON
    metrics_dict = {}
    for model_name, metrics in results.items():
        metrics_dict[model_name] = {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1': float(metrics['f1']),
            'roc_auc': float(metrics['roc_auc']),
            'cv_mean': float(metrics['cv_mean']),
            'cv_std': float(metrics['cv_std']),
            'confusion_matrix': metrics['confusion_matrix'],
            'classification_report': metrics['classification_report']
        }
    
    metrics_path = output_path / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics_dict, f, indent=4)
    print(f"✓ Metrics saved: {metrics_path}")
    
    # Save all models
    all_models_path = output_path / 'all_models.pkl'
    models_to_save = {name: results[name]['model'] for name in results.keys()}
    with open(all_models_path, 'wb') as f:
        pickle.dump(models_to_save, f)
    print(f"✓ All models saved: {all_models_path}")
    
    print()

def load_and_predict(model_path, scaler_path, features):
    """Load saved model and make predictions"""
    print("=" * 80)
    print("LOADING MODEL AND MAKING PREDICTIONS")
    print("=" * 80)
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from {model_path}")
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✓ Scaler loaded from {scaler_path}")
    
    # Scale features and predict
    features_scaled = scaler.transform([features])
    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0]
    
    print(f"\nPrediction Result:")
    print(f"  Status: {'PLACED' if prediction == 1 else 'NOT PLACED'}")
    print(f"  Probability (Not Placed): {probability[0]:.4f}")
    print(f"  Probability (Placed): {probability[1]:.4f}")
    print()

def main():
    """Main execution"""
    print("\n" + "=" * 80)
    print("STUDENT PLACEMENT PREDICTION - MODEL TRAINING")
    print("=" * 80 + "\n")
    
    # Load data
    df = load_data('sample_placement_data.csv')
    
    # Prepare data
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data(df)
    
    # Train models
    results = train_models(X_train, X_test, y_train, y_test)
    
    # Print comparison
    best_model_name = print_comparison(results)
    
    # Save models
    save_models(best_model_name, results, scaler, feature_names)
    
    # Example prediction
    print("=" * 80)
    print("EXAMPLE PREDICTION")
    print("=" * 80)
    print("\nStudent Profile:")
    print("  CGPA: 8.2")
    print("  Internships: 2")
    print("  Projects: 5")
    print("  Coding Score: 85")
    print("  Aptitude Score: 80")
    print("  Communication Score: 82")
    print("  Extra-curricular: 1")
    print()
    
    example_features = [8.2, 2, 5, 85, 80, 82, 1]
    load_and_predict('models/best_model_Random_Forest.pkl', 'models/scaler.pkl', example_features)
    
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
