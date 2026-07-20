# 🎓 Student Placement Prediction System

A complete machine learning project that trains classification models to predict whether a student will be placed based on academic performance, technical skills, and aptitude scores.

## 📋 Project Overview

This system compares multiple machine learning algorithms and deploys the best-performing model through an interactive Streamlit web application. It includes:

- **Three Classification Models**: Logistic Regression, Decision Tree, and Random Forest
- **Comprehensive Model Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Interactive Dashboard**: Real-time predictions and performance visualization
- **Feature Analysis**: Feature importance for tree-based models

## 🛠️ Tech Stack

- **Python 3.8+**
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Scikit-Learn**: Machine learning algorithms
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
- **Streamlit**: Web application framework
- **Matplotlib & Seaborn**: Data visualization

## 📦 Installation

### 1. Clone or Download the Project

```bash
# Create a project directory
mkdir placement-prediction
cd placement-prediction
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install streamlit==1.28.1
pip install pandas==2.1.3
pip install numpy==1.24.3
pip install scikit-learn==1.3.2
pip install matplotlib==3.8.2
pip install seaborn==0.13.0
```

## 🚀 Quick Start

### Run the Streamlit Application

```bash
streamlit run placement_prediction_system.py
```

The application will open in your browser at `http://localhost:8501`

## 📊 Features

### 1. **Data Input**
   - Generate sample dataset with 100-1000 students
   - Upload custom CSV file with your data

### 2. **Model Training**
   - Click "🚀 Train All Models" to train all three algorithms
   - Models are trained on 80% of data, tested on 20%
   - Features are normalized using StandardScaler

### 3. **Performance Comparison**
   - Side-by-side comparison table of all models
   - Bar charts comparing key metrics
   - ROC curves for each model
   - Confusion matrices

### 4. **Model Selection**
   - Best model selected automatically based on F1-Score
   - Detailed metrics for the best model
   - Feature importance visualization

### 5. **Predictions**
   - Interactive form to input student details
   - Real-time placement probability prediction
   - Visual representation of prediction confidence

## 📈 Dataset Structure

The system expects the following columns:

| Column | Type | Range | Description |
|--------|------|-------|-------------|
| `cgpa` | Float | 6.0-10.0 | Cumulative GPA |
| `internships` | Int | 0-5 | Number of internships completed |
| `projects` | Int | 0-10 | Number of projects completed |
| `coding_score` | Int | 0-100 | Programming ability score |
| `aptitude_score` | Int | 0-100 | Logical reasoning score |
| `communication_score` | Int | 0-100 | Communication skills score |
| `extra_curricular` | Int | 0/1 | Participation in activities (0=No, 1=Yes) |
| `placed` | Int | 0/1 | Placement status (0=Not Placed, 1=Placed) |

### Sample Data

A sample dataset with 50 students is included: `sample_placement_data.csv`

## 🔍 Model Details

### Logistic Regression
- **Use Case**: Fast, interpretable linear model
- **Pros**: Low computational cost, works well for binary classification
- **Cons**: Assumes linear relationship between features and outcome

### Decision Tree
- **Use Case**: Interpretable, non-linear model with feature importance
- **Pros**: Captures non-linear patterns, shows decision paths
- **Cons**: Prone to overfitting; limited by max_depth parameter (set to 10)

### Random Forest
- **Use Case**: Ensemble method with improved accuracy and robustness
- **Pros**: Handles non-linearity, reduces overfitting, feature importance
- **Cons**: Slower training and prediction than single tree

## 📊 Evaluation Metrics

### Classification Metrics
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (true positives + false positives)
- **Recall**: True positives / (true positives + false negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve

### Visualizations
- **Confusion Matrix**: Shows true/false positives and negatives
- **ROC Curves**: Plots true positive rate vs false positive rate
- **Feature Importance**: Shows which features contribute most to predictions

## 💡 Usage Example

### Generate Sample Data and Train Models

1. Open the application
2. In the sidebar, select "Generate Sample Data"
3. Set the number of samples (500 is default)
4. Click "🚀 Train All Models"
5. View the comparison table and visualizations

### Make a Prediction

1. Scroll to "🔮 Make Predictions" section
2. Enter a student's details:
   - CGPA: 8.2
   - Internships: 2
   - Projects: 5
   - Coding Score: 85
   - Aptitude Score: 80
   - Communication Score: 82
   - Extra-curricular: Yes
3. Click "📊 Predict Placement Status"
4. View the prediction probability

### Upload Custom Data

1. Prepare a CSV file with columns matching the dataset structure
2. In the sidebar, select "Upload CSV"
3. Upload your file
4. Click "🚀 Train All Models"

## 📁 Project Files

```
placement-prediction/
├── placement_prediction_system.py    # Main Streamlit application
├── sample_placement_data.csv         # Sample dataset (50 students)
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## 🔧 Advanced Features

### Cross-Validation
Models use stratified train-test split (20% test, 80% train) with random state for reproducibility.

### Feature Scaling
StandardScaler normalizes features to have zero mean and unit variance, improving model convergence.

### Model Hyperparameters
- **Logistic Regression**: max_iter=1000
- **Decision Tree**: max_depth=10 (prevents overfitting)
- **Random Forest**: n_estimators=100, n_jobs=-1 (parallel processing)

## 🎯 Expected Results

With the sample dataset, you should see:
- **Best Model**: Usually Random Forest (F1-Score: 0.85-0.90)
- **Accuracy**: 80-90% across all models
- **ROC-AUC**: 0.85-0.95
- **Training Time**: < 1 second for all models

## 🐛 Troubleshooting

### "Module not found" error
```bash
pip install -r requirements.txt
```

### Streamlit not starting
Ensure you're in the correct directory and Python version is 3.8+:
```bash
python --version
streamlit --version
```

### Models not training
- Check that data has the required columns
- Ensure no missing values in critical columns
- Verify placement column (0/1 values)

## 📚 Learning Outcomes

After completing this project, you'll understand:

1. **Data Preprocessing**: Handling datasets, feature scaling
2. **Classification Models**: Logistic Regression, Decision Trees, Ensemble Methods
3. **Model Evaluation**: Metrics, confusion matrices, ROC curves
4. **Model Comparison**: Selecting best model based on performance
5. **Web Deployment**: Building interactive ML applications with Streamlit
6. **Data Visualization**: Creating professional charts and plots

## 🚀 Extension Ideas

### Difficulty: Easy
- Add more models (SVM, Naive Bayes, KNN)
- Implement hyperparameter tuning (GridSearchCV)
- Export trained model as pickle file
- Add data statistics and distributions

### Difficulty: Medium
- Implement cross-validation k-fold
- Add feature engineering and selection
- Create model comparison report (PDF)
- Add SHAP values for model interpretability

### Difficulty: Hard
- Implement automated ML (AutoML)
- Add deep learning models (Neural Networks)
- Deploy to cloud (Heroku, AWS, GCP)
- Create model API with FastAPI

## 📖 References

- [Scikit-Learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Pandas Documentation](https://pandas.pydata.org/)
- [ROC Curves Explained](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_curve.html)

## 📝 License

This project is open-source and available for educational purposes.

## ✨ Key Highlights

✅ **Production-Ready**: Fully functional end-to-end ML pipeline
✅ **User-Friendly**: Interactive Streamlit interface
✅ **Comprehensive**: Multiple models, metrics, and visualizations
✅ **Scalable**: Works with custom datasets
✅ **Educational**: Well-commented code with explanations

---

**Happy Predicting! 🎓**

For questions or improvements, feel free to modify and extend the project.
