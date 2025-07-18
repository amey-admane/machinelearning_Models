# Voice Gender Recognition - Machine Learning Classification Project

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Machine%20Learning-green.svg)](https://scikit-learn.org/)

## 📊 Project Overview

This project implements multiple machine learning algorithms to classify voice samples as male or female based on acoustic properties. The project demonstrates comprehensive machine learning skills including data preprocessing, model implementation, hyperparameter tuning, and performance evaluation across different algorithms.

## 🎯 Objective

To develop and compare various machine learning models for gender recognition using voice and speech features, achieving high accuracy in binary classification tasks.

## 📈 Dataset

The project uses the **Voice Gender Recognition Dataset** containing:
- **3,168 voice samples** (1,584 male, 1,584 female)
- **20 acoustic features** including:
  - `meanfreq`: Mean frequency (kHz)
  - `sd`: Standard deviation of frequency
  - `median`: Median frequency
  - `Q25`, `Q75`: First and third quartiles
  - `IQR`: Interquartile range
  - `skew`: Skewness
  - `kurt`: Kurtosis
  - `sp.ent`: Spectral entropy
  - `sfm`: Spectral flatness
  - And 10 additional acoustic properties

### Model Evaluation Metrics
- **Accuracy Score**
- **Precision, Recall, F1-Score**
- **Confusion Matrix**
- **ROC Curve & AUC**
- **Cross-validation scores**

### Hyperparameter Optimization
- **GridSearchCV** for exhaustive parameter search
- **RandomizedSearchCV** for efficient parameter sampling
- **Cross-validation** for robust model selection

## 📊 Performance Results

| Algorithm | Accuracy | Precision | Recall | F1-Score |
|-----------|----------|-----------|--------|----------|
| SVM (RBF) | **98.3%** | 98.4% | 98.2% | 98.3% |
| Gradient Boosting | 98.1% | 98.2% | 98.0% | 98.1% |
| Random Forest | 97.9% | 98.0% | 97.8% | 97.9% |
| SVM (Linear) | 97.8% | 97.9% | 97.0% | 97.9% |
| KNN | 97.5% | 97.6% | 97.4% | 97.5% |
| Logistic Regression | 97.3% | 97.4% | 97.2% | 97.3% |

## 💻 Technologies Used

- **Python 3.8+**
- **Scikit-learn**: Machine learning algorithms
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Matplotlib & Seaborn**: Data visualization
- **Jupyter Notebook**: Interactive development environment
- **XGBoost**: Gradient boosting framework

## 📁 Project Structure

```
machinelearning_Models/
├── Decision Tree.ipynb          # Decision tree implementation
├── Ensemble.ipynb              # Ensemble methods (voting classifier)
├── Gradient Boosting.ipynb     # Gradient boosting with hyperparameter tuning
├── KNN.ipynb                   # K-Nearest Neighbors with optimization
├── Logistic Regression.ipynb   # Logistic regression with regularization
├── Random Forest.ipynb         # Random forest with feature importance
├── SVM.ipynb                   # Support Vector Machine (all kernels)
├── voice.csv                   # Voice dataset
└── README.md                   # Project documentation
```

## 🔍 Key Insights

1. **SVM with RBF kernel** achieved the highest accuracy (98.3%)
2. **Feature scaling** significantly improved model performance
3. **Hyperparameter tuning** enhanced accuracy by 2-3% across models
4. **Ensemble methods** provided robust and stable predictions
5. **Cross-validation** ensured model generalizability

## 🎓 Learning Outcomes

This project demonstrates proficiency in:
- **Machine Learning Fundamentals**: Understanding of various algorithms
- **Data Preprocessing**: Scaling, encoding, and splitting techniques
- **Model Optimization**: Hyperparameter tuning and cross-validation
- **Performance Evaluation**: Multiple metrics and statistical analysis
- **Python Programming**: Pandas, NumPy, Scikit-learn expertise
- **Data Visualization**: Matplotlib and Seaborn plotting
- **Statistical Analysis**: Understanding of bias-variance tradeoff
