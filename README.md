# Citi Bike Explorer 🚲

An interactive Streamlit dashboard for exploring Citi Bike rider behavior, predicting user type, tracking machine learning experiments, and interpreting model decisions using explainability tools.

## Overview

Citi Bike Explorer is a multi-page data application built to analyze rider behavior using Citi Bike trip data from 2020. The app combines exploratory data analysis, machine learning, experiment tracking, and model explainability in a single interface.

Users can:
- explore demographic and behavioral trends in Citi Bike usage
- visualize ride patterns across age, gender, and user type
- predict whether a rider is more likely to be a **Subscriber** or **Customer**
- compare machine learning models and evaluate performance
- track model runs with MLflow and DagsHub
- interpret feature importance with SHAP and partial dependence plots

## Features

### 1. Introduction
- dataset preview
- column dictionary
- summary statistics
- introductory visualizations on gender, age, and user type

### 2. Visualization
- trip duration by gender
- average trip duration by gender and user type
- trip frequency by gender and user type
- birth year density by user type
- time-of-day usage by age group

### 3. Model Prediction
- baseline linear regression evaluation
- logistic regression prediction interface
- interactive prediction based on:
  - age
  - gender
  - trip duration
- model comparison across:
  - Logistic Regression
  - Decision Tree
  - Random Forest

### 4. Model Tuning
- manual grid search for Decision Tree depth
- experiment logging with MLflow
- DagsHub integration for tracking metrics and parameters
- browsing and comparing past experiment runs

### 5. Explainability
- Random Forest feature importance
- SHAP bar plot
- SHAP beeswarm plot
- SHAP waterfall plot for individual predictions
- partial dependence plots for selected features

---

## Tech Stack

- **Python**
- **Streamlit**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **scikit-learn**
- **MLflow**
- **DagsHub**
- **SHAP**
- **Pillow**

---

## Machine Learning Workflow

The app uses rider features such as:
- age
- gender
- trip duration

to predict:
- **usertype** (`Customer` or `Subscriber`)

### Models included
- Linear Regression (baseline)
- Logistic Regression
- Decision Tree Classifier
- Random Forest Classifier

### Evaluation metrics
- Accuracy
- Precision
- Recall
- F1 Score
- MAE
- RMSE
- R²
- Cross-validation accuracy
- Confusion matrix
- Classification report

---

## Dataset

This project uses Citi Bike trip data from **2020**. The year was selected because it provides demographic fields such as birth year and gender, which are necessary for rider segmentation and prediction.

The app loads and combines multiple CSV files from a root dataset folder:

```bash
CitiBike_Trip_Data/
