# Hybrid Ensemble Model for Breast Cancer Patient Survival Prediction

This repository contains the implementation of a hybrid ensemble machine learning model developed to predict the survival status (alive/deceased) of breast cancer patients in India. The approach combines **Mutual Information (MI)** and **Principal Component Analysis (PCA)** for feature selection, followed by a **stacked ensemble** architecture using Random Forest and XGBoost as base learners with Logistic Regression as the meta-learner.

Due to data use restrictions from the data provider (ICMR-NIRT), only a **sample of the dataset** is included to enable reproducibility of the methodology and results.

## Problem Context

Breast cancer remains a leading cause of cancer mortality in India, with over 221,000 new cases and 82,000 deaths reported in 2023. Accurate survival prediction can inform clinical decisions and public health strategies. However, many existing models are limited by traditional feature selection methods and poor scalability on high-dimensional clinical data.

This study addresses these gaps by:
- Introducing a hybrid MI+PCA feature selection pipeline
- Employing a stacking-based ensemble for robust classification
- Evaluating performance using both train-test split and k-fold cross-validation

## Repository Structure

```
.
├── Hybrid_Ensemble_Final.ipynb          # Full Python Notebook (as provided)
├── Hybrid_Ensemble_Final.py             # Full python Script
├── sample_data.csv                      # 20 Samples from dataset
└── README.md
```

**Note**: The full dataset is proprietary to SEER and was obtained through ICMR-NIRT and cannot be redistributed. `sample_data.csv` contains 20 samples and is for illustrative purposes only.


## Methodology Summary

### Feature Selection
- **Mutual Information (MI)** identifies the top 10 most informative features.
- **PCA** reduces these to 5 principal components to mitigate multicollinearity and dimensionality.
- Final input for modeling: `mipca_bc.csv`.

### Stacked Ensemble Architecture
- **Base learners**: Random Forest, XGBoost
- **Meta-learner**: Logistic Regression trained on predicted probabilities from base models
- Compared against: AdaBoost, Gaussian Naive Bayes, standalone XGBoost, and Logistic Regression

### Evaluation
- Metrics: Accuracy, Precision, Recall, Specificity, F1-Score
- Validation strategies:
  - 80/20 train-test split
  - 10-fold cross-validation (individual models)
  - 5-fold cross-validation (ensemble model)


## Key Results

The proposed ensemble model (**ENSP**) outperformed all baseline methods across all evaluation metrics on the sample dataset:

| Model                | Accuracy (%) | Precision (%) | Recall (%) | Specificity (%) | F1-Score (%) |
|----------------------|--------------|---------------|------------|------------------|--------------|
| **Proposed (ENSP)**  | **94.67**    | **95.54**     | **96.20**  | **91.67**        | **95.87**    |
| XG Boosting          | 88.67        | 88.59         | 94.78      | 77.33            | 91.58        |
| Ada Boosting         | 83.37        | 84.79         | 90.69      | 69.78            | 87.64        |
| Gaussian NB          | 71.16        | 73.14         | 87.94      | 40.00            | 79.86        |
| Logistic Regression  | 71.91        | 73.82         | 87.98      | 42.05            | 80.28        |

These results demonstrate that the hybrid **MI+PCA feature selection** combined with a **stacked ensemble (Random Forest + XGBoost → Logistic Regression)** significantly enhances predictive performance—particularly in specificity and overall accuracy—compared to conventional classifiers.