# **II. Model Training**
## *Lib and Data Imports*
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
XB = pd.read_csv('mipca_bc.csv')
YB = pd.read_csv('BREASTCancerdata.csv',usecols=['STAT_REC'])

XB_train, XB_test, YB_train, YB_test = train_test_split(XB, YB, test_size=0.2, random_state=42)

scores_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall','Specificity','F1-Score'])

XB.columns
## Ada BOOST
from sklearn.ensemble import AdaBoostClassifier
ada_model = AdaBoostClassifier(

    n_estimators=100,  # number of weak learners

    learning_rate=1.0, # learning rate

    random_state=42

)

ada_model.fit(XB_train, YB_train)
## Gaussian NB
from sklearn.naive_bayes import GaussianNB
gnb_model = GaussianNB()

gnb_model.fit(XB_train, YB_train)
## XG BOOST
import xgboost as xgb
xgb_model = xgb.XGBClassifier(

    objective='binary:logistic',

    random_state=42,

    learning_rate=0.1,

    n_estimators=100,

    max_depth=3

)

xgb_model.fit(XB_train, YB_train)
## LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression, Lasso

logis_model = LogisticRegression(random_state=42)

logis_model.fit(XB_train, YB_train)
## Random Forest
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(

    n_estimators=100,  # number of trees

    max_depth=None,    # maximum depth of trees

    random_state=42    # for reproducibility

)

rf_model.fit(XB_train, YB_train)
## **STACKING ENSEMBLE APPROACH**
def create_meta_features(rf_pred, xgb_pred ):

      return np.column_stack((

              rf_pred[:, 1],

                         # Taking probability of class 1

                              xgb_pred[:, 1]
                                  ))
#Base Learners
meta_features_train = create_meta_features(

      rf_model.predict_proba(XB_train),

          xgb_model.predict_proba(XB_train)

              )
#Meta Learner
meta_classifier = LogisticRegression()

meta_classifier.fit(meta_features_train, YB_train)

# **III. Evaluation and Results**
## TRAIN-TEST SPLIT
### *ADA Boost*
YADAB = ada_model.predict(XB_test)
accuracy = accuracy_score(YB_test, YADAB)
precision = precision_score(YB_test, YADAB)
recall = recall_score(YB_test, YADAB)
f1 = f1_score(YB_test, YADAB)
tn, fp, fn, tp = confusion_matrix(YB_test, YADAB).ravel()
specificity = tn / (tn+fp)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1}")


new_row = pd.DataFrame({'Model': ['ADA'],
                        'Accuracy': [accuracy],
                        'Precision': [precision],
                        'Recall': [recall],
                        'Specificity': [specificity],
                        'F1-Score': [f1]})
scores_df = pd.concat([scores_df, new_row], ignore_index=True)

### *Guassian NB*
YGNBB = gnb_model.predict(XB_test)
accuracy = accuracy_score(YB_test, YGNBB)
precision = precision_score(YB_test, YGNBB)
recall = recall_score(YB_test, YGNBB)
f1 = f1_score(YB_test, YGNBB)
tn, fp, fn, tp = confusion_matrix(YB_test, YGNBB).ravel()
specificity = tn / (tn+fp)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1}")


new_row = pd.DataFrame({'Model': ['GNB'],
                        'Accuracy': [accuracy],
                        'Precision': [precision],
                        'Recall': [recall],
                        'Specificity': [specificity],
                        'F1-Score': [f1]})
scores_df = pd.concat([scores_df, new_row], ignore_index=True)

### *XG Boost*
YXGBB = xgb_model.predict(XB_test)
accuracy = accuracy_score(YB_test, YXGBB)
precision = precision_score(YB_test, YXGBB)
recall = recall_score(YB_test, YXGBB)
f1 = f1_score(YB_test, YXGBB)
tn, fp, fn, tp = confusion_matrix(YB_test, YXGBB).ravel()
specificity = tn / (tn+fp)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
new_row = pd.DataFrame({'Model': ['XGB'],
                        'Accuracy': [accuracy],
                        'Precision': [precision],
                        'Recall': [recall],
                        'Specificity': [specificity],
                        'F1-Score': [f1]})
scores_df = pd.concat([scores_df, new_row], ignore_index=True)

### *Logistic Regression*
YLOGISB = logis_model.predict(XB_test)
accuracy = accuracy_score(YB_test, YLOGISB)
precision = precision_score(YB_test, YLOGISB)
recall = recall_score(YB_test, YLOGISB)
f1 = f1_score(YB_test, YLOGISB)
tn, fp, fn, tp = confusion_matrix(YB_test, YLOGISB).ravel()
specificity = tn / (tn+fp)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1}")
new_row = pd.DataFrame({'Model': ['LOG'],
                        'Accuracy': [accuracy],
                        'Precision': [precision],
                        'Recall': [recall],
                        'Specificity': [specificity],
                        'F1-Score': [f1]})
scores_df = pd.concat([scores_df, new_row], ignore_index=True)

### ***Proposed Ensemble Classifier***
ENSB = meta_classifier.predict(meta_features_test)

accuracy = accuracy_score(YB_test, ENSB)
precision = precision_score(YB_test, ENSB)
recall = recall_score(YB_test, ENSB)
f1 = f1_score(YB_test, ENSB)
tn, fp, fn, tp = confusion_matrix(YB_test, ENSB).ravel()
specificity = tn / (tn+fp)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Specificity: {specificity}")
print(f"F1 Score: {f1}")
new_row = pd.DataFrame({'Model': ['ENSPxa'],
                        'Accuracy': [accuracy],
                                                'Precision': [precision],
                                                                        'Recall': [recall],
                                                                                                'Specificity': [specificity],
                                                                                                                        'F1-Score': [f1]})
scores_df = pd.concat([scores_df, new_row], ignore_index=True)
scores_df

## 5-Fold CROSS-VALIDATION

from sklearn.model_selection import cross_val_score, KFold

cv_scores_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall','Specificity','F1-Score'])



def evaluate_model(model, X, y):
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    accuracy = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
    precision = cross_val_score(model, X, y, cv=kf, scoring='precision')
    recall = cross_val_score(model, X, y, cv=kf, scoring='recall')
    f1 = cross_val_score(model, X, y, cv=kf, scoring='f1')

    # Specificity calculation requires confusion matrix for each fold
    specificity_scores = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp)
        specificity_scores.append(specificity)

    return accuracy.mean(), precision.mean(), recall.mean(), np.mean(specificity_scores), f1.mean()




kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_df_cv = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score'])


for fold, (train_index, test_index) in enumerate(kf.split(XB)):
    X_train, X_test = XB.iloc[train_index], XB.iloc[test_index]
    y_train, y_test = YB.iloc[train_index], YB.iloc[test_index]

    # Train the base models
    rf_model.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)

    # Create meta features
    meta_features_train = create_meta_features(rf_model.predict_proba(X_train), xgb_model.predict_proba(X_train))
    meta_features_test = create_meta_features(rf_model.predict_proba(X_test), xgb_model.predict_proba(X_test))

    # Train the meta classifier
    meta_classifier = LogisticRegression()
    meta_classifier.fit(meta_features_train, y_train)

    # Predictions
    y_pred = meta_classifier.predict(meta_features_test)

    # Evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp)

    # Store the results
    new_row = pd.DataFrame({'Model': [f'ENSPxa_Fold_{fold+1}'],
                            'Accuracy': [accuracy],
                            'Precision': [precision],
                            'Recall': [recall],
                            'Specificity': [specificity],
                            'F1-Score': [f1]})
    scores_df_cv = pd.concat([scores_df_cv, new_row], ignore_index=True)

scores_df_cv

average_scores = scores_df_cv[['Accuracy', 'Precision', 'Recall', 'Specificity', 'F1-Score']].mean()
average_scores

new_row = pd.DataFrame({'Model': ['Ensemble_Avg'],
                        'Accuracy': [average_scores['Accuracy']],
                        'Precision': [average_scores['Precision']],
                        'Recall': [average_scores['Recall']],
                        'Specificity': [average_scores['Specificity']],
                        'F1-Score': [average_scores['F1-Score']]})


cv_scores_df = pd.concat([cv_scores_df, new_row], ignore_index=True)
cv_scores_df
## Confusion Matrix


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix



models = {
    'AdaBoost': YADAB,
    'Gaussian NB': YGNBB,
    'XGBoost': YXGBB,
    'Logistic Regression': YLOGISB,
    'Ensemble': ENSB
}

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()  # Flatten the axes array for easier iteration

for i, (model_name, y_pred) in enumerate(models.items()):
    cm = confusion_matrix(YB_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues' if model_name != 'Ensemble' else 'Greens', ax=axes[i])
    axes[i].set_title(model_name)
    axes[i].set_xlabel('Predicted Label')
    axes[i].set_ylabel('True Label')


# Remove any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])


plt.tight_layout()
plt.show()
## *Exporting Results*
cv_scores_df.to_csv('cv_scores_5fold.csv', index=False)
scores_df.to_csv('scores_revised.csv', index=False)