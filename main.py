import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from load_dataset import load_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Transaction Dataset 
# Time -> transaction moment
# It has 27 values V's that are processed data from transactions (PCA)
# Amount -> transaction value
# Class -> indicates that the transaction is a fraud
df = load_dataset()

# Problem of the unbalanced classification
# print(df["Class"].value_counts(normalize=True))
# -> 0,17% of frauds is hard to train the model

# Feature Engineering
# -> Create or transform variables to help to train the model
df["Amount_log"] = np.log1p(df["Amount"]) # lower numbers trains better
scaler = StandardScaler()
# scaler transforms the data this way:
# Mean -> 0
# Standard deviation -> 1
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

x = df.drop("Class", axis=1)
y = df["Class"]
# stratify keeps frauds proportion
x_train, x_test, y_train, y_test = train_test_split(
  x, y, stratify=y, test_size=0.3, random_state=42
) # 30% of the data for testing & 70% for training

# Logistic Regression
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
report = classification_report(y_test, y_pred)
print(f' Logistic Regression Classification Report '.center(60, '='))
print(report)

y_probs = model.predict_proba(x_test)[:,1]

fpr, tpr, _ = roc_curve(y_test, y_probs)

plt.plot(fpr, tpr)
plt.title('ROC Curve')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.show()

print("AUC: ", roc_auc_score(y_test, y_probs))

precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.plot(recall, precision)
plt.title("Precision Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# Balanceamento de dados
# - Undersampling -> Remove frauds from the dataset in a way that we have the same number of frauds & non frauds
frauds = df[df["Class"] == 1]
normals = df[df["Class"] == 0].sample(len(frauds), random_state=42)
df_under = pd.concat([frauds, normals])
# - Oversampling -> Like the Undersample, but we create frauds in order to reach the proportion
smote = SMOTE()
x_res, y_res = smote.fit_resample(x, y)

# Random Forest 
# a model based on decision trees
# every tree learns a diferent data pattern
# at the end the model combines the trees
rf = RandomForestClassifier(
  n_estimators=50,
  max_depth=10,
  class_weight="balanced",
  n_jobs=-1,
  random_state=42
)

rf.fit(x_train, y_train)
y_pred_rf = rf.predict(x_test)
report = classification_report(y_test, y_pred_rf)
print(' Random Forest Classification Report '.center(60, '='))
print(report)

# Pipeline
# It organizers the processment flow
# Here the data are scaled before the regression in a single flow
pipeline = Pipeline([
  ("scaler", StandardScaler()),
  ("model", LogisticRegression(max_iter=1000))
])
pipeline.fit(x_train, y_train)
y_pred = pipeline.predict(x_test)

# If the probability is higher than 0.3 I consider it as a fraud (default = 0.5)
threshold = 0.3
y_pred_custom = (y_probs > threshold).astype(int)
report = classification_report(y_test, y_pred_custom)
print(' Pipeline Report '.center(60, '='))
print(report)

# Advanced Model - XGBoost
# This is an algorithm based on boost
# boosting is a technique like the random forest but it is sequential
# every model is used to train the next model
xgb = XGBClassifier(
  scale_post_weight=10, # helps with unbalancement
  use_label_encoder=False,
  eval_metric="logloss"
)
xgb.fit(x_train, y_train)
y_pred_xgb = xgb.predict(x_test)
report = classification_report(y_test, y_pred_xgb)
print(' XGBoost Report '.center(60, '='))
print(report)

# Variables Importance

importances = xgb.feature_importances_

plt.bar(range(len(importances)), importances)
plt.title("Importância das variáveis")
plt.show()


# Hiperparameters adjustment
# the grid tests all combinations between these values
# it'll find the best model that meets the grid
param_grid = {
  "max_depth": [3, 5],
  "n_estimators": [50, 100]
}
grid = GridSearchCV(
  XGBClassifier(eval_metric="logloss"),
  param_grid,
  scoring="recall",
  cv=3
)
grid.fit(x_train, y_train)
print("Best Model: ", grid.best_params_)

# Explainability
# It shows how each variables contributes for the model learning
explainer = shap.Explainer(xgb)
shap_values = explainer(x_test[:100])
shap.plots.bar(shap_values)