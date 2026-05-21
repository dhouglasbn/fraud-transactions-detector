import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from load_dataset import load_dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_curve, roc_auc_score, precision_recall_curve
from imblearn.over_sampling import SMOTE

# Transaction Dataset 
# Time -> transaction moment
# It has 27 values V's that are processed data from transactions (VCA)
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
print(f' Classification Report '.center(60, '='))
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