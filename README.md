# 💳 Fraud Transactions detector

Fraud transaction detection project developed during the **Accenture Data Analysis, Automation & Machine Learning Bootcamp**.

This project explores machine learning techniques for identifying fraudulent credit card transactions using classification models, feature engineering, imbalance handling, explainability tools, and model evaluation metrics.

---

## 🚀 Overview

Financial fraud detection is a highly imbalanced classification problem, where fraudulent operations represent a very small percentage of all transactions.

This repository demonstrates an end-to-end ML workflow for fraud detection, including:

- Data preprocessing
- Feature engineering
- Imbalanced dataset handling
- Multiple ML models
- Hyperparameter tuning
- Explainability with SHAP
- Performance evaluation with ROC/AUC and Precision-Recall curves

---

## 🧠 Technologies & Libraries

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Imbalanced-learn (SMOTE)

---

## 📂 Project Structure

```bash
frad-transactions-detector/
│
├── main.py
├── load_dataset.py
├── transactions_dataset.csv   # generated automatically
└── README.md
```

---

## ⚙️ Dataset

The project uses the public **Credit Card Fraud Detection Dataset**, originally made available by TensorFlow/Google storage.

If the dataset file does not exist locally, the application automatically downloads it:

```python
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
```

Dataset characteristics:

- Highly imbalanced data
- PCA-anonymized transaction features (`V1` → `V28`)
- `Amount` feature
- `Class` target:
  - `0` → Normal transaction
  - `1` → Fraudulent transaction

---

## 🔍 Features Implemented

### 📌 Feature Engineering

- Log transformation for transaction amounts
- Feature scaling with `StandardScaler`

```python
df["Amount_log"] = np.log1p(df["Amount"])
```

---

### ⚖️ Imbalanced Data Handling

The project addresses class imbalance using:

- **Undersampling**
- **SMOTE Oversampling**

```python
smote = SMOTE()
x_res, y_res = smote.fit_resample(x, y)
```

---

## 🤖 Machine Learning Models

### Logistic Regression

Baseline linear classification model.

### Random Forest

Tree-based ensemble model with balanced class weights.

### XGBoost

Advanced boosting algorithm for fraud prediction.

```python
xgb = XGBClassifier(
  scale_post_weight=10,
  use_label_encoder=False,
  eval_metric="logloss"
)
```

---

## 📊 Evaluation Metrics

The models are evaluated using:

- Classification Report
- ROC Curve
- AUC Score
- Precision-Recall Curve

These metrics are especially important for imbalanced datasets.

---

## 🧪 Hyperparameter Tuning

Grid Search is used to optimize XGBoost parameters:

```python
param_grid = {
  "max_depth": [3, 5],
  "n_estimators": [50, 100]
}
```

---

## 🔬 Explainable AI (XAI)

The project uses **SHAP** to explain feature importance and model behavior.

```python
explainer = shap.Explainer(xgb)
```

This improves interpretability and transparency in fraud prediction systems.

---

## ▶️ Running the Project

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/frad-transactions-detector.git
```

### 2️⃣ Enter the project folder

```bash
cd fraud-transactions-detector
```

### 3️⃣ Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn xgboost shap imbalanced-learn
```

### 4️⃣ Run the application

```bash
python main.py
```

---

## 📈 Concepts Practiced

This project demonstrates practical experience with:

- Machine Learning pipelines
- Fraud detection systems
- Classification problems
- Imbalanced datasets
- Feature engineering
- Model explainability
- Performance evaluation
- Hyperparameter optimization

---

## 🎯 Learning Context

Project developed as part of the:

**Accenture Data Analysis, Automation & Machine Learning Bootcamp**

Focused on practical applications of machine learning in real-world business problems.

---

## 📚 Future Improvements

- Add confusion matrix visualization
- Save trained models with Pickle/Joblib
- Create REST API for predictions
- Deploy model with Docker
- Add notebook version for analysis
- Add experiment tracking

---

## 👨‍💻 Author

Developed by **Dhouglas Bandeira Nobrega**

Feel free to connect and contribute 🚀
