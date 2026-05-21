import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Dataset de transações
# Time -> momento da transação
# Possui 27 valores V's que são processamentos de dados
# das transações (Confidencialidade com técnicas VCA)
# Amount -> valor da transação
# Class -> indica se a transação é fraudulenta
url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

df = pd.read_csv(url)

# Problema de classificação desbalanceada
# print(df["Class"].value_counts(normalize=True))
# -> 0,17% de fraudes é difícil de treinar um modelo

# Feature Engineering
# -> Criar ou transformar variáveis para ajudar o modelo a aprender melhor
df["Amount_log"] = np.log1p(df["Amount"]) # números menores ajudam o modelo
scaler = StandardScaler()
# scaler transforma os dados de forma que:
# Média -> 0
# Desvio padrão -> 1
df["Amount_scaled"] = scaler.fit_transform(df[["Amount"]])

x = df.drop("Class", axis=1)
y = df["Class"]
# stratify mantém a proporção de fraudes
x_train, x_test, y_train, y_test = train_test_split(
  x, y, stratify=y, test_size=0.3, random_state=42
) # 30% dos dados são para teste e 70% para treino