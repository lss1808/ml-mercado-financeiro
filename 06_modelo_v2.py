import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar NOVO dataset
df = pd.read_csv("ouro_features_v2.csv")

# Definir FEATURES (agora com mais colunas)
X = df[[
    "MA_5",
    "MA_20",
    "MA_diff",
    "Retorno",
    "Volatilidade",
    "RSI"
]]

# Target
y = df["Target"]

# Separar treino e teste (IMPORTANTE: shuffle=False para mercado financeiro)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Criar modelo
modelo = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

# Treinar
modelo.fit(X_train, y_train)

# Prever
previsoes = modelo.predict(X_test)

# Avaliar
accuracy = accuracy_score(y_test, previsoes)

print("Nova acurácia:", accuracy)
