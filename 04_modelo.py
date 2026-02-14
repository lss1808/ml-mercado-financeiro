import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar dados com features
df = pd.read_csv("ouro_features.csv")

# Definir FEATURES (X)
X = df[["MA_5", "MA_20", "Retorno", "Volatilidade"]]

# Definir TARGET (y)
y = df["Target"]

# Separar treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Criar modelo
modelo = RandomForestClassifier(n_estimators=100, random_state=42)

# Treinar
modelo.fit(X_train, y_train)

# Previsões
previsoes = modelo.predict(X_test)

# Avaliar
accuracy = accuracy_score(y_test, previsoes)

print("Acurácia do modelo:", accuracy)
