import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Carregar novo dataset filtrado
df = pd.read_csv("ouro_target_filtrado.csv")

# Garantir que Close seja numérico
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Criar features novamente
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_20"] = df["Close"].rolling(20).mean()
df["MA_diff"] = df["MA_5"] - df["MA_20"]
df["Retorno"] = df["Close"].pct_change()
df["Volatilidade"] = df["Close"].rolling(5).std()

df = df.dropna()

# Definir X e y
X = df[["MA_5", "MA_20", "MA_diff", "Retorno", "Volatilidade"]]
y = df["Target"]

# Separar treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Modelo
modelo = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

modelo.fit(X_train, y_train)

previsoes = modelo.predict(X_test)

accuracy = accuracy_score(y_test, previsoes)

print("Accuracy com target filtrado:", accuracy)
