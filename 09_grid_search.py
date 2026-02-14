import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("ouro_target_filtrado.csv")
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_20"] = df["Close"].rolling(20).mean()
df["MA_diff"] = df["MA_5"] - df["MA_20"]
df["Retorno"] = df["Close"].pct_change()
df["Volatilidade"] = df["Close"].rolling(5).std()

df = df.dropna()

X = df[["MA_5", "MA_20", "MA_diff", "Retorno", "Volatilidade"]]
y = df["Target"]

# Separar últimos 20% como teste final
split_index = int(len(df) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "n_estimators": [100, 300],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
}

modelo = RandomForestClassifier(random_state=42)

grid = GridSearchCV(
    modelo,
    param_grid,
    cv=tscv,
    scoring="accuracy",
    n_jobs=-1
)

grid.fit(X_train, y_train)

print("Melhores parâmetros:", grid.best_params_)

best_model = grid.best_estimator_

# Avaliação final no conjunto nunca visto
pred = best_model.predict(X_test)

accuracy_final = accuracy_score(y_test, pred)

print("Accuracy final real:", accuracy_final)
