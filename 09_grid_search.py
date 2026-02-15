import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas_datareader.data as web

# =========================
# 1. Carregar e limpar dados do ouro
# =========================
df = pd.read_csv("ouro_5_anos.csv", skiprows=2)

df.columns = ["Date", "Close", "High", "Low", "Open", "Volume"]

df["Date"] = pd.to_datetime(df["Date"])
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

df = df.sort_values("Date")

# =========================
# 2. Baixar dados macro (FRED)
# =========================
start = df["Date"].min()
end = df["Date"].max()

fed_rate = web.DataReader("FEDFUNDS", "fred", start, end)
public_debt = web.DataReader("GFDEBTN", "fred", start, end)

# Resetar índice para virar coluna
fed_rate = fed_rate.reset_index()
public_debt = public_debt.reset_index()

# Corrigir nome da coluna DATE → Date
fed_rate.rename(columns={"DATE": "Date"}, inplace=True)
public_debt.rename(columns={"DATE": "Date"}, inplace=True)

# Merge com dados do ouro
df = df.merge(fed_rate, on="Date", how="left")
df = df.merge(public_debt, on="Date", how="left")

# Preencher valores faltantes (dados macro são mensais)
df["FEDFUNDS"] = df["FEDFUNDS"].ffill()
df["GFDEBTN"] = df["GFDEBTN"].ffill()

# =========================
# 3. Indicadores técnicos
# =========================
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_20"] = df["Close"].rolling(20).mean()
df["MA_diff"] = df["MA_5"] - df["MA_20"]

df["Retorno"] = df["Close"].pct_change()
df["Volatilidade"] = df["Close"].rolling(5).std()

# =========================
# 4. Lags
# =========================
df["Close_lag1"] = df["Close"].shift(1)
df["Close_lag2"] = df["Close"].shift(2)
df["Close_lag3"] = df["Close"].shift(3)

df["Retorno_lag1"] = df["Retorno"].shift(1)
df["Retorno_lag2"] = df["Retorno"].shift(2)
df["Retorno_lag3"] = df["Retorno"].shift(3)

# =========================
# 5. Target (2 dias à frente)
# =========================
df["Target"] = (df["Close"].shift(-2) > df["Close"]).astype(int)

# Remover NaNs
df = df.dropna()

# =========================
# 6. Selecionar features
# =========================
X = df[
    [
        "MA_5", "MA_20", "MA_diff",
        "Retorno", "Volatilidade",
        "Close_lag1", "Close_lag2", "Close_lag3",
        "Retorno_lag1", "Retorno_lag2", "Retorno_lag3",
        "FEDFUNDS", "GFDEBTN"
    ]
]

y = df["Target"]

# =========================
# 7. Split temporal (80% treino)
# =========================
split_index = int(len(df) * 0.8)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# =========================
# 8. Grid Search com TimeSeriesSplit
# =========================
tscv = TimeSeriesSplit(n_splits=5)

param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5],
    "min_samples_leaf": [1, 2],
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

# =========================
# 9. Avaliação final
# =========================
best_model = grid.best_estimator_

pred = best_model.predict(X_test)

accuracy_final = accuracy_score(y_test, pred)

print("Accuracy final real:", accuracy_final)
