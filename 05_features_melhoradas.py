import pandas as pd
import ta

# Carregar dados originais
df = pd.read_csv("ouro_com_target.csv")

# Médias
df["MA_5"] = df["Close"].rolling(5).mean()
df["MA_20"] = df["Close"].rolling(20).mean()

# Diferença entre médias
df["MA_diff"] = df["MA_5"] - df["MA_20"]

# Retorno
df["Retorno"] = df["Close"].pct_change()

# Volatilidade
df["Volatilidade"] = df["Close"].rolling(5).std()

# RSI
df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi()

# Remover vazios
df = df.dropna()

df.to_csv("ouro_features_v2.csv", index=False)

print("Features melhoradas criadas 🚀")
