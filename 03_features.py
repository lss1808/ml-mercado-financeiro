import pandas as pd

# Carregar dados com target
df = pd.read_csv("ouro_com_target.csv")

# Média móvel curta
df["MA_5"] = df["Close"].rolling(5).mean()

# Média móvel longa
df["MA_20"] = df["Close"].rolling(20).mean()

# Retorno diário
df["Retorno"] = df["Close"].pct_change()

# Volatilidade (desvio padrão 5 dias)
df["Volatilidade"] = df["Close"].rolling(5).std()

# Remover valores vazios
df = df.dropna()

print(df.head())

# Salvar
df.to_csv("ouro_features.csv", index=False)

print("Features criadas 🚀")
