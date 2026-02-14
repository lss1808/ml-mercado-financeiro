import pandas as pd

# Carregar dados
df = pd.read_csv("ouro_5_anos.csv")

# Criar coluna com preço do dia seguinte
df["Close_Amanha"] = df["Close"].shift(-1)

# Criar label (1 = sobe, 0 = cai)
df["Target"] = (df["Close_Amanha"] > df["Close"]).astype(int)

# Remover última linha (vai ficar vazia)
df = df.dropna()

print(df[["Close", "Close_Amanha", "Target"]].head())

# Salvar novo CSV
df.to_csv("ouro_com_target.csv", index=False)

print("Label criada com sucesso 🚀")
