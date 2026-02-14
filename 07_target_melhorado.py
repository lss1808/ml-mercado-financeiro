import pandas as pd

df = pd.read_csv("ouro_5_anos.csv")

# Garantir que Close seja numérico
df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

# Remover valores inválidos
df = df.dropna(subset=["Close"])

# Retorno percentual do dia seguinte
df["Retorno_Amanha"] = df["Close"].pct_change(fill_method=None).shift(-1)

# Criar target com filtro de movimento relevante
limite = 0.003  # 0.3%

df["Target"] = 0
df.loc[df["Retorno_Amanha"] > limite, "Target"] = 1
df.loc[df["Retorno_Amanha"] < -limite, "Target"] = -1

# Remover dias neutros
df = df[df["Target"] != 0]

df.to_csv("ouro_target_filtrado.csv", index=False)

print("Novo target criado 🚀")
