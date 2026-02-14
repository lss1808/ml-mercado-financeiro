import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Ativo: Ouro
ativo = "GC=F"

print("Baixando dados...")

# Últimos 5 anos
df = yf.download(ativo, period="5y")

print(df.head())
print(df.tail())

# Salvar CSV
df.to_csv("ouro_5_anos.csv")

print("Dados salvos com sucesso!")

# Plotar gráfico
df["Close"].plot()
plt.title("Ouro - Últimos 5 anos")
plt.show()
