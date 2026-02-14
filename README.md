# 📈 ML Mercado Financeiro

Projeto de Machine Learning para previsão de movimentos do ouro (GC=F).

## 🎯 Objetivo

Prever se o preço do ouro irá subir ou cair com base em dados históricos
utilizando Random Forest.

## 📊 Dados

- Fonte: Yahoo Finance (yfinance)
- Período: Últimos 5 anos
- Ativo: Ouro (GC=F)

## 🧠 Features utilizadas

- Média móvel 5 dias
- Média móvel 20 dias
- Diferença entre médias
- Retorno diário
- Volatilidade

## 🚀 Modelo

- RandomForestClassifier
- Separação temporal (80% treino / 20% teste)
- Sem shuffle (respeitando ordem temporal)

## 📈 Resultado atual

Accuracy aproximada: ~57% (modelo inicial)

## 🔧 Próximos passos

- Backtesting
- Otimização de hiperparâmetros
- Adição de novos indicadores
- Transformação em API / Web App

---

Projeto educacional para estudo de Machine Learning aplicado ao mercado financeiro.
