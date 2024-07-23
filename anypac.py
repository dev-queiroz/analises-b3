import yfinance as yf
import cryptocompare
import ta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from google.colab import drive

# Montar o Google Drive
drive.mount('/content/drive')

# Função para coletar dados das ações
def coletar_dados_acoes(tickers):
    dados_acoes = {}
    for ticker in tickers:
        dados = yf.download(ticker, period='6mo', interval='1d')  # Ajustar para coletar 6 meses de dados
        dados_acoes[ticker] = dados
    return dados_acoes

# Função para coletar dados das criptomoedas
def coletar_dados_criptos(criptos):
    dados_criptos = {}
    for cripto in criptos:
        dados = cryptocompare.get_historical_price_day(cripto, currency='USD', limit=180)  # Ajustar para coletar 180 dias de dados
        df = pd.DataFrame(dados)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df.rename(columns={'close': 'Close', 'high': 'High', 'low': 'Low', 'open': 'Open', 'volumefrom': 'Volume'}, inplace=True)
        dados_criptos[cripto] = df
    return dados_criptos

# Função para realizar análise técnica
def analise_tecnica(dados):
    for ticker, df in dados.items():
        df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
        df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    return dados

# Função para comparar análises e plotar gráficos
def comparar_analises(dados_atual, dados_anterior):
    for ticker, df in dados_atual.items():
        plt.figure(figsize=(14, 7))
        plt.plot(df['Close'], label='Preço de Fechamento')
        plt.plot(df['SMA50'], label='SMA50')
        plt.title(f'Análise Técnica de {ticker}')
        plt.legend()
        plt.show()

# Função para treinar modelo de predição
def treinar_modelo(dados):
    for ticker, df in dados.items():
        df = df.dropna()
        if len(df) > 1:  # Verificar se há dados suficientes
            X = df[['SMA50', 'RSI']]
            y = df['Close'].shift(-1).dropna()
            X = X[:-1]  # Ajustar X para ter o mesmo tamanho de y
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = LinearRegression()
            modelo.fit(X_train, y_train)
            previsoes = modelo.predict(X_test)
            print(f'Previsões para {ticker}:', previsoes)
        else:
            print(f'Dados insuficientes para {ticker}')

# Lista de tickers de ações e criptomoedas
tickers_acoes = ['SBSP3.SA', 'CPLE6.SA', 'RENT3.SA', 'VALE3.SA', 'EQTL3.SA', 'ITUB4.SA', 'MELI34.SA', 'PETR4.SA', 'ARZZ3.SA', 'CSAN3.SA', 'CYRE3.SA']
tickers_criptos = ['BTC', 'ETH']

# Verificar dados coletados
def verificar_dados(dados):
    for ticker, df in dados.items():
        print(f'{ticker}: {df.shape[0]} linhas coletadas')

# Coletar dados
dados_acoes = coletar_dados_acoes(tickers_acoes)
dados_criptos = coletar_dados_criptos(tickers_criptos)

# Realizar análise técnica
dados_acoes = analise_tecnica(dados_acoes)
dados_criptos = analise_tecnica(dados_criptos)

# Comparar análises e plotar gráficos
comparar_analises(dados_acoes, dados_acoes)  # Aqui você pode passar dados anteriores para comparação
comparar_analises(dados_criptos, dados_criptos)

# Verificar dados coletados
verificar_dados(dados_acoes)
verificar_dados(dados_criptos)

# Treinar modelo de predição
treinar_modelo(dados_acoes)
treinar_modelo(dados_criptos)
