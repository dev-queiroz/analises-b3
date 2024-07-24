# pip install yfinance cryptocompare ta pandas matplotlib scikit-learn numpy keras

import yfinance as yf
import cryptocompare
import ta
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from google.colab import drive
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

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

# Função para preparar os dados para LSTM
def preparar_dados_lstm(df, feature_col, target_col, look_back=1):
    dataX, dataY = [], []
    for i in range(len(df)-look_back-1):
        a = df[feature_col].iloc[i:(i+look_back)].values
        dataX.append(a)
        dataY.append(df[target_col].iloc[i + look_back])
    return np.array(dataX), np.array(dataY)

# Função para treinar modelo LSTM
def treinar_modelo_lstm(df, look_back=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = scaler.fit_transform(df[['Close']])
    df_scaled = pd.DataFrame(df_scaled, columns=['Close'], index=df.index)
    
    X, y = preparar_dados_lstm(df_scaled, 'Close', 'Close', look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(look_back, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X, y, epochs=20, batch_size=1, verbose=2)
    
    return model, scaler

# Função para prever preços futuros com LSTM
def prever_precos_lstm(model, scaler, df, look_back=1):
    df_scaled = scaler.transform(df[['Close']])
    df_scaled = pd.DataFrame(df_scaled, columns=['Close'], index=df.index)
    
    X, _ = preparar_dados_lstm(df_scaled, 'Close', 'Close', look_back)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    previsoes_scaled = model.predict(X)
    previsoes = scaler.inverse_transform(previsoes_scaled)
    
    return previsoes

# Função para gerar hipóteses de curto prazo com gerenciamento de risco
def gerar_hipoteses(dados, modelos, stop_loss_pct=0.02, take_profit_pct=0.04):
    hipoteses = {}
    for ticker, df in dados.items():
        df = df.dropna()
        if len(df) > 1:
            X = df[['SMA50', 'RSI']]
            previsoes = modelos[ticker].predict(X)
            ultima_previsao = previsoes[-1]
            ultima_sma50 = df['SMA50'].iloc[-1]
            ultima_rsi = df['RSI'].iloc[-1]
            preco_atual = df['Close'].iloc[-1]
            
            # Calcular stop loss e take profit
            stop_loss = preco_atual * (1 - stop_loss_pct)
            take_profit = preco_atual * (1 + take_profit_pct)
            
            if ultima_previsao > preco_atual and ultima_rsi < 70:
                hipoteses[ticker] = {
                    "Ação": "Comprar",
                    "Stop Loss": stop_loss,
                    "Take Profit": take_profit
                }
            elif ultima_previsao < preco_atual and ultima_rsi > 30:
                hipoteses[ticker] = {
                    "Ação": "Vender",
                    "Stop Loss": stop_loss,
                    "Take Profit": take_profit
                }
            else:
                hipoteses[ticker] = {
                    "Ação": "Manter",
                    "Stop Loss": None,
                    "Take Profit": None
                }
    return hipoteses

# Função para comparar análises e plotar gráficos com linha de tendência
def comparar_analises(dados_atual, modelos):
    for ticker, df in dados_atual.items():
        plt.figure(figsize=(14, 7))
        plt.plot(df['Close'], label='Preço de Fechamento')
        plt.plot(df['SMA50'], label='SMA50')
        
        # Prever preços futuros
        df = df.dropna()
        if len(df) > 1:
            X = df[['SMA50', 'RSI']]
            previsoes = modelos[ticker].predict(X)
            plt.plot(df.index, previsoes, label='Linha de Tendência', linestyle='--')
        
        plt.title(f'Análise Técnica de {ticker}')
        plt.legend()
        plt.show()

# Treinar modelo de predição e retornar o modelo treinado
def treinar_modelo(dados):
    modelos = {}
    for ticker, df in dados.items():
        df = df.dropna()
        if len(df) > 1:
            X = df[['SMA50', 'RSI']]
            y = df['Close'].shift(-1).dropna()
            X = X[:-1]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            modelo = LinearRegression()
            modelo.fit(X_train, y_train)
            modelos[ticker] = modelo
        else:
            print(f'Dados insuficientes para {ticker}')
    return modelos

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

# Verificar dados coletados
verificar_dados(dados_acoes)
verificar_dados(dados_criptos)

# Treinar modelos para ações e criptomoedas
modelos_acoes = treinar_modelo(dados_acoes)
modelos_criptos = treinar_modelo(dados_criptos)

# Comparar análises e plotar gráficos com linha de tendência
comparar_analises(dados_acoes, modelos_acoes)
comparar_analises(dados_criptos, modelos_criptos)

# Gerar hipóteses de curto prazo com gerenciamento de risco
hipoteses_acoes = gerar_hipoteses(dados_acoes, modelos_acoes)
hipoteses_criptos = gerar_hipoteses(dados_criptos, modelos_criptos)

# Exibir hipóteses
print("Hipóteses para Ações:")
for ticker, hipotese in hipoteses_acoes.items():
    print(f"{ticker}: {hipotese}")

print("\nHipóteses para Criptomoedas:")
for ticker, hipotese in hipoteses_criptos.items():
    print(f"{ticker}: {hipotese}")
    
# Exemplo de uso do modelo LSTM para todas as ações
look_back = 10

for ticker, df in dados_acoes.items():
    # Treinar modelo LSTM
    modelo_lstm, scaler = treinar_modelo_lstm(df, look_back)
    
    # Prever preços futuros com LSTM
    previsoes_lstm = prever_precos_lstm(modelo_lstm, scaler, df, look_back)
    
    # Plotar resultados do modelo LSTM
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'].values, label='Preço de Fechamento')
    plt.plot(previsoes_lstm, label='Previsões LSTM', linestyle='--')
    plt.title(f'Previsões de Preços com LSTM para {ticker}')
    plt.legend()
    plt.show()
    
# Exemplo de uso do modelo LSTM para todas as criptomoedas
for ticker, df in dados_criptos.items():
    # Treinar modelo LSTM
    modelo_lstm, scaler = treinar_modelo_lstm(df, look_back)
    
    # Prever preços futuros com LSTM
    previsoes_lstm = prever_precos_lstm(modelo_lstm, scaler, df, look_back)
    
    # Plotar resultados do modelo LSTM
    plt.figure(figsize=(14, 7))
    plt.plot(df['Close'].values, label='Preço de Fechamento')
    plt.plot(previsoes_lstm, label='Previsões LSTM', linestyle='--')
    plt.title(f'Previsões de Preços com LSTM para {ticker}')
    plt.legend()
    plt.show()
