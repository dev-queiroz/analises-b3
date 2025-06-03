# Analises-B3

Bem-vindo ao **Analises-B3**, um projeto em Python para análise técnica e predição de preços de ações da B3 (Bolsa de Valores do Brasil) e criptomoedas. Este projeto é ideal para desenvolvedores iniciantes, como Douglas, que desejam aprender sobre análise de dados financeiros, machine learning e visualização de dados, utilizando ferramentas gratuitas. Ele pode ser usado em cenários de freelancing, como relatórios financeiros para clientes ou ferramentas de suporte à decisão para investidores.

## Objetivo do Projeto

O Analises-B3 tem como objetivo fornecer uma ferramenta para:

- **Coletar Dados**: Obter dados históricos de ações e criptomoedas usando APIs públicas (`yfinance` e `cryptocompare`).
- **Análise Técnica**: Calcular indicadores como Média Móvel Simples (SMA) e Índice de Força Relativa (RSI).
- **Previsão de Preços**: Utilizar modelos de machine learning (Regressão Linear e LSTM) para prever preços futuros.
- **Gerenciamento de Risco**: Gerar hipóteses de compra/venda com stop loss e take profit.
- **Visualização**: Plotar gráficos com preços e indicadores para análise visual.

Este projeto é educativo e prático, ideal para desenvolver habilidades em Python, análise de dados e machine learning, além de criar um portfólio atrativo para freelancing.

## Funcionalidades

- **Coleta de Dados**: Obtém dados de ações (ex.: PETR4.SA, VALE3.SA) e criptomoedas (ex.: BTC, ETH).
- **Análise Técnica**: Calcula indicadores como SMA50 e RSI para avaliar tendências.
- **Modelos de Previsão**:
  - Regressão Linear para prever preços com base em indicadores técnicos.
  - Redes Neurais LSTM para previsão de séries temporais.
- **Gerenciamento de Risco**: Sugere ações de compra, venda ou manutenção com stop loss e take profit.
- **Visualização**: Gera gráficos com preços, indicadores e previsões.
- **Integração com Google Drive**: Armazena dados no Google Colab (pode ser adaptado para outras plataformas).

## Tecnologias Utilizadas

- **Python**: Linguagem principal.
- **yfinance**: Biblioteca para coleta de dados de ações.
- **cryptocompare**: API para dados de criptomoedas.
- **ta**: Biblioteca para análise técnica.
- **pandas**: Manipulação de dados.
- **matplotlib**: Visualização de gráficos.
- **scikit-learn**: Modelos de machine learning (Regressão Linear).
- **keras**: Redes neurais LSTM.
- **numpy**: Operações numéricas.
- **Google Colab**: Ambiente gratuito para execução do código (pode ser adaptado para VS Code ou WebStorm).

## Pré-requisitos

- Python 3.8+ ([python.org](https://www.python.org/downloads/)).
- Conta no Google Colab (para executar no ambiente atual).
- Bibliotecas Python:
  ```bash
  pip install yfinance cryptocompare ta pandas matplotlib scikit-learn numpy keras
  ```
- Git para versionamento.
- Conta no GitHub para hospedar o repositório.

## Instalação

1. **Clone o Repositório**:
   ```bash
   git clone https://github.com/dev-queiroz/analises-b3.git
   cd analises-b3
   ```

2. **Crie um Ambiente Virtual** (recomendado):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

3. **Instale as Dependências**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Execute o Projeto**:
   - No Google Colab, faça upload do arquivo `analises_b3.py` e execute as células.
   - Localmente, rode:
     ```bash
     python analises_b3.py
     ```

## Como Usar

1. **Coletar Dados**:
   - Edite a lista de `tickers_acoes` e `tickers_criptos` no arquivo `analises_b3.py` para incluir os ativos desejados.
   - Exemplo:
     ```python
     tickers_acoes = ['PETR4.SA', 'VALE3.SA']
     tickers_criptos = ['BTC', 'ETH']
     ```

2. **Executar Análise Técnica**:
   - A função `analise_tecnica` adiciona indicadores SMA50 e RSI aos dados.
   - Exemplo de uso:
     ```python
     dados_acoes = coletar_dados_acoes(tickers_acoes)
     dados_acoes = analise_tecnica(dados_acoes)
     ```

3. **Treinar e Prever com Modelos**:
   - Use `treinar_modelo` para treinar modelos de Regressão Linear.
   - Use `treinar_modelo_lstm` para treinar modelos LSTM.
   - Exemplo de previsão com LSTM:
     ```python
     modelo_lstm, scaler = treinar_modelo_lstm(dados_acoes['PETR4.SA'], look_back=10)
     previsoes = prever_precos_lstm(modelo_lstm, scaler, dados_acoes['PETR4.SA'], look_back=10)
     ```

4. **Gerar Hipóteses**:
   - A função `gerar_hipoteses` sugere ações de compra/venda com base em previsões e indicadores.
   - Exemplo de saída:
     ```
     PETR4.SA: {'Ação': 'Comprar', 'Stop Loss': 35.50, 'Take Profit': 37.80}
     ```

5. **Visualizar Resultados**:
   - A função `comparar_analises` plota gráficos com preços, SMA50 e linhas de tendência.
   - Gráficos LSTM são gerados automaticamente no final do script.

## Exemplo Prático

Abaixo está um exemplo de como usar o projeto para analisar a ação PETR4.SA:

```python
from analises_b3 import coletar_dados_acoes, analise_tecnica, treinar_modelo, comparar_analises, gerar_hipoteses

# Coletar dados
tickers = ['PETR4.SA']
dados = coletar_dados_acoes(tickers)

# Aplicar análise técnica
dados = analise_tecnica(dados)

# Treinar modelo
modelos = treinar_modelo(dados)

# Gerar hipóteses
hipoteses = gerar_hipoteses(dados, modelos)
print(hipoteses)

# Visualizar resultados
comparar_analises(dados, modelos)
```

**Saída Esperada** (exemplo):
```
PETR4.SA: {'Ação': 'Comprar', 'Stop Loss': 35.50, 'Take Profit': 37.80}
```

Gráficos com preços, SMA50 e linhas de tendência serão exibidos.

## Estrutura do Projeto

```
analises-b3/
├── analises_b3.py        # Script principal com toda a lógica
├── requirements.txt      # Lista de dependências
├── README.md            # Este arquivo
└── data/                # (Opcional) Pasta para salvar dados exportados
```

## Documentação do Código

Exemplo de como documentar uma função no `analises_b3.py`:

```python
def coletar_dados_acoes(tickers):
    """
    Coleta dados históricos de ações usando yfinance.

    Args:
        tickers (list): Lista de tickers das ações (ex.: ['PETR4.SA']).

    Returns:
        dict: Dicionário com DataFrames contendo dados das ações.
    """
    dados_acoes = {}
    for ticker in tickers:
        dados = yf.download(ticker, period='6mo', interval='1d')
        dados_acoes[ticker] = dados
    return dados_acoes
```

Crie um arquivo `docs/guia_usuario.md` para clientes não técnicos, explicando como interpretar os gráficos e hipóteses de investimento.

## Recursos Gratuitos

- **yfinance**: [Documentação](https://github.com/ranaroussi/yfinance) – Guia para coleta de dados financeiros.
- **cryptocompare**: [API Docs](https://min-api.cryptocompare.com/documentation) – API gratuita para criptomoedas.
- **FreeCodeCamp**: [Tutorial de Machine Learning](https://www.freecodecamp.org/news/machine-learning-for-finance/) – Introdução a ML em finanças.
- **Stack Overflow**: Comunidade para dúvidas técnicas.
- **Python Discord**: Suporte e networking para desenvolvedores.
- **Kaggle**: Datasets financeiros e tutoriais gratuitos.

## Próximos Passos

Plano de aprendizado de 4 semanas:

- **Semana 1**: Entenda os indicadores técnicos (SMA, RSI) e experimente diferentes tickers. Leia a documentação do `yfinance`.
- **Semana 2**: Adicione novos indicadores (ex.: MACD, Bollinger Bands) usando a biblioteca `ta`. Atualize o README.
- **Semana 3**: Crie uma API REST com Flask para expor as análises como serviço (hospede gratuitamente na Render).
- **Semana 4**: Desenvolva um portfólio no GitHub Pages, destacando o Analises-B3 como um projeto para clientes financeiros.

## Dica de Freelancing

Crie um relatório de exemplo com análises de ações populares (ex.: PETR4.SA) e publique no seu portfólio. Mostre como o projeto pode ajudar clientes a tomar decisões de investimento, destacando a clareza dos gráficos e hipóteses.

## Contribuições

1. Faça um fork do reposit
