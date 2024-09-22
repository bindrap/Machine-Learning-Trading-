import yfinance as yf

def fetch_data(ticker, period='1mo'):
    """Fetch stock data from Yahoo Finance."""
    data = yf.download(ticker, period=period, interval='5m')
    return data
