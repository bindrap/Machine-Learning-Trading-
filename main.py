from data_fetcher import fetch_data
from data_analyzer import analyze_data
from plotter import plot_data
from rl_trading import train_reinforcement_learning_model, load_model
import pandas as pd
from trading_environment import TradingEnvironment
def generate_signals(data):
    """Generate buy and sell signals based on SMA crossovers."""
    buy_signals = pd.DataFrame(index=data.index, columns=['Close'])
    sell_signals = pd.DataFrame(index=data.index, columns=['Close'])

    for i in range(1, len(data)):
        if data['SMA_20'][i] > data['SMA_50'][i] and data['SMA_20'][i-1] <= data['SMA_50'][i-1]:
            buy_signals.loc[data.index[i]] = data['Close'][i]
        elif data['SMA_20'][i] < data['SMA_50'][i] and data['SMA_20'][i-1] >= data['SMA_50'][i-1]:
            sell_signals.loc[data.index[i]] = data['Close'][i]

    return buy_signals.dropna(), sell_signals.dropna()

def main(ticker):
    data = fetch_data(ticker)
    
    # Optionally analyze data and generate signals
    data, model_results = analyze_data(data)

    # Train the RL model
    model = train_reinforcement_learning_model(data)
    
    # Load the trained model (if already trained)
    model = load_model()
    
    # Use the model to make trading decisions
    env = TradingEnvironment(data)
    obs = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

    buy_signals, sell_signals = generate_signals(data)
    
    print("Trading completed.")
    plot_data(data, 'AAPL', buy_signals, sell_signals)

if __name__ == '__main__':
    ticker = 'NVDA'  # Replace with your chosen ticker, can we add apple, tesla, microsoft and google
    main(ticker)

