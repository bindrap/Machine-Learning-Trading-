import tempfile
import webbrowser
import plotly.io as pio
import plotly.graph_objects as go
import pandas as pd

def plot_data(data, ticker, buy_signals, sell_signals):
    """Plot data and save as HTML file."""
    fig = go.Figure()

    # Plot the Close Price
    fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close Price', line=dict(color='blue')))
    
    # Plot the Moving Averages
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='20-Day SMA', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='50-Day SMA', line=dict(color='green')))

    # Plot Portfolio Values for different models
    for model in ['KNN', 'Random Forest', 'SVM']:
        fig.add_trace(go.Scatter(x=data.index, y=data[f'{model}_Portfolio_Value'], mode='lines', name=f'{model} Portfolio Value', line=dict(dash='dash')))

    # Plot Buy and Sell signals
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', name='Buy Signal', marker=dict(color='green', size=10, symbol='triangle-up')))
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', name='Sell Signal', marker=dict(color='red', size=10, symbol='triangle-down')))

    # Update layout for zooming capabilities
    fig.update_layout(title=f'{ticker.upper()} Stock Analysis with Buy/Sell Signals',
                      xaxis_title='Date',
                      yaxis_title='Price/Portfolio Value',
                      xaxis_rangeslider_visible=True,  # Allows zooming and panning
                      xaxis=dict(rangeselector=dict(buttons=list([
                          dict(count=1, label="1m", step="month", stepmode="backward"),
                          dict(count=6, label="6m", step="month", stepmode="backward"),
                          dict(step="all")
                      ])),
                      type="date"))

    # Save the Plotly graph as an HTML file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as temp_file:
        temp_file.close()
        pio.write_html(fig, file=temp_file.name, auto_open=False)
        webbrowser.open(temp_file.name)

    print("Portfolio final values:")
    for model in ['KNN', 'Random Forest', 'SVM']:
        print(f"{model}: ${data[f'{model}_Portfolio_Value'].iloc[-1]:.2f}")

# Example usage:
# Assuming you have your 'data' DataFrame, 'buy_signals' DataFrame, and 'sell_signals' DataFrame ready
# plot_data(data, 'AAPL', buy_signals, sell_signals)
