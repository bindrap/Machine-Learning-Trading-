import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier

def add_features(data):
    """Add additional technical indicators to the data."""
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['MACD'] = data['Close'].ewm(span=12, adjust=False).mean() - data['Close'].ewm(span=26, adjust=False).mean()
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff().clip(lower=0).rolling(window=14).mean() / 
                                     data['Close'].diff().clip(upper=0).abs().rolling(window=14).mean())))
    data['Bollinger_High'] = data['Close'].rolling(window=20).mean() + 2 * data['Close'].rolling(window=20).std()
    data['Bollinger_Low'] = data['Close'].rolling(window=20).mean() - 2 * data['Close'].rolling(window=20).std()
    return data

def analyze_data(data):
    """Analyze data and generate trading signals using ML algorithms."""
    data = add_features(data)  # Add new features

    # Create features and target
    data['Target'] = np.where(data['Close'].shift(-1) > data['Close'], 1, 0)
    data = data.dropna(subset=['SMA_20', 'SMA_50', 'Target'])
    features = data[['SMA_20', 'SMA_50', 'MACD', 'RSI', 'Bollinger_High', 'Bollinger_Low']]
    targets = data['Target']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.3, random_state=42)

    # Initialize dictionary to store results
    results = {}

    # KNN Model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    knn_predictions = knn.predict(X_test)
    knn_accuracy = accuracy_score(y_test, knn_predictions)
    results['KNN'] = {'predictions': knn_predictions, 'accuracy': knn_accuracy}

    # Random Forest Model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_predictions = rf.predict(X_test)
    rf_accuracy = accuracy_score(y_test, rf_predictions)
    results['Random Forest'] = {'predictions': rf_predictions, 'accuracy': rf_accuracy}

    # SVM Model
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)
    svm_predictions = svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_predictions)
    results['SVM'] = {'predictions': svm_predictions, 'accuracy': svm_accuracy}

    # XGBoost Model
    xgb = XGBClassifier()
    xgb.fit(X_train, y_train)
    xgb_predictions = xgb.predict(X_test)
    xgb_accuracy = accuracy_score(y_test, xgb_predictions)
    results['XGBoost'] = {'predictions': xgb_predictions, 'accuracy': xgb_accuracy}

    initial_investment = 10000

    for model in results.keys():
        data.loc[:, f'{model}_Signal'] = np.nan  # Create model-specific Signal column
        data.loc[X_test.index, f'{model}_Signal'] = results[model]['predictions']

        data.loc[:, f'{model}_Position'] = data[f'{model}_Signal'].shift()
        data[f'{model}_Position'].fillna(0, inplace=True)

        data.loc[:, f'{model}_Strategy_Return'] = data[f'{model}_Position'] * data['Close'].pct_change()
        data.loc[:, f'{model}_Cumulative_Strategy_Return'] = (data[f'{model}_Strategy_Return'] + 1).cumprod()
        data.loc[:, f'{model}_Portfolio_Value'] = initial_investment * data[f'{model}_Cumulative_Strategy_Return']

    data = data.dropna()  # Remove rows with NaN values

    return data, results

