import yfinance as yf
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from statsmodels.tsa.stattools import adfuller
from xgboost import XGBClassifier


# getting stock ticker, period and interval for data acquisition from user
def get_user_inputs():

    ticker = str(input("Enter the ticker of the stock data you want to download: \n")).upper()
    period = str(input("Choose a time period. Valid time periods are 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max: \n"))
    interval = str(input("Choose a data interval. Valid intervals are 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo: \n"))
    
    return ticker, period, interval


# using previous user input to download stock data
def download_data(ticker, period, interval):

    stock_data = yf.download(tickers=ticker, period=period, interval=interval, auto_adjust=True, multi_level_index=False)
    stock_data = stock_data["Close"]

    stock_data.to_csv(f"{ticker}_{period}_{interval}_data.csv")


# calculating returns and log returns from "Close" price
def returns(ticker, period, interval, column="Close"):

    stock_df = pd.read_csv(f"{ticker}_{period}_{interval}_data.csv", index_col=0, parse_dates=True)

    stock_df["Simple Return"] = stock_df[column].pct_change()
    stock_df["Log Return"] = np.log(stock_df[column] / stock_df[column].shift(1))

    return stock_df.dropna()


# testing whether data is statistically stationary
def is_stationary(stock_df, column):

    result = adfuller(stock_df[column])
    
    print(f"ADF statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print("Critial values: ")
    for key, value in result[4].items():
        print(f"{key}: {value:.4f}")

    if result[1] < 0.05:
        print(f"We reject the null hypothesis, H0. This suggests that the {column} data is stationary.")
    
    else:
        print(f"We fail to reject the null hypothesis, H0. This suggests that the {column} data is non-stationary.")


# plotting two sets of data on the same graph
def overlap_plot(stock_df, column1, column2):

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(stock_df.index, stock_df[column1], color="r", label=column1)
    ax.plot(stock_df.index, stock_df[column2], color="b", label=column2)

    ax.set_xticks(stock_df.index[::200])
    ax.set_xlabel("Dates (YYYY-MM-DD)")

    ax.set_title(f"{column1} vs {column2}")
    ax.legend()

    plt.tight_layout()
    plt.show()


# plotting two sets of data on two different graphs on the same figure
def separate_plot(stock_df, column1, column2):

    fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax[0].plot(stock_df.index, stock_df[column1], color="r", label=column1)
    ax[1].plot(stock_df.index, stock_df[column2], color="b", label=column2)

    ax[1].set_xticks(stock_df.index[::200])
    ax[1].set_xlabel("Dates (YYYY-MM-DD)")

    fig.suptitle(f"{column1} vs {column2}")
    fig.legend()

    plt.tight_layout()
    plt.show()


# calculating rolling mean, rolling std & RSI as model features
def features(stock_df, window=20):

    rolling_mean = stock_df["Log Return"].rolling(window=window).mean()
    rolling_std = stock_df["Log Return"].rolling(window=window).std()

    price_change = stock_df["Close"].diff()

    gain = price_change.clip(lower=0)
    loss = -price_change.clip(upper=0)

    rolling_avg_gain = gain.rolling(window=window).mean()
    rolling_avg_loss = loss.rolling(window=window).mean()

    rs = rolling_avg_gain / rolling_avg_loss
    rsi = 100 - 100 / (1 + rs)

    stock_df["Rolling Mean"] = rolling_mean
    stock_df["Rolling Std"] = rolling_std
    stock_df["RSI"] = rsi

    return stock_df.dropna()


# converting returns into binary data to represent whether next period returns are positive (1) or negative (0)
def define_target(stock_df):
    
    target_values = stock_df["Simple Return"] > 0
    stock_df["Target"] = target_values.shift(-1)
    stock_df = stock_df.dropna()
    stock_df["Target"] = stock_df["Target"].astype(int)

    return stock_df


# separating features and target columns into different variables and performing a chronological 80/20 train-test split
def split_data(stock_df):
    
    features = ["Log Return", "Rolling Mean", "Rolling Std", "RSI"]
    X = stock_df[features]
    y = stock_df["Target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    return X_train, X_test, y_train, y_test


# defining a scaler based on training data, not test data to avoid data leakage, normalising both training and testing data
def normalise_data(X_train, X_test):

    scaler = StandardScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)

    return X_train_norm, X_test_norm


# using logistic regression model to predict whether next period returns are positive or negative
def logistic_regression(X_train_norm, X_test_norm, y_train, y_test, period, interval):

    model = LogisticRegression(random_state=36)
    model.fit(X_train_norm, y_train)
    y_pred = model.predict(X_test_norm)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Classification Report:\n{classification_report(y_true=y_test, y_pred=y_pred)}")

    joblib.dump(model, f"logistic_regression_{period}_{interval}_{accuracy:.3f}.joblib")


# using random forest model to predict whether next period returns are positive or negative
def random_forest(X_train_norm, X_test_norm, y_train, y_test, period, interval):

    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=36)
    rf.fit(X_train_norm, y_train)
    y_pred = rf.predict(X_test_norm)
    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Classification Report:\n{classification_report(y_true=y_test, y_pred=y_pred)}")

    joblib.dump(rf, f"random_forest_{period}_{interval}_{accuracy:.3f}.joblib")


# using xgboost tree model to predict whether next period returns are positive or negative
def xgboost_tree(X_train_norm, X_test_norm, y_train, y_test, period, interval):

    bst = XGBClassifier(n_estimators=8, max_depth=4, learning_rate=1, objective="binary:logistic", n_jobs=1)
    bst.fit(X_train_norm, y_train)
    y_pred = bst.predict(X_test_norm)

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)

    print(f"Accuracy: {accuracy:.3f}")
    print(f"Classification Report:\n{classification_report(y_true=y_test, y_pred=y_pred)}")

    joblib.dump(bst, f"XGBoost_{period}_{interval}_{accuracy:.3f}.joblib")


# converting data columns into PyTorch tensors
def NN_data(X_train_norm, X_test_norm, y_train, y_test, batch_size):

    X_train_norm_tensor = torch.from_numpy(X_train_norm.copy()).float()
    X_test_norm_tensor = torch.from_numpy(X_test_norm.copy()).float()
    y_train_tensor = torch.from_numpy(y_train.values.copy()).float().unsqueeze(1)
    y_test_tensor = torch.from_numpy(y_test.values.copy()).float().unsqueeze(1)

    train_dataset = TensorDataset(X_train_norm_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    return X_train_norm_tensor, X_test_norm_tensor, y_train_tensor, y_test_tensor, train_loader


# defining neural network with 4 linear layers using ReLU as non-linear activation function for first 3 layers and getting a probability via sigmoid function from last layer
class StockNet(nn.Module):

    def __init__(self, n_features):
        super(StockNet, self).__init__()

        self.fc1 = nn.Linear(n_features, n_features * 4)
        self.fc2 = nn.Linear(n_features * 4, n_features * 16)
        self.fc3 = nn.Linear(n_features * 16, n_features * 8)
        self.fc4 = nn.Linear(n_features * 8, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.sigmoid(self.fc4(x))

        return x


# defining training loop for neural network
def NN_training_loop(X_train_norm_tensor, train_loader, epochs: int):

    NN_model = StockNet(X_train_norm_tensor.shape[1])
    criterion = nn.BCELoss()
    optimiser = optim.Adam(NN_model.parameters(), lr=0.001)

    for epoch in range(epochs):

        NN_model.train()
        running_loss = 0

        for x_batch, y_batch in train_loader:

            optimiser.zero_grad()
            y_preds = NN_model(x_batch)
            loss = criterion(y_preds, y_batch)
            loss.backward()
            optimiser.step()

            running_loss += loss.item()
        
        print(f"Epoch {epoch + 1}: Loss was {running_loss / len(train_loader)}")

    return NN_model, criterion


# using neural network to predict whether next period returns are positive or negative
def NN_evaluation(NN_model, criterion, X_test_norm_tensor, y_test_tensor, period, interval):

    with torch.no_grad():

        NN_model.eval()
        y_preds = NN_model(X_test_norm_tensor)
        loss = criterion(y_preds, y_test_tensor).item()
        accuracy = ((y_preds >= 0.5) == y_test_tensor).float().mean().item()

    print(f"Test Loss: {loss:.4f}")
    print(f"Accuracy: {100 * accuracy:.2f}%")

    joblib.dump(NN_model, f"NN_{period}_{interval}_{accuracy:.3f}.joblib")




#ticker, period, interval = get_user_inputs()
#download_data(ticker, period, interval)
#stock_df1 = returns(ticker, period, interval)

#is_stationary(stock_df1, "Simple Return")
#is_stationary(stock_df1, "Log Return")

#overlap_plot(stock_df1, "Simple Return", "Log Return")

#stock_df2 = features(stock_df1)

#separate_plot(stock_df2, "Close", "RSI")

#is_stationary(stock_df2, "RSI")

#stock_df3 = define_target(stock_df2)
#X_train, X_test, y_train, y_test = split_data(stock_df3)


#X_train_norm, X_test_norm = normalise_data(X_train, X_test)

#logistic_regression(X_train_norm, X_test_norm, y_train, y_test, period, interval)
#random_forest(X_train_norm, X_test_norm, y_train, y_test, period, interval)
#xgboost_tree(X_train_norm, X_test_norm, y_train, y_test, period, interval)

#X_train_norm_tensor, X_test_norm_tensor, y_train_tensor, y_test_tensor, train_loader = NN_data(X_train_norm, X_test_norm, y_train, y_test, 32)
#NN_model, criterion = NN_training_loop(X_train_norm_tensor, train_loader, 50)
#NN_evaluation(NN_model, criterion, X_test_norm_tensor, y_test_tensor, period, interval)

