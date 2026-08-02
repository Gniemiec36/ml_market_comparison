# ml_market_comparison

Machine Learning Model Comparison on Stock Data:

A pipeline for testing whether short term stock price direction can be predicted from technical indicators, comparing four modelling approaches: logistic regression, random forest, XGBoost and a custom PyTorch neural network.


## Overview:
This project downloads historical price data via yfinance, engineers a set of technical features and frames the prediction task as binary classification: will the next period's return be positive or negative? Multiple model classes are trained and evaluated on the same train/test split to allow direct comparison.


## Result: 
Across all four models, test accuracy stayed close to 50% — none showed a meaningful edge over a coin flip. This is treated as a genuine finding rather than a failure: the features used (rolling mean, rolling standard deviation and RSI) combined with each other do not appear to carry a predictive signal for the direction of the market in the next time period, which is broadly consistent with short term price movements behaving close to a random walk.


## Pipeline:
1. Data acquisition — download OHLC data for a given ticker, period, and interval via yfinance
2. Returns — compute simple and log returns from closing price
3. Feature engineering — rolling mean/std of log returns, and RSI (Relative Strength Index), all computed over a configurable window
4. Target definition — binary label: whether the next period's simple return is positive
5. Train/test split — chronological (non-shuffled) 80/20 split to avoid lookahead bias
6. Normalisation — StandardScaler fit on training data only, applied to both sets
7. Modelling — four independent approaches trained and evaluated on identical data:
   - Logistic Regression (scikit-learn)
   - Random Forest (scikit-learn, 200 estimators, max depth 10)
   - XGBoost (gradient boosted trees)
   - Custom feed-forward neural network (PyTorch, 4 layers, BCE loss, Adam optimiser)
8. Evaluation — accuracy and classification report for each model; trained models are serialised with joblib


## Notes:
Stationarity of returns and RSI can be checked via the included is_stationary function (Augmented Dickey-Fuller test)
Diagnostic plotting functions (overlap_plot, separate_plot) are included for visually inspecting feature behaviour over time
All models are saved with their test accuracy in the filename for easy comparison across runs
Limitations & Future Work


Limitations:
Only tested on a single ticker/timeframe configuration in this run; results may vary across assets, timeframes, and market regimes
Feature set is limited to simple technical indicators; incorporating volume, order book, or cross-asset features may reveal different results
