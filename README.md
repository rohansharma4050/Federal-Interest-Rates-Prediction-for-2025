# Federal-Interest-Rates-Prediction-for-2025
Predicting Federal Interest Rates for range or month basis for the year of 2025

### Data Sources -

Federal Interest Rates from 1954 to 2024 - https://fred.stlouisfed.org/series/FEDFUNDS

### Demo -
![Uploading image18.gif…]()

#### Features
Inflation Rates from 1954 to 2024 - https://www.usinflationcalculator.com/inflation/historical-inflation-rates/

Unemployment Rate from 1954 to 2024 - https://fred.stlouisfed.org/series/UNRATE

Moodys Corporate bond yield from 1954 to 2024 - https://fred.stlouisfed.org/series/AAA

### About Data

The Federal Interest Rates data that we have used is a **Vector Multivariate Time Series Dataset** which has the dataset ranging from July, 1954 to October, 2024. There are 844 time points and 4 features. The features are Inflation Rate, Unemployment Rate, Bonds Yield and, Federal Fund Rates. The data is recorded monthly particularly on the 1st of every month. Our goal is to predict the Federal Fund Rates for the months in beginning of 2025 (January, February, March, April and May) considering the other features. We also forecast the interest rates for November and December, 2024 alongwith the first 5 months for January.

### Model to be used
1. Autoregressive Integrated Moving Average (ARIMA)
2. Vector Autoregressive Moving Average (VARMA)
3. Long Short Term Memory (LSTM)
