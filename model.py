from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from statsmodels.tsa.api import VAR
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)

# Load the data from the CSV file
def load_data():
    file_path = 'finaldata.csv'  # Change this to your actual CSV file path
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df.index = pd.to_datetime(df.index)
    return df

# Helper function for VAR forecast
def var_forecast(data, start_date, steps):
    model = VAR(data)
    model_fit = model.fit()
    lag_order = model_fit.k_ar  # Number of lags in the VAR model
    forecast_input = data.values[-lag_order:]  # Last `lag_order` rows of the input data
    forecast = model_fit.forecast(y=forecast_input, steps=steps)
    forecast_index = pd.date_range(start_date, periods=steps, freq='M')
    forecast_df = pd.DataFrame(forecast, index=forecast_index, columns=data.columns)
    return forecast_df['FEDFUNDS'].round(2)  # Round to 2 decimal places

# Helper function for SARIMAX forecast
def sarimax_forecast(data, exog_data, start_date, steps):
    model = SARIMAX(data, exog=exog_data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps, exog=exog_data[-steps:])
    return forecast.round(2)  # Round to 2 decimal places

# Helper function to generate plot
def generate_plot(fedfunds, forecast_values, forecast_dates):
    forecast_series = pd.Series(forecast_values, index=forecast_dates)
    plt.figure(figsize=(10, 6))
    plt.plot(fedfunds.index, fedfunds['FEDFUNDS'], label='Historical FEDFUNDS', color='blue')
    plt.plot(forecast_series.index, forecast_series, label='Forecasted FEDFUNDS', color='red', linestyle='--')
    plt.xlabel('Date')
    plt.ylabel('FEDFUNDS Rate')
    plt.title('FEDFUNDS Forecast')
    plt.legend()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    return f"data:image/png;base64,{plot_url}"

# Helper function to calculate accuracy metrics
def calculate_accuracy(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

# Flask route for the main page
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        month = request.form.get('month')
        forecast_months = request.form.get('forecast_months')
        model_choice = request.form.get('model_choice')

        if not month or not forecast_months:
            return render_template('index.html', error="Please provide valid inputs.")

        forecast_months = int(forecast_months)
        if forecast_months <= 0:
            return render_template('index.html', error="Forecast months must be greater than zero.")

        fedfunds = load_data()
        target = fedfunds['FEDFUNDS']
        exog_data = fedfunds.drop(columns=['FEDFUNDS'])
        forecast_start_date = f'2025-{month.zfill(2)}-01'

        # Perform the forecast
        if model_choice == 'VAR':
            forecast_values = var_forecast(fedfunds, forecast_start_date, forecast_months)
            model_fit = VAR(fedfunds).fit()
            fitted_values = model_fit.fittedvalues['FEDFUNDS']
        elif model_choice == 'SARIMAX':
            forecast_values = sarimax_forecast(target, exog_data, forecast_start_date, forecast_months)
            model_fit = SARIMAX(target, exog=exog_data, order=(1, 1, 1), seasonal_order=(1, 1, 0, 12)).fit()
            fitted_values = model_fit.fittedvalues
        else:
            return render_template('index.html', error="Invalid model choice. Please try again.")

        # Forecast dates
        forecast_dates = pd.date_range(forecast_start_date, periods=forecast_months, freq='M')

        # Generate forecast plot
        plot_url = generate_plot(fedfunds, forecast_values, forecast_dates)

        # Prepare forecast data table
        forecast_data = pd.DataFrame({
            'Month': forecast_dates.strftime('%B %Y'),
            'Predicted FEDFUNDS Rate': forecast_values
        })
        forecast_table = forecast_data.to_dict(orient='records')

        # Calculate accuracy metrics
        mae, rmse, mape = calculate_accuracy(fedfunds['FEDFUNDS'][fitted_values.index], fitted_values)
        accuracy_metrics = {
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2),
            'MAPE': round(mape, 2)
        }

        return render_template('index.html', plot_url=plot_url, forecast_table=forecast_table, 
                               accuracy_metrics=accuracy_metrics, model_choice=model_choice)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
