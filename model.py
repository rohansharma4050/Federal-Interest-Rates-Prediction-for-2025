from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.api import ExponentialSmoothing
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

app = Flask(__name__)

# Load the data from the CSV file directly inside the model
def load_data():
    file_path = 'finaldata.csv'  # Change this to your actual CSV file path
    df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
    df.index = pd.to_datetime(df.index)
    return df

# Helper function to forecast using ARIMA
from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(data, start_date):
    # Fit ARIMA model
    model = ARIMA(data, order=(5,1,0))  # ARIMA model configuration (example: AR(5), I(1), MA(0))
    model_fit = model.fit()
    
    # Forecast the next 12 months
    forecast = model_fit.forecast(steps=12)
    
    return forecast


# Helper function to forecast using Rolling Window (Exponential Smoothing)
def rolling_window_forecast(data, start_date):
    forecast = []
    window_size = 12  # Example rolling window size
    
    for i in range(12):  # Forecasting for the next 12 months
        # Use the last `window_size` data points to forecast the next month
        window_data = data[-window_size:]
        forecast_value = window_data.mean()  # Simple mean of the window for prediction
        forecast.append(forecast_value)
        
        # Instead of using pd.Timedelta('1M'), use pd.DateOffset(months=1)
        data = pd.concat([data, pd.Series([forecast_value], index=[data.index[-1] + pd.DateOffset(months=1)])], ignore_index=False)
        
    return forecast



# Helper function to generate plot
import matplotlib.pyplot as plt
import io
import base64

def generate_plot(fedfunds, forecast_values, forecast_dates):
    # Combine historical and forecast data
    forecast_series = pd.Series(forecast_values, index=forecast_dates)
    
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(fedfunds.index, fedfunds['FEDFUNDS'], label='Historical FEDFUNDS', color='blue')
    plt.plot(forecast_series.index, forecast_series, label='Forecasted FEDFUNDS', color='red', linestyle='--')
    
    plt.xlabel('Date')
    plt.ylabel('FEDFUNDS Rate')
    plt.title('FEDFUNDS Forecast for 2025')
    plt.legend()
    
    # Save the plot to a BytesIO object and convert it to a base64 string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf8')
    
    return f"data:image/png;base64,{plot_url}"


# Flask route to render page and handle POST requests
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        month = request.form.get('month')
        model_choice = request.form.get('model_choice')

        # Validate that the month is a valid integer between 01 and 12
        if not month or int(month) < 1 or int(month) > 12:
            return render_template('index.html', error="Please select a valid month.")

        # Load the data
        fedfunds = load_data()

        # Perform forecast based on model choice
        forecast_start_date = f'2025-{month.zfill(2)}-01'

        # Perform forecasting for 2025
        if model_choice == 'ARIMA':
            forecast_values = arima_forecast(fedfunds['FEDFUNDS'], forecast_start_date)
        elif model_choice == 'Rolling Window':
            forecast_values = rolling_window_forecast(fedfunds['FEDFUNDS'], forecast_start_date)
        else:
            return render_template('index.html', error="Invalid model choice. Please try again.")

        # Generate forecast plot
        forecast_dates = pd.date_range(forecast_start_date, periods=12, freq='M')  # Forecast for the next 12 months
        plot_url = generate_plot(fedfunds, forecast_values, forecast_dates)

        return render_template('index.html', plot_url=plot_url)

    return render_template('index.html')



if __name__ == '__main__':
    app.run(debug=True)
