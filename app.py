from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objs as go
import plotly.io as pio

app = Flask(__name__)

# Load trained ARIMA model and scaler
model = joblib.load('modelling/arima_model.pkl')

# Load data
data_path = 'modelling/transformed_cotton_production.csv'  # path to your data
df = pd.read_csv(data_path)

# Convert the Year column to integers if not already
df['Year'] = df['Year'].astype(int)

@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to get the list of available regions
@app.route('/regions', methods=['GET'])
def get_regions():
    regions = df['Region'].unique().tolist()
    return jsonify({'regions': regions})

# Predict next 5 years total cotton production
@app.route('/predict/whole', methods=['GET'])
def predict_whole():
    forecast = model.forecast(steps=5)
    last_year = df['Year'].max()  # Get the most recent year in the dataset
    years = list(range(int(last_year) + 1, int(last_year) + 6))  # Generate integer years
    
    forecast_df = pd.DataFrame({'Year': years, 'Prediction': forecast})
    
    # Create a Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=forecast_df['Year'], y=forecast_df['Prediction'], mode='lines+markers', name='Prediction'))
    
    # Set the background to be semi-transparent
    fig.update_layout(
        title='Total Cotton Production Forecast',
        xaxis_title='Year',
        yaxis_title='Production (tons)',
        paper_bgcolor='rgba(0,0,0,0.4)',  # Semi-transparent background for the entire chart
        plot_bgcolor='rgba(0,0,0,0)',     # Transparent background for the plotting area
        font_color='white'
    )
    
    # Convert the Plotly figure to JSON
    graphJSON = pio.to_json(fig)
    
    return jsonify({'forecast': forecast_df.to_dict(orient='records'), 'graph': graphJSON})

# Predict cotton production per economic region
@app.route('/predict/economic-region', methods=['GET'])
def predict_economic_region():
    economic_regions = df['Economic_Region'].unique()
    predictions = {}
    
    fig = go.Figure()  # Initialize figure for all regions
    for region in economic_regions:
        region_data = df[df['Economic_Region'] == region].groupby('Year')['Production'].sum()
        if len(region_data) > 1:
            model = ARIMA(region_data, order=(1, 1, 1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=5)

            last_year = region_data.index.max()
            years = list(range(int(last_year) + 1, int(last_year) + 6))
            fig.add_trace(go.Scatter(x=years, y=forecast, mode='lines+markers', name=region))
    
    # Set the background to be semi-transparent
    fig.update_layout(
        title='Cotton Production Forecast per Economic Region',
        xaxis_title='Year',
        yaxis_title='Production (tons)',
        paper_bgcolor='rgba(0,0,0,0.4)',
        plot_bgcolor='rgba(0,0,0,0)',
        font_color='white'
    )
    
    graphJSON = pio.to_json(fig)
    return jsonify({'predictions': predictions, 'graph': graphJSON})

# Predict cotton production per region (with region selection)
@app.route('/predict/region', methods=['GET'])
def predict_region():
    region = request.args.get('region', None)

    # Check if 'Region' column exists and has valid data
    if 'Region' not in df.columns:
        return jsonify({'error': 'Region column missing in dataset.'}), 400

    available_regions = df['Region'].unique()
    if region is None:
        region = available_regions[0]  # Default to the first region if no region provided
    
    # Ensure the region exists
    if region not in available_regions:
        return jsonify({'error': f'Region "{region}" not found.'}), 400

    region_data = df[df['Region'] == region].groupby('Year')['Production'].sum()
    
    if len(region_data) > 1:
        model = ARIMA(region_data, order=(1, 1, 1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=5)

        last_year = region_data.index.max()
        years = list(range(int(last_year) + 1, int(last_year) + 6))
        forecast_df = pd.DataFrame({'Year': years, 'Prediction': forecast})
        
        # Create a Plotly figure for the specific region
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=forecast_df['Year'], y=forecast_df['Prediction'], mode='lines+markers', name='Prediction'))
        
        # Set the background to be semi-transparent
        fig.update_layout(
            title=f'Cotton Production Forecast for {region}',
            xaxis_title='Year',
            yaxis_title='Production (tons)',
            paper_bgcolor='rgba(0,0,0,0.4)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_color='white'
        )
        
        graphJSON = pio.to_json(fig)

        return jsonify({'forecast': forecast_df.to_dict(orient='records'), 'graph': graphJSON})
    else:
        return jsonify({'error': f'Not enough data available to predict for region: {region}'}), 400

if __name__ == '__main__':
    app.run(debug=True)
