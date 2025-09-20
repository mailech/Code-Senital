import dash
from dash import dcc, html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from model3 import prediction

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])
server = app.server

app.layout = html.Div([
    html.Div([
        html.P("Welcome to the Stock Dash App!", className="start"),
        dcc.Input(id='stock-code', type='text', placeholder='Enter stock code'),
        html.Button('Submit', id='submit-button'),
        dcc.DatePickerRange(
            id='date-range', start_date=dt(2020, 1, 1).date(), end_date=dt.now().date()
        ),
        html.Button('Get Stock Price', id='stock-price-button'),
        html.Button('Get Indicators', id='indicators-button'),
        dcc.Input(id='forecast-days', type='number', placeholder='Enter number of days'),
        html.Button('Get Forecast', id='forecast-button')
    ], className="nav"),
    html.Div([
        html.Div(id="description"),
        html.Div(id="graphs-content"),
        html.Div(id="main-content"),
        html.Div(id="forecast-content")
    ], className="content")
])

@app.callback(
    Output("graphs-content", "children"),
    [Input("stock-price-button", "n_clicks")],
    [State("stock-code", "value"), State('date-range', 'start_date'), State('date-range', 'end_date')]
)
def stock_price(n, val, start_date, end_date):
    if n is None or val is None:
        raise PreventUpdate
    print(f"Fetching stock price for {val} from {start_date} to {end_date}")
    df = yf.download(val, start=start_date, end=end_date)
    
    if df.empty:
        print(f"Error: No stock data available for {val}")
        return [html.P("No data available for this stock.")]
    
    print("Stock Data Sample:")
    print(df.head())
    
    df.reset_index(inplace=True)
    df['Close'] = df['Close'].values.flatten()  # FIXED: Convert to 1D
    df['Open'] = df['Open'].values.flatten()    # FIXED: Convert to 1D
    
    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date")
    
    if fig is None:
        return [html.P("Error generating stock price graph.")]
    
    return [dcc.Graph(figure=fig)]

@app.callback(
    Output("forecast-content", "children"),
    [Input("forecast-button", "n_clicks")],
    [State("stock-code", "value"), State("forecast-days", "value")]
)
def get_forecast(n, stock, days):
    if n is None or stock is None or days is None:
        raise PreventUpdate
    print(f"Generating forecast for {stock} for {days} days")
    fig = prediction(stock, days)
    
    if fig is None:
        print("Error: Forecasting function returned None")
        return [html.P("Error generating forecast.")]
    
    return [dcc.Graph(figure=fig)]

if __name__ == '__main__':
    print("Starting Dash App...")
    app.run_server(debug=True)
