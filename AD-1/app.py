import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px

# model
from model8 import prediction
from sklearn.svm import SVR

app = dash.Dash(__name__, external_stylesheets=['assets/styles.css'])
server = app.server

# Define the layout components

# Navigation component
item1 = html.Div(
    [
        html.P("Welcome to the Stock Dash App!", className="start"),

        html.Div([
            # stock code input
            dcc.Input(id='stock-code', type='text', placeholder='Enter stock code'),
            html.Button('Submit', id='submit-button')
        ], className="stock-input"),

        html.Div([
            # Date range picker input
            dcc.DatePickerRange(
                id='date-range', start_date=dt(2020, 1, 1).date(), end_date=dt.now().date(), className='date-input')
        ]),
        html.Div([
            # Stock price button
            html.Button('Get Stock Price', id='stock-price-button'),

            # Indicators button
            html.Button('Get Indicators', id='indicators-button'),

            # Number of days of forecast input
            dcc.Input(id='forecast-days', type='number', placeholder='Enter number of days'),

            # Forecast button
            html.Button('Get Forecast', id='forecast-button')
        ], className="selectors")
    ],
    className="nav"
)

# Content component
item2 = html.Div(
    [
        html.Div(
            [
                html.Img(id='logo', className='logo'),
                html.H1(id='company-name', className='company-name')
            ],
            className="header"),
        html.Div(id="description"),
        html.Div([], id="graphs-content"),
        html.Div([], id="main-content"),
        html.Div([], id="forecast-content")
    ],
    className="content"
)

# Set the layout
app.layout = html.Div(className='container', children=[item1, item2])

# Callbacks

# Callback to update the data based on the submitted stock code
@app.callback(
    [
        Output("description", "children"),
        Output("logo", "src"),
        Output("company-name", "children"),
        Output("stock-price-button", "n_clicks"),
        Output("indicators-button", "n_clicks"),
        Output("forecast-button", "n_clicks")
    ],
    [Input("submit-button", "n_clicks")],
    [State("stock-code", "value")]
)
def update_data(n, val):
    if n is None:
        return "", "", "", 0, 0, 0
    if val is None or str(val).strip() == "":
        raise PreventUpdate
    try:
        ticker = yf.Ticker(val)
        try:
            inf = ticker.get_info()
        except Exception:
            inf = ticker.info
    except Exception:
        inf = {}

    name = inf.get('longName') or str(val).upper()
    logo_url = inf.get('logo_url') or ""
    description = inf.get('longBusinessSummary') or ""
    return description, logo_url, name, 0, 0, 0


# Callback for displaying stock price graphs
@app.callback(
    [Output("graphs-content", "children")],
    [
        Input("stock-price-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def stock_price(n, start_date, end_date, val):
    if not n:
        return [""]
    if val is None or str(val).strip() == "":
        raise PreventUpdate
    try:
        if start_date is not None and end_date is not None:
            df = yf.download(str(val).strip(), start=str(start_date), end=str(end_date))
        else:
            df = yf.download(str(val).strip(), period='6mo')
    except Exception:
        return [html.P("Failed to fetch data. Please try again later.")]

    if df is None or df.empty:
        return [html.P("No data available for this stock/date range.")]

    df.reset_index(inplace=True)
    if not {'Date','Close','Open'}.issubset(df.columns):
        return [html.P("Unexpected data format received.")]

    fig = px.line(df, x="Date", y=["Close", "Open"], title="Closing and Opening Price vs Date")
    return [dcc.Graph(figure=fig)]


# Callback for displaying indicators
@app.callback(
    [Output("main-content", "children")],
    [
        Input("indicators-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def indicators(n, start_date, end_date, val):
    if not n:
        return [""]
    if val is None:
        return [""]
    if start_date is None:
        df_more = yf.download(val)
    else:
        df_more = yf.download(val, str(start_date), str(end_date))

    df_more.reset_index(inplace=True)
    fig = get_more(df_more)
    return [dcc.Graph(figure=fig)]


def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x="Date", y="EWA_20", title="Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig


# Callback for displaying forecast
@app.callback(
    [Output("forecast-content", "children")],
    [Input("forecast-button", "n_clicks")],
    [State("forecast-days", "value"),
     State("stock-code", "value")]
)
def forecast(n, n_days, val):
    if not n:
        return [""]
    if val is None or str(val).strip() == "":
        raise PreventUpdate
    if n_days is None:
        raise PreventUpdate
    try:
        days = int(n_days)
    except Exception:
        raise PreventUpdate
    print("Forecast Days: ", days)
    fig = prediction(val, days + 1)
    return [dcc.Graph(figure=fig)]


if __name__ == '__main__':
    app.run_server(debug=True)
