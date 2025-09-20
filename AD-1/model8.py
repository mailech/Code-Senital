def prediction(stock, n_days):
    import yfinance as yf
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split, GridSearchCV
    from datetime import date, timedelta
    import plotly.graph_objs as go
    import numpy as np

    # Fetch stock data from yfinance
    try:
        df = yf.download(stock, period='6mo')
        if df.empty:
            print(f"No data found for stock {stock}")
            return None
    except Exception as e:
        print(f"Error fetching data for stock {stock}: {e}")
        return None

    print(f"Data for {stock}:")
    print(df.head())

    # Preprocess the data
    df.reset_index(inplace=True)
    df['Day'] = (df['Date'] - df['Date'].min()).dt.days  
    X = df[['Day']].values.reshape(-1, 1)  # FIXED: Ensure X is 2D
    Y = df['Close'].values.ravel()  # FIXED: Ensure Y is 1D

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

    # Train the model
    try:
        gsc = GridSearchCV(estimator=SVR(kernel='rbf'),
                           param_grid={'C': [0.1, 1, 10],
                                       'epsilon': [0.01, 0.1],
                                       'gamma': [0.001, 0.01]},
                           cv=5, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)
        grid_result = gsc.fit(x_train, y_train)
    except Exception as e:
        print(f"Error during model training: {e}")
        return None

    # Best SVR model
    best_svr = SVR(kernel='rbf', C=grid_result.best_params_["C"],
                   epsilon=grid_result.best_params_["epsilon"],
                   gamma=grid_result.best_params_["gamma"])
    best_svr.fit(x_train, y_train)

    # Predicting future days
    future_days = np.array(range(df['Day'].max() + 1, df['Day'].max() + 1 + n_days)).reshape(-1, 1)
    future_dates = [df['Date'].max() + timedelta(days=i) for i in range(1, n_days + 1)]
    predictions = best_svr.predict(future_days).flatten()  # FIXED: Ensure predictions are 1D

    print("Predicted Values:")
    print(predictions)

    # Debugging Output
    print("Future Days:", future_days)
    print("Future Dates:", future_dates)

    # Create prediction plot
    try:
        print("Generating forecast plot...")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='markers', name='Actual'))
        fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Forecast'))
        fig.update_layout(title=f"Stock Price Forecast for {stock}", xaxis_title="Date", yaxis_title="Close Price")
        return fig
    except Exception as e:
        print(f"Error creating figure: {e}")
        return None

# Debugging: Run model separately
if __name__ == "__main__":
    test_fig = prediction("AAPL", 10)
    if test_fig:
        test_fig.show()
    else:
        print("No figure generated!")
