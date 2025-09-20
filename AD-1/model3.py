def prediction(stock, n_days):
    import yfinance as yf
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split, GridSearchCV
    from datetime import date, timedelta
    import plotly.graph_objs as go
    import numpy as np

    # Fetch stock data from yfinance
    try:
        df = yf.download("NFLX", period='6mo')
        if df.empty:
            print(f"No data found for stock {stock}")
            return None
    except Exception as e:
        print(f"Error fetching data for stock {stock}: {e}")
        return None

    # Print the head of the dataframe to ensure data is correct
    print(f"Data for {stock}:\n", df.head())

    # Preprocess the data
    df.reset_index(inplace=True)
    df['Day'] = df.index
    X = [[i] for i in range(len(df))]
    Y = df[['Close']].values.ravel()

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

    # Train the model
    try:
        gsc = GridSearchCV(estimator=SVR(kernel='rbf'),
                           param_grid={'C': [0.001, 0.01, 0.1, 1, 100, 1000],
                                       'epsilon': [0.0001, 0.0005, 0.001, 0.005],
                                       'gamma': [0.0001, 0.001, 0.01, 0.1]},
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
    output_days = [[i + len(df)] for i in range(n_days)]

    # Generate future dates
    dates = [date.today() + timedelta(days=i) for i in range(1, n_days + 1)]

    # Create prediction plot
    try:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['Day'], y=df['Close'], mode='markers', name='Actual'))
        fig.add_trace(go.Scatter(x=output_days, y=best_svr.predict(output_days), mode='lines', name='Forecast'))
        
        fig.update_layout(title=f"Stock Price Forecast for {stock}", xaxis_title="Days", yaxis_title="Close Price")
        
        # Debug print to ensure figure is constructed
        print("Output days: ", output_days)
        print("Predicted values: ", best_svr.predict(output_days))

        print("Figure constructed successfully.")
        
        return fig
    except Exception as e:
        print(f"Error creating figure: {e}")
        return None
