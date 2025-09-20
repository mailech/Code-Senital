def prediction(stock, n_days, start_date=None, end_date=None):
    import yfinance as yf
    from sklearn.svm import SVR
    from sklearn.model_selection import train_test_split, GridSearchCV
    from datetime import timedelta
    import plotly.graph_objs as go
    import numpy as np

    # Guard inputs
    try:
        n_days = int(n_days)
        if n_days <= 0:
            return None
    except Exception:
        return None

    # Load data safely
    try:
        if start_date and end_date:
            df = yf.download(str(stock).strip(), start=start_date, end=end_date)
        else:
            df = yf.download(str(stock).strip(), period='6mo')
    except Exception:
        return None
    if df is None or df.empty:
        return None

    # Prepare features/labels
    df.reset_index(inplace=True)
    if 'Date' not in df.columns or 'Close' not in df.columns:
        return None
    df['Day'] = (df['Date'] - df['Date'].min()).dt.days
    X = df[['Day']].values.reshape(-1, 1)
    Y = df['Close'].values.ravel()

    # Train/test split
    if len(X) < 10:
        return None
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

    # Hyperparameter search (small grid for speed)
    try:
        gsc = GridSearchCV(
            estimator=SVR(kernel='rbf'),
            param_grid={'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1], 'gamma': [0.001, 0.01]},
            cv=5, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1
        )
        grid_result = gsc.fit(x_train, y_train)
    except Exception:
        return None

    best_svr = SVR(kernel='rbf', C=grid_result.best_params_['C'],
                   epsilon=grid_result.best_params_['epsilon'],
                   gamma=grid_result.best_params_['gamma'])
    try:
        best_svr.fit(x_train, y_train)
    except Exception:
        return None

    # Predict next n_days
    last_day = int(df['Day'].max())
    future_days = np.arange(last_day + 1, last_day + 1 + n_days).reshape(-1, 1)
    try:
        pred = best_svr.predict(future_days).flatten()
    except Exception:
        return None
    # Build future date axis
    last_date = df['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, n_days + 1)]
    if len(pred) != len(future_dates):
        return None

    # Build figure: show historical close and forecast
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], mode='lines', name='Actual'))
    fig.add_trace(go.Scatter(x=future_dates, y=pred, mode='lines+markers', name='Forecast'))
    fig.update_layout(title=f"Stock Price Forecast for {stock}", xaxis_title="Date", yaxis_title="Close Price")
    return fig
