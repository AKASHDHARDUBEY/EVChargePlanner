import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


def plot_time_series(df, title="EV Charging Volume Over Time"):
    fig = px.line(df, x='time', y='total_volume', title=title)
    fig.update_layout(xaxis_title="Time", yaxis_title="Charging Volume (kWh)", template="plotly_white")
    return fig


def plot_hourly_pattern(df):
    hourly_avg = df.groupby('hour')['total_volume'].mean().reset_index()
    fig = px.bar(hourly_avg, x='hour', y='total_volume', title="Average Charging Volume by Hour")
    fig.update_layout(xaxis_title="Hour of Day", yaxis_title="Avg Volume (kWh)", template="plotly_white")
    return fig


def plot_daily_pattern(df):
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    daily_avg = df.groupby('day_of_week')['total_volume'].mean().reset_index()
    daily_avg['day_name'] = daily_avg['day_of_week'].map(lambda x: day_names[x])
    fig = px.bar(daily_avg, x='day_name', y='total_volume', title="Average Charging Volume by Day of Week")
    fig.update_layout(xaxis_title="Day of Week", yaxis_title="Avg Volume (kWh)", template="plotly_white")
    return fig


def plot_monthly_pattern(df):
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_avg = df.groupby('month')['total_volume'].mean().reset_index()
    monthly_avg['month_name'] = monthly_avg['month'].map(lambda x: month_names[x - 1])
    fig = px.bar(monthly_avg, x='month_name', y='total_volume', title="Average Charging Volume by Month")
    fig.update_layout(xaxis_title="Month", yaxis_title="Avg Volume (kWh)", template="plotly_white")
    return fig


def plot_heatmap(df):
    pivot_df = df.groupby(['day_of_week', 'hour'])['total_volume'].mean().reset_index()
    pivot_table = pivot_df.pivot(index='day_of_week', columns='hour', values='total_volume')
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    pivot_table.index = [day_names[i] for i in pivot_table.index]
    fig = px.imshow(pivot_table, title="Charging Volume Heatmap (Day vs Hour)",
                    labels=dict(x="Hour", y="Day", color="Volume (kWh)"),
                    color_continuous_scale="Blues")
    return fig


def plot_predictions(actual, predicted, dates, title="Actual vs Predicted"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=actual, mode='lines', name='Actual', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=dates, y=predicted, mode='lines', name='Predicted', line=dict(color='red')))
    fig.update_layout(title=title, xaxis_title="Time", yaxis_title="Charging Volume (kWh)", template="plotly_white")
    return fig


def plot_feature_importance(importance_dict, top_n=10):
    top_features = dict(list(importance_dict.items())[:top_n])
    fig = px.bar(x=list(top_features.values()), y=list(top_features.keys()), orientation='h',
                 title=f"Top {top_n} Feature Importance")
    fig.update_layout(xaxis_title="Importance", yaxis_title="Feature", template="plotly_white")
    return fig


def plot_residuals(actual, predicted):
    residuals = actual - predicted
    fig = px.histogram(residuals, nbins=50, title="Prediction Residuals Distribution")
    fig.update_layout(xaxis_title="Residual", yaxis_title="Frequency", template="plotly_white")
    return fig


def plot_forecast(historical_dates, historical_values, forecast_dates, forecast_values):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=historical_dates, y=historical_values, mode='lines', name='Historical', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=forecast_dates, y=forecast_values, mode='lines', name='Forecast', line=dict(color='red', dash='dash')))
    fig.update_layout(title="Demand Forecast", xaxis_title="Time", yaxis_title="Charging Volume (kWh)", template="plotly_white")
    return fig


def plot_model_comparison(comparison_results, metric='RMSE'):
    model_names = list(comparison_results.keys())
    train_vals = [comparison_results[m]['train_metrics'][metric] for m in model_names]
    test_vals = [comparison_results[m]['test_metrics'][metric] for m in model_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name='Training', x=model_names, y=train_vals, marker_color='#636EFA'))
    fig.add_trace(go.Bar(name='Testing', x=model_names, y=test_vals, marker_color='#EF553B'))
    fig.update_layout(title=f"Model Comparison — {metric}", xaxis_title="Model",
                      yaxis_title=metric, barmode='group', template="plotly_white")
    return fig


def plot_multi_model_predictions(y_test, comparison_results, dates):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=dates, y=y_test, mode='lines', name='Actual', line=dict(color='blue', width=2)))

    colors = ['red', 'green', 'orange', 'purple']
    for i, (name, result) in enumerate(comparison_results.items()):
        fig.add_trace(go.Scatter(x=dates, y=result['predictions'], mode='lines',
                                 name=f'{name}', line=dict(color=colors[i % len(colors)], dash='dash')))

    fig.update_layout(title="All Models — Predictions vs Actual", xaxis_title="Time",
                      yaxis_title="Charging Volume (kWh)", template="plotly_white")
    return fig