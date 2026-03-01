import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_model(model_name="linear_regression"):
    if model_name == "random_forest":
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    return LinearRegression()


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def predict(model, X):
    return model.predict(X)


def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100

    return {
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'R2 Score': round(r2, 4),
        'MAPE (%)': round(mape, 2)
    }


def get_feature_importance(model, feature_names):
    # random forest has feature_importances_, linear regression uses coef_
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    else:
        importance = np.abs(model.coef_)

    importance_dict = dict(zip(feature_names, importance))
    importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
    return importance_dict


def forecast_future(model, last_data, feature_cols, steps=24):
    predictions = []
    current_data = last_data.copy()

    for i in range(steps):
        X = current_data[feature_cols].values.reshape(1, -1)
        pred = model.predict(X)[0]
        predictions.append(pred)

        # shift lag values forward
        current_data['lag_1'] = pred
        current_data['lag_2'] = current_data['lag_1']
        current_data['lag_3'] = current_data['lag_2']

        # advance time by 1 hour
        current_data['hour'] = (current_data['hour'] + 1) % 24
        if current_data['hour'] == 0:
            current_data['day_of_week'] = (current_data['day_of_week'] + 1) % 7

        # update cyclical features
        current_data['hour_sin'] = np.sin(2 * np.pi * current_data['hour'] / 24)
        current_data['hour_cos'] = np.cos(2 * np.pi * current_data['hour'] / 24)
        current_data['day_sin'] = np.sin(2 * np.pi * current_data['day_of_week'] / 7)
        current_data['day_cos'] = np.cos(2 * np.pi * current_data['day_of_week'] / 7)
        current_data['is_weekend'] = 1 if current_data['day_of_week'] >= 5 else 0

    return predictions


def compare_models(X_train, y_train, X_test, y_test):
    """Train both models on same data and return their metrics + predictions."""
    results = {}

    models_to_try = {
        'Linear Regression': 'linear_regression',
        'Random Forest': 'random_forest'
    }

    for name, key in models_to_try.items():
        model = get_model(key)
        model = train_model(model, X_train, y_train)

        pred_train = predict(model, X_train)
        pred_test = predict(model, X_test)

        results[name] = {
            'model': model,
            'train_metrics': evaluate_model(y_train, pred_train),
            'test_metrics': evaluate_model(y_test, pred_test),
            'predictions': pred_test
        }

    return results
