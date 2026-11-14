"""
Machine Learning demand prediction and inventory recommendation module.
Uses Scikit-learn (Random Forest & Linear Regression) to forecast future product demand
based on historical sales time-series data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


def prepare_product_sales_data(sales_df, product_id):
    """
    Extracts daily sales for a specific product and fills missing dates with 0 sales.
    """
    # Filter by product
    p_sales = sales_df[sales_df["product_id"] == product_id].copy()
    if p_sales.empty:
        return pd.DataFrame()

    p_sales["sale_date"] = pd.to_datetime(p_sales["sale_date"])
    daily = p_sales.groupby("sale_date")["quantity"].sum().reset_index()

    # Reindex to full continuous calendar range
    min_date = daily["sale_date"].min()
    max_date = daily["sale_date"].max()
    all_dates = pd.date_range(start=min_date, end=max_date, freq="D")
    
    daily = daily.set_index("sale_date").reindex(all_dates, fill_value=0).reset_index()
    daily.rename(columns={"index": "sale_date"}, inplace=True)
    
    return daily


def build_features(daily_df):
    """
    Constructs feature set for time-series ML regression:
    - day_num: sequential trend index
    - day_of_week: weekly seasonality
    - day_of_month: monthly pattern
    - is_weekend: weekend flag
    - rolling_7d: 7-day moving average of sales
    - lag_1: previous day sales
    """
    df = daily_df.copy()
    start_date = df["sale_date"].min()
    
    df["day_num"] = (df["sale_date"] - start_date).dt.days
    df["day_of_week"] = df["sale_date"].dt.dayofweek
    df["day_of_month"] = df["sale_date"].dt.day
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x in [5, 6] else 0)
    df["rolling_7d"] = df["quantity"].rolling(window=7, min_periods=1).mean()
    df["lag_1"] = df["quantity"].shift(1).fillna(df["quantity"].mean())
    
    return df


def train_demand_model(daily_df, model_type="random_forest"):
    """
    Trains a Scikit-Learn regressor on historical sales.
    Returns: (trained_model, feature_names, metrics_dict, feature_df)
    """
    if len(daily_df) < 5:
        return None, [], {}, daily_df

    df_feat = build_features(daily_df)
    feature_cols = ["day_num", "day_of_week", "day_of_month", "is_weekend", "rolling_7d", "lag_1"]
    
    X = df_feat[feature_cols]
    y = df_feat["quantity"]

    if model_type == "linear_regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, max_depth=6, random_state=42)

    model.fit(X, y)
    preds = model.predict(X)
    
    mae = mean_absolute_error(y, preds)
    r2 = r2_score(y, preds)
    
    metrics = {
        "mae": round(mae, 2),
        "r2": round(max(0.0, r2), 2),
        "model_type": "Random Forest Regressor" if model_type == "random_forest" else "Linear Regression"
    }

    return model, feature_cols, metrics, df_feat


def forecast_future_demand(model, feature_cols, last_date, last_df, forecast_days=7):
    """
    Forecasts demand for the next N days iteratively using the trained model.
    """
    future_records = []
    current_date = last_date
    start_date = last_df["sale_date"].min()
    
    recent_quantities = list(last_df["quantity"].tail(7))
    last_qty = recent_quantities[-1] if recent_quantities else 1.0

    for step in range(1, forecast_days + 1):
        future_date = current_date + timedelta(days=step)
        day_num = (future_date - start_date).days
        day_of_week = future_date.weekday()
        day_of_month = future_date.day
        is_weekend = 1 if day_of_week in [5, 6] else 0
        rolling_7d = float(np.mean(recent_quantities[-7:]))
        lag_1 = float(last_qty)

        feature_values = pd.DataFrame([{
            "day_num": day_num,
            "day_of_week": day_of_week,
            "day_of_month": day_of_month,
            "is_weekend": is_weekend,
            "rolling_7d": rolling_7d,
            "lag_1": lag_1
        }])[feature_cols]

        pred_raw = model.predict(feature_values)[0]
        pred_qty = max(0.0, round(float(pred_raw), 1))

        future_records.append({
            "sale_date": future_date,
            "predicted_demand": pred_qty
        })

        # Slide window for next step
        recent_quantities.append(pred_qty)
        last_qty = pred_qty

    forecast_df = pd.DataFrame(future_records)
    return forecast_df


def generate_inventory_recommendation(product_info, forecast_df, forecast_days):
    """
    Compares predicted demand against current inventory and safety buffer.
    Produces clear actionable recommendation.
    """
    current_stock = int(product_info["stock"])
    min_stock = int(product_info["min_stock"])
    total_predicted_demand = int(round(forecast_df["predicted_demand"].sum()))
    
    # Net balance after satisfying predicted demand
    projected_remaining = current_stock - total_predicted_demand

    if projected_remaining < 0:
        shortfall = abs(projected_remaining)
        recommended_restock = shortfall + min_stock
        status = "URGENT_RESTOCK"
        badge_color = "#e53e3e"  # Red
        title = "Urgent Restock Required"
        message = (
            f"Projected demand ({total_predicted_demand} units) exceeds current stock ({current_stock} units). "
            f"You will stock out by approximately {shortfall} units. "
            f"Recommend ordering at least {recommended_restock} units immediately to fulfill demand and restore safety buffer."
        )
    elif projected_remaining <= min_stock:
        shortfall = min_stock - projected_remaining
        recommended_restock = shortfall + 5
        status = "RESTOCK_ADVISED"
        badge_color = "#dd6b20"  # Orange
        title = "Restock Advised (Low Safety Buffer)"
        message = (
            f"Current stock ({current_stock} units) can cover projected demand ({total_predicted_demand} units), "
            f"but remaining inventory ({projected_remaining} units) will fall below your safety threshold ({min_stock} units). "
            f"Recommend restocking at least {recommended_restock} units soon."
        )
    else:
        recommended_restock = 0
        status = "STOCK_SUFFICIENT"
        badge_color = "#38a169"  # Green
        title = "Stock Sufficient & Healthy"
        message = (
            f"Current stock ({current_stock} units) is sufficient to meet projected demand ({total_predicted_demand} units) "
            f"for the next {forecast_days} days while preserving safety stock buffer."
        )

    return {
        "status": status,
        "badge_color": badge_color,
        "title": title,
        "message": message,
        "current_stock": current_stock,
        "min_stock": min_stock,
        "total_predicted_demand": total_predicted_demand,
        "projected_remaining": projected_remaining,
        "recommended_restock": recommended_restock,
        "forecast_days": forecast_days
    }


def run_demand_forecast(sales_df, product_row, forecast_days=7, model_type="random_forest"):
    """
    Full pipeline to run demand forecast for a product and generate recommendation.
    """
    daily_sales = prepare_product_sales_data(sales_df, product_row["id"])
    
    if daily_sales.empty or len(daily_sales) < 5:
        # Fallback if no or little sales history
        avg_daily = 1.0
        dates = [datetime.now().date() + timedelta(days=i) for i in range(1, forecast_days + 1)]
        forecast_df = pd.DataFrame({
            "sale_date": pd.to_datetime(dates),
            "predicted_demand": [avg_daily] * forecast_days
        })
        metrics = {"mae": 0.0, "r2": 0.0, "model_type": "Baseline Average"}
        recommendation = generate_inventory_recommendation(product_row, forecast_df, forecast_days)
        return daily_sales, forecast_df, metrics, recommendation

    model, feature_cols, metrics, df_feat = train_demand_model(daily_sales, model_type=model_type)
    last_date = daily_sales["sale_date"].max()
    forecast_df = forecast_future_demand(model, feature_cols, last_date, df_feat, forecast_days=forecast_days)
    recommendation = generate_inventory_recommendation(product_row, forecast_df, forecast_days)

    return daily_sales, forecast_df, metrics, recommendation
