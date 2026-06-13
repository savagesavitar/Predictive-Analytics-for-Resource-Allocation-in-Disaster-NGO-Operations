"""
Model Training: Train and evaluate forecasting models for resource demand.
- LightGBM regressor with lag features
- Prophet for time-series forecasting per district
- Cross-validation and accuracy metrics (MAE, RMSE, MAPE)
"""
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import lightgbm as lgb
from prophet import Prophet

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'data', 'processed')
MODELS_DIR = os.path.join(BASE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def load_panel():
    path = os.path.join(PROC, 'odisha_flood_panel.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def prepare_lgb_data(df, target_col='resource_demand_score'):
    """Prepare data for LightGBM model."""
    # Features for modeling
    feature_cols = [
        'month_sin', 'month_cos',
        'DFSI', 'Corrected_Percent_Flooded_Area',
        'census_population', 'households',
        'flood_lag_1m', 'flood_lag_3m', 'flood_lag_6m', 'flood_lag_12m',
        'flood_roll_3m', 'flood_roll_12m',
        'flood_exposed_pop', 'pop_x_flood_severity',
    ]

    # Encode district
    le = LabelEncoder()
    df['district_encoded'] = le.fit_transform(df['district'])

    # One-hot encode season
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)

    feature_cols.extend(['district_encoded'])
    feature_cols.extend([c for c in season_dummies.columns])

    # Ensure all feature columns exist
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    # Handle any remaining NaN
    X = X.fillna(0)

    return X, y, feature_cols, le

def train_lgb_model(X, y, feature_cols):
    """Train LightGBM model with time-series CV."""
    # Sort by date for time-series split
    tscv = TimeSeriesSplit(n_splits=5)

    models = []
    cv_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = lgb.LGBMRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric='mae',
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )

        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        # MAPE only on non-zero actuals
        mask = y_val > 0
        mape = mean_absolute_percentage_error(y_val[mask], y_pred[mask]) if mask.any() else 0

        models.append(model)
        cv_scores.append({'fold': fold, 'mae': mae, 'rmse': rmse, 'mape': mape})
        print(f"Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2%}")

    # Best model (first fold, or ensemble)
    best_model = models[0]

    # Feature importance
    importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)

    return best_model, models, cv_scores, importance

def train_prophet_models(df):
    """Train Prophet model per district."""
    districts = df['district'].unique()
    prophet_models = {}
    prophet_metrics = {}

    for district in districts:
        district_df = df[df['district'] == district].copy()
        district_df = district_df.sort_values('date')

        # Prepare Prophet data
        prophet_df = pd.DataFrame({
            'ds': district_df['date'],
            'y': district_df['resource_demand_score']
        })

        # Train/test split (last 20% for test)
        split_idx = int(len(prophet_df) * 0.8)
        train_df = prophet_df.iloc[:split_idx]
        test_df = prophet_df.iloc[split_idx:]

        if len(train_df) < 12:
            print(f"  Skipping {district}: insufficient data ({len(train_df)} months)")
            continue

        try:
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='multiplicative',
                changepoint_prior_scale=0.05,
            )

            # Add monthly seasonality
            model.add_seasonality(name='monthly', period=30.5, fourier_order=5)

            # Add regressors if available
            regressor_cols = ['DFSI', 'Corrected_Percent_Flooded_Area', 'census_population']
            regressor_data = district_df[regressor_cols].iloc[0]
            for col in regressor_cols:
                if col in district_df.columns:
                    model.add_regressor(col)
                    prophet_df[col] = regressor_data[col]

            model.fit(train_df)

            # Predict on test
            future = model.make_future_dataframe(periods=len(test_df), freq='MS')
            for col in regressor_cols:
                future[col] = regressor_data[col]

            forecast = model.predict(future)
            y_pred = forecast['yhat'].iloc[split_idx:].values
            y_test = test_df['y'].values

            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mask = y_test > 0
            mape = mean_absolute_percentage_error(y_test[mask], y_pred[mask]) if mask.any() else 0

            prophet_models[district] = model
            prophet_metrics[district] = {'mae': float(mae), 'rmse': float(rmse), 'mape': float(mape)}
            print(f"  {district}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2%}")

        except Exception as e:
            print(f"  ERROR {district}: {e}")

    return prophet_models, prophet_metrics

def generate_forecast(model, X_latest, feature_cols, periods=12):
    """Generate future forecast using LightGBM model."""
    forecasts = []
    for i in range(periods):
        pred = model.predict(X_latest)[0]
        forecasts.append(pred)
    return np.array(forecasts)

def main():
    print("=" * 60)
    print("FLOOD RESOURCE DEMAND FORECASTING - MODEL TRAINING")
    print("=" * 60)

    # Load data
    print("\n1. Loading panel data...")
    df = load_panel()
    print(f"   Shape: {df.shape}")

    # Prepare LightGBM data
    print("\n2. Preparing LightGBM data...")
    X, y, feature_cols, le = prepare_lgb_data(df)
    print(f"   Features: {len(feature_cols)}")
    print(f"   X shape: {X.shape}")

    # Train LightGBM
    print("\n3. Training LightGBM model...")
    lgb_model, lgb_models, lgb_cv, lgb_importance = train_lgb_model(X, y, feature_cols)

    print("\n   Top 10 Feature Importances:")
    print(lgb_importance.head(10).to_string(index=False))

    # Save LightGBM model
    import joblib
    model_path = os.path.join(MODELS_DIR, 'lgb_resource_model.pkl')
    joblib.dump({
        'model': lgb_model,
        'feature_cols': feature_cols,
        'label_encoder': le,
        'cv_scores': lgb_cv,
        'feature_importance': lgb_importance.to_dict('records')
    }, model_path)
    print(f"\n   LightGBM model saved to {model_path}")

    # Summary metrics
    avg_mae = np.mean([s['mae'] for s in lgb_cv])
    avg_rmse = np.mean([s['rmse'] for s in lgb_cv])
    avg_mape = np.mean([s['mape'] for s in lgb_cv])
    print(f"\n   LightGBM CV Average: MAE={avg_mae:.4f}, RMSE={avg_rmse:.4f}, MAPE={avg_mape:.2%}")

    # Train Prophet models
    print("\n4. Training Prophet models per district...")
    prophet_models, prophet_metrics = train_prophet_models(df)

    # Save Prophet metrics
    metrics_path = os.path.join(MODELS_DIR, 'prophet_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(prophet_metrics, f, indent=2)

    # Comprehensive metrics report
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE REPORT")
    print("=" * 60)

    all_mae = [s['mae'] for s in lgb_cv]
    all_rmse = [s['rmse'] for s in lgb_cv]
    all_mape = [s['mape'] for s in lgb_cv if s['mape'] > 0]

    report = {
        'lightgbm': {
            'cv_mae_mean': float(np.mean(all_mae)),
            'cv_mae_std': float(np.std(all_mae)),
            'cv_rmse_mean': float(np.mean(all_rmse)),
            'cv_rmse_std': float(np.std(all_rmse)),
            'cv_mape_mean': float(np.mean(all_mape)) if all_mape else 0,
            'cv_scores': lgb_cv,
            'feature_importance': lgb_importance.to_dict('records'),
            'n_features': len(feature_cols),
        },
        'prophet': {
            'districts_trained': list(prophet_metrics.keys()),
            'district_metrics': prophet_metrics,
            'avg_mae': float(np.mean([m['mae'] for m in prophet_metrics.values()])),
            'avg_rmse': float(np.mean([m['rmse'] for m in prophet_metrics.values()])),
            'avg_mape': float(np.mean([m['mape'] for m in prophet_metrics.values() if m['mape'] > 0])),
        },
        'dataset': {
            'n_records': len(df),
            'n_districts': df['district'].nunique(),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
            'flood_event_rate': float(df['flood_occurrence'].mean()),
        }
    }

    report_path = os.path.join(MODELS_DIR, 'metrics_report.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nMetrics report saved to {report_path}")

    # Print final summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS:")
    print(f"  LightGBM CV MAE: {avg_mae:.4f}")
    print(f"  LightGBM CV RMSE: {avg_rmse:.4f}")
    print(f"  LightGBM CV MAPE: {avg_mape:.2%}")
    if prophet_metrics:
        prophet_mae = np.mean([m['mae'] for m in prophet_metrics.values()])
        prophet_rmse = np.mean([m['rmse'] for m in prophet_metrics.values()])
        prophet_mape_list = [m['mape'] for m in prophet_metrics.values() if m['mape'] > 0]
        prophet_mape = np.mean(prophet_mape_list) if prophet_mape_list else 0
        print(f"  Prophet Avg MAE: {prophet_mae:.4f}")
        print(f"  Prophet Avg RMSE: {prophet_rmse:.4f}")
        print(f"  Prophet Avg MAPE: {prophet_mape:.2%}")
    print(f"{'='*60}")

    return report

if __name__ == '__main__':
    report = main()
