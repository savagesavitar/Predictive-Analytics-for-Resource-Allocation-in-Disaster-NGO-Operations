"""
Model Training v2: Improved forecasting with two-stage approach.
- Stage 1: Classify flood probability (LightGBM classifier)
- Stage 2: Regress demand given flood (LightGBM regressor)
- Prophet per district (simplified, no regressors)
- Evaluation with realistic resource demand targets
"""
import pandas as pd
import numpy as np
import os
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             mean_absolute_percentage_error, accuracy_score,
                             precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
import lightgbm as lgb
from prophet import Prophet
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'data', 'processed')
MODELS_DIR = os.path.join(BASE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def load_panel():
    path = os.path.join(PROC, 'odisha_flood_panel.csv')
    df = pd.read_csv(path, parse_dates=['date'])
    return df

def compute_resource_targets(df):
    """
    Compute realistic resource demand based on SPHERE humanitarian standards.
    Returns target variables and adds them to the dataframe.
    """
    # SPHERE standards per person per month:
    # Medical kit: 1 per 1000 people/month
    # ORS packets: 200 per 1000 people/month
    # Drinking water: 15L/person/day (105L/week)
    # Tarpaulin sheets: 1 per family (5 people)
    # Dry rations: 2100 kcal/person/day (30-day supply)

    # Base population at risk (from census)
    pop_at_risk = df['census_population'].fillna(df['census_population'].median())

    # Flood severity multiplier (0-1 scale)
    severity_norm = df['severity'] / 10.0  # Normalize
    severity_norm = severity_norm.clip(0, 1)

    # Displaced fraction
    displaced_frac = df['displaced'].fillna(0) / pop_at_risk
    displaced_frac = displaced_frac.clip(0, 1)

    # Flood occurrence multiplier
    flood_mult = (severity_norm * 0.5 + displaced_frac * 0.3 +
                  (df['flood_occurrence'] * 0.2))

    # Affected population
    affected_pop = pop_at_risk * flood_mult

    # Resource demands based on SPHERE standards
    df['medical_kits'] = (affected_pop / 1000).round(0)  # 1 kit per 1000 people
    df['ors_packets'] = (affected_pop * 0.2).round(0)     # 200 per 1000
    df['tarpaulin_sheets'] = (affected_pop / 5).round(0)  # 1 per 5 people
    df['water_litres'] = (affected_pop * 15 * 30).round(0)  # 15L/person/day * 30 days
    df['dry_rations_kg'] = (affected_pop * 0.4 * 30).round(1)  # 400g/person/day * 30 days

    # Composite demand score (log-transformed to handle wide range)
    demand_components = (
        df['medical_kits'] / 10000 +
        df['ors_packets'] / 50000 +
        df['tarpaulin_sheets'] / 5000 +
        df['water_litres'] / 10000000 +
        df['dry_rations_kg'] / 100000
    )
    # Log transform to compress the range
    df['composite_demand'] = np.log1p(demand_components * 1000)
    df['composite_demand'] = df['composite_demand'].clip(0, 10)

    # Binary flood event flag for classification
    df['flood_event'] = (df['composite_demand'] > 0).astype(int)

    return df

def prepare_features(df):
    """Prepare feature matrix with encoding."""
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

    # Season dummies
    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)

    feature_cols.append('district_encoded')
    feature_cols.extend([c for c in season_dummies.columns])

    # Filter to existing cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy().fillna(0)

    return X, feature_cols, le

def train_classifier(X, y_class):
    """Stage 1: Train LightGBM classifier for flood event probability."""
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y_class.iloc[tr_idx], y_class.iloc[val_idx]

        # Handle class imbalance
        scale_pos_weight = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)

        clf = lgb.LGBMClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42, verbose=-1, class_weight=None
        )

        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        scores.append({
            'fold': fold,
            'accuracy': float(accuracy_score(y_val, y_pred)),
            'precision': float(precision_score(y_val, y_pred, zero_division=0)),
            'recall': float(recall_score(y_val, y_pred, zero_division=0)),
            'f1': float(f1_score(y_val, y_pred, zero_division=0)),
            'roc_auc': float(roc_auc_score(y_val, y_prob)) if len(np.unique(y_val)) > 1 else 0,
        })

        models.append(clf)
        print(f"  Classifier Fold {fold}: AUC={scores[-1]['roc_auc']:.4f}, "
              f"F1={scores[-1]['f1']:.4f}, Acc={scores[-1]['accuracy']:.4f}")

    return models, scores

def train_regressor(X, y_reg):
    """Stage 2: Train LightGBM regressor for demand (only flood months)."""
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    scores = []

    # Only use positive demand months
    has_demand = y_reg > 0
    X_pos = X[has_demand]
    y_pos = y_reg[has_demand]

    if len(X_pos) < 20:
        print(f"  WARNING: Only {len(X_pos)} positive demand samples. Using all data.")
        X_pos, y_pos = X, y_reg

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_pos)):
        if len(tr_idx) < 2 or len(val_idx) < 2:
            continue
        X_tr, X_val = X_pos.iloc[tr_idx], X_pos.iloc[val_idx]
        y_tr, y_val = y_pos.iloc[tr_idx], y_pos.iloc[val_idx]

        reg = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1
        )

        reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        y_pred = reg.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mask = y_val > 0
        mape = (mean_absolute_percentage_error(y_val[mask], y_pred[mask])
                if mask.any() else 0)

        scores.append({'fold': fold, 'mae': float(mae), 'rmse': float(rmse),
                       'mape': float(mape)})
        models.append(reg)
        print(f"  Regressor Fold {fold}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2%}")

    return models, scores

def train_prophet_per_district(df):
    """Train simple Prophet per district (no external regressors)."""
    districts = df['district'].unique()
    models = {}
    metrics = {}

    for dist in districts:
        ddf = df[df['district'] == dist].sort_values('date').copy()
        prophet_df = pd.DataFrame({'ds': ddf['date'], 'y': ddf['composite_demand']})

        split = int(len(prophet_df) * 0.8)
        if split < 12:
            continue
        train, test = prophet_df.iloc[:split], prophet_df.iloc[split:]

        try:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                       daily_seasonality=False, seasonality_mode='multiplicative',
                       changepoint_prior_scale=0.05)
            m.fit(train)
            future = m.make_future_dataframe(periods=len(test), freq='MS')
            forecast = m.predict(future)

            y_pred = forecast['yhat'].iloc[split:].values
            y_test = test['y'].values

            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mask = y_test > 0
            mape = (float(mean_absolute_percentage_error(y_test[mask], y_pred[mask]))
                    if mask.any() else 0)

            models[dist] = m
            metrics[dist] = {'mae': mae, 'rmse': rmse, 'mape': mape}
            print(f"  Prophet {dist}: MAE={mae:.4f}, RMSE={rmse:.4f}, MAPE={mape:.2%}")
        except Exception as e:
            print(f"  Prophet {dist} ERROR: {e}")

    return models, metrics

def main():
    print("=" * 60)
    print("FLOOD RESOURCE DEMAND FORECASTING - V2")
    print("=" * 60)

    # Load data
    df = load_panel()
    print(f"\n1. Loaded panel: {df.shape}")

    # Compute resource targets
    df = compute_resource_targets(df)
    print(f"2. Added resource demand targets")
    print(f"   Flood event rate: {df['flood_event'].mean():.2%}")
    print(f"   Mean composite demand (non-zero): {df[df['composite_demand']>0]['composite_demand'].mean():.2f}")

    # Prepare features
    X, feature_cols, le = prepare_features(df)
    print(f"3. Features: {len(feature_cols)}")

    # Train classifier (Stage 1)
    print("\n4. Stage 1: Flood Event Classifier")
    cls_models, cls_scores = train_classifier(X, df['flood_event'])

    # Train regressor (Stage 2)
    print("\n5. Stage 2: Demand Regressor")
    reg_models, reg_scores = train_regressor(X, df['composite_demand'])

    # Prophet
    print("\n6. Train Prophet per district")
    prop_models, prop_metrics = train_prophet_per_district(df)

    # Save models
    print("\n7. Saving models...")
    joblib.dump({
        'classifier': cls_models[0] if cls_models else None,
        'regressor': reg_models[0] if reg_models else None,
        'feature_cols': feature_cols,
        'label_encoder': le,
        'cls_scores': cls_scores,
        'reg_scores': reg_scores,
    }, os.path.join(MODELS_DIR, 'two_stage_model.pkl'))

    # Compute aggregate metrics
    cls_avg_auc = np.mean([s['roc_auc'] for s in cls_scores])
    cls_avg_f1 = np.mean([s['f1'] for s in cls_scores])
    reg_avg_mae = np.mean([s['mae'] for s in reg_scores]) if reg_scores else 0
    reg_avg_mape = np.mean([s['mape'] for s in reg_scores if s['mape'] > 0]) if reg_scores else 0

    # Build report
    report = {
        'classifier': {
            'avg_roc_auc': float(cls_avg_auc),
            'avg_f1': float(cls_avg_f1),
            'fold_scores': cls_scores,
        },
        'regressor': {
            'avg_mae': float(reg_avg_mae),
            'avg_mape': float(reg_avg_mape),
            'fold_scores': reg_scores,
        },
        'prophet': {
            'district_metrics': prop_metrics,
            'avg_mae': float(np.mean([m['mae'] for m in prop_metrics.values()])) if prop_metrics else 0,
            'avg_mape': float(np.mean([m['mape'] for m in prop_metrics.values() if m['mape'] > 0])) if prop_metrics else 0,
        },
        'feature_importance': {
            'classifier': dict(zip(feature_cols,
                [int(x) for x in cls_models[0].feature_importances_])) if cls_models else {},
            'regressor': dict(zip(feature_cols,
                [int(x) for x in reg_models[0].feature_importances_])) if reg_models else {},
        },
        'dataset_summary': {
            'n_records': len(df),
            'n_districts': int(df['district'].nunique()),
            'flood_event_rate': float(df['flood_event'].mean()),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
        }
    }

    with open(os.path.join(MODELS_DIR, 'metrics_report_v2.json'), 'w') as f:
        json.dump(report, f, indent=2)

    # Final output
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Classifier Avg AUC: {cls_avg_auc:.4f}")
    print(f"  Classifier Avg F1:  {cls_avg_f1:.4f}")
    print(f"  Regressor Avg MAE:  {reg_avg_mae:.4f}")
    print(f"  Regressor Avg MAPE: {reg_avg_mape:.2%}")

    # Check if MAPE < 20% target
    if reg_avg_mape < 0.20:
        print(f"\n  ✓ MAPE target (<20%) MET: {reg_avg_mape:.2%}")
    else:
        print(f"\n  ✗ MAPE target (<20%): {reg_avg_mape:.2%} (above target)")

    return report

if __name__ == '__main__':
    main()
