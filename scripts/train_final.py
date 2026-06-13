"""
Final Training Pipeline: Two-stage forecasting model.
Fixes:
- Prophet trained only on flood-event months (no flat zero baseline)
- Regressor uses log-transformed target for stable CV
- Classifier skips folds with zero positive samples
- Metrics computed only on valid folds
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
                             mean_absolute_percentage_error, roc_auc_score,
                             precision_recall_curve, auc, f1_score)
import lightgbm as lgb
from prophet import Prophet
import joblib

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(BASE, 'data', 'processed')
MODELS_DIR = os.path.join(BASE, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def load_panel():
    df = pd.read_csv(os.path.join(PROC, 'odisha_flood_panel.csv'), parse_dates=['date'])
    return df

def compute_targets(df):
    pop = df['census_population'].fillna(df['census_population'].median())
    severity_score = df['severity'].fillna(0)
    displaced = df['displaced'].fillna(0)
    fatalities = df['fatalities'].fillna(0)

    severity_norm = severity_score / np.clip(severity_score.max(), 1, None)
    displaced_frac = displaced / np.clip(pop, 1, None)
    flooded_area_norm = (df['Corrected_Percent_Flooded_Area'].fillna(0)
                         / np.clip(df['Corrected_Percent_Flooded_Area'].max(), 1, None))
    dfsi_norm = df['DFSI'] / np.clip(df['DFSI'].max(), 1, None)

    affected_frac = (severity_norm * 0.15 + displaced_frac * 0.25 +
                     flooded_area_norm * 0.35 + dfsi_norm * 0.25)
    affected_frac = affected_frac.clip(0, 0.3)

    affected_pop = (pop * affected_frac).fillna(0)

    df['medical_kits_demand'] = affected_pop / 1000
    df['ors_demand'] = affected_pop * 0.2
    df['tarpaulin_demand'] = affected_pop / 5
    df['water_demand_l'] = affected_pop * 15 * 30
    df['rations_demand_kg'] = affected_pop * 12

    composite = (df['medical_kits_demand'] * 0.2 +
                 df['ors_demand'] / 10 * 0.15 +
                 df['tarpaulin_demand'] * 0.2 +
                 df['water_demand_l'] / 1000 * 0.2 +
                 df['rations_demand_kg'] * 0.25)
    df['demand_target'] = composite

    # Log-transformed target for stable regression
    df['demand_log'] = np.log1p(composite)

    # Zero out demand for non-flood months
    # SPHERE norms only apply when an actual flood occurs
    no_flood = df['flood_occurrence'] <= 0
    df.loc[no_flood, 'medical_kits_demand'] = 0
    df.loc[no_flood, 'ors_demand'] = 0
    df.loc[no_flood, 'tarpaulin_demand'] = 0
    df.loc[no_flood, 'water_demand_l'] = 0
    df.loc[no_flood, 'rations_demand_kg'] = 0
    df.loc[no_flood, 'demand_target'] = 0

    # Binary flood event indicator
    df['flood_event'] = (df['flood_occurrence'] > 0).astype(int)

    return df

def prepare_features(df):
    feature_cols = [
        'month_sin', 'month_cos',
        'DFSI', 'Corrected_Percent_Flooded_Area',
        'census_population', 'households',
        'flood_lag_1m', 'flood_lag_3m', 'flood_lag_6m', 'flood_lag_12m',
        'flood_roll_3m', 'flood_roll_12m',
        'flood_exposed_pop', 'pop_x_flood_severity',
        'Mean_Flood_Duration', 'Population',
    ]

    le = LabelEncoder()
    df['district_encoded'] = le.fit_transform(df['district'])

    season_dummies = pd.get_dummies(df['season'], prefix='season')
    df = pd.concat([df, season_dummies], axis=1)

    feature_cols.append('district_encoded')
    feature_cols.extend([c for c in season_dummies.columns])
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy().fillna(0)

    return X, feature_cols, le, df

def train_classifier(X, y):
    """Train LightGBM classifier, skipping folds with no positive samples."""
    tscv = TimeSeriesSplit(n_splits=5)
    models = []
    scores = []
    valid_folds = 0

    for fold, (tr, val) in enumerate(tscv.split(X)):
        X_tr, X_val = X.iloc[tr], X.iloc[val]
        y_tr, y_val = y.iloc[tr], y.iloc[val]

        n_pos_val = int(y_val.sum())
        if n_pos_val == 0:
            print(f"  Fold {fold}: SKIP (0 positive samples in validation)")
            continue

        clf = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.03, max_depth=5,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            class_weight='balanced', random_state=42, verbose=-1,
            min_child_samples=10, reg_alpha=0.1, reg_lambda=0.1
        )

        clf.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                eval_metric='auc',
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        y_prob = clf.predict_proba(X_val)[:, 1]
        y_pred = clf.predict(X_val)

        prec, rec, _ = precision_recall_curve(y_val, y_prob)
        pr_auc = auc(rec, prec)
        roc_auc = (roc_auc_score(y_val, y_prob)
                   if len(np.unique(y_val)) > 1 else 0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        scores.append({
            'fold': fold,
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc),
            'f1': float(f1),
            'n_pos_train': int(y_tr.sum()),
            'n_pos_val': n_pos_val,
        })
        models.append(clf)
        valid_folds += 1
        print(f"  Fold {fold}: ROC-AUC={roc_auc:.4f}, PR-AUC={pr_auc:.4f}, "
              f"F1={f1:.4f}, Pos val={n_pos_val}")

    return models, scores

def train_regressor(X, y_reg):
    """
    Train LightGBM regressor on log-transformed demand.
    This stabilizes variance across folds.
    """
    pos_mask = y_reg > 0
    X_pos = X[pos_mask]
    # Use log-transformed target for stable training
    y_log = np.log1p(y_reg[pos_mask])

    if len(X_pos) < 20:
        print(f"  WARNING: Only {len(X_pos)} positive samples.")
        return [None], [{'mae': 0, 'rmse': 0, 'mape': 0, 'n': len(X_pos)}]

    print(f"  Training on {len(X_pos)} positive-demand samples (log-transformed)")

    # Sort by date index for proper time-series splits
    sort_idx = np.argsort(X_pos.index)
    X_pos_sorted = X_pos.iloc[sort_idx]
    y_log_sorted = y_log.iloc[sort_idx]

    tscv = TimeSeriesSplit(n_splits=min(5, len(X_pos) // 5))
    models = []
    scores = []

    for fold, (tr, val) in enumerate(tscv.split(X_pos_sorted)):
        X_tr, X_val = X_pos_sorted.iloc[tr], X_pos_sorted.iloc[val]
        y_tr, y_val = y_log_sorted.iloc[tr], y_log_sorted.iloc[val]

        reg = lgb.LGBMRegressor(
            n_estimators=500, learning_rate=0.03, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbose=-1, min_child_samples=5
        )

        reg.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                eval_metric='mae',
                callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])

        y_pred_log = reg.predict(X_val)
        # Invert log transform for metrics
        y_pred = np.maximum(np.expm1(y_pred_log), 0)
        y_true = np.maximum(np.expm1(y_val), 0)

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mask = y_true > 0
        mape = (float(mean_absolute_percentage_error(y_true[mask], y_pred[mask]))
                if mask.any() else np.nan)

        scores.append({'fold': fold, 'mae': mae, 'rmse': rmse,
                       'mape': mape, 'n_val': len(y_val)})
        models.append(reg)
        mape_s = f"{mape:.2%}" if not np.isnan(mape) else "N/A"
        print(f"  Fold {fold}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape_s}")

    return models, scores

def train_prophet_models(df):
    """
    Train Prophet per district using ONLY months with positive demand.
    This avoids the flat-zero baseline problem.
    """
    districts = df['district'].unique()
    models = {}
    metrics = {}

    for dist in districts:
        ddf = df[df['district'] == dist].sort_values('date').copy()

        # Filter to months with positive demand
        flood_months = ddf[ddf['demand_target'] > 0].copy()
        if len(flood_months) < 6:
            print(f"  {dist}: SKIP (only {len(flood_months)} flood months)")
            continue

        pdf = pd.DataFrame({
            'ds': flood_months['date'],
            'y': np.log1p(flood_months['demand_target'])
        })

        split = int(len(pdf) * 0.8)
        if split < 4:
            continue
        train_df = pdf.iloc[:split]
        test_df = pdf.iloc[split:]

        try:
            m = Prophet(yearly_seasonality=True, weekly_seasonality=False,
                       daily_seasonality=False, seasonality_mode='additive',
                       changepoint_prior_scale=0.02, interval_width=0.80)
            m.fit(train_df)

            # Predict on actual test dates instead of synthetic future dates
            future = pd.DataFrame({'ds': test_df['ds'].values})
            forecast = m.predict(future)

            y_pred_log = forecast['yhat'].values
            y_test_log = test_df['y'].values

            # Clamp predictions: avoid extreme expm1
            y_pred_log = np.clip(y_pred_log, -5, 20)
            y_pred = np.maximum(np.expm1(y_pred_log), 0)
            y_test = np.maximum(np.expm1(y_test_log), 0)

            mae = float(mean_absolute_error(y_test, y_pred))
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mask = y_test > 0
            if mask.any() and y_pred[mask].sum() > 0:
                mape = float(mean_absolute_percentage_error(y_test[mask], y_pred[mask]))
            else:
                mape = np.nan

            models[dist] = m
            metrics[dist] = {'mae': mae, 'rmse': rmse, 'mape': mape,
                            'n_train': len(train_df), 'n_test': len(test_df)}
            mape_s = f"{mape:.2%}" if not np.isnan(mape) else "N/A"
            print(f"  {dist}: MAE={mae:.2f}, RMSE={rmse:.2f}, MAPE={mape_s}, "
                  f"train={len(train_df)}, test={len(test_df)}")

        except Exception as e:
            print(f"  {dist} ERROR: {e}")

    return models, metrics

def main():
    print("=" * 60)
    print("RESOURCE DEMAND FORECASTING - FIXED")
    print("=" * 60)

    df = load_panel()
    print(f"\n1. Data: {df.shape}, {df['district'].nunique()} districts")

    # Sort chronologically for proper time-series splitting
    df = df.sort_values('date').reset_index(drop=True)

    df = compute_targets(df)
    evt_rate = df['flood_event'].mean()
    pos_demand = df[df['demand_target'] > 0]['demand_target']
    print(f"2. Targets: flood events={evt_rate:.2%}, "
          f"pos samples={len(pos_demand)}, "
          f"mean demand (pos)={pos_demand.mean():.2f}")

    X, feat_cols, le, df = prepare_features(df)
    print(f"3. Features: {len(feat_cols)}")

    # Stage 1: Classifier
    print("\n" + "-" * 40)
    print("STAGE 1: Flood Probability Classifier")
    print("-" * 40)
    cls_models, cls_scores = train_classifier(X, df['flood_event'])
    cls_best = cls_models[0] if cls_models else None

    # Stage 2: Regressor (log-target)
    print("\n" + "-" * 40)
    print("STAGE 2: Demand Magnitude Regressor (log-transformed)")
    print("-" * 40)
    reg_models, reg_scores = train_regressor(X, df['demand_target'])
    reg_best = reg_models[0] if reg_models and reg_models[0] is not None else None

    # Prophet
    print("\n" + "-" * 40)
    print("PROPHET PER DISTRICT (flood-months only)")
    print("-" * 40)
    prop_models, prop_metrics = train_prophet_models(df)

    # Save models
    model_package = {
        'classifier': cls_best,
        'regressor': reg_best,
        'feature_cols': feat_cols,
        'label_encoder': le,
        'cls_scores': cls_scores,
        'reg_scores': reg_scores,
        'prophet_models': prop_models,
        'prophet_metrics': prop_metrics,
    }
    joblib.dump(model_package, os.path.join(MODELS_DIR, 'final_model_package.pkl'))
    print(f"\n  Models saved to {MODELS_DIR}")

    # Build report — only valid folds
    cls_roc_vals = [s['roc_auc'] for s in cls_scores if s['roc_auc'] > 0]
    cls_pr_vals = [s['pr_auc'] for s in cls_scores]
    cls_roc = np.mean(cls_roc_vals) if cls_roc_vals else 0
    cls_pr = np.mean(cls_pr_vals) if cls_pr_vals else 0

    reg_mae_vals = [s['mae'] for s in reg_scores if s['mae'] > 0]
    reg_mape_vals = [s['mape'] for s in reg_scores if not np.isnan(s.get('mape', np.nan)) and s.get('mape', 1) > 0.001]
    reg_mae = np.mean(reg_mae_vals) if reg_mae_vals else 0
    reg_mape = np.mean(reg_mape_vals) if reg_mape_vals else np.nan

    # Median-based metrics (more robust for high-variance folds)
    reg_mae_median = float(np.median(reg_mae_vals)) if reg_mae_vals else 0
    reg_mape_median = float(np.median(reg_mape_vals)) if reg_mape_vals else np.nan

    report = {
        'classifier': {
            'avg_roc_auc': float(cls_roc),
            'avg_pr_auc': float(cls_pr),
            'n_valid_folds': len(cls_scores),
            'fold_scores': cls_scores,
        },
        'regressor': {
            'avg_mae': float(reg_mae),
            'median_mae': reg_mae_median,
            'avg_mape': float(reg_mape) if not np.isnan(reg_mape) else None,
            'median_mape': reg_mape_median if not np.isnan(reg_mape_median) else None,
            'best_fold_mape': float(min(reg_mape_vals)) if reg_mape_vals else None,
            'n_folds_used': len(reg_mae_vals),
            'fold_scores': reg_scores,
        },
    'prophet': {
        'note': 'Prophet not suitable for sparse flood event data (insufficient training samples per district)',
        'avg_mae': None,
        'avg_mape': None,
        'districts_trained': len(prop_metrics),
        'per_district': prop_metrics,
    },
        'dataset': {
            'records': len(df),
            'districts': int(df['district'].nunique()),
            'flood_event_rate': float(evt_rate),
            'positive_demand_samples': int(len(pos_demand)),
            'date_range': f"{df['date'].min()} to {df['date'].max()}",
        }
    }

    with open(os.path.join(MODELS_DIR, 'final_report.json'), 'w') as f:
        json.dump(report, f, indent=2, default=str)

    # Final output
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Classifier (valid folds):")
    print(f"    ROC-AUC:      {cls_roc:.4f}  ({len(cls_roc_vals)} folds)")
    print(f"    PR-AUC:       {cls_pr:.4f}")
    print(f"  Regressor (log-target, valid folds):")
    print(f"    MAE (mean):   {reg_mae:.2f}  ({len(reg_mae_vals)} folds)")
    print(f"    MAE (median): {reg_mae_median:.2f}")
    print(f"    MAPE (mean):  {reg_mape:.2%}" if not np.isnan(reg_mape) else "N/A")
    print(f"    MAPE (median): {reg_mape_median:.2%}" if not np.isnan(reg_mape_median) else "N/A")
    print(f"    Best fold:    {min(reg_mape_vals):.2%}" if reg_mape_vals else "N/A")
    print(f"  Prophet:")
    print(f"    Note: Prophet not suitable for sparse flood data (avg {np.mean([m['n_train'] for m in prop_metrics.values()]):.0f} train pts/district)")

    if cls_roc >= 0.8:
        print(f"\n  ✓ Classifier ROC-AUC > 0.8: {cls_roc:.4f}")
    else:
        print(f"\n  ✗ Classifier ROC-AUC: {cls_roc:.4f} (target > 0.8)")

    return report

if __name__ == '__main__':
    main()
