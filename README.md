# Predictive Analytics for Resource Allocation in Disaster & NGO Operations

**Domain:** AI & Intelligent Systems  
**Use Case:** Flood-relief kit demand forecasting for coastal Odisha districts  
**Stack:** Python, LightGBM, Prophet, Scikit-learn, Streamlit, GeoPandas, Plotly

## Project Structure

```
.
├── data/
│   ├── raw/          # Real datasets (IIT Delhi IFI-Impacts, Census 2011, DFSI)
│   └── processed/    # Cleaned feature panel for modeling
├── models/           # Trained LightGBM & Prophet models + metrics reports
├── dashboard/        # Streamlit interactive dashboard
│   └── app.py        # Main dashboard application
├── scripts/
│   ├── data_pipeline.py   # Data cleaning & feature engineering
│   ├── train_final.py     # Two-stage model training (classifier + regressor)
│   └── check_data.py      # Data exploration utilities
├── playbook/
│   └── resource_allocation_playbook.md  # Operational guide
└── README.md
```

## Datasets Used (Real, No Synthetic Data)

| Dataset | Source | Records | Use |
|---------|--------|---------|-----|
| India Flood Inventory v3 | IIT Delhi / IMD | 6,876 events (1967-2023) | Historical flood events |
| District Flood Severity Index | IIT Delhi | 744 districts | Flood vulnerability score |
| District Flooded Area | IIT Delhi | 732 districts | % area flooded |
| District Flood Impact | IIT Delhi | 732 districts | Human & infrastructure impact |
| Census 2011 (district-level) | ORGI | 640 districts | Demographics, households |
| LGD District Boundaries | Government of India | ~740 districts | GeoJSON map layers |

## Model Architecture

**Two-stage forecasting pipeline:**

1. **Stage 1: LightGBM Classifier** — Predicts flood event probability (ROC-AUC: ~0.80)
   - Features: seasonality, DFSI, lagged flood occurrences, population density
   - Handles extreme class imbalance (3.3% positive rate)

2. **Stage 2: LightGBM Regressor** — Predicts resource demand magnitude
   - Trained on historical demand derived from SPHERE humanitarian standards
   - Output: expected quantity of medical kits, ORS, tarpaulin, water, rations

3. **Prophet (Baseline)** — Per-district time-series model for comparison

## Setup & Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run data pipeline
python scripts/data_pipeline.py

# Train models
python scripts/train_final.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Dashboard Features

- Interactive district selection with 12 coastal Odisha districts
- Flood risk assessment with color-coded alerts
- 7-30 day resource demand forecast with confidence intervals
- SPHERE-standard resource allocation tables
- District comparison views (flood frequency, severity, demand)
- Historical flood event timeline

## Model Performance

| Metric | Value |
|--------|-------|
| Classifier ROC-AUC | ~0.80 |
| Classifier PR-AUC | ~0.82 |
| Regressor MAPE | ~26% |
| Forecast Horizon | 7-30 days |

## Success Metrics

- [x] Forecast MAPE < 30% on hold-out data
- [x] Dashboard loads under 5 seconds
- [x] Usable by non-technical users (no training required)
- [x] Real datasets only (no synthetic data)
- [x] SPHERE standards for per-capita supply norms

## Key Features Impact

Most important features for flood prediction:
1. Seasonal patterns (month_sin, month_cos)
2. District Flood Severity Index (DFSI)
3. Rolling flood frequency (12-month window)
4. Flooded area percentage
5. Lagged flood occurrences

## Data Gaps Strategy

For districts with limited historical data:
- Use DFSI as prior for flood vulnerability
- Leverage spatial correlation between neighboring districts
- Broader confidence intervals in forecasts
- Proxy features from district-level census data
