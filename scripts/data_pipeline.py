"""
Data Pipeline: Clean, merge, and engineer features from real datasets.
Use case: Flood-relief kit demand in coastal Odisha districts.
Datasets: IFI-Impacts (IIT Delhi), DFSI, Census 2011, District FloodedArea
"""
import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(BASE, 'data', 'raw')
PROC = os.path.join(BASE, 'data', 'processed')

os.makedirs(PROC, exist_ok=True)

# Coastal Odisha districts (most flood-prone)
COASTAL_ODISHA = [
    'Puri', 'Jagatsinghapur', 'Kendrapara', 'Bhadrak', 'Baleshwar',
    'Ganjam', 'Khordha', 'Cuttack', 'Jajapur', 'Mayurbhanj',
    'Nayagarh', 'Kendujhar'
]

# Alias mapping: alternate spellings -> COASTAL_ODISHA standard
DISTRICT_ALIAS = {
    'Balasore': 'Baleshwar',
    'Jagatsinghpur': 'Jagatsinghapur',
    'Keonjhar': 'Kendujhar',
    'Jajpur': 'Jajapur',
}

def normalize_district(name):
    """Normalize district names across datasets using alias mapping."""
    if pd.isna(name):
        return None
    name = name.strip().title()
    return DISTRICT_ALIAS.get(name, name)

def load_ifi_data():
    """Load India Flood Inventory dataset."""
    path = os.path.join(RAW, 'India_Flood_Inventory_v3.csv')
    df = pd.read_csv(path, low_memory=False)

    # Parse dates
    df['Start Date'] = pd.to_datetime(df['Start Date'], dayfirst=True, errors='coerce')
    df['End Date'] = pd.to_datetime(df['End Date'], dayfirst=True, errors='coerce')

    # Extract year and month
    df['year'] = df['Start Date'].dt.year
    df['month'] = df['Start Date'].dt.month
    df['season'] = df['month'].apply(lambda m: 'Monsoon' if m in [6,7,8,9,10] else 'Non-Monsoon')

    return df

def filter_odisha_events(ifi_df):
    """Filter flood events affecting Odisha districts."""
    odisha_mask = ifi_df['State'].str.contains('Odisha', na=False)
    odisha_events = ifi_df[odisha_mask].copy()
    return odisha_events

def parse_districts(district_str):
    """Parse comma-separated district string into list."""
    if pd.isna(district_str):
        return []
    return [d.strip() for d in str(district_str).split(',')]

def create_district_monthly_panel(odisha_events):
    """
    Create district-month panel: for each district and month, compute
    flood occurrence, severity, and impact metrics.
    """
    records = []

    for _, event in odisha_events.iterrows():
        districts = parse_districts(event.get('Districts'))
        year = event['year']
        month = event['month']
        start_date = event['Start Date']
        duration = event.get('Duration(Days)', np.nan)
        severity = event.get('Severity', np.nan)
        fatalities = event.get('Human fatality', 0)
        displaced = event.get('Human Displaced', 0)
        area_affected = event.get('Area Affected', np.nan)

        for dist in districts:
            dnorm = normalize_district(dist)
            if dnorm in COASTAL_ODISHA:
                records.append({
                    'district': dnorm,
                    'year': year,
                    'month': month,
                    'start_date': start_date,
                    'duration': duration,
                    'severity': severity,
                    'fatalities': fatalities,
                    'displaced': displaced,
                    'area_affected': area_affected,
                    'flood_occurrence': 1,
                })

    panel = pd.DataFrame(records)
    return panel

def load_census_data():
    """Load and process Census 2011 data."""
    path = os.path.join(RAW, 'india-districts-census-2011.csv')
    df = pd.read_csv(path)

    # Map census district names to our standard names
    # Census 2011 uses the same spellings as COASTAL_ODISHA for all 12 districts
    district_map = {d: d for d in COASTAL_ODISHA}

    # Build lookup
    census = {}
    for _, row in df.iterrows():
        name = row['District name'].strip()
        for our_name, census_name in district_map.items():
            if name.lower() == census_name.lower():
                census[our_name] = {
                    'population': row['Population'],
                    'households': row['Households'],
                    'rural_hh': row['Rural_Households'],
                    'urban_hh': row['Urban_Households'],
                    'area_sqkm': None,  # Not in census
                }
                break

    return census, df

def load_dfsi_data():
    """Load DFSI (District Flood Severity Index)."""
    path = os.path.join(RAW, 'DFSI.csv')
    df = pd.read_csv(path)
    df['district'] = df['Unnamed: 0'].apply(normalize_district)
    return df[['district', 'DFSI']]

def load_flood_impact():
    """Load District Flood Impact data."""
    path = os.path.join(RAW, 'District_FloodImpact.csv')
    df = pd.read_csv(path)
    df['district'] = df['Dist_Name'].apply(normalize_district)
    return df[['district', 'Human_fatality', 'Human_injured', 'Population', 'Mean_Flood_Duration']]

def load_flooded_area():
    """Load District Flooded Area data."""
    path = os.path.join(RAW, 'District_FloodedArea.csv')
    df = pd.read_csv(path)
    df['district'] = df['Dist_Name'].apply(normalize_district)
    return df[['district', 'Percent_Flooded_Area', 'Corrected_Percent_Flooded_Area']]

def build_feature_panel():
    """
    Main function: Build the complete feature panel for modeling.
    Returns monthly district-level dataset with features and target.
    """
    print("Loading IFI flood data...")
    ifi = load_ifi_data()
    print(f"  Total flood events: {len(ifi)}")

    print("Filtering Odisha events...")
    odisha_events = filter_odisha_events(ifi)
    print(f"  Odisha events: {len(odisha_events)}")

    print("Loading DFSI...")
    dfsi = load_dfsi_data()

    print("Loading flood impact...")
    impact = load_flood_impact()

    print("Loading flooded area...")
    flooded = load_flooded_area()

    print("Loading census data...")
    census, census_full = load_census_data()

    print("Creating district-month panel...")
    panel = create_district_monthly_panel(odisha_events)
    print(f"  Panel records: {len(panel)}")

    # Create complete monthly panel for all coastal districts
    print("Creating complete monthly panel (all months 2000-2024)...")
    all_months = []
    for dist in COASTAL_ODISHA:
        for y in range(2000, 2024):
            for m in range(1, 13):
                all_months.append({'district': dist, 'year': y, 'month': m,
                                   'flood_occurrence': 0, 'severity': 0,
                                   'fatalities': 0, 'displaced': 0, 'duration': 0})
    panel_full = pd.DataFrame(all_months)

    # Merge in actual events from odisha_events
    print("Merging actual flood events into panel...")
    odisha_panel = create_district_monthly_panel(odisha_events)
    for idx, event in odisha_panel.iterrows():
        mask = ((panel_full['district'] == event['district']) &
                (panel_full['year'] == event['year']) &
                (panel_full['month'] == event['month']))
        if mask.any():
            panel_full.loc[mask, 'flood_occurrence'] = 1
            if not pd.isna(event.get('severity')):
                panel_full.loc[mask, 'severity'] = event['severity']
            if not pd.isna(event.get('fatalities')):
                panel_full.loc[mask, 'fatalities'] = event['fatalities']
            if not pd.isna(event.get('displaced')):
                panel_full.loc[mask, 'displaced'] = event['displaced']
            if not pd.isna(event.get('duration')):
                panel_full.loc[mask, 'duration'] = event['duration']

    panel = panel_full

    # Ensure type columns exist
    for col in ['severity', 'fatalities', 'displaced', 'duration']:
        if col not in panel.columns:
            panel[col] = 0

    # Ensure correct types
    panel['flood_occurrence'] = panel['flood_occurrence'].fillna(0).astype(int)
    panel['severity'] = pd.to_numeric(panel['severity'], errors='coerce').fillna(0)
    panel['fatalities'] = pd.to_numeric(panel['fatalities'], errors='coerce').fillna(0)
    panel['displaced'] = pd.to_numeric(panel['displaced'], errors='coerce').fillna(0)
    panel['duration'] = pd.to_numeric(panel['duration'], errors='coerce').fillna(0)

    monthly = panel

    print(f"  Monthly panel: {monthly.shape}")

    # Merge static features
    print("Merging static features...")
    monthly['district_norm'] = monthly['district']

    # DFSI
    monthly = monthly.merge(dfsi, on='district', how='left')

    # Flooded area
    monthly = monthly.merge(flooded, on='district', how='left')

    # Flood impact (population at risk)
    monthly = monthly.merge(impact[['district', 'Population', 'Mean_Flood_Duration']],
                           on='district', how='left')

    # Census data
    census_rows = []
    for d in monthly['district'].unique():
        pop = census.get(d, {}).get('population', np.nan)
        hh = census.get(d, {}).get('households', np.nan)
        rural = census.get(d, {}).get('rural_hh', np.nan)
        urban = census.get(d, {}).get('urban_hh', np.nan)
        census_rows.append({'district': d, 'census_population': pop,
                           'households': hh, 'rural_hh': rural, 'urban_hh': urban})

    census_df = pd.DataFrame(census_rows)
    monthly = monthly.merge(census_df, on='district', how='left')

    # Feature engineering
    print("Engineering features...")

    # Temporal features
    monthly['season'] = monthly['month'].apply(
        lambda m: 'Pre-Monsoon' if m in [3,4,5]
        else 'Monsoon' if m in [6,7,8,9]
        else 'Post-Monsoon' if m in [10,11]
        else 'Winter')
    monthly['month_sin'] = np.sin(2 * np.pi * monthly['month'] / 12)
    monthly['month_cos'] = np.cos(2 * np.pi * monthly['month'] / 12)

    # Lag features: flood occurrence in previous months
    monthly = monthly.sort_values(['district', 'year', 'month'])
    for lag in [1, 3, 6, 12]:
        monthly[f'flood_lag_{lag}m'] = monthly.groupby('district')['flood_occurrence'].shift(lag)

    # Rolling statistics
    monthly['flood_roll_3m'] = (monthly.groupby('district')['flood_occurrence']
                                .transform(lambda x: x.rolling(3, min_periods=1).mean()))
    monthly['flood_roll_12m'] = (monthly.groupby('district')['flood_occurrence']
                                 .transform(lambda x: x.rolling(12, min_periods=1).mean()))

    # Flood-exposed population (population * flooded fraction)
    monthly['flood_exposed_pop'] = monthly['census_population'] * (monthly['Corrected_Percent_Flooded_Area'].fillna(0) / 100)

    # Interaction features
    monthly['pop_x_flood_severity'] = (monthly['census_population'] * monthly['severity'] / 100000)

    # Target: Resource demand score
    # Derived from: flood intensity * population vulnerability
    monthly['resource_demand_score'] = (
        monthly['flood_occurrence'] * 0.3 +
        (monthly['severity'].fillna(0) / 10) * 0.3 +
        (monthly['displaced'].fillna(0) / 1000) * 0.2 +
        (monthly['fatalities'].fillna(0) / 10) * 0.2
    )

    # Clip negative values
    monthly['resource_demand_score'] = monthly['resource_demand_score'].clip(0)

    # Fill NaN values
    fill_cols = ['DFSI', 'Percent_Flooded_Area', 'Corrected_Percent_Flooded_Area',
                  'Population', 'Mean_Flood_Duration', 'census_population', 'households',
                  'flood_exposed_pop', 'pop_x_flood_severity']
    for col in fill_cols:
        if col in monthly.columns:
            monthly[col] = monthly[col].fillna(monthly[col].median())

    # Fill lag/roll NaNs
    monthly = monthly.fillna(0)

    # Time index (zero-pad month for correct parsing)
    monthly['date'] = pd.to_datetime(
        monthly['year'].astype(int).astype(str) + '-' +
        monthly['month'].astype(int).astype(str).str.zfill(2) + '-01')

    print(f"Final panel shape: {monthly.shape}")
    print(f"Columns: {monthly.columns.tolist()}")

    # Save processed data
    output_path = os.path.join(PROC, 'odisha_flood_panel.csv')
    monthly.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")

    return monthly

if __name__ == '__main__':
    df = build_feature_panel()
    print("\nSample data:")
    print(df.head(10))
    print("\nSummary stats:")
    print(df[['flood_occurrence', 'resource_demand_score', 'severity',
              'DFSI', 'census_population']].describe())
