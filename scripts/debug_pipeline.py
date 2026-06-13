import pandas as pd
import numpy as np
import os

BASE = 'B:\\Predictive Analytics for Resource Allocation'
RAW = os.path.join(BASE, 'data', 'raw')

# Replicate just the key section of the pipeline
COASTAL_ODISHA = [
    'Puri', 'Jagatsinghapur', 'Kendrapara', 'Bhadrak', 'Baleshwar',
    'Ganjam', 'Khordha', 'Cuttack', 'Jajapur', 'Mayurbhanj',
    'Nayagarh', 'Kendujhar'
]
DISTRICT_ALIAS = {
    'Jagatsinghapur': 'Jagatsinghpur',
    'Baleshwar': 'Balasore',
    'Kendujhar': 'Keonjhar',
    'Jajapur': 'Jajpur',
}

def normalize_district(name):
    if pd.isna(name):
        return None
    name = name.strip().title()
    return DISTRICT_ALIAS.get(name, name)

def parse_districts(district_str):
    if pd.isna(district_str):
        return []
    return [d.strip() for d in str(district_str).split(',')]

# Load and parse
ifi = pd.read_csv(f'{RAW}/India_Flood_Inventory_v3.csv', low_memory=False)
ifi['Start Date'] = pd.to_datetime(ifi['Start Date'], dayfirst=True, errors='coerce')
ifi['year'] = ifi['Start Date'].dt.year
ifi['month'] = ifi['Start Date'].dt.month

# Filter Odisha
odisha_mask = ifi['State'].str.contains('Odisha', na=False)
odisha_events = ifi[odisha_mask].copy()
print(f"Total Odisha events: {len(odisha_events)}")
print(f"Events with valid year: {odisha_events['year'].notna().sum()}")
print(f"Events with year >= 2000: {(odisha_events['year'] >= 2000).sum()}")
print(f"Events with Districts: {odisha_events['Districts'].notna().sum()}")

# Create panel  
records = []
for _, event in odisha_events.iterrows():
    districts = parse_districts(event.get('Districts'))
    year = event['year']
    month = event['month']
    severity = event.get('severity', np.nan)
    fatalities = event.get('Human fatality', 0)
    displaced = event.get('Human Displaced', 0)
    duration = event.get('Duration(Days)', np.nan)

    if pd.isna(year) or pd.isna(month):
        continue
    year = int(year)
    month = int(month)

    for dist in districts:
        dnorm = normalize_district(dist)
        if dnorm in COASTAL_ODISHA:
            records.append({
                'district': dnorm,
                'year': year,
                'month': month,
                'flood_occurrence': 1,
                'severity': severity if not pd.isna(severity) else 0,
                'fatalities': fatalities if not pd.isna(fatalities) else 0,
                'displaced': displaced if not pd.isna(displaced) else 0,
                'duration': duration if not pd.isna(duration) else 0,
            })

odisha_panel = pd.DataFrame(records)
print(f"\nOdisha panel records: {len(odisha_panel)}")
for d in sorted(odisha_panel['district'].unique()):
    count = len(odisha_panel[odisha_panel['district'] == d])
    print(f"  {d}: {count}")

# Now create complete panel and merge
print(f"\n=== Creating complete panel ===")
all_months = []
for dist in COASTAL_ODISHA:
    for y in range(2000, 2024):
        for m in range(1, 13):
            all_months.append({'district': dist, 'year': y, 'month': m,
                               'flood_occurrence': 0, 'severity': 0,
                               'fatalities': 0, 'displaced': 0, 'duration': 0})
panel_full = pd.DataFrame(all_months)
print(f"Full panel before merge: {len(panel_full)}")

# Merge
merged_count = 0
for idx, event in odisha_panel.iterrows():
    mask = ((panel_full['district'] == event['district']) &
            (panel_full['year'] == event['year']) &
            (panel_full['month'] == event['month']))
    if mask.any():
        panel_full.loc[mask, 'flood_occurrence'] = 1
        panel_full.loc[mask, 'severity'] = event['severity']
        merged_count += 1
    else:
        print(f"  FAILED TO MERGE: {event['district']} {event['year']}-{event['month']}")

print(f"\nMerged successfully: {merged_count} / {len(odisha_panel)}")

# Check final counts
for d in sorted(panel_full[panel_full['flood_occurrence'] > 0]['district'].unique()):
    count = len(panel_full[(panel_full['district'] == d) & (panel_full['flood_occurrence'] > 0)])
    print(f"  {d}: {count} flood months")
