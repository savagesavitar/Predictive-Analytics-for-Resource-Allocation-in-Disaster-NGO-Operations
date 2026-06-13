import pandas as pd
import numpy as np
from datetime import datetime

RAW = 'B:\\Predictive Analytics for Resource Allocation\\data\\raw'

ifi = pd.read_csv(f'{RAW}/India_Flood_Inventory_v3.csv', low_memory=False)
odisha_mask = ifi['State'].str.contains('Odisha', na=False)
odisha_events = ifi[odisha_mask].copy()

# Parse dates
odisha_events['Start Date'] = pd.to_datetime(odisha_events['Start Date'], dayfirst=True, errors='coerce')
odisha_events['year'] = odisha_events['Start Date'].dt.year
odisha_events['month'] = odisha_events['Start Date'].dt.month

print("=== Year distribution of Odisha events ===")
year_counts = odisha_events['year'].value_counts().sort_index()
for y, c in year_counts.items():
    marker = " *** PRE-2000" if y < 2000 else ""
    print(f"  {int(y)}: {c} events{marker}")

pre_2000 = len(odisha_events[odisha_events['year'] < 2000])
post_2000 = len(odisha_events[odisha_events['year'] >= 2000])
print(f"\nPre-2000: {pre_2000} events")
print(f"Post-2000: {post_2000} events")

# Check which events for Baleshwar are pre/post 2000
print("\n=== Baleshwar events by decade ===")
for _, event in odisha_events.iterrows():
    dist_str = event.get('Districts')
    if pd.isna(dist_str):
        continue
    y = event['year']
    parts = [d.strip() for d in str(dist_str).split(',')]
    for p in parts:
        if p.strip().lower() == 'baleshwar':
            decade = (y // 10) * 10
            print(f"  Baleshwar event in {int(y)}")

# Check what create_district_monthly_panel actually produces
print("\n=== Testing create_district_monthly_panel logic ===")
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

records = []
for _, event in odisha_events.iterrows():
    districts = parse_districts(event.get('Districts'))
    y = event['year']
    m = event['month']
    if y < 2000:
        continue  # Panel only covers 2000+
    for dist in districts:
        dnorm = normalize_district(dist)
        if dnorm in COASTAL_ODISHA:
            records.append((dnorm, y, m))

records_df = pd.DataFrame(records, columns=['district', 'year', 'month'])
print(f"Total post-2000 event-district pairs: {len(records_df)}")
for d in sorted(records_df['district'].unique()):
    count = len(records_df[records_df['district'] == d])
    print(f"  {d}: {count} events")
