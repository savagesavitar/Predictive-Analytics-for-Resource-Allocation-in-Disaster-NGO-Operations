import pandas as pd
import numpy as np

RAW = 'B:\\Predictive Analytics for Resource Allocation\\data\\raw'
PROC = 'B:\\Predictive Analytics for Resource Allocation\\data\\processed'

# 1. Check IFI for these districts
ifi = pd.read_csv(f'{RAW}/India_Flood_Inventory_v3.csv', low_memory=False)
print("=== Checking IFI for Balasore, Jagatsinghpur, Jajpur, Keonjhar ===")
districts_to_check = ['Baleshwar', 'Jagatsinghapur', 'Jajapur', 'Kendujhar',
                       'Balasore', 'Jagatsinghpur', 'Jajpur', 'Keonjhar',
                       'Bhadrak', 'Cuttack', 'Puri', 'Ganjam']

# Check in State column
state_mask = ifi['State'].str.contains('Odisha', na=False)
odisha = ifi[state_mask]
print(f"Total Odisha events in IFI: {len(odisha)}")

# Check district column for these names
for d in districts_to_check:
    count = odisha['Districts'].dropna().apply(lambda x: d.lower() in str(x).lower()).sum()
    print(f"  '{d}' appears in: {count} events")

# 2. Check panel data
panel = pd.read_csv(f'{PROC}/odisha_flood_panel.csv')
print(f"\n=== Panel data ===")
print(f"Total records: {len(panel)}")
print(f"Districts: {panel['district'].unique()}")
for d in panel['district'].unique():
    n_events = panel[panel['flood_occurrence'] > 0].shape[0]
    d_events = panel[(panel['district'] == d) & (panel['flood_occurrence'] > 0)]
    print(f"  {d}: {len(d_events)} flood months out of {len(panel[panel['district']==d])} total")

# 3. Check DFSI for these districts
dfsi = pd.read_csv(f'{RAW}/DFSI.csv')
print(f"\n=== DFSI for the 4 districts ===")
for d in ['Baleshwar', 'Balasore', 'Jagatsinghapur', 'Jagatsinghpur', 'Jajapur', 'Jajpur', 'Kendujhar', 'Keonjhar']:
    match = dfsi[dfsi['Unnamed: 0'].str.contains(d, case=False, na=False)]
    if len(match) > 0:
        print(f"  DFSI '{d}': {match.iloc[0]['DFSI']:.2f}")

# 4. Check how data_pipeline parses district names
print(f"\n=== Checking district name parsing ===")
sample_dists = odisha['Districts'].dropna().iloc[:5]
for s in sample_dists:
    print(f"  Raw: {s[:200]}...")

# 5. Check the "Districts_FloodImpact.csv" for these 4
impact = pd.read_csv(f'{RAW}/District_FloodImpact.csv')
print(f"\n=== Flood Impact for missing districts ===")
for d in ['Baleshwar', 'Jagatsinghapur', 'Jajapur', 'Kendujhar']:
    match = impact[impact['Dist_Name'].str.contains(d, case=False, na=False)]
    if len(match) > 0:
        print(f"  {d}: found")
    else:
        # Try alternate names
        for alt in ['Balasore', 'Jagatsinghpur', 'Jajpur', 'Keonjhar', 'Jajpur']:
            m2 = impact[impact['Dist_Name'].str.contains(alt, case=False, na=False)]
            if len(m2) > 0:
                print(f"  {d} -> {alt}: found")
                break
        else:
            print(f"  {d}: NOT FOUND")
