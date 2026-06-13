import json
import pandas as pd

# Check GeoJSONL
print("=== LGD_Districts.geojsonl ===")
with open('B:\\Predictive Analytics for Resource Allocation\\data\\raw\\LGD_Districts.geojsonl', 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i < 3:
            data = json.loads(line.strip())
            print(f'Record {i}: keys={list(data.keys())}')
            props = data.get('properties', {})
            print(f'  properties keys: {list(props.keys())}')
            print(f'  properties sample: {json.dumps(props, indent=2)[:500]}')

# Check Odisha-specific data
print("\n=== Odisha Flood Events from IFI ===")
ifi = pd.read_csv('B:\\Predictive Analytics for Resource Allocation\\data\\raw\\India_Flood_Inventory_v3.csv')
odisha_events = ifi[ifi['State'].str.contains('Odisha', na=False)]
print(f'Total Odisha events: {len(odisha_events)}')
print(odisha_events[['Start Date', 'End Date', 'Districts', 'Severity', 'Human fatality', 'Human Displaced']].head(10))

# Check unique states in IFI
print(f"\nUnique states: {ifi['State'].dropna().unique()[:20]}")

# Check DFSI for Odisha districts
print("\n=== DFSI for Odisha ===")
dfsi = pd.read_csv('B:\\Predictive Analytics for Resource Allocation\\data\\raw\\DFSI.csv')
print(f'DFSI columns: {dfsi.columns.tolist()}')
print(dfsi.head(10))
