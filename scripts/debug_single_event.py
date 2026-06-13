import pandas as pd
import numpy as np

RAW = 'B:\\Predictive Analytics for Resource Allocation\\data\\raw'

ifi = pd.read_csv(f'{RAW}/India_Flood_Inventory_v3.csv', low_memory=False)
odisha_mask = ifi['State'].str.contains('Odisha', na=False)
odisha_events = ifi[odisha_mask].copy()
odisha_events['Start Date'] = pd.to_datetime(odisha_events['Start Date'], dayfirst=True, errors='coerce')
odisha_events['year'] = odisha_events['Start Date'].dt.year
odisha_events['month'] = odisha_events['Start Date'].dt.month

# Find a 2001 event with Baleshwar
for idx, event in odisha_events.iterrows():
    y = event['year']
    ds = event.get('Districts')
    if pd.isna(ds):
        continue
    if y == 2001 and 'baleshwar' in str(ds).lower():
        print(f"Found 2001 Baleshwar event:")
        print(f"  UEI: {event.get('UEI')}")
        print(f"  Start Date: {event['Start Date']}")
        print(f"  Districts (raw): [{ds}]")
        parts = [d.strip() for d in str(ds).split(',')]
        print(f"  Parts: {parts[:10]}...")
        for p in parts:
            normalized = p.strip().title()
            in_list = normalized in [
                'Puri', 'Jagatsinghapur', 'Kendrapara', 'Bhadrak', 'Baleshwar',
                'Ganjam', 'Khordha', 'Cuttack', 'Jajapur', 'Mayurbhanj',
                'Nayagarh', 'Kendujhar'
            ]
            print(f"    '{p}' -> normalized='{normalized}', in_list={in_list}")
        break

# Check all 2001 events
print(f"\n=== All 2001 events ===")
for idx, event in odisha_events[odisha_events['year'] == 2001].iterrows():
    ds = event.get('Districts')
    if pd.isna(ds):
        print(f"  Event {idx}: No Districts column")
        continue
    print(f"  Event {idx}: Districts prefix: {str(ds)[:200]}...")
    parts = [d.strip() for d in str(ds).split(',')]
    coastal_found = []
    for p in parts:
        normalized = p.strip().title()
        coastal_list = [
            'Puri', 'Jagatsinghapur', 'Kendrapara', 'Bhadrak', 'Baleshwar',
            'Ganjam', 'Khordha', 'Cuttack', 'Jajapur', 'Mayurbhanj',
            'Nayagarh', 'Kendujhar'
        ]
        if normalized in coastal_list:
            coastal_found.append(normalized)
    print(f"    Coastal districts found: {coastal_found}")
