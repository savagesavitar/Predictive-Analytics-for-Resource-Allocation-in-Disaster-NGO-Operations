import pandas as pd
import numpy as np

RAW = 'B:\\Predictive Analytics for Resource Allocation\\data\\raw'

# Load IFI
ifi = pd.read_csv(f'{RAW}/India_Flood_Inventory_v3.csv', low_memory=False)
odisha_mask = ifi['State'].str.contains('Odisha', na=False)
odisha_events = ifi[odisha_mask].copy()

# Parse all district names from Odisha events
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

all_parsed = set()
district_found = {d: 0 for d in COASTAL_ODISHA}
total_with_districts = 0

for _, event in odisha_events.iterrows():
    dist_str = event.get('Districts')
    if pd.isna(dist_str):
        continue
    total_with_districts += 1
    parts = [d.strip() for d in str(dist_str).split(',')]
    for p in parts:
        all_parsed.add(p)
        # Check exact match or partial match
        for cd in COASTAL_ODISHA:
            if cd.lower() in p.lower() or p.lower() in cd.lower():
                district_found[cd] += 1

print(f"Total Odisha events: {len(odisha_events)}")
print(f"Events with Districts column: {total_with_districts}")
print(f"\nUnique district names parsed (sample of 30):")
sorted_parsed = sorted(all_parsed)
for p in sorted_parsed[:30]:
    print(f"  '{p}'")

print(f"\nMatching counts for COASTAL_ODISHA:")
for d, count in sorted(district_found.items(), key=lambda x: -x[1]):
    print(f"  {d}: {count}")

print(f"\n--- Exact string matching debug ---")
# Check the 4 missing ones more carefully
missing = ['Baleshwar', 'Jagatsinghapur', 'Jajapur', 'Kendujhar']
for m in missing:
    for _, event in odisha_events.iterrows():
        dist_str = event.get('Districts')
        if pd.isna(dist_str):
            continue
        parts = [d.strip() for d in str(dist_str).split(',')]
        for p in parts:
            after_title = p.strip().title()
            if m.lower() in after_title.lower() or after_title.lower() in m.lower():
                print(f"  '{m}' matched by: '{p}' (after title: '{after_title}')")
                break

# Check normalization issues
print(f"\n--- Checking exact case/whitespace issues ---")
raw_examples = set()
for _, event in odisha_events.iterrows():
    dist_str = event.get('Districts')
    if pd.isna(dist_str):
        continue
    parts = [d.strip() for d in str(dist_str).split(',')]
    for p in parts:
        raw_examples.add(p)
    # Look for Baleshwar variants
    for p in parts:
        if 'baleshwar' in p.lower() or 'balasore' in p.lower():
            print(f"  Found Baleshwar/Balasore variant: '{p}'")
        if 'jajapur' in p.lower() or 'jajpur' in p.lower():
            print(f"  Found Jajapur/Jajpur variant: '{p}'")
        if 'kendujhar' in p.lower() or 'keonjhar' in p.lower():
            print(f"  Found Kendujhar/Keonjhar variant: '{p}'")
        if 'jagatsinghapur' in p.lower() or 'jagatsinghpur' in p.lower():
            print(f"  Found Jagatsinghapur variant: '{p}'")
