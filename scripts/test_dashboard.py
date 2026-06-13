import sys, os
sys.path.insert(0, 'B:\\Predictive Analytics for Resource Allocation')

# Test imports
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
print('All imports OK')

# Test data loading
from dashboard.app import load_data, load_demo_forecast, compute_resource_allocation
print('Functions imported OK')

# Test basic functions
df, report, geo = load_data()
n_districts = df['district'].nunique()
print(f'Data loaded: {len(df)} rows, {n_districts} districts')

# Test forecast generation
fdf, hist = load_demo_forecast('Puri')
print(f'Forecast: {len(fdf)} months, Historical: {len(hist)} months')

# Test resource allocation
res, aff_pop = compute_resource_allocation('Puri', 5.0)
print(f'Affected pop: {aff_pop}, Resources: {len(res)} types')
for k, v in res.items():
    print(f'  {k}: {v["amount"]} {v["unit"]}')

print('\nAll tests passed!')
