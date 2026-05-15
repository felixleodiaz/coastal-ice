# input parameters

date_start: str = input("Enter starting date in format YYYY-MM-DD: ")
date_end: str = input("Enter ending date in format YYYY-MM-DD: ")
year = date_end[0:4]

# import libraries

import os
import earthaccess
import xarray as xr
import dask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj, Transformer
from pathlib import Path
import glob
import re
from shapely.geometry import Polygon
from pyproj import Geod

# colormap for plotting sea ice throughout rest of project

cmap = plt.get_cmap("Blues_r").copy()
cmap.set_bad(color='lightgray')

## PASSIVE MICROWAVE SECTION ##

print('Getting NASA Team file list')
auth = earthaccess.login(strategy='interactive', persist=True)

# search NASA database for Team

results = earthaccess.search_data(
    short_name='NSIDC-0051',
    temporal=(date_start, date_end),
    bounding_box=(-180, 0, 180, 90),
    cloud_hosted=True
)

filtered_results = [
    g for g in results
    if re.search(r'_20\d{6}_', g.data_links(access='external')[0])
]

files_team = earthaccess.open(filtered_results)

# search NASA database for Bootstrap

print('Getting NASA Bootstrap file list')

results = earthaccess.search_data(
    short_name='NSIDC-0079',
    temporal=(date_start, date_end),
    bounding_box=(-180, 0, 180, 90),
    cloud_hosted=True
)

filtered_results = [
    g for g in results
    if re.search(r'_20\d{6}_', g.data_links(access='external')[0])
]

files_bootstrap = earthaccess.open(filtered_results)

# stream team into xarray
print('Opening daily NASA Team data into array')

ds = xr.open_mfdataset(
    files_team,
    parallel=True,
    concat_dim="time",
    combine="nested",
    data_vars='minimal',
    coords='minimal',
    compat='override'
)

# change ice concentration variable to something universal

icecon_vars = sorted([var for var in ds.data_vars if 'ICECON' in var], reverse=True)
icecon = ds[icecon_vars].to_array("source").max("source", skipna=True)

# add back to dataset and clean up

ds = ds.assign(team_icecon=icecon).drop_vars(icecon_vars)

# stream bootstrap into xarray
print('Opening daily NASA Bootstrap data into array')

ds_bootstrap = xr.open_mfdataset(
    files_bootstrap,
    parallel=True,
    concat_dim="time",
    combine="nested",
    data_vars='minimal',
    coords='minimal',
    compat='override'
)

# change ice concentration variable to something universal

icecon_vars = sorted([var for var in ds_bootstrap.data_vars if 'ICECON' in var], reverse=True)
icecon = ds_bootstrap[icecon_vars].to_array("source").max("source", skipna=True)

# add back to dataset and clean up

ds = ds.assign(bootstrap_icecon=icecon)

# strip any hours/minutes/seconds from the passive microwave time dimension

ds['time'] = ds.time.dt.floor('D')

# distance from land load

print('Loading cached geodesic distance-to-land')
distance_xr = xr.load_dataarray('../../local_data/nsidc_distance_to_land.nc')

nrows = ds.sizes['y']
ncols = ds.sizes['x']

ds = ds.assign_coords({
    'row': ('y', np.arange(nrows)),
    'col': ('x', np.arange(ncols))
}).swap_dims({'y': 'row', 'x': 'col'})

ds['edtl'] = distance_xr

## VISUAL ICE SECTION ##
print('Reading in and engineering visual data')

# read in all files, skipping empty ones

folder_path = '../../local_data/AutomaticProcessingResults'
all_files = glob.glob(os.path.join(folder_path, "*.csv"))

df_list = []
print(f"Reading in {len(all_files)} files")
for f in all_files:
    try:
        df = pd.read_csv(f)
        if not df.empty:
            df_list.append(df)
    except pd.errors.EmptyDataError:
        continue
    except Exception as e:
        print(f"Error reading {f}: {e}")

if not df_list:
    print("No data found in directory")
    raise SystemExit

visual = pd.concat(df_list, ignore_index=True)

# parse date and filter to requested date range

visual["time"] = pd.to_datetime(visual["date"], yearfirst=True).dt.floor('D')
visual = visual[(visual["time"] >= date_start) & (visual["time"] <= date_end)]

print('Merging coordinate mappings for area calculation...')
coord_mapping = pd.read_csv('../../local_data/CoastCellInfoJan5_10.csv')
coord_mapping.columns = coord_mapping.columns.str.strip()

# Merge the Lat/Lon corners into your visual dataframe based on the row/column identifiers
visual = visual.merge(
    coord_mapping[['Row', 'Col', 'Lat1', 'Lon1', 'Lat2', 'Lon2', 'Lat3', 'Lon3', 'Lat4', 'Lon4']],
    left_on=['row', 'column'], 
    right_on=['Row', 'Col'],
    how='inner'
)

# do area calculations for later filtering
# absolute value is necessary here because opp. winding order could result in negative area

geod = Geod(ellps="WGS84")

def grid_cell_area(row):
    coords = [
        (row['Lon1'], row['Lat1']),
        (row['Lon2'], row['Lat2']),
        (row['Lon3'], row['Lat3']),
        (row['Lon4'], row['Lat4'])
    ]
    poly = Polygon(coords)
    poly_area, _ = geod.geometry_area_perimeter(poly)
    
    return abs(poly_area)

visual['expected_area'] = visual.apply(grid_cell_area, axis=1)
visual['coverage_frac'] = visual['area'] / visual['expected_area']

# the RF pipeline outputs lowercase 'row' and 'column' as grid identifiers.
# rename these before applying the NSIDC index mapping to avoid a column name
# collision when we set visual['row'] and visual['col'] below.

visual = visual.rename(columns={'row': 'grid_row', 'column': 'grid_col'})

# preserve original index mapping: grid column -> NSIDC row, grid row -> NSIDC col

visual['row'] = visual['grid_col'].astype(int)
visual['col'] = visual['grid_row'].astype(int)

# drop duplicates

visual = visual.drop_duplicates(subset=["time", "row", "col"])

# convert to xarray

da_sparse = visual.set_index(['time', 'row', 'col']).to_xarray()
da_full = da_sparse.reindex_like(ds, method=None)
da_full = da_full.chunk({'time': 2})

# assign RF pipeline variables to dataset

ds = ds.assign(**{'visual_ice':      da_full['sea_ice_frac']})
ds = ds.assign(**{'melt_frac':       da_full['melt_frac']})
ds = ds.assign(**{'thin_ice_frac':   da_full['thin_ice_frac']})
ds = ds.assign(**{'water_frac':      da_full['water_frac']})
ds = ds.assign(**{'land_frac':       da_full['land_frac']})
ds = ds.assign(**{'snow_frac':       da_full['snow_frac']})
ds = ds.assign(**{'sea_ice_se':      da_full['sea_ice_se']})
ds = ds.assign(**{'melt_se':         da_full['melt_se']})
ds = ds.assign(**{'thin_ice_se':     da_full['thin_ice_se']})
ds = ds.assign(**{'water_se':        da_full['water_se']})
ds = ds.assign(**{'cloud_qa_pixels': da_full['cloud_qa_pixels']})
ds = ds.assign(**{'total_pixels':    da_full['total_pixels']})
ds = ds.assign(**{'sensor':          da_full['sensor']})
ds = ds.assign(**{'area':            da_full['area']})
ds = ds.assign(**{'expected_area':   da_full['expected_area']})
ds = ds.assign(**{'coverage_frac':   da_full['coverage_frac']})

# sanity check to make sure everything works

col_min, col_max = visual['col'].min(), visual['col'].max()
row_min, row_max = visual['row'].min(), visual['row'].max()

ds_subset = ds.sel(col=slice(col_min, col_max), row=slice(row_min, row_max))
ds_subset.team_icecon.mean(dim='time').plot(cmap=cmap, figsize=(6, 6))

plt.scatter(visual['col'], visual['row'], color='black', s=1, alpha=0.6)
plt.title(f"Where we Have Visual Data in Year {year}")
plt.savefig(f'../../figures/VisualDataExtents/{year}_visual_extent.png')
plt.close()

# print dataset

print('Printing dataset with visual, team, bootstrap, and distance from land')
print(ds)

# crop the giant grid down to just the bounding box of visual data

ds_cropped = ds.sel(col=slice(col_min, col_max), row=slice(row_min, row_max))

# single shared mask

condition = (
    ds_cropped.visual_ice.notnull() &
    (ds_cropped.team_icecon < 1.001) &
    (ds_cropped.bootstrap_icecon < 1.001)
)

ds_clean = ds_cropped.where(condition)

print("Converting to dataframe and dropping NaNs")

# export and save ds as a dataframe

df = ds_clean.to_dataframe().dropna().reset_index()

print(f"Exporting CSV with {len(df)} valid overlap points!")
df.to_csv(f'../../local_data/DataFrames/validation_{year}.csv', index=False)