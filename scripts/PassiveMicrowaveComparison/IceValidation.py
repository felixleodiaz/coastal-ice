# input parameters
date_start: str = input("Enter starting date in format YYYY-MM-DD: ")
date_end: str = input("Enter ending date in format YYYY-MM-DD: ")
year = date_end[0:4]

# import libraries

import earthaccess
import xarray as xr
import dask
import numpy as np
import pandas as pd
import geopandas as gpd
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from rasterio import features
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import seaborn as sns
from pyproj import Proj, Transformer
from pathlib import Path
import glob
import re

# colormap for plotting sea ice throughout rest of project

cmap = plt.get_cmap("Blues_r").copy()
cmap.set_bad(color='lightgray')

## PASSIVE MICROWAVE SECTION ##

# authenticate NASA earth access
print('Getting NASA Team file list')

auth = earthaccess.login(strategy='interactive', persist = True)

# search NASA database for Team

results = earthaccess.search_data(
    short_name= 'NSIDC-0051',
    temporal=(date_start, date_end),
    bounding_box=(-180, 0, 180, 90),
    cloud_hosted=True
)

filtered_results = [
    g for g in results
    if re.search(r'_20\d{6}_', g.data_links(access='external')[0])
]

# get files from NSIDC

files_team = earthaccess.open(filtered_results)

# search NASA database for Bootstrap
print('Getting NASA Bootstrap file list')

results = earthaccess.search_data(
    short_name= 'NSIDC-0079',
    temporal=(date_start, date_end),
    bounding_box=(-180, 0, 180, 90),
    cloud_hosted=True
)

filtered_results = [
    g for g in results
    if re.search(r'_20\d{6}_', g.data_links(access='external')[0])
]

# get files from NSIDC

files_bootstrap = earthaccess.open(filtered_results)

# stream team into xarray
print('Opening daily NASA Team data into array')

ds = xr.open_mfdataset(
    files_team, 
    parallel = True, 
    concat_dim="time", 
    combine="nested", 
    data_vars='minimal', 
    coords='minimal', 
    compat='override'
)

# change ice concentration variable to something universal

icecon_vars = sorted([var for var in ds.data_vars if 'ICECON'in var], reverse=True)
icecon = ds[icecon_vars].to_array("source").max("source", skipna=True)

# add back to dataset and clean up
ds = ds.assign(team_icecon=icecon).drop_vars(icecon_vars)

# stream bootstrap into xarray
print('Opening daily NASA Bootstrap data into array')
ds_bootstrap = xr.open_mfdataset(
    files_bootstrap, 
    parallel = True, 
    concat_dim="time", 
    combine="nested", 
    data_vars='minimal', 
    coords='minimal', 
    compat='override'
)

# change ice concentration variable to something universal

icecon_vars = sorted([var for var in ds_bootstrap.data_vars if 'ICECON'in var], reverse=True)
icecon = ds_bootstrap[icecon_vars].to_array("source").max("source", skipna=True)

# add back to dataset and clean up
ds = ds.assign(bootstrap_icecon=icecon)

# read in land file from geopandas and initialize transform (from NSIDC metadata)
print('Calculating distance from land for each pixel')

land = gpd.read_file("../data/ne_10m_land/ne_10m_land.shp")
land = land.to_crs(epsg=3411)
transform = [25000, 0, -3830000, 0, -25000, 5830000]

# use transform to mask out coastal cells

land_mask = features.rasterize(
    ((geom, 1) for geom in land.geometry),
    out_shape=(448, 304),
    transform=transform,
    fill=0,
    dtype=np.uint8
)

# calculate distance from land using euclidian distance transform

distance_from_land = distance_transform_edt(land_mask == 0)

# convert to xarray.DataArray

distance_xr = xr.DataArray(
    distance_from_land,
    coords={'y': ds.y, 'x': ds.x},
    dims=('y', 'x'),
    name='distance_to_land_cells'
)

# add as data variable in ds

ds['edtl'] = distance_xr

# reindex data array

nrows = ds.sizes['y']
ncols = ds.sizes['x']

ds = ds.assign_coords({
    'row': ('y', np.arange(nrows)),
    'col': ('x', np.arange(ncols))
}).swap_dims({'y': 'row', 'x': 'col'})

## VISUAL ICE SECTION ##
print('Reading in and engineering visual data')

# read in files

folderpath = '../local_data/earth_engine_demos/'
paths = Path(folderpath).glob(f'*{year}*.csv')
pathlist = list(paths)

# data cleaning of visual datasets

visual = pd.concat(map(pd.read_csv, pathlist), ignore_index=True)

# convert things for xarray

visual["time"] = pd.to_datetime(visual["Date"], yearfirst=True)
visual['row'] = visual['Column'] + 1
visual['col'] = visual['Row'] + 1

# drop duplicates

visual = visual.drop_duplicates(subset=["time", "row", "col"])

# convert to xarray

da_sparse = visual.set_index(['time', 'row', 'col']).to_xarray()
da_full = da_sparse.reindex_like(ds, method=None)

da_full = da_full.chunk({'time': 2})
ds = ds.assign(**{'visual_ice': da_full['SI frac']})
ds = ds.assign(**{'ocean_fraction': da_full['Ocean frac']})
ds = ds.assign(**{'land_fraction': da_full['Land frac']})
ds = ds.assign(**{'snow_fraction': da_full['Snow frac']})
ds = ds.assign(**{'sun_elevation': da_full['Sun elevation']})
ds = ds.assign(**{'cloud_cover': da_full['Cloud cover']})
ds = ds.assign(**{'pixels': da_full['total pixels']})

# sanity check to make sure everything works

col_min, col_max = visual['col'].min(), visual['col'].max()
row_min, row_max = visual['row'].min(), visual['row'].max()

ds_subset = ds.sel(col=slice(col_min, col_max), row=slice(row_min, row_max))
ax = ds_subset.team_icecon.mean(dim='time').plot(
    cmap=cmap,
    figsize=(6,6)
)

plt.scatter(
    visual['col'],
    visual['row'],
    color='black',
    s=1,
    alpha=0.6
)
plt.title(f"Where we Have Visual Data in Year {year}")
plt.savefig(f'../figures/visual_data_extent/{year}_visual_extent.png')
plt.close()

# print dataset

print('Printing dataset with visual, team, bootstrap, and distance from land')
print(ds)

## ERROR CALCULATIONS ##
print('Starting error calculations')

# ERROR FOR TEAM

condition = ((ds.visual_ice.notnull()) & (ds.team_icecon < 1.001))
ds_clean = ds.where(condition, other=np.nan).compute()

# calc error

error_team = (((ds_clean['team_icecon'] - ds_clean['visual_ice'])**2)**0.5)
error_avg = error_team.mean(dim=['time', 'col', 'row'], skipna=True)
print('RMS error for NASA Team is', error_avg.compute().item())

# save a data cleaned pandas dataframe for team with everything (1.012 = coast, 1.016 = land)

df = ds_clean.to_dataframe().reset_index().dropna()
df.to_csv(f'../data/data_frames/team_validation_{year}.csv')

# ERROR FOR BOOTSTRAP

condition = ((ds.visual_ice.notnull()) & (ds.bootstrap_icecon < 1.001))
ds_clean = ds.where(condition, other=np.nan).compute()

# error calculation bootstrap

error_bootstrap = (((ds_clean['bootstrap_icecon'] - ds_clean['visual_ice'])**2)**0.5)
error_avg = error_bootstrap.mean(dim=['time', 'col', 'row'], skipna=True)
print('RMS error for NASA Bootstrap is', error_avg.compute().item())

# save a data cleaned pandas dataframe for boostrap with everything (1.012 = coast, 1.016 = land)

df = ds_clean.to_dataframe().reset_index().dropna()
df.to_csv(f'../data/data_frames/bootstrap_validation_{year}.csv')

# MAP ERROR

# map error NASA team

sns.set_style('darkgrid')
ax = error_team.mean(dim='time', skipna=True).plot(cmap = 'RdBu', figsize=(6,6))
plt.title(f"Error Between NASA Team and Visual Mapped 2023-04-01 to 2023-05-31")
plt.savefig(f'../figures/error_maps/team_{year}_error.png')
plt.close()

# map error NASA bootstrap

ax = error_bootstrap.mean(dim='time', skipna=True).plot(cmap = 'RdBu', figsize=(6,6))
plt.title(f"Error Between NASA Team and Visual Mapped 2023-04-01 to 2023-05-31")
plt.savefig(f'../figures/error_maps/bootstrap_{year}_error.png')
plt.close()