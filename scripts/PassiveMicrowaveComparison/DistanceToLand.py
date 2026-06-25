# import libraries
# run this script once prior to running validation comparison and error analysis scripts
# it calculates and saves a distance to land dataset

import numpy as np
import geopandas as gpd
import xarray as xr

# NSIDC NH polar stereographic grid parameters (EPSG:3411)
# 304 cols x 448 rows at 25 km resolution

NROWS      = 448
NCOLS      = 304
PIXEL_SIZE = 25000   # metres
X_ORIGIN   = -3830000
Y_ORIGIN   =  5830000

print('Loading and projecting land mask to EPSG:3411')
land = gpd.read_file("../../data/ne_10m_land/ne_10m_land.shp", engine='pyogrio')
land = land.to_crs(epsg=3411)

print('Generating grid cell centres')
cols = np.arange(NCOLS)
rows = np.arange(NROWS)
col_grid, row_grid = np.meshgrid(cols, rows)   # both (NROWS, NCOLS)

x_centres = X_ORIGIN + (col_grid + 0.5) * PIXEL_SIZE
y_centres = Y_ORIGIN + (row_grid + 0.5) * (-PIXEL_SIZE)

# create a GeoDataFrame of all the grid points

points_gdf = gpd.GeoDataFrame(
    geometry=gpd.points_from_xy(x_centres.ravel(), y_centres.ravel()),
    crs="EPSG:3411"
)

# sjoin_nearest automatically builds a spatial index and efficiently calculates 
# the distance from every point to the nearest polygon

print(f'Computing EPSG:3411 distance to land for {NROWS * NCOLS:,} grid cells')
joined = points_gdf.sjoin_nearest(land, how='left', distance_col='dist_m')
joined = joined[~joined.index.duplicated(keep='first')]
joined = joined.sort_index()

# extract the distances and reshape back to the 2D grid

distances_2d = joined['dist_m'].values.reshape((NROWS, NCOLS))

print('Building NetCDF dataset...')
distance_xr = xr.DataArray(
    distances_2d,
    coords={
        'row': np.arange(NROWS),
        'col': np.arange(NCOLS)
    },
    dims=('row', 'col'),
    name='distance_to_land_m',
    attrs={
        'units': 'metres',
        'description': (
            'Distance from NSIDC NH 25 km polar stereographic '
            'grid cell centre to nearest NaturalEarth 10 m land polygon, '
            'calculated natively in EPSG:3411'
        )
    }
)

# export and save

output_path = '../../local_data/nsidc_distance_to_land.nc'
distance_xr.to_netcdf(output_path)
print(f'saved to {output_path}')