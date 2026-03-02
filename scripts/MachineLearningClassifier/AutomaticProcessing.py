# MACHINE LEARNING BASED CLASSIFICATION SCRIPT

import ee
import numpy as np
import calendar
import sys

# initialize earth engine

project_id = 'gee-personal-483416'

try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

# load classifier

classifier = ee.Classifier.load('projects/gee-personal-483416/assets/random_forest_seaice_classifier')
classifier = classifier.setOutputMode('MULTIPROBABILITY')

# load water mask
waterMask = ee.Image('projects/gee-personal-483416/assets/connected_water_mask_2015').unmask(0)

# function one
# creates polygons in GEE from grid coordinates

def coastal_polygon(feature, year, month):

    # set dates

    starttime = f'{year}-{month:02d}-01'
    endtime = f'{year}-{month:02d}-{calendar.monthrange(year, month)[1]}'

    # read in grid polygon coordinates

    coordinates = [
        [feature.get('Lon1'), feature.get('Lat1')],
        [feature.get('Lon2'), feature.get('Lat2')],
        [feature.get('Lon3'), feature.get('Lat3')],
        [feature.get('Lon4'), feature.get('Lat4')]
    ]

    # construct grid in earth engine

    gridbox = ee.Feature(ee.Geometry.Polygon([coordinates]), {
        'Column': feature.get('Col'), 
        'Row': feature.get('Row'), 
        'Start': starttime, 
        'End': endtime
    })

    return gridbox

# function two
# gets images from Landsat 8/9 and Sentinel 2 and clips to grid

def image_clipping(grid):

    gridGeom = grid.geometry()
    startdate = grid.get('Start')
    enddate = grid.get('End')

    # get landsat 8 images TOA
    
    L8 = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
        .filterDate(startdate, enddate)
        .filterBounds(gridGeom)
        .filter(ee.Filter.lt('CLOUD_COVER', 5))
        .map(
            lambda img: (
                img
                .select(
                    ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
                )
                .set('cloud', img.get('CLOUD_COVER'))
                .set('sensor', 'Landsat8')
                .set('area', img.geometry().intersection(gridGeom, 1).area())
            )
        )
    )

    # get landsat 9 images TOA

    L9 = (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA')
        .filterDate(startdate, enddate)
        .filterBounds(gridGeom)
        .filter(ee.Filter.lt('CLOUD_COVER', 5))
        .map(
            lambda img: (
                img
                .select(
                    ['B2', 'B3', 'B4', 'B5', 'B6', 'B7'],
                    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
                )
                .set('cloud', img.get('CLOUD_COVER'))
                .set('sensor', 'Landsat9')
                .set('area', img.geometry().intersection(gridGeom, 1).area())
            )
        )
    )

    # get sentinel 2 images TOA

    S2 = (
        ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filterDate(startdate, enddate)
        .filterBounds(gridGeom)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
        .map(
            lambda img: (
                img.addBands(
                    img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])
                    .divide(10000),
                    overwrite=True
                )
                .select(
                    ['B2', 'B3', 'B4', 'B8', 'B11', 'B12'],
                    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
                )
                .set('cloud', img.get('CLOUDY_PIXEL_PERCENTAGE'))
                .set('sensor', 'Sentinel2')
                .set('area', img.geometry().intersection(gridGeom, 1).area())
            )
        )
    )

    # merge datasets

    merged = L8.merge(L9).merge(S2)
    nimages = merged.size()

    # sort by lowest cloud fraction and pick the lowest

    merged = merged.sort('cloud').sort('area', False)

    # make sure not null, attach metadata and export
    
    safe_image = ee.Image(ee.Algorithms.If(
        nimages.gt(0), 
        merged.first(), 
        ee.Image.constant(0)
    ))
    
    best = safe_image.clip(gridGeom).set({
        'Row': grid.get('Row'),
        'Column': grid.get('Column'),
        'system:time_start': safe_image.get('system:time_start')
    })

    return ee.Image(ee.Algorithms.If(nimages.gt(0), best, None))


# function three
# surface classification using a previously saved Random Forest classifier

def surface_calculations(image):

    image = ee.Image(image)
    geom = image.geometry()

    # attach sensor as band

    sensor_name = ee.String(image.get('sensor'))
    sensor_val = ee.Number(ee.Algorithms.If(sensor_name.compareTo('Sentinel2').eq(0), 1, 0))
    sensor_band = ee.Image.constant(sensor_val).rename('sensor').toByte()
    image = image.addBands(sensor_band)

    # water mask

    waterMask_clipped = waterMask.clip(geom)

    # static NDSI for land pixels

    NDSI = image.normalizedDifference(['green', 'swir1']).rename('NDSI')
    landSnow = NDSI.gt(0.4)

    # multiprobability classification over water pixels

    prob_image = image.classify(classifier)

    p_seaice    = prob_image.arrayGet(0).rename('seaice')
    p_water     = prob_image.arrayGet(2).rename('water')
    p_melt      = prob_image.arrayGet(1).rename('melt')
    p_thinice   = prob_image.arrayGet(3).rename('thinice')
    p_hazywater = prob_image.arrayGet(4).rename('hazywater')
    p_hazyice   = prob_image.arrayGet(5).rename('hazyice')
    p_cloud     = prob_image.arrayGet(6).rename('cloud')

    prob_bands = (
        p_seaice.addBands([p_water, p_melt, p_thinice, p_hazywater, p_hazyice, p_cloud])
        .updateMask(waterMask_clipped.eq(1))
    )

    # compute p * (1 - p) per pixel per class for Bernoulli variance and calc Var[Y_i] = p_i * (1 - p_i)

    var_bands = prob_bands.multiply(prob_bands.subtract(1).multiply(-1)).rename(
        ['var_seaice', 'var_water', 'var_melt', 'var_thinice', 'var_hazywater', 'var_hazyice', 'var_cloud']
    )

    # sum probabilities across water pixels per class: S = sum(p_i)

    water_sums = prob_bands.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=30,
        maxPixels=1e8
    )

    # sum variances across water pixels per class: sum(p_i * (1 - p_i))

    water_vars = var_bands.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=30,
        maxPixels=1e8
    )

    sum_seaice    = ee.Number(water_sums.get('seaice'))
    sum_water     = ee.Number(water_sums.get('water'))
    sum_melt      = ee.Number(water_sums.get('melt'))
    sum_thinice   = ee.Number(water_sums.get('thinice'))
    sum_hazywater = ee.Number(water_sums.get('hazywater'))
    sum_hazyice   = ee.Number(water_sums.get('hazyice'))
    sum_cloud     = ee.Number(water_sums.get('cloud'))

    var_seaice    = ee.Number(water_vars.get('var_seaice'))
    var_water     = ee.Number(water_vars.get('var_water'))
    var_melt      = ee.Number(water_vars.get('var_melt'))
    var_thinice   = ee.Number(water_vars.get('var_thinice'))
    var_hazywater = ee.Number(water_vars.get('var_hazywater'))
    var_hazyice   = ee.Number(water_vars.get('var_hazyice'))
    var_cloud     = ee.Number(water_vars.get('var_cloud'))

    # count land and snow pixels via NDSI, restricted to land mask only

    landMask = waterMask_clipped.eq(0)
    land_snow_bands = (
        landMask.And(landSnow.Not()).rename('land')
        .addBands(landMask.And(landSnow).rename('snow'))
    )

    land_stats = land_snow_bands.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=30,
        maxPixels=1e8
    )

    nLAND = ee.Number(land_stats.get('land'))
    nSNOW = ee.Number(land_stats.get('snow'))

    # total combines probability sums (water) and pixel counts (land/snow)

    total = (
        sum_seaice.add(sum_water).add(sum_melt).add(sum_thinice)
        .add(sum_hazywater).add(sum_hazyice).add(sum_cloud)
        .add(nLAND).add(nSNOW)
    )

    total_sq = total.multiply(total)

    # SE[f] = sqrt(sum(p_i * (1 - p_i))) / N_total

    se_seaice    = var_seaice.divide(total_sq).sqrt()
    se_water     = var_water.divide(total_sq).sqrt()
    se_melt      = var_melt.divide(total_sq).sqrt()
    se_thinice   = var_thinice.divide(total_sq).sqrt()
    se_hazywater = var_hazywater.divide(total_sq).sqrt()
    se_hazyice   = var_hazyice.divide(total_sq).sqrt()
    se_cloud     = var_cloud.divide(total_sq).sqrt()

    return ee.Feature(geom, {
        'total_pixels':       total,
        'sea_ice_frac':       sum_seaice.divide(total),
        'sea_ice_se':         se_seaice,
        'water_frac':         sum_water.divide(total),
        'water_se':           se_water,
        'melt_frac':          sum_melt.divide(total),
        'melt_se':            se_melt,
        'thin_ice_frac':      sum_thinice.divide(total),
        'thin_ice_se':        se_thinice,
        'hazy_water_frac':    sum_hazywater.divide(total),
        'hazy_water_se':      se_hazywater,
        'hazy_ice_frac':      sum_hazyice.divide(total),
        'hazy_ice_se':        se_hazyice,
        'cloud_frac':         sum_cloud.divide(total),
        'cloud_se':           se_cloud,
        'land_frac':          nLAND.divide(total),
        'snow_frac':          nSNOW.divide(total),
        'date':   ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
        'sensor': image.get('sensor'),
        'area':   image.get('area'),
        'row':    image.get('Row'),
        'column': image.get('Column')
    })

# workflow
# run using python AutomaticProcessing.py <year>

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Please use input format: python script.py <year>")
        sys.exit(1)

    year = int(sys.argv[1])

    grid = ee.FeatureCollection('projects/gee-personal-483416/assets/CoastCellInfoJan5_10')

    lon_min = int(-180)
    lon_max = int(180)
    step = int(4)

    # create list of lon steps

    lons = np.arange(lon_min, lon_max, step)

    for month in range(1, 13):
        for lon in lons:

            lon_start = int(lon)
            lon_end = int(lon_start + step)
            
            # create filter

            grid_slice = grid.filter(
                ee.Filter.And(
                    ee.Filter.gte('Lon1', lon_start),
                    ee.Filter.lt('Lon1', lon_end)
                )
            )
            
            filename = f'classified_{lon_start}_{lon_end}_{year}_{month:02d}'
            
            # run functions

            grid_cutouts = grid_slice.map(lambda f: coastal_polygon(f, year, month))
            one_grid = grid_cutouts.map(image_clipping, dropNulls=True)
            ice_class = one_grid.map(surface_calculations)
            
            # prepare export

            task = ee.batch.Export.table.toDrive(
                collection=ice_class,
                description=filename,
                folder='EarthEngineResultsRF',
                fileFormat='CSV'
            )
            
            # start task

            task.start()
            print(f'started {filename}')

    print("\n All tasks started successfully")