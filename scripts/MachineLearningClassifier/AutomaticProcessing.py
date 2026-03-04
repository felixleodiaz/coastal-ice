# AUTOMATIC PROCESSING PIPELINE

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

# load training data and train classifier

training_data = ee.FeatureCollection('projects/gee-personal-483416/assets/training_asset_sample')

best_params = {
    'numberOfTrees': 198, 
    'variablesPerSplit': 3, 
    'minLeafPopulation': 5, 
    'bagFraction': 0.9428329774159232, 
    'seed': 12
}

classifier = (ee.Classifier.smileRandomForest(**best_params)
    .setOutputMode('MULTIPROBABILITY')
    .train(
        features=training_data,
        classProperty='class_id',
        inputProperties=['blue', 'green', 'red', 'nir', 'swir1', 'swir2', 'sensor']
    )
)

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

from ImageClipping import image_clipping

# function three
# surface classification using a previously saved Random Forest classifier

def surface_calculations(image):

    image = ee.Image(image)
    geom = image.geometry()

    # count and then mask high-confidence opaque cloud pixels

    cloud_qa = image.select('cloud_qa')

    nCLOUD_QA = ee.Number(cloud_qa.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=geom,
        scale=30,
        maxPixels=1e8
    ).get('cloud_qa'))

    image = image.updateMask(cloud_qa.Not())

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
    p_melt      = prob_image.arrayGet(1).rename('melt')
    p_water     = prob_image.arrayGet(2).rename('water')
    p_thinice   = prob_image.arrayGet(3).rename('thinice')

    prob_bands = (
        p_seaice.addBands([p_melt, p_water, p_thinice])
        .updateMask(waterMask_clipped.eq(1))
    )

    # compute p * (1 - p) per pixel per class for Bernoulli variance and calc Var[Y_i] = p_i * (1 - p_i)

    var_bands = prob_bands.multiply(prob_bands.subtract(1).multiply(-1)).rename(
        ['var_seaice', 'var_melt', 'var_water', 'var_thinice']
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

    var_seaice    = ee.Number(water_vars.get('var_seaice'))
    var_water     = ee.Number(water_vars.get('var_water'))
    var_melt      = ee.Number(water_vars.get('var_melt'))
    var_thinice   = ee.Number(water_vars.get('var_thinice'))

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
        .add(nLAND).add(nSNOW)
    )

    total_sq = total.multiply(total)

    # SE[f] = sqrt(sum(p_i * (1 - p_i))) / N_total

    se_seaice    = var_seaice.divide(total_sq).sqrt()
    se_water     = var_water.divide(total_sq).sqrt()
    se_melt      = var_melt.divide(total_sq).sqrt()
    se_thinice   = var_thinice.divide(total_sq).sqrt()

    return ee.Feature(geom, {
        'total_pixels':       total,
        'cloud_qa_pixels': nCLOUD_QA,
        'sea_ice_frac':       sum_seaice.divide(total),
        'sea_ice_se':         se_seaice,
        'water_frac':         sum_water.divide(total),
        'water_se':           se_water,
        'melt_frac':          sum_melt.divide(total),
        'melt_se':            se_melt,
        'thin_ice_frac':      sum_thinice.divide(total),
        'thin_ice_se':        se_thinice,
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