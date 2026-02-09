# MACHINE LEARNING BASED CLASSIFICATION SCRIPT

import ee
import numpy as np

# initialize earth engine

project_id = 'gee-personal-483416'

try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

# load classifier

classifier = ee.Classifier.load('projects/gee-personal-483416/assets/random_forest_seaice_classifier')

# function one
# creates polygons in GEE from grid coordinates

def coastal_polygon(feature, year):

    # set dates

    starttime = f'{year}-01-01'
    endtime = f'{year}-12-31'

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

    # get landsat 8 images

    L8 = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_L2')
        .filterDate(startdate, enddate)
        .filterBounds(gridGeom)
        .filter(ee.Filter.lt('CLOUD_COVER', 5))
        .map(
            lambda img: (
                img.addBands(
                    img.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
                    .multiply(0.0000275)
                    .add(-0.2),
                    overwrite=True
                )
                .select(
                    ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
                    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
                )
                .set('cloud', img.get('CLOUD_COVER'))
                .set('sensor', 'Landsat8')
                .set('area', img.geometry().intersection(gridGeom, 1).area())
            )
        )
    )

    # get landsat 9 images

    L9 = (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_L2')
        .filterDate(startdate, enddate)
        .filterBounds(gridGeom)
        .filter(ee.Filter.lt('CLOUD_COVER', 5))
        .map(
            lambda img: (
                img.addBands(
                    img.select(['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'])
                    .multiply(0.0000275)
                    .add(-0.2),
                    overwrite=True
                )
                .select(
                    ['SR_B2', 'SR_B3', 'SR_B4', 'SR_B5', 'SR_B6', 'SR_B7'],
                    ['blue', 'green', 'red', 'nir', 'swir1', 'swir2']
                )
                .set('cloud', img.get('CLOUD_COVER'))
                .set('sensor', 'Landsat9')
                .set('area', img.geometry().intersection(gridGeom, 1).area())
            )
        )
    )

    # get sentinel 2 images

    S2 = (
        ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
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

    waterMask = ee.Image('projects/gee-personal-483416/assets/connected_water_mask_2015').clip(geom)

    # calculate NDSI for land

    NDSI = image.normalizedDifference(['green', 'swir1']).rename('NDSI')
    landSnow = NDSI.gt(0.4)

    # machine learning classification for ocean cells

    classified = image.classify(classifier)

    # final classification using ML over water and NDSI over land

    classification = (
        ee.Image(0)
        .where(waterMask.eq(1), classified)
        .where(waterMask.eq(0).And(landSnow.Not()), 7)
        .where(waterMask.eq(0).And(landSnow), 8)
        .rename('class')
        .clip(geom)
    )

    # single pass reduction

    stats = (
        classification
        .reduceRegion(
            reducer=ee.Reducer.frequencyHistogram(),
            geometry=geom,
            scale=30,
            maxPixels=1e8
        )
        .get('class')
    )

    stats_dict = ee.Dictionary(stats)

    def getCount(key):
        return ee.Number(ee.Algorithms.If(stats_dict.contains(key), stats_dict.get(key), 0))
    
    # get counts and return

    nSEAICE = getCount('1')
    nOCEAN  = getCount('2')
    nPOND  = getCount('3')
    nHAZYWATER = getCount('4')
    nHAZYICE = getCount('5')
    nCLOUD = getCount('6')
    nLAND   = getCount('7')
    nSNOW   = getCount('8')
    total = nSEAICE.add(nOCEAN).add(nLAND).add(nSNOW)

    return ee.Feature(geom, {
        'total_pixels': total,
        'sea_ice_frac': nSEAICE.divide(total),
        'ocean_frac': nOCEAN.divide(total),
        'land_frac': nLAND.divide(total),
        'snow_frac': nSNOW.divide(total),
        'pond_frac': nPOND.divide(total),
        'hazy_ice_frac': nHAZYICE.divide(total),
        'hazy_water_frac': nHAZYWATER.divide(total),
        'cloud_frac': nCLOUD.divide(total),
        'date': ee.Date(image.get('system:time_start')).format('YYYY-MM-dd'),
        'sensor': image.get('sensor'),
        'area' : image.get('area'),
        'row' : image.get('Row'), 
        'column' : image.get('Column')
    })

# workflow

grid = ee.FeatureCollection('projects/gee-personal-483416/assets/CoastCellInfoJan5_10')

lon_min = int(-180)
lon_max = int(180)
step = int(4)

# create list of lon steps

lons = np.arange(lon_min, lon_max, step)

for year in range(2013, 2024):

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
        
        filename = f'classified_{lon_start}_{lon_end}_{year}'
        
        # run functions

        grid_cutouts = grid_slice.map(lambda f: coastal_polygon(f, year))
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