# SCRIPT TO GENERATE CLASSIFIED IMAGES FROM SPECIFIED ROWS, COLS, AND DATES

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
# create images

def coastal_polygon(feature):

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
        'Row': feature.get('Row')
    })

    return gridbox

# function two
# pull from automatic process, get images

from AutomaticProcessing import image_clipping

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

    waterMask = ee.Image('projects/gee-personal-483416/assets/connected_water_mask_2015').clip(geom).unmask(0)

    # calculate NDSI for land

    NDSI = image.normalizedDifference(['green', 'swir1']).rename('NDSI')
    landSnow = NDSI.gt(0.4)

    # machine learning classification for ocean cells

    classified = image.classify(classifier)

    # final classification using ML over water and NDSI over land

    classification = (
        ee.Image(0)
        .where(waterMask.eq(1), classified)
        .where(waterMask.Not().And(landSnow.Not()), 8)
        .where(waterMask.Not().And(landSnow), 9)
        .rename('class')
        .clip(geom)
    )

    # generate images

    rgb = image.visualize(**{'bands': ['nir', 'swir1', 'blue'], 'min': 0, 'max': 0.4, 'gamma': 1.5})
    rgb = rgb.rename(['rgb_R', 'rgb_G', 'rgb_B'])
    params = {
    'min': 0,
    'max': 9,
    'palette': [
        'b10000', # 0: NA value (bright red)
        '6DAEDB', # 1: Ice
        '120D31', # 2: Water
        '0a29c2', # 3: Melt
        '029e73', # 4: Thin Ice
        '120D31', # 5: Hazy Water (visually Water)
        '6DAEDB', # 6: Hazy Ice (visually Ice)
        'ffffff', # 7: Clouds
        'ca9161', # 8: Land
        'FFE8D1'  # 9: Snow on Land
    ]
}
    class_map = classification.visualize(**params)
    class_map = class_map.rename(['class_R', 'class_G', 'class_B'])
    
    # composite image and return

    final = rgb.addBands(class_map).reproject(**{'crs': image.select(0).projection(), 'scale': 30})
    return final

# function four
# error image in case of error

def create_error_image(region):
  return ee.Image(0).visualize(**{'palette':['red']}).paint(region, 1, 5)

# load things

print('loading and processing assets')
grid = ee.FeatureCollection('projects/gee-personal-483416/assets/CoastCellInfoJan5_10')
samplesCollection = ee.FeatureCollection('projects/gee-personal-483416/assets/sample_images')
samplesList = samplesCollection.toList(samplesCollection.size()).getInfo()
print(f'Processing {len(samplesList)} samples')

for feature in samplesList:

    sample = feature['properties']

    # server side code

    raw_feature = grid.filter(ee.Filter.And(
        ee.Filter.eq('Col', sample['col']),
        ee.Filter.eq('Row', sample['row'])
    )).first()

    dummy_feature = ee.Feature(None, {'Col': sample['col'], 'Row': sample['row']})
    safe_raw_feature = ee.Feature(ee.Algorithms.If(raw_feature, raw_feature, dummy_feature))

    # run function one

    cell_feature = coastal_polygon(safe_raw_feature)

    # get dates and run function two

    start_date = sample['date']
    end_date = ee.Date(start_date).advance(1, 'day').format('YYYY-MM-dd')
    cell_with_date = cell_feature.set('Start', start_date).set('End', end_date)

    img = image_clipping(cell_with_date)
    img = ee.Image(img)

    # run function three (and four if necessary)

    combined_visuals = ee.Algorithms.If(
        img, 
        surface_calculations(img), 
        create_error_image(cell_feature.geometry())
    )
    combined_visuals = ee.Image(combined_visuals)

    export_crs = 'EPSG:3413'

    # export satellite image

    export_name_lines = f"Visual_{sample['row']}_{sample['col']}_{sample['date']}"
    
    task_lines = ee.batch.Export.image.toDrive(
        image=combined_visuals.select(['rgb_R', 'rgb_G', 'rgb_B']),
        description=export_name_lines,
        folder='EarthEngineVisualsRF_testTOA',
        region=cell_feature.geometry(),
        scale=30,
        crs=export_crs,
        fileFormat='GeoTIFF'
    )

    task_lines.start()

    # export classification map

    export_name_class = f"Class_{sample['row']}_{sample['col']}_{sample['date']}"
    
    task_class = ee.batch.Export.image.toDrive(
        image=combined_visuals.select(['class_R', 'class_G', 'class_B']),
        description=export_name_class,
        folder='EarthEngineClassmapsRF_testTOA',
        region=cell_feature.geometry(),
        scale=30,
        crs=export_crs,
        fileFormat='GeoTIFF'
    )
    task_class.start()
    print(f'task {export_name_class} started')

print('tasks exported!')