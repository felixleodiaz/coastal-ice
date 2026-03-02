# SCRIPT TO GENERATE PROBABILITY MAPS AND TRUE COLOR IMAGES FOR SAMPLE CELLS

import ee
import numpy as np

# initialize earth engine

project_id = 'gee-personal-483416'

try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

# load classifier in multiprobability mode

classifier = ee.Classifier.load('projects/gee-personal-483416/assets/random_forest_seaice_classifier')
classifier = classifier.setOutputMode('MULTIPROBABILITY')

# load water mask

waterMask = ee.Image('projects/gee-personal-483416/assets/connected_water_mask_2015').unmask(0)

# class definitions
# arrayGet indices are 0-based, mapping to class labels 1-7

CLASSES = [
    (0, 'seaice',    'Sea Ice'),
    (1, 'water',     'Water'),
    (2, 'melt',      'Melt Ponds'),
    (3, 'thinice',   'Thin Ice'),
    (4, 'hazywater', 'Hazy Water'),
    (5, 'hazyice',   'Hazy Ice'),
    (6, 'cloud',     'Cloud'),
]

# blues palette for probability maps: white (p=0) to dark blue (p=1)

BLUES_PALETTE = ['ffffff', 'ddeeff', 'aaccee', '6699cc', '3366aa', '003388', '001155']

# function one
# create grid polygon from feature

def coastal_polygon(feature):

    coordinates = [
        [feature.get('Lon1'), feature.get('Lat1')],
        [feature.get('Lon2'), feature.get('Lat2')],
        [feature.get('Lon3'), feature.get('Lat3')],
        [feature.get('Lon4'), feature.get('Lat4')]
    ]

    gridbox = ee.Feature(ee.Geometry.Polygon([coordinates]), {
        'Column': feature.get('Col'),
        'Row': feature.get('Row')
    })

    return gridbox

# function two
# pull image clipping from main processing script

from AutomaticProcessing import image_clipping

# function three
# generate true color and per-class probability images

def generate_visuals(image):

    image = ee.Image(image)
    geom = image.geometry()

    # attach sensor as band

    sensor_name = ee.String(image.get('sensor'))
    sensor_val = ee.Number(ee.Algorithms.If(sensor_name.compareTo('Sentinel2').eq(0), 1, 0))
    sensor_band = ee.Image.constant(sensor_val).rename('sensor').toByte()
    image = image.addBands(sensor_band)

    # true color: NIR / SWIR1 / Blue false color to distinguish ice and water clearly

    rgb = image.visualize(**{
        'bands': ['nir', 'swir1', 'blue'],
        'min': 0,
        'max': 0.4,
        'gamma': 1.5
    })

    # water mask clipped to geometry

    waterMask_clipped = waterMask.clip(geom)

    # run multiprobability classification

    prob_image = image.classify(classifier)

    # extract per-class probability bands and mask to water only

    prob_bands = (
        prob_image.select('classification').arrayGet([0]).rename('seaice')
        .addBands(prob_image.select('classification').arrayGet([1]).rename('water'))
        .addBands(prob_image.select('classification').arrayGet([2]).rename('melt'))
        .addBands(prob_image.select('classification').arrayGet([3]).rename('thinice'))
        .addBands(prob_image.select('classification').arrayGet([4]).rename('hazywater'))
        .addBands(prob_image.select('classification').arrayGet([5]).rename('hazyice'))
        .addBands(prob_image.select('classification').arrayGet([6]).rename('cloud'))
        .updateMask(waterMask_clipped.eq(1))
    )

    # land / snow classification via NDSI for land pixels only
    # land = 8, snow = 9 (matching AutomaticProcessing convention)

    NDSI = image.normalizedDifference(['green', 'swir1']).rename('NDSI')
    landSnow = NDSI.gt(0.4)
    landMask = waterMask_clipped.eq(0)

    land_class = (
        ee.Image(0)
        .where(landMask.And(landSnow.Not()), 8)
        .where(landMask.And(landSnow), 9)
        .rename('land_snow')
        .clip(geom)
    )

    land_vis = land_class.visualize(**{
        'min': 0,
        'max': 9,
        'palette': [
            '000000',  # 0: ocean/no data (black, won't be visible over water mask)
            '000000', '000000', '000000',
            '000000', '000000', '000000', '000000',
            'ca9161',  # 8: land
            'FFE8D1'   # 9: snow on land
        ]
    })

    return rgb, prob_bands, land_vis

# function four
# error image in case of failure

def create_error_image(region):
    return ee.Image(0).visualize(**{'palette': ['ff0000']}).paint(region, 1, 5)

# load assets

print('loading and processing assets')
grid = ee.FeatureCollection('projects/gee-personal-483416/assets/CoastCellInfoJan5_10')
samplesCollection = ee.FeatureCollection('projects/gee-personal-483416/assets/sample_images')
samplesList = samplesCollection.toList(samplesCollection.size()).getInfo()
print(f'processing {len(samplesList)} samples')

for feature in samplesList:

    sample = feature['properties']

    row  = sample['row']
    col  = sample['col']
    date = sample['date']

    # one subfolder per sample image - all outputs for this sample go here

    sample_folder = f'EarthEngineVisuals_{row}_{col}_{date}'

    # server side: find grid cell

    raw_feature = grid.filter(ee.Filter.And(
        ee.Filter.eq('Col', col),
        ee.Filter.eq('Row', row)
    )).first()

    dummy_feature = ee.Feature(None, {
        'Col': col, 
        'Row': row,
        'Lon1': 0, 'Lat1': 0, 'Lon2': 0, 'Lat2': 1,
        'Lon3': 1, 'Lat3': 1, 'Lon4': 1, 'Lat4': 0
    })
    safe_raw_feature = ee.Feature(ee.Algorithms.If(raw_feature, raw_feature, dummy_feature))

    cell_feature = coastal_polygon(safe_raw_feature)

    # set date window and retrieve best image

    end_date = ee.Date(date).advance(1, 'day').format('YYYY-MM-dd')
    cell_with_date = cell_feature.set('Start', date).set('End', end_date)

    img = ee.Image(image_clipping(cell_with_date))

    export_crs    = 'EPSG:3413'
    export_region = cell_feature.geometry()

    # shared export settings

    def make_export_params(image, description):
        return {
            'image':       image,
            'description': description,
            'folder':      sample_folder,
            'region':      export_region,
            'scale':       30,
            'crs':         export_crs,
            'fileFormat':  'GeoTIFF'
        }

    # generate visuals

    rgb, prob_bands, land_vis = generate_visuals(img)

    # export 1: true color

    ee.batch.Export.image.toDrive(**make_export_params(
        image=rgb,
        description=f'TrueColor_{row}_{col}_{date}'
    )).start()

    # export 2: land and snow map

    ee.batch.Export.image.toDrive(**make_export_params(
        image=land_vis,
        description=f'LandSnow_{row}_{col}_{date}'
    )).start()

    # export 3: one probability map per class, graduated blues

    for idx, band_name, class_label in CLASSES:

        prob_vis = prob_bands.select(band_name).visualize(**{
            'min':     0,
            'max':     1,
            'palette': BLUES_PALETTE
        })

        ee.batch.Export.image.toDrive(**make_export_params(
            image=prob_vis,
            description=f'Prob_{class_label.replace(" ", "")}_{row}_{col}_{date}'
        )).start()

        print(f'  queued: {class_label} probability map')

    print(f'all tasks started for sample {row}_{col}_{date} -> folder: {sample_folder}')

print('all samples submitted!')