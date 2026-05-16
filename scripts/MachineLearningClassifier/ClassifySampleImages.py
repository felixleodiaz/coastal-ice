# IMAGE GENERATION PIPELINE
import ee
import pandas as pd
from ImageClipping import image_clipping

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

waterMask = ee.Image('projects/gee-personal-483416/assets/connected_water_mask_2015').unmask(0)

CLASSES = [
    (0, 'seaice',  'Sea Ice'),
    (1, 'melt',    'Melt Ponds'),
    (2, 'water',   'Water'),
    (3, 'thinice', 'Thin Ice'),
]

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
        'Row':    feature.get('Row')
    })
    return gridbox

# function two
# generate true color and per-class probability images

def generate_visuals(image):
    image = ee.Image(image)
    geom  = image.geometry()

    image_unmasked = image

    cloud_qa = image.select('cloud_qa')
    image = image.updateMask(cloud_qa.Not())

    sensor_name = ee.String(image.get('sensor'))
    sensor_val  = ee.Number(ee.Algorithms.If(sensor_name.compareTo('Sentinel2').eq(0), 1, 0))
    sensor_band = ee.Image.constant(sensor_val).rename('sensor').toByte()
    image = image.addBands(sensor_band)

    rgb = image_unmasked.visualize(**{
        'bands': ['red', 'green', 'blue'],
        'min':   0,
        'max':   0.4,
        'gamma': 1.5
    })

    waterMask_clipped = waterMask.clip(geom)

    prob_image = image.classify(classifier)

    prob_bands = (
        prob_image.select('classification').arrayGet([0]).rename('seaice')
        .addBands(prob_image.select('classification').arrayGet([1]).rename('melt'))
        .addBands(prob_image.select('classification').arrayGet([2]).rename('water'))
        .addBands(prob_image.select('classification').arrayGet([3]).rename('thinice'))
        .updateMask(waterMask_clipped.eq(1))
    )

    NDSI     = image.normalizedDifference(['green', 'swir1']).rename('NDSI')
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
            '000000',
            '000000', '000000', '000000',
            '000000', '000000', '000000', '000000',
            'ca9161',
            'FFE8D1'
        ]
    })

    return rgb, prob_bands, land_vis

# function three
# error image in case of failure

def create_error_image(region):
    return ee.Image(0).visualize(**{'palette': ['ff0000']}).paint(region, 1, 5)

# function four
# makes export parameters for a given image and region

def make_export_params(image, description, export_region):
    return {
        'image':       image,
        'description': description,
        'folder':      'HighErrorSubsetImages',
        'region':      export_region,
        'scale':       30,
        'crs':         'EPSG:3413',
        'fileFormat':  'GeoTIFF'
    }

# iterate through samples
# generates images
# submits export tasks to GEE

if __name__ == '__main__':
    sample_df = pd.read_csv('../../local_data/high_errors_subset_all_years.csv')
    grid = ee.FeatureCollection('projects/gee-personal-483416/assets/CoastCellInfoJan5_10')

    print(f'processing {len(sample_df)} samples:')

    for _, row_data in sample_df.iterrows():
        row  = row_data['row']
        col  = row_data['col']
        date = row_data['time']

        try:
            raw_feature = grid.filter(ee.Filter.And(
                ee.Filter.eq('Col', col),
                ee.Filter.eq('Row', row)
            )).first()

            dummy_feature = ee.Feature(None, {
                'Col': col, 'Row': row,
                'Lon1': 0, 'Lat1': 0, 'Lon2': 0, 'Lat2': 1,
                'Lon3': 1, 'Lat3': 1, 'Lon4': 1, 'Lat4': 0
            })
            safe_feature = ee.Feature(ee.Algorithms.If(raw_feature, raw_feature, dummy_feature))

            cell_feature = coastal_polygon(safe_feature)

            end_date     = ee.Date(date).advance(1, 'day').format('YYYY-MM-dd')
            cell_with_date = cell_feature.set('Start', date).set('End', end_date)

            img           = ee.Image(image_clipping(cell_with_date))
            export_region = cell_feature.geometry()

            rgb, prob_bands, land_vis = generate_visuals(img)

        except Exception as e:
            print(f'  Failed for {row}/{col}/{date}: {e} — skipping')
            continue

        print(f'submitting tasks for row: {row}, col: {col}, date: {date}')

        ee.batch.Export.image.toDrive(**make_export_params(
            rgb, f'TrueColor_{row}_{col}_{date}', export_region
        )).start()

        ee.batch.Export.image.toDrive(**make_export_params(
            land_vis, f'LandSnow_{row}_{col}_{date}', export_region
        )).start()

        for idx, band_name, class_label in CLASSES:
            prob_vis = prob_bands.select(band_name).visualize(**{
                'min': 0, 'max': 1, 'palette': BLUES_PALETTE
            })
            ee.batch.Export.image.toDrive(**make_export_params(
                prob_vis, f'Prob_{class_label.replace(" ", "")}_{row}_{col}_{date}', export_region
            )).start()

    print('all tasks submitted to GEE!')