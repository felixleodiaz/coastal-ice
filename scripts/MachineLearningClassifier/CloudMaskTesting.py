# CLOUD MASK TESTING SCRIPT
# exports per sample: NIR-SWIR-Blue, true color, and cloud mask visualizations
# at varying confidence levels to assess QA60 / QA_PIXEL performance

import ee

# initialize earth engine

project_id = 'gee-personal-483416'

try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

# colors for cloud mask visualization

MASKED_COLOR = '2D2B2A'
CLOUD_COLOR  = 'FFFFFF'

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
# gets best image from Landsat 8/9 and Sentinel 2 and clips to grid
# renames spectral bands to consistent names across sensors
# retains QA_PIXEL (Landsat) or QA60 (Sentinel) for cloud mask visualization

def image_clipping(grid):

    gridGeom  = grid.geometry()
    startdate = grid.get('Start')
    enddate   = grid.get('End')

    # get landsat 8 images TOA

    L8 = (
        ee.ImageCollection('LANDSAT/LC08/C02/T1_TOA')
        .filterDate(startdate, enddate)
        .filterBounds(gridGeom)
        .filter(ee.Filter.lt('CLOUD_COVER', 5))
        .map(lambda img: (
            img
            .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
            .addBands(img.select('QA_PIXEL'))
            .set('cloud', img.get('CLOUD_COVER'))
            .set('sensor', 'Landsat8')
            .set('area', img.geometry().intersection(gridGeom, 1).area())
        ))
    )

    # get landsat 9 images TOA

    L9 = (
        ee.ImageCollection('LANDSAT/LC09/C02/T1_TOA')
        .filterDate(startdate, enddate)
        .filterBounds(gridGeom)
        .filter(ee.Filter.lt('CLOUD_COVER', 5))
        .map(lambda img: (
            img
            .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
            .addBands(img.select('QA_PIXEL'))
            .set('cloud', img.get('CLOUD_COVER'))
            .set('sensor', 'Landsat9')
            .set('area', img.geometry().intersection(gridGeom, 1).area())
        ))
    )

    # get sentinel 2 images TOA

    S2 = (
        ee.ImageCollection('COPERNICUS/S2_HARMONIZED')
        .filterDate(startdate, enddate)
        .filterBounds(gridGeom)
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 5))
        .map(lambda img: (
            img
            .addBands(
                img.select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12']).divide(10000),
                overwrite=True
            )
            .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
            .addBands(img.select('QA60'))
            .set('cloud', img.get('CLOUDY_PIXEL_PERCENTAGE'))
            .set('sensor', 'Sentinel2')
            .set('area', img.geometry().intersection(gridGeom, 1).area())
        ))
    )

    # merge datasets

    merged  = L8.merge(L9).merge(S2)
    nimages = merged.size()

    # sort by lowest cloud fraction and pick the lowest

    merged = merged.sort('area', False).sort('cloud')

    # make sure not null, attach metadata and return

    safe_image = ee.Image(ee.Algorithms.If(
        nimages.gt(0),
        merged.first(),
        ee.Image.constant(0)
    ))

    best = safe_image.clip(gridGeom).set({
        'Row':               grid.get('Row'),
        'Column':            grid.get('Column'),
        'system:time_start': safe_image.get('system:time_start')
    })

    return ee.Image(ee.Algorithms.If(nimages.gt(0), best, None))

# function three
# generate cloud flag visualizations at varying confidence levels
# white = cloud, dark gray/brown = clear
# note: QA60 is known to have commission errors over bright Arctic surfaces

def get_cloud_mask_variants(img, sensor):

    is_sentinel = ee.String(sensor).compareTo('Sentinel2').eq(0)

    # sentinel 2: QA60 - bit 10 = opaque cloud, bit 11 = cirrus

    qa60      = img.select('QA60')
    s2_opaque = qa60.bitwiseAnd(1 << 10).gt(0)
    s2_cirrus = qa60.bitwiseAnd(1 << 11).gt(0)
    s2_both   = s2_opaque.Or(s2_cirrus)

    # landsat: QA_PIXEL - bit 3 = high conf cloud, bit 2 = cirrus, bit 4 = shadow

    qa       = img.select('QA_PIXEL')
    l_cloud  = qa.bitwiseAnd(1 << 3).gt(0)
    l_cirrus = qa.bitwiseAnd(1 << 2).gt(0)
    l_shadow = qa.bitwiseAnd(1 << 4).gt(0)
    l_both   = l_cloud.Or(l_cirrus)
    l_all    = l_cloud.Or(l_cirrus).Or(l_shadow)

    variants = {
        'OpaqueOnly':       ee.Image(ee.Algorithms.If(is_sentinel, s2_opaque, l_cloud)),
        'CirrusOnly':       ee.Image(ee.Algorithms.If(is_sentinel, s2_cirrus, l_cirrus)),
        'OpaquePlusCirrus': ee.Image(ee.Algorithms.If(is_sentinel, s2_both,   l_both)),
        'AllFlags':         ee.Image(ee.Algorithms.If(is_sentinel, s2_both,   l_all)),
    }

    return {
        name: flag.rename('cloud_flag').visualize(**{
            'min':     0,
            'max':     1,
            'palette': [MASKED_COLOR, CLOUD_COLOR]
        })
        for name, flag in variants.items()
    }

# load assets

print('loading assets...')
grid              = ee.FeatureCollection('projects/gee-personal-483416/assets/CoastCellInfoJan5_10')
samplesCollection = ee.FeatureCollection('projects/gee-personal-483416/assets/sample_images')
samplesList       = samplesCollection.toList(samplesCollection.size()).getInfo()
print(f'processing {len(samplesList)} samples')

for feature in samplesList:

    sample = feature['properties']
    row    = sample['row']
    col    = sample['col']
    date   = sample['date']

    # dynamic folder per sample

    sample_folder = f'CloudMask_{row}_{col}_{date}'

    # find grid cell

    raw_feature = grid.filter(ee.Filter.And(
        ee.Filter.eq('Col', col),
        ee.Filter.eq('Row', row)
    )).first()

    dummy_feature    = ee.Feature(None, {'Col': col, 'Row': row})
    safe_raw_feature = ee.Feature(ee.Algorithms.If(raw_feature, raw_feature, dummy_feature))
    cell_feature     = coastal_polygon(safe_raw_feature)

    end_date       = ee.Date(date).advance(1, 'day').format('YYYY-MM-dd')
    cell_with_date = cell_feature.set('Start', date).set('End', end_date)

    # get best image with renamed spectral bands and QA band retained

    img           = ee.Image(image_clipping(cell_with_date))
    sensor        = img.get('sensor')
    export_region = cell_feature.geometry()
    export_crs    = 'EPSG:3413'

    # shared export helper

    def export(image, description):
        ee.batch.Export.image.toDrive(
            image=image,
            description=description,
            folder=sample_folder,
            region=export_region,
            scale=30,
            crs=export_crs,
            fileFormat='GeoTIFF'
        ).start()
        print(f'  queued: {description}')

    # export 1: true color RGB

    export(img.visualize(**{
        'bands': ['red', 'green', 'blue'],
        'min':   0,
        'max':   0.3,
        'gamma': 1.4
    }), f'TrueColor_{row}_{col}_{date}')

    # export 2: NIR-SWIR-Blue false color

    export(img.visualize(**{
        'bands': ['nir', 'swir1', 'blue'],
        'min':   0,
        'max':   0.4,
        'gamma': 1.5
    }), f'FalseColor_{row}_{col}_{date}')

    # export 3-6: cloud mask variants at varying confidence levels

    for variant_name, variant_img in get_cloud_mask_variants(img, sensor).items():
        export(variant_img, f'CloudMask_{variant_name}_{row}_{col}_{date}')

    print(f'all tasks started for {row}_{col}_{date} -> {sample_folder}')

print('all samples submitted!')