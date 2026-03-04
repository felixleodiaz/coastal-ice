# initialize earth engine

import ee

project_id = 'gee-personal-483416'

try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

# image clipping function for automatic processing and image gen pipelines

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
                .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
                .addBands(
                    img.select('QA_PIXEL').bitwiseAnd(1 << 3).neq(0)
                    .And(img.select('QA_PIXEL').rightShift(8).bitwiseAnd(3).eq(3))
                    .rename('cloud_qa')
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
                .select(['B2', 'B3', 'B4', 'B5', 'B6', 'B7'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
                .addBands(
                    img.select('QA_PIXEL').bitwiseAnd(1 << 3).neq(0)
                    .And(img.select('QA_PIXEL').rightShift(8).bitwiseAnd(3).eq(3))
                    .rename('cloud_qa')
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
                .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'], ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'])
                .addBands(
                    img.select('QA60').bitwiseAnd(1 << 10).neq(0)
                    .rename('cloud_qa')
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