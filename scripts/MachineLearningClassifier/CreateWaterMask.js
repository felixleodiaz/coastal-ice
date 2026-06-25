// import mask and QA layer from MODIS

var rawWaterMask = ee.ImageCollection('MODIS/006/MOD44W')
                  .filter(ee.Filter.date('2015-01-01', '2015-05-01'))
                  .select('water_mask').mosaic().eq(1);
                  

var rawOceanMask = ee.ImageCollection('MODIS/006/MOD44W')
                  .filter(ee.Filter.date('2015-01-01', '2015-05-01'))
                  .select('water_mask_QA')
                  .mosaic()
                  .remap([4, 5], [1, 1], 0)
                  .eq(1);

                  

// RUN MORPHOLOGICAL RECONSTRUCTION

var confirmedSeeds = rawWaterMask.and(rawOceanMask);

// create blobs with max size = 1024 pixels (the max this function will take)
// any water body smaller than this gets an id. 
// any water body larger than this gets masked out and set to 1 later

var maxSize = 1024;
var rawWaterBlobs = rawWaterMask.selfMask()
    .connectedComponents({
        connectedness: ee.Kernel.plus(1),
        maxSize: maxSize
    });

// check for blobs containing a "seed pixel" (1 if blob touches seed, 0 if note)

var blobAnalysis = confirmedSeeds.addBands(rawWaterBlobs.select('labels'));

var blobHasSeed = blobAnalysis.reduceConnectedComponents({
    reducer: ee.Reducer.max(),
    labelBand: 'labels'
});

// if a blob was so big that connectedComponents turned it into NA assume it's ocean

var largeWater = rawWaterMask.eq(1).and(rawWaterBlobs.select('labels').mask().not());

// create final mask

var finalOceanMask = blobHasSeed.unmask(0)
    .or(largeWater)
    .selfMask();

// VISUALIZE

Map.addLayer(rawWaterMask.selfMask(), {palette: ['red']}, 'raw water');
Map.addLayer(confirmedSeeds.selfMask(), {palette: ['green']}, 'confirmed seeds');
Map.addLayer(finalOceanMask.selfMask(), {palette: ['blue']}, 'ocean mask');

// EXPORT ARCTIC
// define the Arctic as everything North of 50 degrees

var arcticGeom = ee.Geometry.Rectangle({
  coords: [-180, 50, 180, 90],
  proj: 'EPSG:4326',
  geodesic: false
});

// export increased to 10 trillion pixels to ensure it fits the whole arctic

Export.image.toAsset({
  image: finalOceanMask,            
  description: 'Export_Water_Mask_Arctic',
  assetId: 'connected_water_mask_arctic_2015',
  scale: 250,
  region: arcticGeom,
  maxPixels: 1e13 
});

// check water mask

// load your water mask assset

var waterMask = ee.Image('projects/gee-personal-483416/assets/connected_water_mask_2015');

// visualization parameters
// 0 is Land and Gray
// 1 is Water and Blue
var visParams = {
  min: 0,
  max: 1,
  palette: ['gray', '0000FF'] // Land is gray, Water is bright blue
};

// Center the Map
Map.centerObject(waterMask, 4); 

// Add the Layer
// use 'randomVisualizer' for the raw check to see everything
// and the 'visParams' to see the logical classification.

Map.addLayer(waterMask, visParams, 'Water Mask (Blue=Water)');

// we also add a "Safety" layer to see where the mask might be Null/Missing
// If this layer shows up in Red, it means the mask has no data there which crashes calculations

var nullMask = waterMask.unmask(-9999).eq(-9999);
Map.addLayer(nullMask.updateMask(nullMask), {palette:['red']}, 'Missing/Null Data (Red)', false);

print('Water Mask Info:', waterMask);