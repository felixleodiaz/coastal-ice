// MANUAL CLASSIFICATION OF SATELLITE IMAGERY
// also good for finding row / col and superimposing sat imagery on a given date

var targetRow  = 42;
var targetCol  = 167;
var targetDate = '2024-01-22'

// imports

var tools = require('users/fld1/visual_coastal_sea_ice:ProcessingFunctions');
var grid = ee.FeatureCollection('projects/gee-personal-483416/assets/CoastCellInfoJan5_10'); 

// safety check for empty layers

var icePoly = typeof Ice !== 'undefined' ? Ice : ee.FeatureCollection([]);
var waterPoly = typeof Water !== 'undefined' ? Water : ee.FeatureCollection([]);
var meltPoly = typeof Melt !== 'undefined' ? Melt : ee.FeatureCollection([]);
var thinIcePoly = typeof ThinIce !== 'undefined' ? ThinIce : ee.FeatureCollection([]);
var hazyWaterPoly = typeof HazyWater !== 'undefined' ? HazyWater : ee.FeatureCollection([]);
var hazyIcePoly = typeof HazyIce !== 'undefined' ? HazyIce : ee.FeatureCollection([]);
var cloudPoly = typeof Cloud !== 'undefined' ? Cloud : ee.FeatureCollection([]);

// create grid polygon and build geometry

var coastalpolygon = function(feature) {
  var coordinates = [
    [feature.get('Lon1'), feature.get('Lat1')],
    [feature.get('Lon2'), feature.get('Lat2')],
    [feature.get('Lon3'), feature.get('Lat3')],
    [feature.get('Lon4'), feature.get('Lat4')]
  ]; 
  return ee.Feature(ee.Geometry.Polygon([coordinates]),
    {'Column': feature.get('Col'), 'Row': feature.get('Row')});
}

var rawFeature = grid.filter(ee.Filter.and(
    ee.Filter.eq('Row', targetRow),
    ee.Filter.eq('Col', targetCol)
  )).first();

var cellFeature = coastalpolygon(rawFeature);

// set the date window

var startDate = targetDate;
var endDate = ee.Date(startDate).advance(1, 'day').format('YYYY-MM-dd');

// update the grid feature with dates

var cellWithDate = cellFeature.set('Start', startDate).set('End', endDate);

// get satellite image

var image = tools.imageClippingToGrid(cellWithDate);

// visualization
  
var vizParams = {
  bands: ['nir', 'swir1', 'blue'],
  min: 0,
  max: 0.4,
  gamma: 1.5
};

var exportParams = {
  bands: ['swir1', 'nir', 'blue'],
  min: 0,
  max: 0.4,
  gamma: 1.5
};

Map.centerObject(cellFeature, 10);
Map.addLayer(image, vizParams, 'False Color Image');
Map.addLayer(image, exportParams, 'True-er Color Image');

// polygon extraction logic

var mergePolygons = function() {
  var s1 = icePoly.map(function(f) { return f.set('class', 'Ice').set('class_id', 1); });
  var s2 = meltPoly.map(function(f) { return f.set('class', 'Melt').set('class_id', 2); });
  var s3 = waterPoly.map(function(f) { return f.set('class', 'Water').set('class_id', 3); });
  var s4 = thinIcePoly.map(function(f) { return f.set('class', 'ThinIce').set('class_id', 4); });
  var s5 = hazyWaterPoly.map(function(f) { return f.set('class', 'HazyWater').set('class_id', 5); });
  var s6 = hazyIcePoly.map(function(f) { return f.set('class', 'HazyIce').set('class_id', 6); });
  var s7 = cloudPoly.map(function(f) { return f.set('class', 'Cloud').set('class_id', 7); });
  return ee.FeatureCollection([s1, s2, s3, s4, s5, s6, s7]).flatten();
};

var allPolygons = mergePolygons();

// fetch information

var dataFetch = ee.Dictionary({
  count: allPolygons.size(),
  sensor: image.get('sensor')
});

// check if any polygons were drawn

dataFetch.evaluate(function(result) {
  
  // unwrap the results from the dictionary
  
  var polyCount = result.count;
  var sensorName = result.sensor || 'UnknownSensor';

  if (polyCount > 0) {
    print('Polygons found! Extracting spectra from ' + sensorName + '...');
    
    // band names
    
    var bandsToSample = ['blue', 'green', 'red', 'nir', 'swir1', 'swir2'];
    
    var samples = image.select(bandsToSample)
      .sampleRegions({
         collection: allPolygons,
         properties: ['class', 'class_id'], 
         scale: 30, 
         geometries: true 
      });
    
    // export pixel data
    
    var desc = 'Spectra_' + sensorName + '_R' + targetRow + '_C' + targetCol + '_' + targetDate;
    Export.table.toDrive({
      collection: samples,
      description: desc,
      folder: 'Training_Data_Data',
      fileNamePrefix: 'Spectra_' + sensorName + '_R' + targetRow + '_C' + targetCol + '_' + targetDate,
      fileFormat: 'CSV',
      selectors: ['class', 'class_id', 'blue', 'green', 'red', 'nir', 'swir1', 'swir2', '.geo']
    });
    
    // export and create reference image
    
    var outlines = ee.Image(0).mask(0).paint({
      featureCollection: allPolygons,
      color: 'class_id',
      width: 2
    });

    var linePalette = ['d55e00', 'de8f05', 'ca9161', '029e73', 'cc78bc', '0173b2', '000000'];
    
    var referenceImage = image.visualize(exportParams)
        .blend(outlines.visualize({palette: linePalette, min: 1, max: 7}));

    Export.image.toDrive({
      image: referenceImage,
      description: 'RefMap_' + sensorName + '_R' + targetRow + '_C' + targetCol + '_' + targetDate,
      folder: 'Training_Data_Images',
      fileNamePrefix: 'RefMap_' + sensorName + '_R' + targetRow + '_C' + targetCol + '_' + targetDate,
      region: cellFeature.geometry(),
      scale: 30,
      crs: 'EPSG:3413'
    });
    
    print('Thanks! Exporting task: ' + desc);
    
  } else {
    print('Input needed. Please draw polygons and run again.');
  }
});