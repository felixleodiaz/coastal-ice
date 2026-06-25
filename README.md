## Coastal Sea Ice Detection Algorithm Comparison
# Felix Diaz and Alice Bradley

This is data analysis for my senior thesis at Williams College, Massachussets, advised by Dr Alice Bradley. I am looking at error between Passive Microwave (PMW) coastal sea ice products and visual data derived in Google Earth Engine. Scripts used for the project can be found in scripts/ and figures are sorted into folders in figures/. The small datasets  I could fit into this repository can be found in data/ alongside cleaned dataframes with final error calculations.

## How to use this code

The code in this repo is organized in the scripts/ folder. Inside the folder all of the scripts needed to run the machine learning classification are inside the MachineLearningClassifier/ folder and all of the scripts needed to download NASA Team and Bootstrap, and run the data analysis we did, are in the PassiveMicrowaveComparison/ folder. Below are two sections describing in detail how to run the scripts in these two folders. 

Before doing anything please ensure you have figures/, scripts/, environments/, and data/, and create alongside them a directory called local_data/. Once you have that you are all set. Thanks for reading!

# How to Run the Machine Learning Classifier

This code needs a Google Earth Engine account. One can be aquired at https://www.code.earthengine.google.com. Below are the steps to run the classifier.

The script ClassifySpectra.js is the script we use to create a training dataset. It should be copied and paste into a file online at code.earthengine.google.com. Then follow the below steps.

1. Run the script once with a valid row, col, and time that you know contains a satellite image from Landsat 8/9 or Sentinel-2. The image will pop up in the map environment in google earth engine.
2. Create a polygon feature collection by clicking the polygon icon in the top left of the screen.
3. It will be called "geometry" by default so change that to either "Ice" "Melt" "Water" or "ThinIce" and change the type to FeatureCollection. Both of these options are accessed in the setting menu of that layer.
4. Click on the image to draw polygons once the layer has been setup. Repeat these steps as necessary for all of the surface cover types in the image.
5. Once you are done save using the button at the top of the script and rerun the script. It will now recognize that features have been drawn and pull the pixels underneath the polygons. Run the two tasks in the task manager panel. 
6. To repeat on another image wait until the tasks have finished then delete the FeatureCollection layers. Run the task again with a new row, col, and time combination and classify again.

You will also need a water mask. This can be created using the CreateWaterMask.js script. This should also be copied and pasted into a file on code.earthengine.google.com. Run the script and ensure that connected_water_mask_2015 is saved to your assets.

Once you have a training dataset and water mask, you can create the random forest model by running the optuna optimization script called CreateRandomForest.py. This will save the best parameters into the same folder in a text file called best_gee_rf_params.txt and produce a testing dataset on which it will test the model. You can also just use our best parameters and skip this step.

The AutomaticProcessing.py script is the main processing script of this analysis. To run this script you will need to have the lgb-env conda environment activated. The yaml for this environment can be found in the environments/ folder of this repo. When running this script specify the year of data you want to classify at the end of the command e.g. python AutomaticProcessing 2026. 

Repeat this for all the years you want to classify. We use 2013 to 2025. Finally, save the folder AutomaticProcessingResults which is now in your google drive to a folder called local_data which you should create in the main coastal-ice folder.

# How to Calculate Errors

For this you need the classified results calculated in the above section saved in the folder  coastal-ice/local_data/AutomaticProcessingResults/. You also need to have the coastal-ice conda environment activated. You can again find the yaml for this in the environments/ directory.

Before running the actual error calculation script and making figures you need a couple dependancies. Below we list them and explain where to find them or how to calculate them.

1. First you need a distance to land raster in the local_data folder. To calculate this run the DistanceToLand.py script. 
2. Next you need a file that list the latitude and longitudes of every coastal NSIDC grid cell in the arctic. We have saved this to google drive in a file called CoastCellInfoJan5_10.csv. Here is the link to download that: 
3. You also need a NASA Earthaccess account. You can find info on how to get one of those here: https://www.earthdata.nasa.gov/data/tools/earthaccess

Now run the ValidationComparison.py script with the coastal-ice environment activated. You will be prompted to enter a start and end date. We suggest chunking your analysis into sections of about one year. For our analysis we entered 01/01/year as a start date and 12/31/year as the end date. The results are saved into the local_data directory in a folder called DataFrames/ so make sure that is created before running the script.

Lastly, the ErrorAnalysis.ipynb notebook runs you through all of the figures we created for this thesis.

# Read our analysis

We have saved the thesis and working paper to this repo! Please enjoy











