# import libraries

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 12
np.random.seed(RANDOM_SEED)

# get rid of the map() for class_id once all classes are in use
# read in files L8 = 0

folderpath = '../../local_data/Training_Data/'
paths = Path(folderpath).glob('*Landsat8*.csv')
pathlist = list(paths)
dataL8 = pd.concat(map(pd.read_csv, pathlist), ignore_index=True)
dataL8['sensor'] = 0

# read in files L9 = 1

folderpath = '../../local_data/Training_Data/'
paths = Path(folderpath).glob('*Landsat9*.csv')
pathlist = list(paths)
dataL9 = pd.concat(map(pd.read_csv, pathlist), ignore_index=True)
dataL9['sensor'] = 0

# # read in files S2 = 2

folderpath = '../../local_data/Training_Data/'
paths = Path(folderpath).glob('*_Sentinel2_*.csv')
pathlist = list(paths)
dataS2 = pd.concat(map(pd.read_csv, pathlist), ignore_index=True)
dataS2['sensor'] = 1

data = pd.concat((dataL8, dataL9, dataS2), ignore_index=True)
data.to_csv('../../local_data/RFTrainingAsset.csv', index=False)

# ensure that we have all the class IDs

print(f"Current class IDs are {data.class_id.unique()}")