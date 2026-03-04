# Create Training Asset for Random Forest Classifier
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

# concatenate all data and remap Hazy classes to their base classes (eg Hazy Ice to Ice)

data = pd.concat((dataL8, dataL9, dataS2), ignore_index=True)
data['class_id'] = data['class_id'].map({
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 3,
    6: 1
})

# save data to csv

data.to_csv('../../local_data/RFTrainingAsset.csv', index=False)
print(f"Current class IDs are {data.class_id.unique()}")

# calculate and print class distribution

class_counts = data['class_id'].value_counts().sort_index()
class_percentages = (class_counts / class_counts.sum()) * 100
print("Class Distribution:")
for class_id, count in class_counts.items():
    percentage = class_percentages[class_id]
    print(f"Class {class_id}: {count} samples ({percentage:.2f}%)")

# and get stratified sample

SAMPLES_PER_CLASS = 16000
print(f"\ngenerating {SAMPLES_PER_CLASS} sample stratified subset")

# get sample

sampled_data = data.groupby('class_id', group_keys=False).apply(
    lambda x: x.sample(n=min(len(x), SAMPLES_PER_CLASS), random_state=RANDOM_SEED)
)

# shuffle final dataset and save to csv

sampled_data = sampled_data.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
sampled_path = '../../local_data/training_asset_sample.csv'
sampled_data.to_csv(sampled_path, index=False)

# verify class distribution of subset

sampled_counts = sampled_data['class_id'].value_counts().sort_index()
print("\nSubset Class Distribution:")
for class_id, count in sampled_counts.items():
    print(f"Class {class_id}: {count} samples")

print(f"\nSuccess! Total subset rows: {len(sampled_data)}. Saved to {sampled_path}")