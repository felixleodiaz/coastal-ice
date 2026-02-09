# import libraries

import ee
import optuna
import joblib

# initialize earth engine

project_id = 'gee-personal-483416'

try:
    ee.Initialize(project=project_id)
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project=project_id)

# constants and datasetup

RANDOM_SEED = 12
ASSET_ID = "projects/gee-personal-483416/assets/SmileTrainingAsset"

features = [
    "red", 
    "green",
    "blue", 
    "nir", 
    "swir1", 
    "swir2", 
    "sensor"
]
target = "class_id"

# define these for later

specific_class_ids = [1, 2, 3, 4, 5, 6]
class_names = ['Ice', 'Water', 'Melt Ponds', 'Hazy Water', 'Hazy Ice', 'Clouds']

# load asset

print(f"Loading asset {ASSET_ID}")
raw_data = ee.FeatureCollection(ASSET_ID)

# split data to avoid leakage later

data_with_folds = raw_data.randomColumn(columnName='fold_rand', seed=RANDOM_SEED)
train_data = data_with_folds.filter(ee.Filter.lt('fold_rand', 0.8))
test_data = data_with_folds.filter(ee.Filter.gte('fold_rand', 0.8))

# create stratified random sample for optuna (1 million datapoints ran into time constraints from GEE)

SAMPLES_PER_CLASS = 5000
stratified_subsets = []

for class_id in specific_class_ids:

    class_collection = train_data.filter(ee.Filter.eq(target, class_id))
    
    subset = class_collection.randomColumn('subsample_rand', seed=RANDOM_SEED)\
        .sort('subsample_rand')\
        .limit(SAMPLES_PER_CLASS)
    
    stratified_subsets.append(subset)

opt_data = stratified_subsets[0]
for i in range(1, len(stratified_subsets)):
    opt_data = opt_data.merge(stratified_subsets[i])

print("Asset loaded, 80/20 train/test split created, and SRS for optuna generated")

# define Optuna objective

def objective(trial):

    # hyperparameter search space

    params = {
        "kernelType": "RBF",
        "cost": trial.suggest_float("cost", 1e-1, 1e3, log=True),
        "gamma": trial.suggest_float("gamma", 1e-2, 1e2, log=True)
    }

    # three fold cross val and randomization col creation

    cv_col = 'inner_cv_rand'
    opt_data_cv = opt_data.randomColumn(columnName=cv_col, seed=trial.number)

    fold_accuracies = []
    splits = [0.0, 0.33, 0.66, 1.0]

    for i in range(3):
        lower_bound = splits[i]
        upper_bound = splits[i+1]

        val_set = opt_data_cv.filter(
            ee.Filter.And(
                ee.Filter.gte('inner_cv_rand', lower_bound),
                ee.Filter.lt('inner_cv_rand', upper_bound)
            )
        )

        train_set = opt_data_cv.filter(
            ee.Filter.Or(
                ee.Filter.lt('inner_cv_rand', lower_bound),
                ee.Filter.gte('inner_cv_rand', upper_bound)
            )
        )

        # train classifier

        classifier = ee.Classifier.libsvm(**params)\
            .train(
                features=train_set,
                classProperty=target,
                inputProperties=features
            )

        # validate

        validated = val_set.classify(classifier)
        
        # calculate accuracy

        accuracy = validated.errorMatrix(target, 'classification').accuracy().getInfo()
        fold_accuracies.append(accuracy)

    # return mean accuracy across 3 folds

    mean_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    return mean_accuracy

# run study

print("Starting Optuna optimization")

study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
study.optimize(objective, n_trials=40) 

print("Best params from Optuna:", study.best_params)
print("Best accuracy achieved:", study.best_value)

# map params back to GEE useable values for easy use later

final_gee_params = {
    "kernelType": "RBF",
    "cost": study.best_params["cost"],
    "gamma": study.best_params["gamma"]
}

# save study object

joblib.dump(study, "optuna_gee_svm_study.pkl")
print("study saved to 'optuna_gee_svm_study.pkl'")

# save interpretable params

with open("best_gee_svm_params.txt", "w") as f:
    f.write(str(final_gee_params))
print("best parameters saved to 'best_gee_svm_params.txt'")


# train a diagnostic model using best params

final_gee_params = {
    "kernelType": "RBF",
    "cost": study.best_params["cost"],
    "gamma": study.best_params["gamma"]
}

eval_classifier = ee.Classifier.libsvm(**final_gee_params)\
    .train(
        features=train_data,
        classProperty=target,
        inputProperties=features
    )

# Classify the test set

validated_test = test_data.classify(eval_classifier)

# TASK 1
# export predictions to csv

export_columns = [target, 'classification', 'fold_rand'] 

task_predictions = ee.batch.Export.table.toDrive(
    collection=validated_test.select(export_columns),
    folder = 'EarthEngineResults',
    description='TestDataPredictionsSVM',
    fileFormat='CSV'
)
task_predictions.start()
print("task started and saving to 'EarthEngineResults.csv'")

# TASK 2
# export final model to asset

final_classifier = ee.Classifier.libsvm(**final_gee_params)\
    .train(
        features=raw_data,
        classProperty=target,
        inputProperties=features
    )

classifier_asset_id = 'projects/gee-personal-483416/assets/svm_seaice_classifier'

task_model = ee.batch.Export.classifier.toAsset(
    final_classifier, 
    'Saved-svm-IGBP-classification', 
    classifier_asset_id
)
task_model.start()

print(f"/nstarted task to save final classifier to {classifier_asset_id}")
print('Done!')