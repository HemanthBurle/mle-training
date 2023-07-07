import argparse
import logging
import os
import pickle

import numpy as np
from scipy.stats import randint
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

parser = argparse.ArgumentParser()
parser.add_argument(
    "-i",
    "--input",
    help="Please give the appropriate input path for the CSV file",
    # choices=["datasets"],
    default="datasets",
)
parser.add_argument(
    "-o",
    "--output",
    help="Please give the appropriate output path for the pickle file",
    # choices=["artifacts"],
    default="artifacts",
)
parser.add_argument(
    "--loglevel",
    help="Please specify the log level",
    choices=["INFO", "DEBUG", "ERROR", "CRITICAL", "WARNING"],
    default="INFO",
)
parser.add_argument(
    "--logpath",
    help="Please specify the directory for the log file ",
    # choices=["", "logs"],
    default="logs",
)
args = parser.parse_args()
os.makedirs("../" + args.logpath, exist_ok=True)
log_dict = {
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "WARNING": logging.WARNING,
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
}
if args.logpath == "":
    file_name = ""
else:
    file_name = "./" + args.logpath + "/train.log"
logging.basicConfig(
    level=log_dict[args.loglevel],
    filename=file_name,
    format="%(asctime)s-%(levelname)s-[%(filename)s:%(lineno)s]  %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


ARTIFACTS_PATH = os.path.join("./" + args.output)
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "housing_prepared.pkl")
with open(PICKLE_PATH, "rb") as file:
    housing_prepared = pickle.load(file)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "housing_labels.pkl")
with open(PICKLE_PATH, "rb") as file:
    housing_labels = pickle.load(file)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "strat_train_set.pkl")
with open(PICKLE_PATH, "rb") as file:
    strat_train_set = pickle.load(file)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "strat_test_set.pkl")
with open(PICKLE_PATH, "rb") as file:
    strat_test_set = pickle.load(file)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "X_test_prepared.pkl")
with open(PICKLE_PATH, "rb") as file:
    X_test_prepared = pickle.load(file)

PICKLE_PATH = os.path.join(ARTIFACTS_PATH, "y_test.pkl")
with open(PICKLE_PATH, "rb") as file:
    y_test = pickle.load(file)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)


housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_mae = mean_absolute_error(housing_labels, housing_predictions)

output_file = os.path.join(ARTIFACTS_PATH, "lin_reg_op.pkl")
with open(output_file, "wb") as f:
    pickle.dump([lin_rmse, lin_mae], f)


tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)

output_file = os.path.join(ARTIFACTS_PATH, "tree_reg_op.pkl")
with open(output_file, "wb") as f:
    pickle.dump(tree_rmse, f)

param_distribs = {
    "n_estimators": randint(low=1, high=200),
    "max_features": randint(low=1, high=8),
}

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(
    forest_reg,
    param_distributions=param_distribs,
    n_iter=10,
    cv=5,
    scoring="neg_mean_squared_error",
    random_state=42,
)
rnd_search.fit(housing_prepared, housing_labels)
cvres = rnd_search.cv_results_

output_file = os.path.join(ARTIFACTS_PATH, "cvres.pkl")
with open(output_file, "wb") as f:
    pickle.dump(cvres, f)

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {"bootstrap": [False], "n_estimators": [3, 10], "max_features": [2, 3, 4]},
]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training
grid_search = GridSearchCV(
    forest_reg,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error",
    return_train_score=True,
)
grid_search.fit(housing_prepared, housing_labels)

grid_search.best_params_
cvres = grid_search.cv_results_

output_file = os.path.join(ARTIFACTS_PATH, "new_cvres.pkl")
with open(output_file, "wb") as f:
    pickle.dump(cvres, f)

feature_importances = grid_search.best_estimator_.feature_importances_
sorted(zip(feature_importances, housing_prepared.columns), reverse=True)


final_model = grid_search.best_estimator_

final_predictions = final_model.predict(X_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)

output_file = os.path.join(ARTIFACTS_PATH, "final_rmse.pkl")
with open(output_file, "wb") as f:
    pickle.dump(final_rmse, f)

logging.info("The train script ran successfully")
