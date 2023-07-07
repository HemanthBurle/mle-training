import argparse
import logging
import os
import pickle
import tarfile

import numpy as np
import pandas as pd
from six.moves import urllib
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--CSVPath",
    help="Please specify the output directory",
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
os.makedirs("../../" + args.logpath, exist_ok=True)
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
    file_name = "../../" + args.logpath + "/ingest_data.log"
logging.basicConfig(
    level=log_dict[args.loglevel],
    filename=file_name,
    format="%(asctime)s-%(levelname)s-[%(filename)s:%(lineno)s]  %(message)s",
    datefmt="%d-%b-%y %H:%M:%S",
)


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("../../" + args.CSVPath, "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    """This function fetches data from web and is converted to a CSV File

    Parameters
    ----------
    housing_url : string
                URL of the website where the data is there

    housing_path: string
                Path in which we need to store the CSV File

    Returns
    -------
                Creates a Directory and stores the file as "housing.csv"

    """

    os.makedirs(HOUSING_PATH, exist_ok=True)
    tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=HOUSING_PATH)
    housing_tgz.close()


fetch_housing_data()


def load_housing_data(housing_path=HOUSING_PATH):
    """This converts the csv file into pandas dataframe

    Parameters
    ----------
    download_root   : string
                    URL to the dataset
    housing_path    : string
                    Directory in which the CSV file resides

    Returns
    -------
                Returns a pandas DataFrame of the dataset

    """
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


housing = load_housing_data()

ARTIFACTS_PATH = os.path.join("../../", args.output)
os.makedirs(ARTIFACTS_PATH, exist_ok=True)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = pd.cut(
    housing["median_income"],
    bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
    labels=[1, 2, 3, 4, 5],
)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


def income_cat_proportions(data):
    """This function distrbutes data by income into catogories

    Parameters
    ----------
    data    :   pandas dataframe
            This dataframe should already have column "income_cat"

    Returns
    -------
            A series containing the ratios of the categorical income

    """
    return data["income_cat"].value_counts() / len(data)


def data_transformation():
    global housing
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    compare_props = pd.DataFrame(
        {
            "Overall": income_cat_proportions(housing),
            "Stratified": income_cat_proportions(strat_test_set),
            "Random": income_cat_proportions(test_set),
        }
    ).sort_index()

    compare_props["Rand. %error"] = (
        100 * compare_props["Random"] / compare_props["Overall"] - 100
    )
    compare_props["Strat. %error"] = (
        100 * compare_props["Stratified"] / compare_props["Overall"] - 100
    )

    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    output_file = os.path.join(ARTIFACTS_PATH, "strat_train_set.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(strat_train_set, f)

    output_file = os.path.join(ARTIFACTS_PATH, "strat_test_set.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(strat_test_set, f)

    housing = strat_train_set.copy()
    housing.plot(kind="scatter", x="longitude", y="latitude")
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

    corr_matrix = housing.corr(numeric_only=True)
    corr_matrix["median_house_value"].sort_values(ascending=False)
    housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
    housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
    housing["population_per_household"] = housing["population"] / housing["households"]

    housing = strat_train_set.drop(
        "median_house_value", axis=1
    )  # drop labels for training set
    housing_labels = strat_train_set["median_house_value"].copy()

    output_file = os.path.join(ARTIFACTS_PATH, "housing_labels.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(housing_labels, f)

    imputer = SimpleImputer(strategy="median")

    housing_num = housing.drop("ocean_proximity", axis=1)

    imputer.fit(housing_num)
    X = imputer.transform(housing_num)

    housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing.index)
    housing_tr["rooms_per_household"] = (
        housing_tr["total_rooms"] / housing_tr["households"]
    )
    housing_tr["bedrooms_per_room"] = (
        housing_tr["total_bedrooms"] / housing_tr["total_rooms"]
    )
    housing_tr["population_per_household"] = (
        housing_tr["population"] / housing_tr["households"]
    )

    housing_cat = housing[["ocean_proximity"]]
    housing_prepared = housing_tr.join(pd.get_dummies(housing_cat, drop_first=True))

    output_file = os.path.join(ARTIFACTS_PATH, "housing_prepared.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(housing_prepared, f)

    X_test = strat_test_set.drop("median_house_value", axis=1)
    y_test = strat_test_set["median_house_value"].copy()

    output_file = os.path.join(ARTIFACTS_PATH, "y_test.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(y_test, f)

    X_test_num = X_test.drop("ocean_proximity", axis=1)
    X_test_prepared = imputer.transform(X_test_num)
    X_test_prepared = pd.DataFrame(
        X_test_prepared, columns=X_test_num.columns, index=X_test.index
    )
    X_test_prepared["rooms_per_household"] = (
        X_test_prepared["total_rooms"] / X_test_prepared["households"]
    )
    X_test_prepared["bedrooms_per_room"] = (
        X_test_prepared["total_bedrooms"] / X_test_prepared["total_rooms"]
    )
    X_test_prepared["population_per_household"] = (
        X_test_prepared["population"] / X_test_prepared["households"]
    )

    X_test_cat = X_test[["ocean_proximity"]]
    X_test_prepared = X_test_prepared.join(pd.get_dummies(X_test_cat, drop_first=True))

    output_file = os.path.join(ARTIFACTS_PATH, "X_test_prepared.pkl")
    with open(output_file, "wb") as f:
        pickle.dump(X_test_prepared, f)


data_transformation()
logging.info("Ingesting Data was successful")
