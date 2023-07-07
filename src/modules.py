import os
import tarfile

import pandas as pd
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing").replace("\\", "/")
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
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz").replace("\\", "/")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(download_root=DOWNLOAD_ROOT, housing_path=HOUSING_PATH):
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
    csv_path = os.path.join(download_root, housing_path, "housing.csv").replace(
        "\\", "/"
    )
    return pd.read_csv(csv_path)


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
